"""One-machine 2..8 GPU launcher using private Ray and verified native storage."""
import argparse
import json
import os
from pathlib import Path
import re
try:
    import resource
except ImportError:
    resource = None
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT
sys.path.insert(0,str(ROOT))
from response_rs.config import PROFILES, render as render_method


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def available_port():
    # Ray canonicalizes loopback to the host IP. Reserve across interfaces so
    # an existing host-IP listener is also excluded when choosing a port.
    with socket.socket() as s:s.bind(('0.0.0.0',0));return s.getsockname()[1]


def environment(plan,cpu=True):
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONUTF8='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
             PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             RAY_USAGE_STATS_ENABLED='0',VLLM_NO_USAGE_STATS='1',
             RAY_memory_usage_threshold='0.95',
             WEBSHOP_ASSET_ROOT=plan.get('webshop_assets', str(ROOT/'assets/webshop')),
             ALFWORLD_DATA=plan.get('alfworld_data',os.environ.get('ALFWORLD_DATA','')), 
             WEBSHOP_RUNTIME_ROOT=plan['short']+'/webshop',
             WEBSHOP_RUNTIME_LINK=plan['short']+'/ws',
             WEBSHOP_DURABLE_PLAN=plan['root']+'/plan.json')
    env.pop('RAY_ADDRESS',None)
    # All ranks managed here share a host; honor caller-supplied interfaces.
    env.setdefault('NCCL_SOCKET_IFNAME','lo')
    env.setdefault('GLOO_SOCKET_IFNAME','lo')
    env.setdefault('NCCL_IB_DISABLE','1')
    for key,relative in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','HF_HOME':'hf',
        'HF_HUB_CACHE':'hf/hub','HF_DATASETS_CACHE':'hf/datasets',
        'XDG_CACHE_HOME':'cache','TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor',
        'CUDA_CACHE_PATH':'cuda','TORCH_HOME':'torch','VLLM_RPC_BASE_PATH':'tmp',
        'VLLM_CACHE_ROOT':'vllm'}.items():
        path=Path(plan['short'])/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    env['WEBSHOP_SPACY_MODEL']=os.environ.get('WEBSHOP_SPACY_MODEL','en_core_web_sm')
    prefix=Path(sys.executable).resolve().parent.parent
    env['PATH']=str(prefix/'bin')+':'+env.get('PATH','')
    for key,name in [('CC','x86_64-conda-linux-gnu-cc'),('CXX','x86_64-conda-linux-gnu-c++')]:
        if (prefix/'bin'/name).is_file():env[key]=str(prefix/'bin'/name)
    lib=prefix/'lib';env['LD_LIBRARY_PATH']=str(lib)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if (lib/'libstdc++.so.6').is_file():env['LD_PRELOAD']=str(lib/'libstdc++.so.6')
    java=str(prefix/'bin/java') if (prefix/'bin/java').is_file() else os.environ.get('JAVA_HOME') or shutil.which('java')
    if java:
        value=Path(java)
        home=value if value.is_dir() else value.resolve().parent.parent
        env.update(JAVA_HOME=str(home),PATH=str(home/'bin')+':'+env.get('PATH',''))
    env['CUDA_VISIBLE_DEVICES']='' if cpu else ','.join(map(str,plan['gpus']))
    return env


def _complete_checkpoint(run):
    storage=run/'checkpoints';pointer=storage/'latest_durable_checkpoint.json'
    value=json.loads(pointer.read_text()) if pointer.is_file() else None
    if value and Path(value['checkpoint']).parent!=storage:
        raise RuntimeError('Checkpoint pointer is outside this run')
    folders=[folder for folder in storage.glob('global_step_*')
             if re.fullmatch(r'global_step_[0-9]+',folder.name) and (folder/'STORAGE_ARCHIVE_VERIFIED.json').is_file()]
    if value:folders.append(Path(value['checkpoint']))
    if not folders:return None
    # A crash can occur after the complete receipt fsync but before the latest
    # pointer replace. The newer complete state must win, without rolling back.
    folder=max(folders,key=lambda p:int(p.name.removeprefix('global_step_')))
    step=int(folder.name.removeprefix('global_step_'))
    assert not folder.is_symlink()
    receipt=json.loads((folder/'STORAGE_ARCHIVE_VERIFIED.json').read_text())
    assert receipt['step']==step and (folder/'CHECKPOINT_COMPLETE').read_text().strip()==str(step)
    assert receipt['all_bytes_equal'] and receipt['fsync']
    for row in receipt['files']:
        file=folder/row['name']
        assert file.resolve().is_relative_to(folder.resolve()) and file.stat().st_size==row['bytes']
    return dict(step=step,checkpoint=str(folder),time=receipt.get('time'))


def complete_checkpoint(run):
    for attempt in range(3):
        try:return _complete_checkpoint(run)
        except FileNotFoundError:
            # A new commit may rotate the old point after this read began.
            # Re-read the current receipt; never fall back to an older state.
            if attempt==2:raise


def retain_incomplete_saves(plan):
    storage=Path(plan['archive_storage']);run=Path(plan['run_root']);retained=[]
    destination=run/'retained_partials'/Path(plan['root']).name
    for folder in storage.glob('global_step_*'):
        if not re.fullmatch(r'global_step_[0-9]+',folder.name):continue
        if int(folder.name.removeprefix('global_step_'))<=plan['source_step']:continue
        assert not (folder/'STORAGE_ARCHIVE_VERIFIED.json').exists()
        assert folder.is_dir() and not folder.is_symlink() and folder.parent.resolve()==storage.resolve()
        target=destination/folder.name
        assert target.resolve().is_relative_to(run.resolve()) and not target.exists()
        destination.mkdir(parents=True,exist_ok=True)
        folder.rename(target)
        retained.append(str(target))
    plan['retained_partial_checkpoints']=retained
    save(Path(plan['root'])/'retained_partial_checkpoints.json',dict(source_step=plan['source_step'],retained=retained))
    save(Path(plan['root'])/'plan.json',plan)
    return retained


def status(run):
    import psutil
    result=dict(run=str(run),latest_complete=complete_checkpoint(run))
    points={}
    for attempt in sorted((run/'attempts').glob('attempt_*')) if (run/'attempts').is_dir() else []:
        if (attempt/'launch.json').is_file():
            record=json.loads((attempt/'launch.json').read_text())
            try:
                proc=psutil.Process(record['pid'])
                alive=abs(proc.create_time()-record['create_time'])<.02 and proc.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:alive=False
            result.update(attempt=str(attempt),active=alive)
            driver_path=attempt/'driver_identity.json'
            if driver_path.is_file():
                driver_id=json.loads(driver_path.read_text())
                try:
                    driver=psutil.Process(driver_id['pid'])
                    driver_alive=abs(driver.create_time()-driver_id['create_time'])<.02 and driver.status()!=psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:driver_alive=False
                result.update(driver_alive=driver_alive,active=alive or driver_alive)
            result.pop('exit',None)
        if (attempt/'exit.json').is_file():result['exit']=json.loads((attempt/'exit.json').read_text())
        log=attempt/'train.log'
        if log.is_file():
            for line in log.read_text(errors='replace').splitlines():
                values={key:float(value) for key,value in re.findall(r'([A-Za-z0-9_/]+):(-?[0-9]+(?:\.[0-9]+)?)',line)}
                if 'training/global_step' in values:points[int(values['training/global_step'])]=values
    last=(result.get('latest_complete') or {}).get('step',0)
    complete=[(step,values) for step,values in sorted(points.items()) if step<=last and 'episode/success_rate' in values]
    window=complete[-10:]
    if window:
        result.update(latest_train_success_rate_approx=window[-1][1]['episode/success_rate'],
            recent_complete_step_range=[window[0][0],window[-1][0]],
            recent_train_mean_approx=sum(row[1]['episode/success_rate'] for row in window)/len(window))
    validation=[(step,value) for step,value in sorted(points.items()) if 'val/success_rate' in value]
    if validation:result['last_fixed128_validation_approx']=dict(step=validation[-1][0],success_rate=validation[-1][1]['val/success_rate'])
    return result


def select_gpus(args):
    if args.gpus!='auto':
        values=[int(value) for value in args.gpus.split(',')]
        if not 2<=len(values)<=8 or len(set(values))!=len(values) or any(value<0 for value in values):raise ValueError('Use 2..8 distinct nonnegative GPU IDs')
        if args.num_gpus and len(values)!=args.num_gpus:raise ValueError('--gpus and --num-gpus disagree')
        return values
    command=['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    candidates=[]
    for line in result.stdout.splitlines():
        index,free,util=[int(value.strip()) for value in line.split(',')]
        if free>=48000:candidates.append((util,-free,index))
    number=args.num_gpus or min(8,len(candidates))
    if not 2<=number<=8 or len(candidates)<number:
        raise RuntimeError('Not enough GPUs with startup memory. Free the GPUs or select --gpus explicitly; other jobs are never stopped.')
    return [row[2] for row in sorted(candidates)[:number]]


def make_plan(run,gpus,latest,total_steps,args):
    run.mkdir(parents=True,exist_ok=True);(run/'checkpoints').mkdir(exist_ok=True)
    attempts=run/'attempts';attempts.mkdir(exist_ok=True)
    n=len(list(attempts.glob('attempt_*')))+1
    attempt=attempts/f'attempt_{n:03d}';attempt.mkdir()
    short=Path('/tmp')/('ws3b_'+str(os.getuid())+'_'+uuid.uuid4().hex[:8]);short.mkdir()
    for name in ['output','receiver','first_verifier']:(attempt/name).mkdir()
    token=attempt/'archive_token';token.write_text(secrets.token_urlsafe(32));token.chmod(0o600)
    prior=json.loads((run/'run.json').read_text()) if (run/'run.json').is_file() else None
    if latest and prior and prior['world_size']!=len(gpus):
        raise RuntimeError('Resume requires the saved physical GPU count. Use the documented native-to-portable conversion before changing it; never rebuild Adam.')
    step=latest['step'] if latest else 0
    plan=dict(run_id=run.name,root=str(attempt),run_root=str(run),short=str(short),
              repo=str(CODE),output=str(attempt/'output'),python=sys.executable,model=str(args.model.resolve()),profile=args.config,
              alfworld_data=str(args.alfworld_data.resolve()) if args.alfworld_data else os.environ.get('ALFWORLD_DATA',''),
              webshop_assets=str(args.webshop_assets.resolve()) if args.webshop_assets else os.environ.get('WEBSHOP_ASSET_ROOT',str(ROOT/'assets/webshop')),
              vllm_memory_utilization=args.vllm_memory_utilization,
              world_size=len(gpus),logical_world_size=8,gpus=gpus,gpu_groups=[len(gpus)],
              nodes=[dict(host=socket.gethostname(),ip='127.0.0.1',gpus=gpus)],
              source_ip='127.0.0.1',source_ips=['127.0.0.1'],source_step=step,
              resume_source_step=step,resume_from_path=latest['checkpoint'] if latest else None,
              resource_changes=dict(release_verified_tmpfs_portable_payloads_after_load=False),
              steps=total_steps,direct_native_save=True,reserve_bytes=80_000_000_000,
              environment_state_file='alfworld_env.pt' if args.config.startswith('alfworld') else 'webshop_env.pt',archive_host='localhost',archive_ip='127.0.0.1',
              archive_http_port=available_port(),archive_storage=str(run/'checkpoints'),
              archive_service_root=str(attempt/'receiver'),archive_token_file=str(token),
              first_verifier_root=str(run/'first_verifier'),resume_verifier_root=str(attempt/'resume_verifier'),
              ray_port=available_port())
    Path(plan['first_verifier_root']).mkdir(exist_ok=True)
    Path(plan['resume_verifier_root']).mkdir(exist_ok=True)
    save(attempt/'plan.json',plan);save(run/'run.json',dict(world_size=len(gpus),total_steps=total_steps,latest_attempt=str(attempt)))
    return plan


def render_config(plan):
    import yaml
    cfg=render_method(plan)
    env=environment(plan)
    runtime={key:value for key,value in env.items() if key in [
        'PYTHONPATH','PYTHONUTF8','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JAVA_HOME',
        'PATH','CC','CXX','LD_LIBRARY_PATH','LD_PRELOAD','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE',
        'TOKENIZERS_PARALLELISM','RAY_memory_usage_threshold','RAY_USAGE_STATS_ENABLED','VLLM_NO_USAGE_STATS']
        or key.startswith(('WEBSHOP_','HF_','TMP','XDG_','TRITON_','TORCH','CUDA_CACHE','VLLM_','NCCL_','GLOO_','TRAINING_'))}
    cfg['ray_init']=dict(address='127.0.0.1:'+str(plan['ray_port']),namespace=plan['run_id'],
        runtime_env=dict(env_vars=runtime,worker_process_setup_hook='webshop_runtime.setup_worker'))
    path=Path(plan['root'])/'config/response_rs.yaml';path.parent.mkdir()
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return path


def discover_private_node(plan):
    """Ray canonicalizes loopback to the host address; use its actual resource."""
    script="""
import json,os,ray
ray.init(address=os.environ['WS3B_RAY_ADDRESS'],log_to_driver=False)
nodes=[n for n in ray.nodes() if n['Alive']]
assert len(nodes)==1,nodes
node=nodes[0]
print('PRIVATE_NODE='+json.dumps(dict(ip=node['NodeManagerAddress'],node_id=node['NodeID'],resources=node['Resources'])),flush=True)
ray.shutdown()
"""
    env=environment(plan);env['WS3B_RAY_ADDRESS']='127.0.0.1:'+str(plan['ray_port'])
    result=subprocess.run([sys.executable,'-X','utf8','-c',script],cwd=CODE,env=env,
                          capture_output=True,text=True,check=True,timeout=90)
    return json.loads(next(line.partition('=')[2] for line in result.stdout.splitlines() if line.startswith('PRIVATE_NODE=')))


def stop_identity(record):
    import psutil
    try:
        proc=psutil.Process(record['pid'])
        if abs(proc.create_time()-record['create_time'])<.02:
            proc.terminate()
            try:proc.wait(timeout=20)
            except psutil.TimeoutExpired:proc.kill();proc.wait(timeout=10)
    except psutil.NoSuchProcess:pass


def retire_private_ray(short):
    import psutil
    pattern=str(Path(short)/'ray');rows=[]
    for proc in psutil.process_iter(['pid','create_time','cmdline']):
        try:
            if proc.pid!=os.getpid() and pattern in ' '.join(proc.info['cmdline'] or []):
                rows.append(dict(pid=proc.pid,create_time=proc.info['create_time']))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    for row in rows:stop_identity(row)
    return rows


def start_update_observer(plan,reference,env):
    import psutil
    step=plan['source_step']+1
    proof_root=Path(plan['resume_verifier_root'] if plan['source_step'] else plan['first_verifier_root'])
    receipt='resume_real_update_verified.json' if plan['source_step'] else 'first_real_update_verified.json'
    source=Path(plan['archive_storage'])/f'global_step_{step}/STORAGE_ARCHIVE_VERIFIED.json'
    if not source.is_file() or (proof_root/receipt).is_file():return None,None,None
    check=dict(plan,root=str(proof_root),output=plan['archive_storage'])
    path=proof_root/'observer_plan.json';save(path,check)
    log=(Path(plan['root'])/'native_update_observer.log').open('x')
    command=[sys.executable,'-X','utf8',str(ROOT/'tools/verify_native_update.py'),
        '--plan',str(path),'--step',str(step),'--receipt-name',receipt]
    if reference:command+=['--reference-samples',str(reference)]
    observer=subprocess.Popen(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
    identity=dict(pid=observer.pid,create_time=psutil.Process(observer.pid).create_time())
    save(Path(plan['root'])/'observer_identity.json',identity)
    return observer,log,identity


def supervise(plan):
    import psutil
    attempt=Path(plan['root']);env=environment(plan)
    (attempt/'launch.json').write_text(json.dumps(dict(pid=os.getpid(),create_time=psutil.Process().create_time(),time=time.time())))
    receiver_log=(attempt/'receiver.log').open('x')
    receiver=subprocess.Popen([sys.executable,'-X','utf8',str(CODE/'durable_native_store.py'),'--plan',str(attempt/'plan.json')],
                               cwd=CODE,env=env,stdout=receiver_log,stderr=subprocess.STDOUT)
    receiver_id=dict(pid=receiver.pid,create_time=psutil.Process(receiver.pid).create_time())
    save(attempt/'receiver_identity.json',receiver_id)
    driver=None;observer=None;observer_id=None;observer_log=None;exit_code=1;error=None
    try:
        reference=None
        if plan['source_step']:
            reference=attempt/'resume_verifier/source_samples.pt'
            capture=dict(plan,root=plan['resume_verifier_root'],output=plan['archive_storage'])
            capture_path=Path(plan['resume_verifier_root'])/'source_plan.json';save(capture_path,capture)
            with (attempt/'resume_source_capture.log').open('x') as stream:
                subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/verify_native_update.py'),
                    '--plan',str(capture_path),'--step',str(plan['source_step']),
                    '--capture-samples',str(reference)],cwd=CODE,env=env,
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
        import http.client
        ready=False
        for _ in range(60):
            if receiver.poll() is not None:raise RuntimeError('Native checkpoint receiver exited')
            connection=http.client.HTTPConnection('127.0.0.1',plan['archive_http_port'],timeout=1)
            try:
                connection.request('GET','/status',headers={'Authorization':'Bearer '+Path(plan['archive_token_file']).read_text()})
                response=connection.getresponse();response.read();ready=response.status==200
            except OSError:pass
            finally:connection.close()
            if ready:break
            time.sleep(.5)
        if not ready:raise RuntimeError('Checkpoint receiver did not become ready')
        ray_command=[sys.executable,'-X','utf8','-m','ray.scripts.scripts','start','--head',
            '--node-ip-address=127.0.0.1','--port='+str(plan['ray_port']),
            '--num-gpus='+str(plan['world_size']),'--num-cpus='+str(os.cpu_count() or 32),
            '--include-dashboard=false','--dashboard-port='+str(available_port()),
            '--dashboard-agent-listen-port='+str(available_port()),
            '--disable-usage-stats','--temp-dir='+plan['short']+'/ray']
        with (attempt/'ray_start.log').open('x') as stream:
            subprocess.run(ray_command,env=environment(plan,cpu=False),stdout=stream,stderr=subprocess.STDOUT,check=True)
        node=discover_private_node(plan)
        assert node['resources'].get('GPU',0)==plan['world_size'],node
        plan['node_ip']=node['ip'];plan['nodes'][0]['ip']=node['ip']
        save(attempt/'ray_node_observed.json',node);save(attempt/'plan.json',plan)
        config=render_config(plan)
        command=[sys.executable,'-X','utf8','-u','-m','verl.trainer.main_grsd_reflect',
                 '--config-path',str(config.parent),'--config-name',config.stem]
        with (attempt/'train.log').open('x') as log:
            driver=subprocess.Popen(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            save(attempt/'driver_identity.json',dict(pid=driver.pid,create_time=psutil.Process(driver.pid).create_time()))
            while driver.poll() is None:
                if observer is None:
                    observer,observer_log,observer_id=start_update_observer(plan,reference,env)
                save(attempt/'status.json',dict(time=time.time(),driver_alive=True,world_size=plan['world_size'],
                    latest_complete=complete_checkpoint(Path(plan['run_root'])),
                    first_update_observer_exit=observer.poll() if observer else None))
                time.sleep(10)
            exit_code=driver.returncode
        # A one-step trial can finish between supervisor polls. Its saved state
        # still requires the same real observer before reporting completion.
        if observer is None:observer,observer_log,observer_id=start_update_observer(plan,reference,env)
        if observer is not None:
            code=observer.wait()
            save(attempt/'observer_exit.json',dict(time=time.time(),exit_code=code))
            if code:error='Native update observer exited '+str(code)+'; preserve its raw logs and do not claim verification.'
    except BaseException as exc:
        error=repr(exc)
        if driver and driver.poll() is None:
            driver.terminate();driver.wait()
        raise
    finally:
        if observer is not None and observer.poll() is None:
            stop_identity(observer_id)
            save(attempt/'observer_exit.json',dict(time=time.time(),exit_code=observer.poll(),
                note='Stopped this run observer while preserving the failed attempt and checkpoint'))
        ray_logs=Path(plan['short'])/'ray/session_latest/logs'
        try:
            if ray_logs.is_dir():shutil.copytree(ray_logs,attempt/'ray_logs_final')
        except OSError as exc:
            error=(error+'; ' if error else '')+'Ray log preservation: '+repr(exc)
        retired=retire_private_ray(plan['short'])
        stop_identity(receiver_id);receiver_log.close()
        if observer_log:observer_log.close()
        save(attempt/'exit.json',dict(time=time.time(),exit_code=exit_code,error=error,
            supervisor_exit_code=exit_code if not error else exit_code or 1,
            latest_complete=complete_checkpoint(Path(plan['run_root'])),private_ray_retired=retired,receiver_retired=True))
    return exit_code if not error else exit_code or 1


def main():
    parser=argparse.ArgumentParser(description='Action-command Teacher (0.01) + joint R/S on 2..8 GPUs')
    parser.add_argument('--config',choices=PROFILES,default='alfworld_3b')
    parser.add_argument('--model',type=Path,help='Matching local SFT checkpoint600')
    parser.add_argument('--alfworld-data',type=Path)
    parser.add_argument('--webshop-assets',type=Path)
    parser.add_argument('--vllm-memory-utilization',type=float,help='Resource budget only; does not change the method')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--gpus',default='auto',help='auto or physical IDs, e.g. 0,2,3')
    parser.add_argument('--num-gpus',type=int,choices=range(2,9))
    parser.add_argument('--run-dir',type=Path,default=None)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--smoke-steps',type=int)
    parser.add_argument('--supervise',type=Path)
    args=parser.parse_args()
    if args.supervise:return supervise(json.loads(args.supervise.read_text()))
    run=(args.run_dir or ROOT/'runs'/args.config).resolve()
    if args.status:print(json.dumps(status(run),indent=2));return 0
    if args.model is None:parser.error('--model is required for training and configuration rendering')
    if args.dry_run:
        import yaml
        gpus=[int(v) for v in args.gpus.split(',')] if args.gpus!='auto' else list(range(args.num_gpus or 8))
        if not 2<=len(gpus)<=8 or len(set(gpus))!=len(gpus) or min(gpus)<0:raise ValueError('Choose 2..8 distinct GPUs')
        plan=dict(profile=args.config,model=str(args.model.resolve()),run_id=run.name,
                  output=str(run/'attempts/attempt_001/output'),world_size=len(gpus),steps=args.smoke_steps or 150,
                  resume_from_path=None,vllm_memory_utilization=args.vllm_memory_utilization)
        print(yaml.safe_dump(render_method(plan),sort_keys=False));return 0
    if os.name!='posix':raise RuntimeError('GPU training requires Linux; --dry-run also works on Windows')
    import fcntl
    if not args.model.is_dir():raise FileNotFoundError(args.model)
    model_config=json.loads((args.model/'config.json').read_text())
    expected_hidden=2048 if args.config.endswith('3b') else 3584
    if model_config.get('hidden_size')!=expected_hidden:raise ValueError('Model size does not match --config')
    if args.config.startswith('alfworld') and not (args.alfworld_data or os.environ.get('ALFWORLD_DATA')):
        parser.error('ALFWorld requires --alfworld-data or ALFWORLD_DATA')
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/'run.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps(status(run),indent=2))
        raise RuntimeError('This run already has a supervisor; no second writer was started.')
    current=status(run)
    if current.get('active'):
        print(json.dumps(current,indent=2));raise RuntimeError('This run is already active; no duplicate trainer was submitted.')
    if current.get('attempt') and not (Path(current['attempt'])/'exit.json').is_file():
        raise RuntimeError('The previous supervisor has no exit receipt. Inspect its exact driver/Ray/service identities before restarting; no new trainer was submitted.')
    latest=complete_checkpoint(run)
    if latest and not args.resume:raise RuntimeError('Saved state exists. Use --resume or select a new --run-dir.')
    if args.resume and not latest:raise RuntimeError('No complete native checkpoint exists; do not pretend to resume from SFT weights.')
    total=args.smoke_steps or 150
    if latest and latest['step']>=total:print('This run already reached its target.');return 0
    if args.smoke_steps and not 1<=args.smoke_steps<=5:raise ValueError('--smoke-steps must be 1..5')
    if shutil.disk_usage(run if run.exists() else run.parent if run.parent.exists() else ROOT).free<120_000_000_000:
        raise RuntimeError('Below the120GB startup floor. Preserve the80GB archive reserve and enough space for Model/Adam milestones; see README for3B/7B capacity estimates.')
    gpus=select_gpus(args)
    if latest:
        previous=json.loads((run/'run.json').read_text())
        if previous['world_size']!=len(gpus):raise RuntimeError('Resume with the saved GPU count; no optimizer state is rebuilt.')
        old_plan=json.loads(Path(previous['latest_attempt'],'plan.json').read_text())
        if old_plan['profile']!=args.config or Path(old_plan['model'])!=args.model.resolve():
            raise ValueError('Resume must use the saved profile and matching SFT reference model')
    plan=make_plan(run,gpus,latest,total,args)
    retain_incomplete_saves(plan)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,65536) if hard==resource.RLIM_INFINITY else min(max(soft,65536),hard),hard))
    if args.background:
        with (Path(plan['root'])/'supervisor.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-X','utf8',__file__,'--supervise',plan['root']+'/plan.json'],
                                   env=environment(plan),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                   pass_fds=(lock.fileno(),))
        print(json.dumps(dict(supervisor_pid=child.pid,run=str(run),attempt=plan['root'],gpus=gpus,logical_gpus=8),indent=2))
        return 0
    return supervise(plan)


if __name__=='__main__':raise SystemExit(main())
