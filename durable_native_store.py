"""Synchronous, byte-verified native checkpoints on this run's durable host."""
import argparse
from contextlib import closing
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.parse
import zipfile

_metadata_write_lock = threading.Lock()

def write(path, value):
    # HTTP handlers share error metadata. Serialize its existing atomic replace
    # so concurrent failures cannot rename another handler's temporary file.
    with _metadata_write_lock:
        temporary = path.with_name(path.name + '.partial')
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def serve(plan):
    import psutil
    storage = Path(plan['archive_storage']); storage.mkdir(parents=True, exist_ok=True)
    token = Path(plan['archive_token_file']).read_text().strip()
    class Handler(BaseHTTPRequestHandler):
        def authorized(self):
            sources = plan.get('source_ips', [plan['source_ip']])
            return self.client_address[0] in sources and self.headers.get('Authorization') == 'Bearer ' + token
        def target(self):
            parts = urllib.parse.unquote(self.path).strip('/').split('/')
            assert len(parts) > 1 and re.fullmatch(r'global_step_[0-9]+', parts[0])
            step = int(parts[0].removeprefix('global_step_'))
            assert plan.get('source_step', 0) < step <= plan['steps']
            assert all(p not in ('', '..', '.') for p in parts)
            path = storage.joinpath(*parts)
            assert path.resolve().is_relative_to(storage.resolve())
            return path, step
        def send_json(self, value, code=200):
            data = json.dumps(value).encode(); self.send_response(code)
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if not self.authorized(): self.send_error(403); return
            if self.path == '/status':
                self.send_json(dict(free=shutil.disk_usage(storage).free, time=time.time())); return
            try:
                target, _ = self.target()
                if not target.is_file(): self.send_error(404); return
                self.send_response(200); self.send_header('Content-Length', str(target.stat().st_size)); self.end_headers()
                with target.open('rb') as stream:
                    while block := stream.read(8*1024**2): self.wfile.write(block)
            except Exception as error:
                self.log_error('%r', error); self.close_connection = True
        def do_PUT(self):
            if not self.authorized(): self.send_error(403); return
            try:
                target, _ = self.target(); size = int(self.headers['Content-Length'])
                assert size >= 0 and not target.exists()
                assert shutil.disk_usage(storage).free > size + plan.get('reserve_bytes', 80*10**9)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.partial')
                with temporary.open('xb') as stream:
                    remaining = size
                    while remaining:
                        block = self.rfile.read(min(8*1024**2, remaining)); assert block
                        stream.write(block); remaining -= len(block)
                    stream.flush(); os.fsync(stream.fileno())
                temporary.replace(target)
                fd = os.open(target.parent, os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
                self.send_json(dict(saved=True), 201)
            except Exception as error:
                write(storage / 'archival_error.json', dict(time=time.time(), error=repr(error), path=self.path))
                self.send_error(409, repr(error))
        def do_POST(self):
            if not self.authorized(): self.send_error(403); return
            try:
                target, step = self.target()
                assert target.name == 'STORAGE_ARCHIVE_VERIFIED.json' and not target.exists()
                receipt = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert receipt['step'] == step and receipt['all_bytes_equal'] and receipt['fsync']
                assert (target.parent / 'CHECKPOINT_COMPLETE').read_text().strip() == str(step)
                expected = ['CHECKPOINT_COMPLETE', 'data.pt', plan.get('environment_state_file', 'webshop_env.pt')]
                expected += [f'actor/{prefix}_world_size_{plan["world_size"]}_rank_{rank}.pt'
                             for rank in range(plan['world_size']) for prefix in ['model', 'optim', 'extra_state']]
                names = {item['name'] for item in receipt['files']}
                assert set(expected).issubset(names)
                for item in receipt['files']:
                    path = target.parent / item['name']
                    assert path.resolve().is_relative_to(target.parent.resolve()) and path.stat().st_size == item['bytes']
                write(target, receipt)
                write(storage / 'latest_durable_checkpoint.json', dict(step=step, checkpoint=str(target.parent), time=time.time()))
                removed = []
                for old in storage.glob('global_step_*'):
                    if not old.is_dir() or not re.fullmatch(r'global_step_[0-9]+', old.name): continue
                    old_step = int(old.name.removeprefix('global_step_'))
                    if old_step >= step or old_step % 20 == 0: continue
                    if old_step == 1 and not (Path(plan['first_verifier_root']) / 'first_real_update_verified.json').is_file(): continue
                    if not (old / 'STORAGE_ARCHIVE_VERIFIED.json').is_file(): continue
                    assert (target.parent / 'STORAGE_ARCHIVE_VERIFIED.json').is_file()
                    assert not old.is_symlink() and old.parent.resolve() == storage.resolve()
                    shutil.rmtree(old); removed.append(old_step)
                self.send_json(dict(committed=step, removed=removed), 201)
            except Exception as error:
                write(storage / 'archival_error.json', dict(time=time.time(), error=repr(error), path=self.path))
                self.send_error(409, repr(error))
        def log_message(self, *args): pass
    server = ThreadingHTTPServer((plan['archive_ip'], plan['archive_http_port']), Handler)
    write(Path(plan['archive_service_root']) / 'service_identity.json',
          dict(pid=os.getpid(),create_time=psutil.Process().create_time(),time=time.time(),
               ip=plan['archive_ip'],port=plan['archive_http_port'],storage=str(storage)))
    server.serve_forever()


def checkpoint_raw(plan, source):
    """Keep all accumulated rollout/validation records in the latest durable point."""
    destination = source / 'attempt_raw.zip'
    if destination.exists(): return
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        root = Path(plan['root']); output = Path(plan['output'])
        for path in root.rglob('*'):
            # Portable runs nest output under the attempt directory. Archive
            # output separately below, excluding native payloads and this ZIP.
            if path.resolve().is_relative_to(output.resolve()):
                continue
            if path.is_file() and not path.is_symlink() and 'token' not in path.name.lower():
                archive.write(path, 'attempt/' + path.relative_to(root).as_posix())
        for item in output.iterdir():
            if item.name.startswith('global_step_'): continue
            for path in item.rglob('*') if item.is_dir() else [item]:
                if path.is_file() and not path.is_symlink():
                    archive.write(path, 'training/' + path.relative_to(output).as_posix())
        logs = Path(plan['short']) / 'ray/session_latest/logs'
        if logs.is_dir():
            for path in logs.rglob('*'):
                if path.is_file() and not path.is_symlink(): archive.write(path, 'ray/' + path.relative_to(logs).as_posix())
    with destination.open('r+b') as stream: os.fsync(stream.fileno())


def evict_state_cache(plan, source):
    """Release only verified local model/Adam copies; the native point is durable."""
    source=Path(source)
    assert source.parent.resolve()==Path(plan['output']).resolve() and not source.is_symlink()
    receipt=json.loads((source/'DURABLE_CHECKPOINT_VERIFIED.json').read_text())
    assert receipt['all_bytes_equal'] and receipt['fsync'] and receipt['target']==plan['archive_storage']+'/'+source.name
    removed=[]
    for item in receipt['files']:
        if not item['name'].endswith('.pt') or not (item['name'].startswith('actor/model_world_size_') or item['name'].startswith('actor/optim_world_size_')):continue
        path=source/item['name']
        assert path.parent.resolve()==(source/'actor').resolve() and not path.is_symlink()
        if path.exists():
            assert path.stat().st_size==item['bytes']
            path.unlink()
        else:
            direct=json.loads(path.with_name(path.name+'.DIRECT_SAVED.json').read_text())
            assert direct['name']==item['name'] and direct['bytes']==item['bytes']
            assert direct['all_bytes_equal'] and direct['fsync']
            assert direct['target']==receipt['target']+'/'+item['name']
        removed.append(item['name'])
    assert len(removed)==plan['world_size']*2
    write(source/'LOCAL_STATE_ARCHIVED.json',dict(time=time.time(),step=receipt['step'],target=receipt['target'],host=plan['archive_host'],removed=removed,all_bytes_equal=True,fsync=True))


def persist_checkpoint(plan, source, include_raw=True):
    source = Path(source); output = Path(plan['output'])
    step = int(source.name.removeprefix('global_step_'))
    assert source.parent.resolve() == output.resolve() and not source.is_symlink()
    assert (source / 'CHECKPOINT_COMPLETE').read_text().strip() == str(step)
    assert not (source / 'DURABLE_CHECKPOINT_VERIFIED.json').exists()
    if include_raw: checkpoint_raw(plan, source)
    token = Path(plan['archive_token_file']).read_text().strip()
    headers = {'Authorization': 'Bearer ' + token}
    def connect():
        return closing(http.client.HTTPConnection(plan['archive_ip'], plan['archive_http_port'], timeout=300))
    files = sorted([p for p in source.rglob('*') if p.is_file()], key=lambda p: (p.name == 'CHECKPOINT_COMPLETE', str(p)))
    with connect() as connection:
        connection.request('GET', '/status', headers=headers); response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())['free'] > sum(p.stat().st_size for p in files) + plan.get('reserve_bytes', 80*10**9)
    manifest = []
    if plan.get('direct_native_save'):
        for marker in source.glob('actor/*.DIRECT_SAVED.json'):
            direct=json.loads(marker.read_text())
            assert direct['step']==step and direct['all_bytes_equal'] and direct['fsync']
            assert direct['target']==plan['archive_storage']+'/'+source.name+'/'+direct['name']
            assert not (source/direct['name']).exists()
            manifest.append(dict(name=direct['name'],bytes=direct['bytes']))
        assert len(manifest)==plan['world_size']*2
        assert len({item['name'] for item in manifest})==len(manifest)
    total_files=len(files)+len(manifest)
    for path in files:
        assert not path.is_symlink()
        name = path.relative_to(source).as_posix(); size = path.stat().st_size
        url = '/' + source.name + '/' + urllib.parse.quote(name, safe='/')
        with path.open('rb') as stream, connect() as connection:
            connection.putrequest('PUT', url)
            for key, value in dict(headers, **{'Content-Length':str(size)}).items(): connection.putheader(key, value)
            connection.endheaders()
            while block := stream.read(8*1024**2): connection.send(block)
            response = connection.getresponse(); assert response.status == 201, (response.status, response.read()); response.read()
        with path.open('rb') as stream, connect() as connection:
            connection.request('GET', url, headers=headers); response = connection.getresponse()
            assert response.status == 200
            while True:
                block = stream.read(8*1024**2); saved = response.read(len(block)) if block else response.read(1)
                assert saved == block
                if not block: break
        manifest.append(dict(name=name, bytes=size))
        write(Path(plan['root']) / 'durable_save_progress.json', dict(time=time.time(),step=step,files=len(manifest),total_files=total_files,bytes=sum(x['bytes'] for x in manifest)))
    receipt = dict(step=step,files=manifest,bytes=sum(x['bytes'] for x in manifest),all_bytes_equal=True,
                   fsync=True,time=time.time(),target_host=plan['archive_host'],target=plan['archive_storage']+'/'+source.name)
    data = json.dumps(receipt).encode()
    url = '/' + source.name + '/STORAGE_ARCHIVE_VERIFIED.json'
    with connect() as connection:
        connection.request('POST', url, body=data, headers=dict(headers, **{'Content-Length':str(len(data))}))
        response = connection.getresponse(); assert response.status == 201, (response.status,response.read()); response.read()
    with connect() as connection:
        connection.request('GET', url, headers=headers); response = connection.getresponse()
        assert response.status == 200 and json.loads(response.read()) == receipt
    write(source / 'DURABLE_CHECKPOINT_VERIFIED.json', receipt)
    # First resumed native point stays readable until its existing CPU verifier
    # finishes; that verifier then releases the same verified cache. Later points
    # release their large local copies immediately after the durable commit.
    if plan.get('direct_native_save') or step != plan.get('resume_source_step',36)+1:
        evict_state_cache(plan,source)
    # A restored source can be portable or a verified same-world native copy.
    # Release either only after the newer native commit has been read back.
    for old in output.glob('global_step_*'):
        if not old.is_dir() or not re.fullmatch(r'global_step_[0-9]+', old.name): continue
        old_step = int(old.name.removeprefix('global_step_'))
        if old_step >= step: continue
        if old_step == plan.get('resume_source_step'):
            if not (old / 'NATIVE_PORTABLE_VERIFIED.json').is_file():
                old_receipt = json.loads((old / 'DURABLE_CHECKPOINT_VERIFIED.json').read_text())
                assert old_receipt['step'] == old_step
                assert old_receipt['all_bytes_equal'] and old_receipt['fsync']
                assert old_receipt['target'] == plan['archive_storage'] + '/' + old.name
                assert (old / 'CHECKPOINT_COMPLETE').read_text().strip() == str(old_step)
        elif not (old / 'DURABLE_CHECKPOINT_VERIFIED.json').is_file():
            continue
        assert old.parent.resolve() == output.resolve() and not old.is_symlink()
        if old_step % 20 == 0:
            if not (old / 'DURABLE_CHECKPOINT_VERIFIED.json').is_file():
                # A portable resume milestone is retained with its original
                # proof; it has no native receipt for this attempt's archive.
                portable = json.loads((old / 'NATIVE_PORTABLE_VERIFIED.json').read_text())
                assert old_step == plan.get('resume_source_step')
                assert portable['step'] == old_step and portable['all_values_equal'] and portable['fsync']
                continue
            old_receipt = json.loads((old / 'DURABLE_CHECKPOINT_VERIFIED.json').read_text())
            write(output / (old.name + '.ARCHIVED.json'), old_receipt)
        shutil.rmtree(old)
    print('[Durable checkpoint]', step, receipt['bytes'], flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan',type=Path,required=True)
    serve(json.loads(parser.parse_args().plan.read_text()))
