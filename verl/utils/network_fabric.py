"""Apply caller-selected network settings before process-group initialization."""
import json
import os
from pathlib import Path
import subprocess


def configure_fabric():
    keys=('NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','NCCL_IB_HCA','NCCL_CROSS_NIC','NCCL_IB_DISABLE')
    network={key:os.environ[key] for key in keys if key in os.environ}
    if network.get('NCCL_SOCKET_IFNAME') and network.get('GLOO_SOCKET_IFNAME'):
        return network
    prefix=os.environ.get('TRAINING_IP_PREFIX')
    if not prefix:
        return network  # Use the standard torch/NCCL interface selection.
    interfaces = json.loads(subprocess.check_output(['ip', '-j', 'address'], text=True))
    candidates = [n['ifname'] for n in interfaces
                  if any(a.get('family') == 'inet' and a['local'].startswith(prefix)
                         for a in n.get('addr_info', []))]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one training interface for {prefix}, found {candidates}')
    interface = candidates[0]
    root=Path('/sys/class/infiniband')
    devices = [p.name for p in root.iterdir()
               if (p / 'device/net' / interface).exists()] if root.is_dir() else []
    if len(devices) > 1:
        raise RuntimeError(f'Expected at most one HCA for {interface}, found {devices}')
    network.update(NCCL_SOCKET_IFNAME='='+interface, GLOO_SOCKET_IFNAME=interface)
    if devices:network['NCCL_IB_HCA']='='+devices[0]+':1'
    os.environ.update(network)
    return network
