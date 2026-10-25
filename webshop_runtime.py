"""Portable single-host WebShop cache setup; no fixed cluster interfaces."""
import os
from pathlib import Path
import socket


def configure(root):
    runtime=Path(root).resolve()/socket.gethostname()
    runtime.mkdir(parents=True,exist_ok=True)
    link=Path(os.environ['WEBSHOP_RUNTIME_LINK'])
    if link.is_symlink():
        if link.resolve()!=runtime:raise ValueError('Runtime link belongs to another run')
    elif link.exists():raise FileExistsError(link)
    else:
        try:link.symlink_to(runtime,target_is_directory=True)
        except FileExistsError:
            if not link.is_symlink() or link.resolve()!=runtime:raise
    for name in ['tmp','cache','triton','hf','vllm']:(runtime/name).mkdir(exist_ok=True)
    return dict(TMPDIR=str(link/'tmp'),TMP=str(link/'tmp'),TEMP=str(link/'tmp'),
        VLLM_RPC_BASE_PATH=str(link/'tmp'),VLLM_CONFIG_ROOT=str(runtime/'vllm'),
        HF_HOME=str(runtime/'hf'),XDG_CACHE_HOME=str(runtime/'cache'),
        TRITON_CACHE_DIR=str(runtime/'triton'),VLLM_NO_USAGE_STATS='1')


def setup_worker():
    os.environ.update(configure(os.environ['WEBSHOP_RUNTIME_ROOT']))
