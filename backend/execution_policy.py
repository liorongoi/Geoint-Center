"""Dedicated CUDA GPUs or user-authorized Apple Silicon MPS; explicit CPU fallback."""
from functools import lru_cache
import os
import re
import platform
if platform.system()=='Darwin':os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')

def dedicated_name(name):
    if re.search(r'orin|tegra|jetson|integrated',name,re.I): return False
    return bool(re.search(r'nvidia|geforce|tesla|quadro|radeon (rx|pro w)|amd instinct|amd.*(mi[0-9]|w[0-9])',name,re.I))

@lru_cache(maxsize=1)
def torch_device():
    import torch
    if os.getenv('SPECTRASCOPE_DEVICE','auto')!='cpu' and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if dedicated_name(torch.cuda.get_device_name(i)): return torch.device(f'cuda:{i}')
    if os.getenv('SPECTRASCOPE_DEVICE','auto')!='cpu' and platform.system()=='Darwin' and hasattr(torch.backends,'mps') and torch.backends.mps.is_available():return torch.device('mps')
    return torch.device('cpu')

@lru_cache(maxsize=1)
def configure_tensorflow():
    import tensorflow as tf
    permitted=[]
    if os.getenv('SPECTRASCOPE_DEVICE','auto')!='cpu':
        for gpu in tf.config.list_physical_devices('GPU'):
            name=tf.config.experimental.get_device_details(gpu).get('device_name','')
            if dedicated_name(name): permitted.append(gpu)
    tf.config.set_visible_devices(permitted,'GPU')
    for gpu in permitted: tf.config.experimental.set_memory_growth(gpu,True)
    return tf


def onnx_providers(available):
    """Dedicated CUDA or available Apple CoreML provider; CPU fallback otherwise."""
    device=torch_device()
    if device.type=='cuda' and 'CUDAExecutionProvider' in available:
        return [('CUDAExecutionProvider',{'device_id':device.index or 0}),'CPUExecutionProvider']
    if device.type=='mps' and 'CoreMLExecutionProvider' in available:return ['CoreMLExecutionProvider','CPUExecutionProvider']
    return ['CPUExecutionProvider']


def available_ram_bytes():
    """Available host/container RAM, rather than installed capacity."""
    import platform
    import subprocess
    from pathlib import Path
    try:
        if platform.system()=='Windows':
            import ctypes
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),('available_page',ctypes.c_ulonglong),('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),('reserved',ctypes.c_ulonglong)]
            status=Memory();status.length=ctypes.sizeof(status)
            return status.available if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
        if platform.system()=='Darwin':
            import subprocess
            text=subprocess.check_output(['/usr/bin/vm_stat'],timeout=3,text=True,stderr=subprocess.DEVNULL)
            size=int(re.search(r'page size of (\d+) bytes',text)[1]);counts=[]
            for label in ('Pages free','Pages inactive','Pages speculative'):
                match=re.search(re.escape(label)+r':\s+(\d+)',text)
                if match:counts.append(int(match[1]))
            return sum(counts)*size if counts else None
        info=Path('/proc/meminfo').read_text();available=int(re.search(r'MemAvailable:\s+(\d+)',info)[1])*1024
        for limit_name,current_name in (('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')):
            try:
                limit=int(Path(limit_name).read_text());current=int(Path(current_name).read_text())
                if limit<1<<50:available=min(available,max(0,limit-current))
            except (OSError,ValueError):pass
        return available
    except (OSError,ValueError,TypeError,AttributeError,IndexError,subprocess.SubprocessError):return None
