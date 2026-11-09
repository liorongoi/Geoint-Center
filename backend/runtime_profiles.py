"""Portable hardware hints and conservative LLM profiles (standard library only)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import subprocess
import shutil
import re
from functools import lru_cache

PROFILES = {
    'compact': {'label':'Compact · 16 GB','model':'qwen3.5:2b','context':4096,'tokens':768,'keep_alive':0,'minimum_ram_gb':12},
    'balanced': {'label':'Balanced · 24–32 GB','model':'qwen3.5:4b','context':8192,'tokens':1024,'keep_alive':'30s','minimum_ram_gb':24},
    'quality': {'label':'Higher quality · 48 GB+','model':'qwen3.5:9b','context':16384,'tokens':1024,'keep_alive':'1m','minimum_ram_gb':48},
}

def dedicated_gpus():
    devices=[]
    command=shutil.which('nvidia-smi')
    if command:
        try:
            output=subprocess.check_output([command,'--query-gpu=name,memory.total','--format=csv,noheader,nounits'],timeout=4,text=True,stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                name,mib=line.rsplit(',',1)
                if not re.search(r'orin|tegra|jetson',name,re.I): devices.append({'name':name.strip(),'vram_gb':round(float(mib)/1024,1),'vendor':'NVIDIA','detected_by':'nvidia-smi'})
        except (OSError,ValueError,subprocess.SubprocessError): pass
    if not devices and platform.system()=='Windows':
        try:
            output=subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command','Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress'],timeout=6,text=True,stderr=subprocess.DEVNULL)
            values=json.loads(output);values=values if isinstance(values,list) else [values]
            for d in values:
                name=d.get('Name','')
                if re.search(r'NVIDIA|Radeon (RX|Pro W)|AMD Instinct|Intel.*Arc.*A[0-9]',name,re.I):
                    devices.append({'name':name,'vram_gb':None,'vendor':'NVIDIA' if 'NVIDIA' in name.upper() else 'other','detected_by':'Windows display inventory; VRAM unverified'})
        except (OSError,ValueError,subprocess.SubprocessError): pass
    if not devices and platform.system()=='Darwin' and platform.machine()=='arm64':
        devices.append({'name':'Apple Silicon GPU','vram_gb':None,'vendor':'Apple','unified_memory':True,'detected_by':'native Apple Silicon host; framework support checked by each adapter'})
    if not devices and platform.system()=='Darwin' and platform.machine()!='arm64':
        try:
            output=json.loads(subprocess.check_output(['system_profiler','SPDisplaysDataType','-json'],timeout=8,text=True,stderr=subprocess.DEVNULL))
            for d in output.get('SPDisplaysDataType',[]):
                name=d.get('sppci_model','')
                if re.search(r'Radeon|NVIDIA|GeForce|Quadro',name,re.I): devices.append({'name':name,'vram_gb':None,'vendor':'AMD' if 'Radeon' in name else 'NVIDIA','detected_by':'macOS display inventory'})
        except (OSError,ValueError,subprocess.SubprocessError): pass
    if not devices and platform.system()=='Linux' and shutil.which('lspci'):
        try:
            text=subprocess.check_output(['lspci','-mm'],timeout=3,text=True,stderr=subprocess.DEVNULL)
            for line in text.splitlines():
                if re.search(r'VGA|3D controller|Display controller',line,re.I) and re.search(r'Radeon (RX|Pro W)|AMD Instinct|Arc.*A[0-9]',line,re.I):
                    devices.append({'name':line[:200],'vram_gb':None,'vendor':'other','detected_by':'PCI display inventory; runtime support unverified'})
        except (OSError,subprocess.SubprocessError): pass
    return devices

def native_hardware(detect_gpu=True):
    system=platform.system(); total=None
    try:
        if system=='Darwin': total=int(subprocess.check_output(['sysctl','-n','hw.memsize'],timeout=3,text=True,stderr=subprocess.DEVNULL).strip())
        elif system=='Windows':
            import ctypes
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),('available_page',ctypes.c_ulonglong),('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),('reserved',ctypes.c_ulonglong)]
            status=Memory();status.length=ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)): total=status.total
        else: total=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (OSError,ValueError,subprocess.SubprocessError,AttributeError): pass
    container=Path('/.dockerenv').exists()
    if container:
        for p in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            try:
                limit=int(Path(p).read_text().strip())
                if limit<1<<50: total=min(total or limit,limit)
            except (ValueError,OSError): pass
    return {'os':system,'architecture':platform.machine(),'ram_gb':round(total/1024**3,1) if total else None,
            'source':'container memory limit (host RAM unknown)' if container else 'native host detection','host_detected':not container,'dedicated_gpu_required':False,'dedicated_gpus':dedicated_gpus() if detect_gpu else []}

PERFORMANCE_PROFILES = {
    'compact': {'label':'Conservative · 16 GB','max_raster_elements':40_000_000,'gdal_cache_mb':64},
    'balanced': {'label':'Balanced · 24–32 GB','max_raster_elements':80_000_000,'gdal_cache_mb':128},
    'extended': {'label':'Extended · 48 GB+','max_raster_elements':160_000_000,'gdal_cache_mb':256},
}

@lru_cache(maxsize=1)
def _processing_hardware():
    # Processing runs inside this process/container: a host inventory file must
    # never grant memory beyond the container's effective limit.
    return native_hardware(detect_gpu=False)

def performance_profile(detected=None, mode=None):
    h=detected if detected is not None else _processing_hardware()
    ram=h.get('ram_gb') or 0
    # Windows can report slightly less usable RAM than the installed capacity.
    capacity='extended' if ram>=47.5 else 'balanced' if ram>=23.5 else 'compact'
    # A host launcher may reserve OS/LLM memory and give this container only
    # part of a larger machine. Require both the host tier and enough actual
    # container memory; never let a host hint bypass the effective limit.
    if not h.get('host_detected',True):
        try:host_ram=float(os.getenv('SPECTRASCOPE_HOST_RAM_GB','0'))
        except ValueError:host_ram=0
        if host_ram>=47.5 and ram>=20:capacity='extended'
        elif host_ram>=23.5 and ram>=8:capacity='balanced'
    requested=mode if mode is not None else os.getenv('SPECTRASCOPE_MEMORY_PROFILE','auto')
    requested={'low':'compact','standard':'balanced','quality':'extended'}.get(requested,requested)
    if requested=='auto':requested=capacity
    if requested not in PERFORMANCE_PROFILES:raise ValueError('Unknown processing memory profile: '+requested)
    order=list(PERFORMANCE_PROFILES)
    chosen=order[min(order.index(requested),order.index(capacity))]
    profile={'id':chosen,**PERFORMANCE_PROFILES[chosen],'effective_ram_gb':h.get('ram_gb'),'source':h.get('source'),
             'requested':requested,'clamped':chosen!=requested,'automatic_resolution_change':False}
    override=os.getenv('SPECTRASCOPE_MAX_RASTER_ELEMENTS')
    if override:
        value=int(override)
        if value<=0:raise ValueError('SPECTRASCOPE_MAX_RASTER_ELEMENTS must be positive')
        profile['max_raster_elements']=min(value,profile['max_raster_elements'])
    # A single local accelerator (including unified-memory Apple GPUs)
    # needs one shared analysis admission slot. Graph branches may queue
    # together; explicitly CPU-only extended hosts can use two slots.
    profile['local_analysis_slots']=2 if chosen=='extended' and os.getenv('SPECTRASCOPE_DEVICE','auto')=='cpu' else 1
    return profile

def docker_memory_plan(ram):
    """Host-side limits, in GiB; reserves stay outside the Docker VM."""
    if not isinstance(ram,(float,int)) or not 12<=ram<=4096:
        raise ValueError('Detect at least 12 GB host RAM before configuring local Docker analysis')
    if ram>=47.5:
        vm=int(ram*.60);app=vm-4;maps=2;threads=4;profile='extended'
    elif ram>=31.5:
        vm=18;app=14;maps=1;threads=4;profile='balanced'
    elif ram>=23.5:
        vm=12;app=9;maps=1;threads=3;profile='balanced'
    elif ram>=15.5:
        vm=8;app=6;maps=1;threads=2;profile='compact'
    else:
        vm=5;app=3;maps=.5;threads=2;profile='compact'
    return {'host_ram_gb':ram,'docker_vm_gb':vm,'app_gb':app,'map_gb':maps,
            'docker_overhead_gb':vm-app-maps,'outside_docker_gb':round(ram-vm,1),
            'threads':threads,'processing_profile':profile}

def hardware(data):
    path=Path(data)/'runtime-hardware.json'
    try:
        if path.stat().st_size<64000:
            h=json.loads(path.read_text(encoding='utf-8-sig'));ram=h.get('ram_gb')
            if isinstance(ram,(float,int)) and 1<=ram<=4096:
                return {k:h.get(k) for k in ('os','architecture','ram_gb','source','host_detected','dedicated_gpu_required','dedicated_gpus')}
    except (OSError,ValueError,TypeError): pass
    return native_hardware()

def select(h,mode='auto'):
    aliases={'low':'compact','standard':'balanced'}
    mode=aliases.get(mode,mode)
    ram=h.get('ram_gb')
    if mode=='auto':
        # Simple spatial planning uses the smallest adequate model, even on large hosts.
        mode='balanced' if h.get('host_detected') and ram and ram>=23.5 else 'compact'
    profile={'id':mode,**PROFILES.get(mode,PROFILES['compact'])}
    profile['hardware_below_minimum']=bool(h.get('host_detected') and ram and ram+.5<profile['minimum_ram_gb'])
    gpus=h.get('dedicated_gpus') or []
    profile['execution_mode']='dedicated GPU when supported; CPU fallback' if gpus else 'CPU · no dedicated GPU detected'
    profile['gpu_policy']='Dedicated CUDA GPUs and native Apple Silicon MPS/Metal when supported; CPU fallback. Docker on Mac cannot access the Apple GPU.'
    profile['selection_reason']='Conservative automatic selection for simple planning; 9B is an opt-in quality upgrade.' if mode!='quality' else 'User-selected quality profile.'
    return profile
