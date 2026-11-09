"""Authorize private chat only to the launcher-owned local model listener."""
import os
import re
import subprocess
import sys
from urllib.parse import urlsplit


def managed_url():
    url = os.getenv('SPECTRASCOPE_LLM_URL', '')
    pid = os.getenv('SPECTRASCOPE_MANAGED_LLM_PID', '')
    if not re.fullmatch(r'[1-9][0-9]{0,9}', pid):
        return ''
    try:
        endpoint = urlsplit(url)
        if endpoint.port and url == f'http://127.0.0.1:{endpoint.port}':
            return url
    except ValueError:
        pass
    return ''


def owns_listener(pid, port, peer_port=None):
    address = f'127.0.0.1:{port}'
    peer = f'127.0.0.1:{peer_port}' if peer_port is not None else '0.0.0.0:0'
    if sys.platform == 'darwin':
        state = 'ESTABLISHED' if peer_port is not None else 'LISTEN'
        command = ['/usr/sbin/lsof', '-nP', '-a', '-p', str(pid), f'-iTCP:{port}', f'-sTCP:{state}', '-Fn']
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
        identity = f'n{address}->{peer}' if peer_port is not None else f'n{address}'
        return result.returncode == 0 and identity in result.stdout.splitlines()
    if sys.platform == 'win32':
        executable = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'netstat.exe')
        result = subprocess.run([executable,'-ano','-p','TCP'],capture_output=True,text=True,timeout=5,check=False)
        # The listener has an unspecified remote address with port zero.
        # Match the address and PID without relying on localized state names.
        return result.returncode == 0 and any(
            len(row) == 5 and row[:3] == ['TCP',address,peer] and row[-1] == str(pid)
            for row in (line.split() for line in result.stdout.splitlines())
        )
    return False


def require_managed_model(url, provider, *, check_listener=True):
    approved = managed_url()
    if provider != 'ollama' or not approved or url.rstrip('/') != approved:
        raise ValueError('Private model requests require Geo Intelligence’s managed local Ollama service. Independent endpoints are blocked until their network boundary is verified.')
    if check_listener:
        require_listener(approved,'SPECTRASCOPE_MANAGED_LLM_PID','local model')
    return approved


def require_listener(url, setting, service, *, peer_port=None):
    pid = os.getenv(setting, '')
    try:
        port = urlsplit(url).port
        if (port and url == f'http://127.0.0.1:{port}' and re.fullmatch(r'[1-9][0-9]{0,9}',pid)
                and owns_listener(int(pid),port,peer_port)):
            return
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    raise ValueError(f'The managed {service} listener could not be verified. Restart the managed service.')
