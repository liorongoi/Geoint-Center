"""OS loopback containment for managed macOS services."""
import os
from pathlib import Path
import sys


def enforce_mac_loopback():
    if sys.platform != 'darwin' or os.getenv('GEO_MAC_LOOPBACK_APPLIED') == '1':
        return
    root = Path(__file__).resolve().parent
    profile = next((p for p in (root/'mac-loopback.sb', root/'desktop'/'mac-loopback.sb') if p.is_file()), None)
    sandbox = Path('/usr/bin/sandbox-exec')
    if profile is None or not sandbox.is_file():
        raise RuntimeError('Strict offline macOS runtime requires its local network policy and sandbox-exec')
    env = {**os.environ, 'GEO_MAC_LOOPBACK_APPLIED':'1'}
    os.execve(sandbox,[str(sandbox),'-f',str(profile),sys.executable,*sys.orig_argv[1:]],env)
