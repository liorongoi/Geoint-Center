"""Process-shared, crash-recoverable admission for native analysis services."""
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

GIB = 1024**3


class ResourceUnavailable(ValueError):
    pass


def _lock(stream):
    stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno not in (13, 36):raise
            return False
    else:
        import fcntl
        try:fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True


def _open(path):
    import stat
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    if path.is_symlink():raise ResourceUnavailable('Resource ledger may not contain links')
    fd = os.open(path, flags, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise ResourceUnavailable('Resource ledger requires private, regular lock files')
    return os.fdopen(fd, 'r+b', buffering=0)


def _active(root):
    # The registry lock is held when inspecting claims. A claim lock is
    # released automatically by the OS when its owning process exits.
    records=[]
    for path in root.glob('*.claim'):
        with _open(path) as stream:
            if _lock(stream):
                stale=True
            else:
                stale=False
                stream.seek(1)
                records.append(json.loads(stream.read(4096)))
        if stale:path.unlink(missing_ok=True)
    return records


def _capacity():
    from runtime_profiles import _processing_hardware
    effective = _processing_hardware().get('ram_gb') or 0
    cap = max(0, math.floor(effective * 2/3) - 1)
    return min(cap, 28) if effective >= 47.5 else cap


def accelerator():
    # Unknown or automatic accelerator choices reserve the single host-wide
    # accelerator slot. CPU overrides still share the host RAM budget.
    return os.getenv('SPECTRASCOPE_DEVICE', 'auto') != 'cpu'


def analysis_gib():
    from runtime_profiles import performance_profile
    return {'compact':4, 'balanced':6, 'extended':8}[performance_profile()['id']]


@contextmanager
def _reserve(root, name, ram_gib, gpu, checkpoint, timeout):
    from execution_policy import available_ram_bytes
    ram_bytes = math.ceil(ram_gib * GIB)
    capacity = _capacity() * GIB
    if not 0 < ram_bytes <= capacity:
        raise ResourceUnavailable(f'{name} exceeds this host’s shared analysis RAM budget')
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if root.is_symlink() or (os.name != 'nt' and (root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077)):
        raise ResourceUnavailable('The host resource directory must be private to the application account')
    deadline = time.monotonic() + timeout
    token = root / (uuid4().hex + '.claim')
    claim = None
    try:
        while claim is None:
            if checkpoint is not None:checkpoint()
            with _open(root / 'admission.lock') as registry:
                if _lock(registry):
                    active = _active(root)
                    reserved = sum(item['ram_bytes'] for item in active)
                    if reserved + ram_bytes <= capacity and not (gpu and any(item['gpu'] for item in active)):
                        live = available_ram_bytes()
                        # In-flight claims may not have allocated their arrays
                        # yet. Charge their full estimates against this sample
                        # too, so simultaneous starts cannot race the OS gauge.
                        if live is not None and live >= reserved + ram_bytes + GIB:
                            claim = _open(token)
                            if not _lock(claim):raise ResourceUnavailable('Unable to own the host resource reservation')
                            claim.seek(1)
                            claim.write(json.dumps({'name':name,'ram_bytes':ram_bytes,'gpu':bool(gpu)}).encode())
            if claim is None:
                if time.monotonic() >= deadline:
                    raise ResourceUnavailable(f'{name} timed out waiting for host RAM or accelerator capacity. Retry after active work finishes.')
                time.sleep(.2)
        yield
    finally:
        if claim is not None:claim.close()
        # The next holder of the registry lock removes released claims.


def configure_native_host():
    import sys
    if sys.platform in ('darwin','win32'):
        os.environ.setdefault('GEO_RESOURCE_ROOT',str(Path.home()/'.geo-intelligence-resources'))


def configuration():
    enabled=bool(os.getenv('GEO_RESOURCE_ROOT'))
    return {'enabled':enabled,'ram_budget_gib':_capacity() if enabled else None,'accelerator_slots':1 if enabled else None}


def reserve(name, ram_gib, *, gpu=False, checkpoint=None, timeout=120):
    value = os.getenv('GEO_RESOURCE_ROOT')
    if not value:return nullcontext()
    root = Path(value)
    if not root.is_absolute() or '..' in root.parts:raise ResourceUnavailable('Use an absolute private host resource directory')
    return _reserve(root, name, ram_gib, gpu, checkpoint, timeout)


def run_reserved(name, ram_gib, function, *args, gpu=False, checkpoint=None, **kwargs):
    with reserve(name, ram_gib, gpu=gpu, checkpoint=checkpoint):
        return function(*args, **kwargs)
