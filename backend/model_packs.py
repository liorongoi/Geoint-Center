"""Prepare only the local model pack requested by a worker, never at app launch."""
from pathlib import Path
import os
import shutil
import threading
from uuid import uuid4
from offline_policy import models_local_only as enabled
_LOCK=threading.Lock()
PACKS={'dynamicworld','openearthmap','prithvi','prithvi-crop','sam3-tracker'}
def ensure_model_pack(destination,pack):
    if pack not in PACKS:raise ValueError('Unknown offline model pack: '+pack)
    if pack=='sam3-tracker':
        from sam3_runtime import model_source
        source=model_source()
        if source is None:raise ValueError('SAM 3 component is missing. Install the separate offline SAM 3 component; no download fallback is allowed.')
        source_root=source.parent
    else:source_root=os.getenv('SPECTRASCOPE_BUNDLED_MODELS')
    if (not enabled() and pack!='sam3-tracker') or not source_root:return
    source=Path(source_root).resolve()/pack;destination=Path(destination).resolve()
    if not source.is_dir():raise ValueError('Required offline model pack is missing: '+pack+'. Restore the complete offline media; no download fallback is allowed.')
    with _LOCK:
        for path in sorted(source.rglob('*')):
            if not path.is_file():continue
            if path.is_symlink() or not path.resolve().is_relative_to(source):raise ValueError('Offline model pack contains an external link')
            relative=path.relative_to(source);target=destination/relative
            if target.exists():continue # Preserve existing checkpoints/caches.
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.'+uuid4().hex+'.part')
            try:shutil.copyfile(path,temp);temp.replace(target)
            finally:temp.unlink(missing_ok=True)
