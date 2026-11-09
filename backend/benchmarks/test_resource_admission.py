"""Independent processes must honor the same RAM/accelerator admission ledger."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import resource_admission as resources

HOLDER = """
import execution_policy, resource_admission as r, sys
execution_policy.available_ram_bytes = lambda: 64 * r.GIB
r._capacity = lambda: 8
with r.reserve('Independent process', float(sys.argv[1]), gpu=bool(int(sys.argv[2]))):
    print('admitted', flush=True)
    sys.stdin.readline()
"""


class HostAdmission(unittest.TestCase):
    def test_competing_processes_and_crash_recovery(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'GEO_RESOURCE_ROOT':str(Path(folder)/'ledger')}), \
                patch('execution_policy.available_ram_bytes',return_value=64*resources.GIB),patch.object(resources,'_capacity',return_value=8):
            child=subprocess.Popen([sys.executable,'-B','-c',HOLDER,'6','1'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(),'admitted')
                with resources.reserve('Map render',.5,timeout=.5):
                    with self.assertRaisesRegex(resources.ResourceUnavailable,'timed out'):
                        with resources.reserve('Worker CPU',2,timeout=.2):pass
                    with self.assertRaisesRegex(resources.ResourceUnavailable,'timed out'):
                        with resources.reserve('Local model',1,gpu=True,timeout=.2):pass
                with resources.reserve('CPU worker',2,timeout=.5):pass
                child.kill()
                child.wait(timeout=5)
                with resources.reserve('Local model',8,gpu=True,timeout=.5):pass
            finally:
                if child.poll() is None:child.kill();child.wait(timeout=5)
                child.stdin.close();child.stdout.close()

    def test_low_live_memory_and_cancelled_wait_fail_without_a_claim(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'GEO_RESOURCE_ROOT':str(Path(folder)/'ledger')}), \
                patch('execution_policy.available_ram_bytes',return_value=resources.GIB),patch.object(resources,'_capacity',return_value=8):
            with self.assertRaisesRegex(resources.ResourceUnavailable,'timed out'):
                with resources.reserve('Worker',4,timeout=.2):pass
            with self.assertRaisesRegex(RuntimeError,'stopped'):
                def stopped():raise RuntimeError('stopped')
                with resources.reserve('Worker',4,checkpoint=stopped):pass
            self.assertEqual(list((Path(folder)/'ledger').glob('*.claim')),[])

    def test_live_headroom_counts_claims_before_their_allocations_land(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'GEO_RESOURCE_ROOT':str(Path(folder)/'ledger')}), \
                patch('execution_policy.available_ram_bytes',return_value=7*resources.GIB),patch.object(resources,'_capacity',return_value=20):
            with resources.reserve('Worker',4):
                with self.assertRaisesRegex(resources.ResourceUnavailable,'timed out'):
                    with resources.reserve('Local analysis',3,timeout=.2):pass
