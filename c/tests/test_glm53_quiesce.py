#!/usr/bin/env python3
"""Only exact, owner-verified disposable workload units can be terminated."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

script=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/glm53_quiesce.py"
sp=importlib.util.spec_from_file_location("glm53_quiesce_tests",script)
assert sp and sp.loader
q=importlib.util.module_from_spec(sp);sp.loader.exec_module(q)

GOOD="glm53-disposable-a1b2c3d4e5f6.service"
BAD="fleet-r9700-toolcall.service"

class QuiesceTests(unittest.TestCase):
    def make_proc(self,p,*,unit=GOOD,uid=None):
        (p/"124").mkdir(parents=True)
        (p/"124"/"status").write_text(f"Name:\tbench\nUid:\t{os.getuid() if uid is None else uid} {os.getuid()} {os.getuid()} {os.getuid()}\n")
        (p/"124"/"cgroup").write_text(f"0::/user.slice/app.slice/{unit}\n")
    def test_owned_exact_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_proc(root)
            self.assertEqual(q.process_unit(124,root),GOOD)
    def test_other_explicit_disposable_fleet_namespaces(self):
        for name in ("colibri","opencode","hermes"):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                unit=name+"-disposable-a1b2c3d4e5f6.scope"
                self.make_proc(root,unit=unit)
                self.assertEqual(q.process_unit(124,root),unit)
    def test_other_user_denied(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_proc(root,uid=os.getuid()+1)
            self.assertIsNone(q.process_unit(124,root))
    def test_production_service_denied(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_proc(root,unit=BAD)
            self.assertIsNone(q.process_unit(124,root))
    def test_lookalike_not_approved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_proc(root,unit="glm53-disposable-a1b2c3d4e5f6.service-backup")
            self.assertIsNone(q.process_unit(124,root))
    def test_mixed_candidate_set_must_not_stop_anything(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_proc(root)
            (root/"125").mkdir()
            (root/"125"/"status").write_text(f"Uid:\t{os.getuid()} {os.getuid()} {os.getuid()} {os.getuid()}\n")
            (root/"125"/"cgroup").write_text("0::/user.slice/app.slice/sshd.service\n")
            with patch.object(q.subprocess,"run") as run:
                result=q.stop_units([124,125],root=root)
                self.assertEqual(result["status"],"HOLD")
                self.assertEqual(result["unknown_pids"],[125])
                run.assert_not_called()
    def test_dry_run_never_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_proc(root)
            with patch.object(q.subprocess,"run") as run:
                r=q.stop_units([124],root=root,execute=False)
                self.assertEqual(r["would_stop_units"],[GOOD])
                run.assert_not_called()
    def test_valid_disposable_service_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_proc(root)
            def do_stop(cmd,**kw):
                self.assertEqual(cmd,["systemctl","--user","stop",GOOD])
                (root/"124"/"cgroup").write_text("0::/user.slice/app.slice/exited.scope\n")
                return SimpleNamespace(returncode=0)
            with patch.object(q.subprocess,"run",side_effect=do_stop) as run:
                r=q.stop_units([124],root=root)
            self.assertEqual(r["status"],"STOPPED")
            self.assertEqual(r["stopped_units"],[GOOD])
            self.assertEqual(run.call_count,1)
    def test_stopped_service_must_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_proc(root)
            with patch.object(q.subprocess,"run",return_value=SimpleNamespace(returncode=0)):
                r=q.stop_units([124],root=root,timeout_s=0.0)
            self.assertEqual(r["status"],"HOLD")
            self.assertEqual(r["failed_unit"],"verified_PIDs_still_alive")
if __name__=="__main__":
    unittest.main()
