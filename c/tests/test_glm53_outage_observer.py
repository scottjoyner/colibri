#!/usr/bin/env python3
"""Production-style fault injection for read-only out-of-band GLM observation."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016"
sys.path.insert(0,str(SCRIPT))
spec=importlib.util.spec_from_file_location("glm53_outage_observer_test",SCRIPT/"glm53_outage_observer.py")
observer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=observer;spec.loader.exec_module(observer)

def obs(remote="UNREACHABLE",healthy=True,route=False):
    return {"remote_state":remote,"local_host_healthy":healthy,
            "local_gateway_advertised_route":route,"decision":"HOLD"}

class OutageTests(unittest.TestCase):
    def make_dir(self):
        td=tempfile.TemporaryDirectory()
        p=Path(td.name)/"private-watch"
        return td,p
    def test_heartbeat_escalation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"watch"
            a=observer.record_once(folder,obs(),now_s=100)
            b=observer.record_once(folder,obs(),now_s=200)
            self.assertEqual(a["alarm"],"REMOTE_UNREACHABLE_PENDING_CONFIRMATION")
            self.assertEqual(b["alarm"],"REMOTE_UNREACHABLE")
            self.assertEqual(b["consecutive_failures"],2)
            self.assertFalse(b["eligible_for_dispatch"])
            self.assertEqual(len((folder/"observations.jsonl").read_text().splitlines()),2)
    def test_http200_cannot_recover_model_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"watch"
            observer.record_once(folder,obs(),now_s=100)
            observer.record_once(folder,obs(),now_s=200)
            for ts in (300,400,500):
                r=observer.record_once(folder,obs("MODEL_CATALOG_UNVERIFIED",route=True),now_s=ts)
                self.assertFalse(r["model_loaded_proven"])
                self.assertFalse(r["failover_authorized"])
                self.assertEqual(r["decision"],"HOLD")
            self.assertEqual(r["transport_recovered_samples"],3)
            self.assertEqual(r["alarm"],"MODEL_GENERATION_UNVERIFIED")
    def test_failed_health_response_counts_as_outage(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"w"
            observer.record_once(p,obs("HEALTH_FAILED"),now_s=100)
            r=observer.record_once(p,obs("HEALTH_FAILED"),now_s=200)
            self.assertEqual(r["alarm"],"REMOTE_UNREACHABLE")
    def test_bad_state_denied(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w";d.mkdir(mode=0o700)
            (d/"state.json").write_text("not-json")
            with self.assertRaises(ValueError):
                observer.record_once(d,obs(),now_s=100)
    def test_replayed_timestamp_denied(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w"
            observer.record_once(d,obs(),now_s=100)
            with self.assertRaises(ValueError):
                observer.record_once(d,obs(),now_s=100)
            self.assertEqual(len((d/"observations.jsonl").read_text().splitlines()),1)
    def test_symlink_journal_denied(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w";d.mkdir(mode=0o700)
            target=Path(td)/"innocent.txt";target.write_text("DO_NOT_TOUCH")
            (d/"observations.jsonl").symlink_to(target)
            with self.assertRaises(ValueError):observer.record_once(d,obs(),now_s=1)
            self.assertEqual(target.read_text(),"DO_NOT_TOUCH")
    def test_symlink_state_denied(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w";d.mkdir(mode=0o700)
            target=Path(td)/"innocent.txt";target.write_text("DO_NOT_TOUCH")
            (d/"state.json").symlink_to(target)
            with self.assertRaises(ValueError):observer.record_once(d,obs(),now_s=1)
            self.assertEqual(target.read_text(),"DO_NOT_TOUCH")
    def test_insecure_directory_denied(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"w";p.mkdir(mode=0o777);p.chmod(0o755)
            with self.assertRaises(ValueError):observer.record_once(p,obs(),now_s=1)
    def test_journal_capacity_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w";d.mkdir(mode=0o700)
            # Patch limit to verify the same branch without allocating 8 MiB.
            with patch.object(observer,"MAX_JOURNAL",10):
                with self.assertRaises(ValueError):
                    observer.record_once(d,obs(),now_s=1)
            self.assertFalse((d/"state.json").exists())
    def test_lock_contention_denies_double_observer(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w";d.mkdir(mode=0o700)
            import os
            fd=os.open(d/"observer.lock",os.O_CREAT|os.O_RDWR,0o600)
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    observer.record_once(d,obs(),now_s=1)
            finally:
                fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
    def test_rotates_when_full_without_losing_current_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w"
            # Record at least one event before enforcing a smaller bound.
            observer.record_once(d,obs(),now_s=100)
            first=(d/"observations.jsonl").read_bytes()
            with patch.object(observer,"MAX_JOURNAL",len(first)+10):
                observer.record_once(d,obs(),now_s=200)
            older=(d/"observations.previous.jsonl").read_text()
            newest=(d/"observations.jsonl").read_text()
            self.assertIn('"epoch_s":100',older)
            self.assertIn('"epoch_s":200',newest)
            self.assertEqual((d/"observations.previous.jsonl").stat().st_mode&0o777,0o600)
    def test_unsafe_rotation_target_denied(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w"
            observer.record_once(d,obs(),now_s=100)
            first=(d/"observations.jsonl").read_bytes()
            innocent=Path(td)/"other";innocent.write_text("KEEP_ME")
            (d/"observations.previous.jsonl").symlink_to(innocent)
            with patch.object(observer,"MAX_JOURNAL",len(first)+10):
                with self.assertRaises(ValueError):
                    observer.record_once(d,obs(),now_s=200)
            self.assertEqual(innocent.read_text(),"KEEP_ME")
            self.assertEqual(len((d/"observations.jsonl").read_text().splitlines()),1)
    def test_journal_private(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w"
            observer.record_once(d,obs(),now_s=1)
            for name in ("observer.lock","state.json","observations.jsonl"):
                self.assertEqual((d/name).stat().st_mode&0o777,0o600)
    def test_missing_or_corrupt_remote_state_denied(self):
        with self.assertRaises(ValueError):observer.decide(None,{"remote_state":"AVAILABLE"},now_s=100)
    def test_broken_atomic_state_does_not_promote(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/"w"
            observer.record_once(d,obs(),now_s=100)
            with patch.object(observer,"_atomic_state",side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    observer.record_once(d,obs("MODEL_CATALOG_UNVERIFIED"),now_s=200)
            old=json.loads((d/"state.json").read_text())
            self.assertEqual(old["remote_state"],"UNREACHABLE")
            self.assertEqual(old["decision"],"HOLD")
if __name__=="__main__": unittest.main()
