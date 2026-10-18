#!/usr/bin/env python3
"""Two independent witness paths can corroborate outages, never admit GLM."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/glm53_cross_witness.py"
sp=importlib.util.spec_from_file_location("glm53_cross_witness_tests",MODULE)
q=importlib.util.module_from_spec(sp);sys.modules[sp.name]=q;sp.loader.exec_module(q)
NOW=1792341000

def receipt(epoch=NOW-15,**changes):
    data={"schema":1,"last_epoch_s":epoch,
          "remote_state":"UNREACHABLE","alarm":"REMOTE_UNREACHABLE",
          "consecutive_failures":9,"decision":"HOLD",
          "eligible_for_dispatch":False,"failover_authorized":False,
          "model_loaded_proven":False}
    data.update(changes)
    return json.dumps(data).encode()

def witness(host,kind,**changes):
    return q.parse(receipt(**changes),host=host,path_kind=kind,now_s=NOW)

class CrossWitnessTests(unittest.TestCase):
    def both(self,**peer):
        return [witness("x1-370","TAILNET"),witness("xwing","LAN",**peer)]
    def test_independent_dual_failure_corroborated(self):
        r=q.correlate(self.both(),now_s=NOW)
        self.assertEqual(r["corroboration"],"JOYNER_UNREACHABLE_BOTH_NETWORK_PATHS")
        self.assertEqual(r["decision"],"HOLD")
        self.assertFalse(r["failover_authorized"])
    def test_http503_is_not_reported_as_transport_outage(self):
        ws=self.both(remote_state="HEALTH_FAILED")
        r=q.correlate(ws,now_s=NOW)
        self.assertEqual(r["corroboration"],"JOYNER_HEALTH_OR_TRANSPORT_FAILURE_BOTH_PATHS")
        self.assertFalse(r["failover_authorized"])
    def test_one_fresh_observer_denied(self):
        r=q.correlate([witness("x1-370","TAILNET")],now_s=NOW)
        self.assertEqual(r["corroboration"],"NO_QUORUM")
    def test_duplicate_physical_observer_denied(self):
        w=witness("x1-370","TAILNET")
        self.assertEqual(q.correlate([w,w],now_s=NOW)["corroboration"],"NO_QUORUM")
    def test_network_paths_not_exchangeable(self):
        w=[witness("x1-370","LAN"),witness("xwing","TAILNET")]
        self.assertEqual(q.correlate(w,now_s=NOW)["corroboration"],"NO_QUORUM")
    def test_remote_recovered_no_model_proof(self):
        ws=self.both(remote_state="MODEL_CATALOG_UNVERIFIED",
                     alarm="MODEL_GENERATION_UNVERIFIED",consecutive_failures=0)
        r=q.correlate(ws,now_s=NOW)
        self.assertEqual(r["corroboration"],"NETWORK_PATH_DISAGREEMENT")
        self.assertFalse(r["eligible_for_dispatch"])
    def test_both_transports_up_still_denied(self):
        changes=dict(remote_state="HOST_HTTP_ONLY",alarm="MODEL_GENERATION_UNVERIFIED",
                     consecutive_failures=0)
        ws=[witness("x1-370","TAILNET",**changes),
            witness("xwing","LAN",**changes)]
        r=q.correlate(ws,now_s=NOW)
        self.assertEqual(r["corroboration"],"MODEL_GENERATION_UNVERIFIED")
        self.assertFalse(r["model_loaded_proven"])
    def test_stale_peer_rejected(self):
        with self.assertRaises(ValueError):
            witness("xwing","LAN",epoch=NOW-190)
    def test_future_peer_rejected(self):
        with self.assertRaises(ValueError):
            witness("xwing","LAN",epoch=NOW+10)
    def test_skewed_but_fresh_readings_no_quorum(self):
        ws=[witness("x1-370","TAILNET",epoch=NOW-10),
            witness("xwing","LAN",epoch=NOW-160)]
        self.assertEqual(q.correlate(ws,now_s=NOW)["corroboration"],"NO_QUORUM")
    def test_forged_ready_state_rejected(self):
        with self.assertRaises(ValueError):
            witness("xwing","LAN",failover_authorized=True)
    def test_missing_lineage_rejected(self):
        with self.assertRaises(ValueError):
            q.parse(b'{"schema":1}',host="xwing",path_kind="LAN",now_s=NOW)
    def test_forged_alert_rejected(self):
        with self.assertRaises(ValueError):
            witness("xwing","LAN",remote_state="MODEL_NOT_LISTED",
                    alarm="REMOTE_UNREACHABLE")
    def test_private_file_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"test.json"
            path.write_bytes(receipt())
            path.chmod(0o600)
            self.assertEqual(q.read_local(path),receipt())
    def test_symlink_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            original=Path(temp)/"test.json";original.write_bytes(receipt())
            alias=Path(temp)/"alias.json";alias.symlink_to(original)
            with self.assertRaises(ValueError):
                q.read_local(alias)
    def test_exposed_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"test.json";path.write_bytes(receipt());path.chmod(0o644)
            with self.assertRaises(ValueError):
                q.read_local(path)
    def test_ssh_command_is_fixed_and_nonmutating(self):
        from types import SimpleNamespace
        def exec_mock(args,**kwargs):
            self.assertEqual(args[-5:],["xwing","head","-c",str(q.MAX_STATE+1),q.LAN_STATE])
            self.assertIn("StrictHostKeyChecking=yes",args)
            self.assertIn("BatchMode=yes",args)
            self.assertFalse(kwargs["check"])
            return SimpleNamespace(stdout=receipt(),returncode=0)
        with patch.object(q.subprocess,"run",side_effect=exec_mock):
            self.assertEqual(q.read_xwing(),receipt())
    def test_ssh_failure_fails_closed(self):
        from types import SimpleNamespace
        with patch.object(q.subprocess,"run",
                          return_value=SimpleNamespace(stdout=b"",returncode=255)):
            with self.assertRaises(RuntimeError):q.read_xwing()
    def test_bad_clock_denied(self):
        with self.assertRaises(ValueError):
            witness("xwing","LAN",epoch="1792340900")
if __name__=="__main__": unittest.main()
