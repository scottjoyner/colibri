import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime,timedelta,timezone

f=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/glm53_production_gate.py"
spec=importlib.util.spec_from_file_location("glm53_production_gate_test",f)
g=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=g
spec.loader.exec_module(g)
NOW=datetime(2026,10,17,22,tzinfo=timezone.utc)
SHA="a"*64

def trial(root,case,round_id="abc123abc123",**updates):
    data={"mode":f"exclusive4{case}-{round_id}","fair_layers":int(case=="fair"),
          "completed":True,"isolated":True,"competitors_detected":[],
          "greedy_ids":g.EXPECTED,"gpu_routed":3696,"exit_code":0,
          "binary_sha256":SHA,"tok_s":.4,"min_available_kib":9000000,
          "max_psi_full_avg10":0.0}
    data.update(updates)
    p=root/(data["mode"]+".json")
    p.write_text(json.dumps(data))
    return p

def soak(root,**updates):
    data={"schema":1,"start_utc":(NOW-timedelta(hours=74)).isoformat(),
          "end_utc":NOW.isoformat(),"requests":2000,"failures":0,
          "natural_language_requests":500,"p95_ms_per_token":500,
          "max_restart_recovery_seconds":30,
          "unauthorized_denials_verified":12,
          "tenant_isolation_failures":0,"unrecovered_crashes":0}
    data.update(updates)
    p=root/"soak.json";p.write_text(json.dumps(data))
    return p
class GateTest(unittest.TestCase):
    def test_real_receipts_hold(self):
        r=g.report(f.parent,now=NOW)
        self.assertEqual(r["decision"],"HOLD")
        self.assertEqual(r["benchmark"]["paired_runs"],1)
        self.assertFalse(r["can_deploy"])
    def test_missing_receipts_hold(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(g.report(Path(td))["benchmark"]["paired_runs"],0)
    def test_paired_control_fair(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);trial(root,"control");trial(root,"fair")
            self.assertEqual(g.evaluate_benchmarks(root)[0]["paired_runs"],1)
    def test_reject_wrong_policy(self):
        with tempfile.TemporaryDirectory() as td:
            p=trial(Path(td),"fair",fair_layers=0)
            with self.assertRaises(ValueError):g.receipt(p)
    def test_reject_contended_trial(self):
        with tempfile.TemporaryDirectory() as td:
            p=trial(Path(td),"fair",competitors_detected=[{"pid":123}])
            with self.assertRaises(ValueError):g.receipt(p)
    def test_reject_wrong_token(self):
        with tempfile.TemporaryDirectory() as td:
            p=trial(Path(td),"fair",greedy_ids="wrong")
            with self.assertRaises(ValueError):g.receipt(p)
    def test_reject_binary_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);trial(root,"control")
            trial(root,"fair",binary_sha256="b"*64)
            self.assertEqual(g.evaluate_benchmarks(root)[0]["paired_runs"],0)
    def test_reject_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=trial(root,"fair")
            link=root/"link";link.symlink_to(p)
            with self.assertRaises(ValueError):g.load_json(link)
    def test_good_soak_still_requires_external_review(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for rid in ("abc123abc123","def456def456","aaa111aaa111"):
                trial(root,"control",rid,tok_s=3.1)
                trial(root,"fair",rid,tok_s=3.5)
            result=g.report(root,soak(root),now=NOW)
            self.assertEqual(result["benchmark"]["paired_runs"],3)
            self.assertTrue(result["soak"]["valid"])
            self.assertEqual(result["decision"],"HOLD")
            self.assertFalse(result["can_start_server"])
    def test_short_soak_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            p=soak(root,start_utc=(NOW-timedelta(hours=3)).isoformat())
            self.assertIn("72 hours"," ".join(g.evaluate_soak(p,NOW)[1]))
    def test_access_failures_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            p=soak(Path(td),unauthorized_denials_verified=0)
            self.assertIn("authentication"," ".join(g.evaluate_soak(p,NOW)[1]))
    def test_tenant_violation_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            p=soak(Path(td),tenant_isolation_failures=1)
            self.assertIn("tenant"," ".join(g.evaluate_soak(p,NOW)[1]))
    def test_future_soak_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            p=soak(Path(td),end_utc=(NOW+timedelta(hours=1)).isoformat())
            self.assertFalse(g.evaluate_soak(p,NOW)[0]["valid"])
    def test_nonfinite_latency_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            p=soak(Path(td),p95_ms_per_token=float("nan"))
            self.assertFalse(g.evaluate_soak(p,NOW)[0]["valid"])
if __name__=="__main__":
    unittest.main()
