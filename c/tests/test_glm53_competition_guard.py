#!/usr/bin/env python3
"""The mutually exclusive watchdog modes must not test downed endpoints."""
import importlib.util
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

p=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/run_joyner_native_tier_ab.py"
sys.path.insert(0,str(p.parent))
s=importlib.util.spec_from_file_location("glm53_native_bench",p)
assert s and s.loader
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class CompetitionGuardTest(unittest.TestCase):
    def test_isolation_does_not_contact_stopped_servers(self):
        with patch.object(m,"unit_active",return_value=False), patch.object(m,"check_endpoint",side_effect=AssertionError("must not check")):
            self.assertIsNone(m.competition_violation(isolated=True))
    def test_isolation_fails_on_unexpected_restart(self):
        with patch.object(m,"unit_active",return_value=True):
            self.assertEqual(m.competition_violation(isolated=True),"competing_service_restarted")
    def test_nonisolated_detects_health_failure(self):
        with patch.object(m,"check_endpoint",return_value=False):
            self.assertEqual(m.competition_violation(isolated=False),"production_health_lost")
    def test_nonisolated_detects_health_exception(self):
        with patch.object(m,"check_endpoint",side_effect=OSError("connection")):
            self.assertEqual(m.competition_violation(isolated=False),"production_health_check_failed")
    def test_dynamic_scanner_flags_late_cpu_workload(self):
        from unittest.mock import patch
        with patch.object(m,"scan_processes",return_value=({123:"snapshot"},0)),             patch.object(m,"observe_processes",return_value=[{"pid":123,"reasons":["cpu"]}]),             patch.object(m,"exclusive_gpu_competitors",return_value=[]):
            _,report=m.exclusive_competition_scan({},4.,
                                                   "glm53-tier-ab-123456789abc")
        self.assertEqual(report["cpu_or_ssd"][0]["pid"],123)
    def test_dynamic_scanner_flags_unexpected_gpu_owner(self):
        with patch.object(m,"scan_processes",return_value=({},0)),             patch.object(m,"observe_processes",return_value=[]),             patch.object(m,"exclusive_gpu_competitors",return_value=[555]):
            _,report=m.exclusive_competition_scan({},4.,"glm53-tier-ab-123456789abc")
        self.assertEqual(report["gpu_users"],[555])
    def test_unique_isolated_round_label_accepted(self):
        args=["glm53","--binary","/nonexistent","--out-dir","/nonexistent",
              "--tier-gib","4","--rate","32","--dense-host","0",
              "--text-only","1","--fair-layers","0",
              "--label","exclusive4control-abcdef123456","--isolated"]
        with patch("sys.argv",args),patch.object(m,"validate",
                                                side_effect=RuntimeError("PINNED_LABEL_ACCEPTED")):
            with self.assertRaisesRegex(RuntimeError,"PINNED_LABEL_ACCEPTED"):
                m.main()
    def test_unique_round_label_denies_shared_gpu_run(self):
        args=["glm53","--binary","/nonexistent","--out-dir","/nonexistent",
              "--tier-gib","4","--rate","32","--dense-host","0",
              "--text-only","1","--fair-layers","0",
              "--label","exclusive4control-abcdef123456"]
        with patch("sys.argv",args),patch.object(m,"validate") as val:
            with self.assertRaisesRegex(SystemExit,"pinned"):
                m.main()
            val.assert_not_called()
    def test_nonisolated_passes_both_healthy(self):
        with patch.object(m,"check_endpoint",return_value=True) as c:
            self.assertIsNone(m.competition_violation(isolated=False))
            self.assertEqual(c.call_count,2)
if __name__=="__main__":
    unittest.main()
