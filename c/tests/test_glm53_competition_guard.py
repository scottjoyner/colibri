#!/usr/bin/env python3
"""The mutually exclusive watchdog modes must not test downed endpoints."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

p=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/run_joyner_native_tier_ab.py"
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
    def test_nonisolated_passes_both_healthy(self):
        with patch.object(m,"check_endpoint",return_value=True) as c:
            self.assertIsNone(m.competition_violation(isolated=False))
            self.assertEqual(c.call_count,2)
if __name__=="__main__":
    unittest.main()
