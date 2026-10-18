#!/usr/bin/env python3
"""Pure bounds checks for the isolated Joyner full-GLM watchdog."""
import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/run_joyner_native_tier_ab.py"
spec=importlib.util.spec_from_file_location("glm53_tier_watchdog",p)
assert spec and spec.loader
wd=importlib.util.module_from_spec(spec);spec.loader.exec_module(wd)

class Watchdog(unittest.TestCase):
    def setUp(self):
        self.p={"mem_available_kib":3_000_000,"full_avg10":0.,
                "some_avg10":0.,"gpu_vram_total_bytes":32*wd.HOST_LIMIT//6,
                "gpu_vram_used_bytes":8*wd.HOST_LIMIT//6}
    def test_healthy(self):self.assertIsNone(wd.pressure_violation(self.p))
    def test_low_host_ram(self):
        self.p["mem_available_kib"]=899999
        self.assertEqual(wd.pressure_violation(self.p),"host_memory_floor")
    def test_high_psi_full(self):
        self.p["full_avg10"]=4.
        self.assertEqual(wd.pressure_violation(self.p),"full_PSI")
    def test_high_psi_some(self):
        self.p["some_avg10"]=8.
        self.assertEqual(wd.pressure_violation(self.p),"some_PSI")
    def test_low_vram(self):
        self.p["gpu_vram_used_bytes"]=30*wd.HOST_LIMIT//6
        self.assertEqual(wd.pressure_violation(self.p),"gpu_vram_floor")
    def test_expected_tokens_are_exact(self):
        self.assertEqual(wd.EXPECTED_GREEDY,"16 11 21 13 17 11 21 13")

if __name__=="__main__":unittest.main()
