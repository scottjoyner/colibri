#!/usr/bin/env python3
"""Independent read-only GLM fleet readiness and deny-by-default tests."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

HERE=Path(__file__).resolve().parents[2]
SOURCE=HERE/"benchmarks/joyner-glm53-20261016/glm53_fleet_readiness.py"
spec=importlib.util.spec_from_file_location("glm53_fleet_readiness",SOURCE)
mod=importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

class FleetPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.mem=self.root/"proc"
        self.mem.mkdir(parents=True)
        (self.mem/"meminfo").write_text(
            "MemTotal: 16000000 kB\nMemAvailable: 8000000 kB\n"
            "SwapTotal: 4000000 kB\nSwapFree: 4000000 kB\n")
        (self.mem/"pressure").mkdir()
        (self.mem/"pressure/memory").write_text(
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
            "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        card=self.root/"sys/class/drm/card1/device"
        card.mkdir(parents=True)
        (card/"vendor").write_text("0x1002\n")
        (card/"device").write_text("0x7551\n")
        (card/"mem_info_vram_total").write_text(str(32*mod.GIB)+"\n")
        (card/"mem_info_vram_used").write_text(str(1*mod.GIB)+"\n")
    def test_meminfo_and_psi(self):
        m=mod.read_meminfo(self.root)
        self.assertEqual(m["MemAvailable"],8_000_000*1024)
        self.assertEqual(mod.read_pressure(self.root)["some"]["avg10"],0)
    def test_r9700_identity_and_allocation(self):
        report=mod.physical_gpu(self.root)
        self.assertEqual(report["candidates"],1)
        self.assertEqual(report["selected"]["available_bytes"],31*mod.GIB)
    def test_unknown_gpu_fails_closed(self):
        (self.root/"sys/class/drm/card1/device/device").write_text("0x0000\n")
        report=mod.physical_gpu(self.root)
        self.assertIsNone(report["selected"])
    def test_invalid_gpu_memory_fails(self):
        (self.root/"sys/class/drm/card1/device/mem_info_vram_used").write_text(str(40*mod.GIB))
        with self.assertRaisesRegex(ValueError,"invalid VRAM"):
            mod.physical_gpu(self.root)
    def test_missing_checkpoint_refused(self):
        with self.assertRaisesRegex(ValueError,"source model directory"):
            mod.inspect_shards(self.root/"notpresent")
    def test_too_few_shards_refused(self):
        model=self.root/"model"
        model.mkdir()
        with (model/"model-00001-of-00062.safetensors").open("wb") as out:
            out.write(b"12345678")
        with self.assertRaisesRegex(ValueError,"exactly 62"):
            mod.inspect_shards(model)
    def test_nan_budget_refused(self):
        with self.assertRaisesRegex(ValueError,"budget"):
            mod.readiness(self.root,self.root,staging_gib=float("nan"))
    def test_report_does_not_claim_live_registry_state(self):
        from unittest import mock
        fake_layout={"shards":62,"categories":{"routed_experts":{"tensors":74304}}}
        with mock.patch.object(mod,"inspect_shards",return_value=fake_layout):
            receipt=mod.readiness(self.root,self.root)
        self.assertIsNone(receipt["production_model_loaded"])
        self.assertIsNone(receipt["fleet_admitted"])
        self.assertFalse(receipt["inference_started_by_preflight"])
        self.assertFalse(receipt["fleet_admission_modified"])
        self.assertEqual(receipt["decision"],"HOLD")

    def test_swap_measurement(self):
        (self.mem/"meminfo").write_text(
            "MemTotal: 16000000 kB\nMemAvailable: 8000000 kB\n"
            "SwapTotal: 4000000 kB\nSwapFree: 2800000 kB\n")
        m=mod.read_meminfo(self.root)
        self.assertEqual(m["SwapTotal"]-m["SwapFree"],1_200_000*1024)

if __name__=="__main__":
    unittest.main()
