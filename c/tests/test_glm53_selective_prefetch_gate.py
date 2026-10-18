#!/usr/bin/env python3
"""Only positive-utility nominations with independent and fresh witnesses."""
import sys
from dataclasses import replace
from pathlib import Path
import unittest

B=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016"
sys.path.insert(0,str(B))
from selective_prefetch_gate import Witness,Confidence,candidate,BYTES_PER_EXPERT
from shadow_tier_plan import Limits
PIN="46a050bec13002ad29ee795c16e61cf4fa3c4a9c"

def make_input():
    limit=Limits(session="shadow-A",pin=PIN,expected_pin=PIN,
        generation=42,expected_generation=42,enabled=True,
        nvme_budget=2*BYTES_PER_EXPERT,gpu_budget=2*BYTES_PER_EXPERT,
        gpu_headroom=2*BYTES_PER_EXPERT,deadline_ms=20,
        nvme_GBps=8.0,upload_GBps=8.0,overhead_ms=.1)
    witness=Witness(model_pin=PIN,session="shadow-A",generation=42,
        observed_at_ms=10000,current_time_ms=10100,
        available_vram_bytes=3*1024**3,
        owner_resident_bytes=4*1024**3,
        reserved_for_service_bytes=1024**3,
        speculative_bytes_inflight=0,queue_depth=0,max_queue_depth=2,
        gpu_healthy=True,ssd_healthy=True,memory_psi_full_avg10=0.0)
    confidence=Confidence(model_pin=PIN,source_runs=4,validation_rows=256,
        verified_split=True,predicted_probability={7:.85,8:.76,9:.3},
        validation_precision=.75)
    snap={"v":2,"n_experts":288,"host_mask":bytes(36).hex(),
          "gpu_mask":bytes(36).hex(),
          "ids":[7,8,11,13,14,17,18,19]}
    return snap,[7,8,9],limit,witness,confidence

class SelectiveGateTests(unittest.TestCase):
    def test_good_shadow_only_nominations(self):
        s,r,l,w,c=make_input()
        got,status=candidate(s,r,l,w,c)
        self.assertEqual(status,"shadow_proposed_no_transfer")
        self.assertEqual([e["expert"] for e in got],[7,8])
        self.assertEqual(sum(e["nominal_gpu_bytes"] for e in got),2*BYTES_PER_EXPERT)
    def test_disabled(self):
        s,r,l,w,c=make_input()
        got,status=candidate(s,r,replace(l,enabled=False),w,c)
        self.assertEqual((got,status),([],"disabled"))
    def test_stale_lease(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,replace(w,generation=41),c)[1],"identity_or_lease_mismatch")
    def test_stale_inventory(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,replace(w,current_time_ms=12500),c)[1],"stale_resource_witness")
    def test_future_inventory(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,replace(w,current_time_ms=9950),c)[1],"stale_resource_witness")
    def test_gpu_failure(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,replace(w,gpu_healthy=False),c)[1],"device_not_healthy")
    def test_positive_memory_pressure_denied(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,replace(w,memory_psi_full_avg10=.2),c)[1],
                         "resource_pressure_or_telemetry_invalid")
    def test_no_evictions_to_admit(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,replace(w,available_vram_bytes=1024**3),c)[1],
                         "no_spare_vram_without_eviction")
    def test_unverified_validation(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,w,replace(c,verified_split=False))[1],
                         "unverified_or_low_precision_prediction")
    def test_insufficient_validation(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,w,replace(c,validation_rows=2))[1],
                         "unverified_or_low_precision_prediction")
    def test_low_confidence(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,w,replace(c,predicted_probability={7:.5,8:.55}))[1],
                         "no_confident_candidates")
    def test_known_gpu_resident_filtered(self):
        s,r,l,w,c=make_input()
        mask=bytearray(36);mask[7>>3]|=1<<(7&7)
        s["gpu_mask"]=mask.hex()
        got,status=candidate(s,r,l,w,c)
        self.assertEqual(status,"shadow_proposed_no_transfer")
        self.assertEqual([e["expert"] for e in got],[8])
    def test_bad_bitmap_denied(self):
        s,r,l,w,c=make_input()
        s["gpu_mask"]="not hex"
        self.assertEqual(candidate(s,r,l,w,c)[1],"no_safe_candidate_bad_cache_mask")
    def test_bounded_per_event(self):
        s,r,l,w,c=make_input()
        got,_=candidate(s,r,l,w,c)
        self.assertLessEqual(sum(e["nominal_nvme_bytes"] for e in got),2*BYTES_PER_EXPERT)
    def test_queue_full_denied(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,replace(w,queue_depth=2),c)[1],
                         "resource_pressure_or_telemetry_invalid")
    def test_pin_mismatch_denied(self):
        s,r,l,w,c=make_input()
        self.assertEqual(candidate(s,r,l,w,replace(c,model_pin="a"*40))[1],"identity_or_lease_mismatch")
    def test_no_peeking_at_route_labels(self):
        s,r,l,w,c=make_input()
        first=candidate(s,r,l,w,c)
        s["ids"]=[9,10,11,12,13,14,15,16]
        second=candidate(s,r,l,w,c)
        self.assertEqual(first,second)
if __name__=="__main__":unittest.main()
