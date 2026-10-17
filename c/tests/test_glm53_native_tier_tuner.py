#!/usr/bin/env python3
"""Causal offline cache-policy simulator invariants (private trace not required)."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/tune_native_expert_tier.py"
spec=importlib.util.spec_from_file_location("glm53_tier_shadow_tuner",SOURCE)
assert spec and spec.loader
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class ShadowAdmit(unittest.TestCase):
    def test_no_future_labels_and_single_forward_duplicates(self):
        # A prefill's repeated rows must count as ONE distinct forward.
        prefill=[(3,2)]*3
        r=mod.simulate([prefill],2,32,"repeat_gate")
        self.assertEqual(r["hits"],0)
        self.assertEqual(r["uploads"],0)
        self.assertEqual(r["repeat_first_refused"],1)
        next_frame=[(3,2)]
        r=mod.simulate([prefill,next_frame,[(3,2)]],2,32,"repeat_gate")
        self.assertEqual(r["uploads"],1)
        self.assertEqual(r["hits"],1)
    def test_baseline_has_future_first_use_miss(self):
        r=mod.simulate([[(3,7)],[(3,7)]],2,16,"first")
        self.assertEqual(r["hits"],1)
        self.assertEqual(r["uploads"],1)
    def test_capped_resident_allocation(self):
        frames=[[(3,x)] for x in (1,2,3,4,1)]
        r=mod.simulate(frames,2,8,"first")
        self.assertLessEqual(r["remaining_resident"],2)
        self.assertGreater(r["evictions"],0)
    def test_zero_uploads_has_no_gpu_hits(self):
        r=mod.simulate([[(3,1)],[(3,1)],[(3,1)]],100,0,"repeat_gate")
        self.assertEqual(r["hits"],0)
        self.assertEqual(r["uploads"],0)
    def test_reject_invalid_policy(self):
        with self.assertRaisesRegex(ValueError,"parameter"):
            mod.simulate([],0,1,"first")
        with self.assertRaisesRegex(ValueError,"parameter"):
            mod.simulate([],8,1,"oracle_future")
    def test_reject_incomplete_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"run.jsonl"
            p.write_text(json.dumps(dict(call=0,row=0,layer=3,ids=[2],
                                         n_experts=288,topk=1))+"\n")
            with self.assertRaisesRegex(ValueError,"incomplete forward"):
                mod.load_forwards(p)
    def test_realistic_frame_causal_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"run.jsonl"
            lines=[]
            for forward in range(2):
                for layer in range(3,45):
                    lines.append(json.dumps(dict(call=42*forward+layer-3,
                           row=0,layer=layer,ids=[layer%288],
                           n_experts=288,topk=1)))
            p.write_text("\n".join(lines)+"\n")
            frames,sha=mod.load_forwards(p)
            self.assertEqual(len(frames),2)
            self.assertEqual(len(frames[0]),42)
            self.assertEqual(len(sha),64)
            self.assertEqual(mod.simulate(frames,64,64,"repeat_gate")["uploads"],42)
if __name__=="__main__":
    unittest.main()
