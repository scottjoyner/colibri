"""Safety and no-route-leakage tests for physical shadow expert-read replay."""
import pathlib
import sys
import unittest
from collections import Counter,defaultdict

ROOT=pathlib.Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016"
sys.path.insert(0,str(ROOT))
from benchmark_trace_predicted_reads import build_policy, FROZEN, PIN, percentile, read_one
from shadow_tier_plan import Limits,plan

def mask(ids):
    m=bytearray(36)
    for i in ids:m[i>>3]|=1<<(i&7)
    return m.hex()

class PredictedReadTests(unittest.TestCase):
    def test_rank_independent_current_native_ids(self):
        base={"call":3,"layer":3,"row":0,"ids":[7,4,6,8,9,10,11,12]}
        alternate=dict(base,ids=[20,21,22,23,24,25,26,27])
        prior={("token",3,0):{"ids":[4,5,6,7,8,9,10,11]},
               ("call",2,0):{"layer":2,"ids":[12,13,14,15,16,17,18,19]}}
        freq=defaultdict(Counter,{3:Counter({9:7,17:2,4:1})})
        cooc={(3,12):Counter({7:5,3:1})}
        for method in ("fusion","prior_token","cross_layer"):
            x=build_policy(method,base,prior,freq,cooc)
            y=build_policy(method,alternate,prior,freq,cooc)
            self.assertEqual(x,y)
            self.assertEqual(len(set(x)),288)
            self.assertEqual(len(x),288)

    def test_nonfuture_plan_honors_masks_and_deadline(self):
        event={"v":2,"n_experts":288,
               "host_mask":mask([3,4]),"gpu_mask":mask([2]),
               "ids":[7,8,9,10,11,12,13,14]}
        limit=Limits(session="physical-shadow",pin=PIN,expected_pin=PIN,
                     generation=1,expected_generation=1,enabled=True,
                     nvme_GBps=11.319,upload_GBps=8.0,deadline_ms=10.)
        a,status=plan([2,3,4,5,7],event,limit)
        self.assertEqual(status,"proposed")
        self.assertEqual([i["expert"] for i in a],[3,4])
        self.assertTrue(all(i["nominal_nvme_bytes"]==0 for i in a))
        self.assertEqual(plan([3,5],event,Limits(**{**limit.__dict__,"generation":2}))[1],
                         "stale_session_or_generation")
        self.assertEqual(plan([3,5],event,Limits(**{**limit.__dict__,"deadline_ms":0.01}))[1],
                         "outside_envelope")
    def test_read_rejects_invalid_offset_before_fd(self):
        with self.assertRaisesRegex(ValueError,"not aligned"):
            read_one(-1,123)
    def test_percentile_is_bounded(self):
        self.assertIsNone(percentile([],.99))
        self.assertEqual(percentile([3,1,2],.5),2)
        self.assertEqual(percentile([3,1,2],.99),3)

if __name__=="__main__":
    unittest.main()
