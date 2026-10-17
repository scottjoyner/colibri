import unittest
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016"))
from shadow_tier_plan import Limits,plan,BYTES_PER_EXPERT

PIN="46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
def mask(ids):
    b=bytearray(36)
    for i in ids:b[i>>3]|=(1<<(i&7))
    return b.hex()
def event():
    return dict(v=2,n_experts=288,host_mask=mask([2,3]),gpu_mask=mask([1,5]),ids=[1,2,7,8,9,10,11,12])
def limits(**kwargs):
    v=dict(session="test",pin=PIN,expected_pin=PIN,generation=5,expected_generation=5,
           enabled=True,nvme_GBps=9.496,upload_GBps=8.,deadline_ms=12.)
    v.update(kwargs)
    return Limits(**v)

class ShadowTests(unittest.TestCase):
    def test_disabled(self):
        self.assertEqual(plan([2,4],event(),limits(enabled=False))[1],"disabled")
    def test_tier_aware(self):
        x,status=plan([1,2,4,5],event(),limits())
        self.assertEqual(status,"proposed")
        self.assertEqual([p["expert"] for p in x],[2,4])
        self.assertEqual(x[0]["nominal_nvme_bytes"],0)
        self.assertEqual(x[1]["nominal_nvme_bytes"],BYTES_PER_EXPERT)
    def test_no_target_leakage(self):
        a=event()
        b=dict(a,ids=[18,17,16,15,14,13,12,11])
        self.assertEqual(plan([4,2,1],a,limits()),plan([4,2,1],b,limits()))
    def test_stale(self):
        self.assertEqual(plan([2,4],event(),limits(generation=4))[1],"stale_session_or_generation")
        self.assertEqual(plan([2,4],event(),limits(expected_pin="f"*40))[1],"pin_mismatch")
    def test_budget_deadline(self):
        self.assertFalse(plan([2,4],event(),limits(gpu_headroom=0))[0])
        self.assertFalse(plan([2,4],event(),limits(deadline_ms=.01))[0])
        p,_=plan([4,2],event(),limits(nvme_budget=0))
        self.assertEqual([q["expert"] for q in p],[2])
    def test_invalid_inputs(self):
        self.assertEqual(plan([2,4],event(),limits(upload_GBps=0))[1],"no_valid_resource_envelope")
        self.assertEqual(plan([2],dict(event(),v=1),limits())[1],"missing_full_inventory")
        self.assertEqual(plan([2],dict(event(),host_mask="ff"),limits())[1],"bad_cache_mask")
        self.assertEqual(plan([2,2],event(),limits())[1],"invalid_candidates")
    def test_cap(self):
        p,_=plan(list(range(32)),event(),limits())
        self.assertLessEqual(len(p),2)
        self.assertLessEqual(sum(x["nominal_gpu_bytes"] for x in p),2*BYTES_PER_EXPERT)
if __name__=="__main__":unittest.main()
