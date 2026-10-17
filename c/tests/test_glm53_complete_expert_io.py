"""Fail-closed complete GLM expert span tests, independent of checkpoint files."""
from __future__ import annotations
import pathlib,sys,unittest
HERE=pathlib.Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016"
sys.path.insert(0,str(HERE))
from benchmark_complete_expert_qd2 import safe_aligned_span,FOUR,HALF,LOGICAL_EXPERT_BYTES,TENSORS,WINDOW

class CompleteExpertIOTests(unittest.TestCase):
    def test_six_required_tensors_and_exact_logical_size(self):
        self.assertEqual(len(TENSORS),6)
        self.assertEqual(sum(n for _,n in TENSORS),LOGICAL_EXPERT_BYTES)
        self.assertEqual(LOGICAL_EXPERT_BYTES,14155776)
        self.assertEqual(sum(name.endswith(".qs") for name,_ in TENSORS),3)
    def test_alignment_interior(self):
        start=2*WINDOW+127
        a,n,b,tail=safe_aligned_span(start,FOUR,start+FOUR+WINDOW)
        self.assertEqual(a,2*WINDOW)
        self.assertEqual(n,FOUR+WINDOW)
        self.assertEqual(tail,0)
        self.assertEqual(b,a+n)
    def test_real_eof_non_aligned_tail(self):
        start=WINDOW+127
        end=start+HALF
        a,n,b,tail=safe_aligned_span(start,HALF,end)
        self.assertLessEqual(b,end)
        self.assertEqual(tail,end-b)
        self.assertEqual(tail,127)
        self.assertLess(tail,WINDOW)
        self.assertEqual(n,HALF)
    def test_reject_reads_beyond_end(self):
        with self.assertRaisesRegex(ValueError,"exceeds source"):
            safe_aligned_span(WINDOW+1,HALF,WINDOW+HALF)
        with self.assertRaisesRegex(ValueError,"exceeds source"):
            safe_aligned_span(-1,HALF,WINDOW+HALF)
    def test_fail_when_subblock_only(self):
        with self.assertRaisesRegex(ValueError,"alignment"):
            safe_aligned_span(100,20,200)
if __name__=="__main__":unittest.main()
