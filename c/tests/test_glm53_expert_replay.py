import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TARGET=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/replay_expert_prefetch.py"
spec=importlib.util.spec_from_file_location("replay_expert_prefetch",TARGET)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class PredictiveReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def write(self,name,shift=0,bad=False):
        path=self.root/name
        rows=[]
        for call in range(6):
            for row in range(3):
                layer=3+call
                selected=[(shift+layer*4+row+i)%288 for i in range(8)]
                ev={"v":1,"call":call,"row":row,"layer":layer,
                    "n_experts":288,"topk":8,"host_slots":1,
                    "host_capacity":1,"ids":selected,"gates":[.125]*8,
                    "resident":[0]*8,"sketch":[round((i+layer)/16,4) for i in range(16)]}
                rows.append(ev)
        if bad:rows[0]["ids"][0]=999
        path.write_text("".join(json.dumps(e)+"\n" for e in rows))
        return path
    def test_holdout_replay(self):
        a=self.write("train-1.jsonl")
        b=self.write("train-2.jsonl")
        c=self.write("holdout.jsonl")
        result=mod.replay([a,b],[c],16)
        self.assertEqual(result["rows_evaluated"],18)
        self.assertEqual({p["policy"] for p in result["policies"]},
                         {"freq","prior-token","cross-layer"})
        self.assertTrue(all(0<=p["selected_recall"]<=1 for p in result["policies"]))
        self.assertEqual(result,mod.replay([a,b],[c],16))
    def test_no_train_eval_overlap(self):
        a=self.write("a.jsonl")
        b=self.write("b.jsonl")
        with self.assertRaisesRegex(ValueError,"evaluation files"):
            mod.replay([a,b],[a],16)
    def test_missing_independent_training_run(self):
        a=self.write("a.jsonl")
        b=self.write("b.jsonl")
        with self.assertRaisesRegex(ValueError,"requires >=2"):
            mod.replay([a],[b],16)
    def test_invalid_expert_refused(self):
        a=self.write("bad.jsonl",bad=True)
        with self.assertRaisesRegex(ValueError,"invalid"):
            mod.read_trace(a)
    def test_missing_trace_refused(self):
        a=self.write("empty.jsonl")
        a.write_text("")
        with self.assertRaisesRegex(ValueError,"empty"):
            mod.read_trace(a)

if __name__=="__main__":unittest.main()
