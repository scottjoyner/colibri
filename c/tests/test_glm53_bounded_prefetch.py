"""Deterministic safety tests for validation-calibrated QD2 shadow reader."""
from __future__ import annotations
import pathlib,sys,threading,time,unittest

DIR=pathlib.Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016"
sys.path.insert(0,str(DIR))
from bounded_shadow_reader import AdmissionDenied,BoundedShadowReader
from calibrated_prefetch_budget import fit

PIN="46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
def new(read=lambda fd,off:off,**opts):
    return BoundedShadowReader(session="s1",pin=PIN,generation=3,read_fn=read,**opts)
class ShadowReaderTests(unittest.TestCase):
    def test_owner_generation_and_pin_denial(self):
        with new() as reader:
            for key in (dict(session="s2",pin=PIN,generation=3),
                        dict(session="s1",pin="f"*40,generation=3),
                        dict(session="s1",pin=PIN,generation=4)):
                with self.assertRaises(AdmissionDenied):
                    reader.read_batch(**key,requests=[(5,4096)])
            self.assertEqual(reader.submitted,0)
            self.assertEqual(reader.read_batch(session="s1",pin=PIN,generation=3,
                                                requests=[(1,1234)]),[1234])
            reader.revoke()
            with self.assertRaises(AdmissionDenied):
                reader.read_batch(session="s1",pin=PIN,generation=3,requests=[(1,1)])
    def test_queue_and_budget_rejection(self):
        with new() as reader:
            with self.assertRaises(AdmissionDenied):
                reader.read_batch(session="s1",pin=PIN,generation=3,
                                  requests=[(1,1),(1,2),(1,3)])
            self.assertEqual(reader.read_batch(session="s1",pin=PIN,generation=3,
                                                requests=[(1,1),(1,2)]),[1,2])
            self.assertEqual(reader.completed,2)
    def test_reads_drain_before_returning_failure(self):
        started=threading.Event();done=threading.Event();second_started=threading.Event()
        def fake(fd,off):
            started.set()
            if off==1:
                if not second_started.wait(timeout=1):
                    raise RuntimeError("second worker did not start")
                raise RuntimeError("simulated read failure")
            second_started.set()
            time.sleep(.025)
            done.set()
            return off
        with new(read=fake) as reader:
            with self.assertRaisesRegex(RuntimeError,"simulated read failure"):
                reader.read_batch(session="s1",pin=PIN,generation=3,
                                  requests=[(1,1),(1,2)])
            self.assertTrue(started.is_set())
            self.assertTrue(done.is_set())
            with self.assertRaises(AdmissionDenied):
                reader.read_batch(session="s1",pin=PIN,generation=3,requests=[])
    def test_no_work_after_close(self):
        reader=new()
        reader.close()
        with self.assertRaises(AdmissionDenied):
            reader.read_batch(session="s1",pin=PIN,generation=3,requests=[])
    def test_gate_cannot_see_current_validation_label_before_ranking(self):
        events=[dict(layer=3,call=i,row=0,ids=[1,2] if i%2 else [3,4])
                for i in range(30)]
        observed=[]
        def rank(ev,history):
            observed.append(dict(history))
            return [1,2,3,4]
        x=fit(events,rank,threshold=.40)
        self.assertEqual(x.train_rows,30)
        self.assertEqual(x.layer_second_probability[3],.5)
        self.assertTrue(x.permitted(3,1))
        self.assertFalse(x.permitted(3,2))
        self.assertEqual(observed[0],{})
        self.assertNotIn(("token",3,0),observed[0])
    def test_calibration_invalid_empty(self):
        with self.assertRaises(ValueError):
            fit([],lambda ev,h:[1,2])
        with self.assertRaises(ValueError):
            fit([dict(layer=2,call=i,row=0,ids=[1,2]) for i in range(30)],
                lambda ev,h:[1,2],threshold=1.1)
if __name__=="__main__":unittest.main()
