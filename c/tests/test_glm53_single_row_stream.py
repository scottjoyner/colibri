"""The new cold-expert decode path remains isolated, pinned and default OFF."""
import importlib.util
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

B=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016"
sys.path.insert(0,str(B))
sp=importlib.util.spec_from_file_location("glm53_single_stream_runner",B/"run_joyner_native_tier_ab.py")
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
SOURCE=(Path(__file__).resolve().parents[1]/"vk_tier.c").read_text()

class DecodeStreamingGuard(unittest.TestCase):
    def argv(self,label,enabled=0,isolated=True,fair=1):
        args=["run","--binary","/nonexistent","--out-dir","/nonexistent",
              "--tier-gib","4","--rate","32","--dense-host","0",
              "--text-only","1","--fair-layers",str(fair),
              "--single-row-stream",str(enabled),"--label",label]
        return args+["--isolated"] if isolated else args
    def test_stream_case_with_explicit_optin_is_pinned(self):
        with patch.object(sys,"argv",self.argv("exclusive4stream-123abc123abc",1)),patch.object(m,"validate",side_effect=RuntimeError("VALID_CASE")):
            with self.assertRaisesRegex(RuntimeError,"VALID_CASE"):m.main()
    def test_stream_case_missing_flag_denied(self):
        with patch.object(sys,"argv",self.argv("exclusive4stream-123abc123abc",0)),patch.object(m,"validate") as check:
            with self.assertRaises(SystemExit):m.main()
            check.assert_not_called()
    def test_stream_flag_without_isolation_denied(self):
        with patch.object(sys,"argv",self.argv("exclusive4stream-123abc123abc",1,isolated=False)),patch.object(m,"validate") as check:
            with self.assertRaises(SystemExit):m.main()
            check.assert_not_called()
    def test_stream_flag_wrong_fair_share_denied(self):
        with patch.object(sys,"argv",self.argv("exclusive4stream-123abc123abc",1,fair=0)),patch.object(m,"validate") as check:
            with self.assertRaises(SystemExit):m.main()
            check.assert_not_called()
    def test_flag_cannot_escape_into_normal_control(self):
        with patch.object(sys,"argv",self.argv("exclusive4control-123abc123abc",1,fair=0)),patch.object(m,"validate") as check:
            with self.assertRaises(SystemExit):m.main()
            check.assert_not_called()
    def test_runner_exports_feature_to_child_not_only_receipt(self):
        src=(B/"run_joyner_native_tier_ab.py").read_text()
        self.assertIn('"COLI_VK_TIER_SINGLE_ROW_STREAM","USAGE_SAVE"',src)
        self.assertIn('experimental_feature_not_activated',src)
    def test_default_off_and_glm_only(self):
        self.assertIn('T.single_row_stream=!strcmp(T.engine,"glm53")',SOURCE)
        self.assertIn('!strcmp(single_flag,"1")',SOURCE)
        self.assertIn('S == 1 && T.single_row_stream',SOURCE)
    def test_no_change_to_regular_prefill(self):
        self.assertIn('S >= T.gemm_rows || (S == 1 && T.single_row_stream)',SOURCE)
    def test_exact_one_row_threshold(self):
        self.assertIn('R = (S == 1 && T.single_row_stream) ? 1 : stream_min_rows()',SOURCE)
if __name__=="__main__":unittest.main()
