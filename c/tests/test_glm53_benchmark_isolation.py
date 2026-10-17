#!/usr/bin/env python3
"""Selection of benchmark scopes is exact; no incidental systemd services."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SRC=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/exclusive_joyner_restore.py"
sp=importlib.util.spec_from_file_location("glm53_isolation",SRC)
assert sp and sp.loader
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

class IsolationRestore(unittest.TestCase):
    def test_exact_scope_pattern(self):
        lines="""glm53-tier-ab-a1b2c3d4e5f6.scope loaded active running
fleet-r9700-toolcall.service loaded active running
glm53-tier-ab-123456789abc.scope loaded inactive dead
glm53-tier-ab-123456789abz.scope loaded active running
other-glm53-tier-ab-123456789abc.scope loaded active running
glm53-tier-ab-123456789abc.service loaded active running"""
        self.assertEqual(m.owned_scopes(lines),[
            "glm53-tier-ab-123456789abc.scope",
            "glm53-tier-ab-a1b2c3d4e5f6.scope"])
    def test_no_generic_glm_or_python_match(self):
        self.assertEqual(m.owned_scopes("python 1234 glm53 --model x\nsshd.service\n"),[])
    def test_restore_targets_are_exact(self):
        self.assertEqual(m.UNITS,("fleet-r9700-vulkan-canary.service",
                                   "fleet-r9700-toolcall.service"))
    def test_missing_scope_is_noop(self):
        class Dummy:
            returncode=0
            stdout="unrelated.scope loaded active\n"
            stderr=""
        with patch.object(m,"run",return_value=Dummy()) as runner:
            self.assertEqual(m.stop_owned_scopes(),[])
            self.assertEqual(runner.call_count,2)
if __name__=="__main__":
    unittest.main()
