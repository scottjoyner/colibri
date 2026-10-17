#!/usr/bin/env python3
"""Safety and accounting controls for benchmark process interference."""
import importlib.util
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[2]/"benchmarks/joyner-glm53-20261016/glm53_competitors.py"
spec=importlib.util.spec_from_file_location("glm53_competitors_test",SCRIPT)
assert spec and spec.loader
mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)

def item(pid=123,start=100,cpu=0,io=0,rss=30,scope=""):
    return mod.ProcessSample(pid,start,cpu,io,rss,"worker",scope)

class Interference(unittest.TestCase):
    def test_cpu_core_threshold(self):
        clk=os.sysconf("SC_CLK_TCK")
        start={123:item(cpu=0)}
        end={123:item(cpu=clk*5)}
        self.assertIn("cpu",mod.observe(start,end,5)[0]["reasons"])
    def test_disk_io_deltas(self):
        before={123:item(io=0)}
        after={123:item(io=80*mod.MIB)}
        got=mod.observe(before,after,2)
        self.assertEqual(got[0]["reasons"],["disk_io"])
    def test_memory_pressure(self):
        got=mod.observe({123:item()}, {123:item(rss=3*1024*1024)},1)
        self.assertEqual(got[0]["reasons"],["memory"])
    def test_same_scope_never_competes_with_itself(self):
        clk=os.sysconf("SC_CLK_TCK")
        initial={123:item(cpu=0,io=0,scope="0::/app.slice/glm53-tier-ab-123456789abc.scope")}
        final={123:item(cpu=clk*10,io=500*mod.MIB,
                       scope="0::/app.slice/glm53-tier-ab-123456789abc.scope")}
        self.assertEqual(mod.observe(initial,final,2,
            excluded_scope="glm53-tier-ab-123456789abc.scope"),[])
    def test_pid_reuse_does_not_generate_false_sample(self):
        self.assertEqual(mod.observe({123:item(start=1)},
                                     {123:item(start=2,cpu=10**9,io=10**9)},1),[])
    def test_unknown_io_does_not_crash(self):
        initial={123:item(io=None)}
        final={123:item(io=None)}
        self.assertEqual(mod.observe(initial,final,1),[])
    def test_excluded_pids(self):
        self.assertEqual(mod.observe({123:item()}, {123:item(rss=10**9)},1,
                        exclude_pids=frozenset({123})),[])
    def test_reject_zero_seconds(self):
        with self.assertRaises(ValueError):
            mod.observe({}, {},0)
    def test_process_stat_parse_and_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/"111"
            parent.mkdir()
            fields=["S"]+["0"]*50
            fields[11]="180";fields[12]="20";fields[19]="4567"
            (parent/"stat").write_text("111 (an odd proc name) "+" ".join(fields)+"\n")
            (parent/"statm").write_text("500 7 0 0 0 0 0\n")
            (parent/"cgroup").write_text("0::/app.slice/test.scope\n")
            (parent/"io").write_text("read_bytes: 2300\nwrite_bytes: 100\n")
            r=mod.process_sample(111,Path(tmp))
            self.assertEqual(r.cpu_ticks,200)
            self.assertEqual(r.starttime,4567)
            self.assertEqual(r.io_bytes,2400)
            self.assertEqual(r.comm,"an odd proc name")
    def test_nvme_block_stat_is_512_byte_sectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=Path(tmp)/"stat"
            x.write_text("100 0 2048 0 20 0 1024 0 0 0 0\n")
            self.assertEqual(mod.block_bytes(x),3072*512)
            self.assertEqual(mod.block_rate(0,3072*512,2),786432)
    def test_invalid_nvme_sample_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=Path(tmp)/"stat"
            x.write_text("1 2 3\n")
            with self.assertRaisesRegex(ValueError,"NVMe"):
                mod.block_bytes(x)
            with self.assertRaisesRegex(ValueError,"disk"):
                mod.block_rate(100,99,1)
    def test_syscall_cpu_sampling_reports_only_competitors(self):
        a={123:item(cpu=0),456:item(cpu=0)}
        clk=os.sysconf("SC_CLK_TCK")
        b={123:item(cpu=clk*5),456:item(cpu=clk)}
        with patch.object(mod.os,"sysconf",return_value=clk):
            report=mod.observe(a,b,5)
        self.assertEqual([r["pid"] for r in report],[123])
if __name__=="__main__":
    unittest.main()
