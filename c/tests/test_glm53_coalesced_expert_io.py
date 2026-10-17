"""Source-immutable, offline tests of exact six-tensor grouped expert reads."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[2] / "benchmarks" / "joyner-glm53-20261016"
sys.path.insert(0, str(HERE))
import benchmark_coalesced_complete_expert as bench

class SegmentationTests(unittest.TestCase):
    def test_one_six_tensor_run(self):
        p = Path("/tmp/fixture-shard")
        a = [(p, i * 4096, 4096) for i in range(6)]
        grouped = bench.extents(a, True)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0][2]-grouped[0][1], 24576)
        self.assertEqual(len(grouped[0][3]), 6)
        self.assertEqual(len(bench.extents(a, False)), 6)

    def test_two_runs_when_there_is_a_gap(self):
        p = Path("/tmp/fixture-shard")
        a = [(p, i*4096, 4096) for i in range(3)]
        a += [(p, 16384+i*4096, 4096) for i in range(3)]
        self.assertEqual(len(bench.extents(a, True)), 2)

    def test_four_runs_across_shards(self):
        p = Path("/tmp/fixture-shard-A")
        q = Path("/tmp/fixture-shard-B")
        a = [(p, 0, 4096), (p, 4096, 4096),
             (p, 12288, 4096), (q, 0, 4096),
             (q, 8192, 4096), (q, 12288, 4096)]
        self.assertEqual(len(bench.extents(a, True)), 4)

    def test_overlap_fails_closed(self):
        p = Path("/tmp/fixture-shard")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            bench.extents([(p, 0, 8192), (p, 4096, 4096)], True)

    def test_no_file_crossing_or_gap_overfetch(self):
        p, q = Path("/tmp/a"), Path("/tmp/b")
        got = bench.extents([(q, 0, 4096), (p, 0, 4096),
                             (q, 4096, 4096), (p, 8192, 4096)], True)
        self.assertEqual([(str(e[0]), e[1], e[2]) for e in got],
            [("/tmp/a",0,4096), ("/tmp/a",8192,12288), ("/tmp/b",0,8192)])

class ExactBytesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.safetensors"
        self.prefix = 97
        self.parts = [(self.path, 97+i*4096, 4096) for i in range(6)]
        self.path.write_bytes(os.urandom(self.prefix+6*4096))

    def open_fixture(self):
        # Do not depend on ephemeral CI filesystem O_DIRECT support.
        patch = mock.patch.object(bench.os, "O_DIRECT", 0)
        patch.start()
        self.addCleanup(patch.stop)
        fd, tails, snapshot = bench.open_checked([self.path])
        def cleanup():
            for handle in [*fd.values(), *tails.values()]:
                os.close(handle)
        self.addCleanup(cleanup)
        return fd, tails, snapshot

    def test_grouped_exact_six_tensor_bytes_including_eof(self):
        fd, tails, snapshot = self.open_fixture()
        grp = {"calls":0,"direct_bytes":0,"tail_bytes":0,"tail_calls":0}
        six = dict(grp)
        _, a = bench.checked_read(bench.extents(self.parts, True), fd, tails,
                                  grp, verify=True)
        _, b = bench.checked_read(bench.extents(self.parts, False), fd, tails,
                                  six, verify=True)
        self.assertEqual(a,b)
        self.assertEqual(len(a),6)
        self.assertEqual(grp["calls"],1)
        self.assertEqual(six["calls"],6)
        self.assertEqual(grp["tail_bytes"],self.prefix)
        bench.check_sources(fd, snapshot)

    def test_symlinks_rejected(self):
        link = Path(self.temp.name) / "symlink"
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            bench.open_checked([link])

    def test_mutation_rejected(self):
        fd, tails, snap = self.open_fixture()
        self.path.write_bytes(b'Z' * self.path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "source changed"):
            bench.check_sources(fd, snap)

if __name__ == "__main__":
    unittest.main()
