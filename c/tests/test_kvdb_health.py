import json
import unittest

from openai_server import Engine


class StatsParsingTest(unittest.TestCase):
    def test_stats_parses_kvdb_hot_trailer(self):
        # mux-mode STAT line carries the kvdb=1 / kvhot_* trailer
        line = "STAT 12 3.5 0.0 24.1 7 0 kvdb=1 kvhot_n=3 kvhot_mb=12.4 kvhot_hits=9 kvhot_loads=2"
        stats = Engine._stats(line.split())
        hot = stats["kv_cache_hot"]
        self.assertTrue(hot["enabled"])
        self.assertEqual(hot["prompts"], 3)
        self.assertEqual(hot["size_mb"], 12.4)
        self.assertEqual(hot["hits"], 9)
        self.assertEqual(hot["promoted_from_ssd"], 2)

    def test_stats_without_kvdb_trailer_has_no_hot(self):
        line = "STAT 12 3.5 0.0 24.1 7 0"
        stats = Engine._stats(line.split())
        self.assertNotIn("kv_cache_hot", stats)
