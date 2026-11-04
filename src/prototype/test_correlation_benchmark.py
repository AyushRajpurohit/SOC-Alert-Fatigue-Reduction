import tempfile
import unittest
from pathlib import Path
from correlation_benchmark import (aisoc_style_groups, all_methods, fixture_events,
                                   pair_metrics, run, stage_diagnostic)


class BenchmarkTests(unittest.TestCase):
    def test_fixture_membership(self):
        events, labels = fixture_events()
        self.assertEqual(len(events), 8)
        self.assertEqual(set(labels.values()), {'A', 'B', 'C'})
        self.assertEqual(len({e.uid for e in events}), 8)

    def test_linked_engine_synthetic(self):
        events, labels = fixture_events()
        groups = all_methods(events)['evidence_linked']()
        m = pair_metrics(groups, labels)
        self.assertEqual(m['pairwise_fp'], 0)
        self.assertEqual(m['pairwise_fn'], 0)
        self.assertEqual(m['pairwise_tp'], 7)

    def test_baseline_false_merges_synthetic(self):
        events, labels = fixture_events()
        for name in ('host_time', 'aisoc_style_key_window'):
            m = pair_metrics(all_methods(events)[name](), labels)
            self.assertGreater(m['pairwise_fp'], 0)

    def test_stage_window_empty(self):
        self.assertIsNone(stage_diagnostic([],[])['mean_groups_per_populated_window'])

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)/'data'; data.mkdir()
            out = Path(d)/'out'; out.mkdir()
            (out/'existing').write_text('do not erase')
            with self.assertRaises(FileExistsError): run(data,out)
            self.assertEqual((out/'existing').read_text(),'do not erase')

    def test_missing_data_not_fabricated(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)/'data'; data.mkdir()
            out = Path(d)/'out'
            real, synthetic = run(data,out)
            self.assertEqual(real, [])
            self.assertEqual(len(synthetic),3)
            import json
            m=json.loads((out/'manifest.json').read_text())
            self.assertEqual(len(m['files_missing']),4)
            self.assertFalse((out/'lab_grouping.csv').exists())

if __name__ == '__main__': unittest.main()
