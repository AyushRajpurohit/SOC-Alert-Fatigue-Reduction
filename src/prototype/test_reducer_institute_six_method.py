import csv
import json
import tempfile
import unittest
from pathlib import Path

from reducer_institute_six_method import METHODS, run


class InstituteAdapterTests(unittest.TestCase):
    def make_fixture(self, root):
        folder = root / 'sanitized'
        folder.mkdir()
        for sample in ('A','B','C'):
            rows = []
            for j, (tech, level) in enumerate((('T1059', 3),('T1059', 3),('T1110',10),('',4))):
                obj = {'id': f'event_{j+1:08d}',
                       'timestamp':f'2024-01-01T00:00:{j:02d}+00:00',
                       'agent':{'id':'host_example'},
                       'rule':{'id':str(100+j//2),'level':level},
                       'data':{'dstuser':'user_pseudonym'}}
                if tech:
                    obj['rule']['mitre']={'id':[tech]}
                rows.append(obj)
            (folder/f'sample_{sample}.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
        return folder

    def test_three_sample_aggregate_and_privacy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = self.make_fixture(root)
            dest = root/'out'
            rows, manifest = run(folder, dest)
            self.assertEqual(len(rows), 24)
            by = {(r['sample'],r['method']):r for r in rows}
            self.assertEqual(by[('TOTAL_SEPARATE_WINDOWS','all')]['alerts'],12)
            for name in ('A','B','C','TOTAL_SEPARATE_WINDOWS'):
                self.assertTrue(by[(name,'greedy_cover')]['all_constraints_preserved'])
                self.assertTrue(by[(name,'first_cover')]['all_constraints_preserved'])
                self.assertEqual(by[(name,'severity_budget')]['visible'],by[(name,'greedy_cover')]['visible'])
            text = (dest/'method_comparison.csv').read_text()+(dest/'manifest.json').read_text()
            self.assertNotIn('host_example',text)
            self.assertNotIn('user_pseudonym',text)
            self.assertNotIn('event_000',text)
            self.assertNotIn(str(folder),text)
            self.assertEqual(manifest['source'], 'locally sanitized, authorized institute Wazuh fixed two-minute samples')
            with self.assertRaises(FileExistsError):
                run(folder,dest)

    def test_missing_file_refused_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = self.make_fixture(root)
            (folder/'sample_C.jsonl').unlink()
            with self.assertRaises(FileNotFoundError):
                run(folder,root/'out')
            self.assertFalse((root/'out').exists())

    def test_duplicate_label_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            folder=self.make_fixture(root)
            with self.assertRaises(ValueError):
                run(folder,root/'out',names=('A','A'))

if __name__ == '__main__':
    unittest.main()
