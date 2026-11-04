import json
import tempfile
import unittest
from pathlib import Path

from correlation_engine import parse_wazuh
from integrated_replay import make_view, run


def rec(i, sec, rule='5501', level=5, host='01', src='10.0.0.1', mitre=None):
    obj = {'timestamp':f'2026-11-02T12:00:{sec:02d}Z','agent':{'id':host},
           'rule':{'id':rule,'level':level,'mitre':{'id':mitre or []}},
           'data':{'srcip':src,'dstuser':'u'}}
    return parse_wazuh(obj,'smoke',i)


class IntegrationTests(unittest.TestCase):
    def test_replay_conserves_alerts_and_safety_policy(self):
        events = [rec(1,0),rec(2,1),rec(3,2,level=10),rec(4,3)]
        x = make_view(events, scorer_mode='rules', model='/nonexistent')
        s=x['stats']
        self.assertEqual(s['alerts'],4)
        self.assertEqual(s['visible']+s['hidden'],4)
        self.assertEqual(s['requirements'],s['covered'])
        self.assertEqual(s['protected'],s['protected_retained'])
        self.assertEqual(s['protected'],1)
        self.assertEqual(sum(len(i['alerts']) for i in x['incidents']),4)
        self.assertTrue(any(i['decision']!='DEMOTE' for i in x['incidents']))

    def test_both_correlation_modes_and_all_reducer(self):
        events=[rec(1,0),rec(2,1),rec(3,2,host='02')]
        for correlation in ('host_time','evidence_linked'):
            x=make_view(events,correlation=correlation,method='all',scorer_mode='rules',model='/no')
            self.assertEqual(x['stats']['visible'],3)
            self.assertEqual(x['stats']['hidden'],0)

    def test_missing_model_refused(self):
        with self.assertRaises(FileNotFoundError):
            make_view([rec(1,0)],scorer_mode='learned',model='/no')

    def test_writes_product_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.jsonl'
            obj={'timestamp':'2026-11-02T12:00:00Z','agent':{'id':'01'},'rule':{'id':'5501','level':10}}
            path.write_text(json.dumps(obj)+'\n')
            out=Path(d)/'out'
            run([path],out,scorer='rules',model='/no')
            self.assertTrue((out/'index.html').is_file())
            self.assertEqual(json.loads((out/'state.json').read_text())['stats']['alerts'],1)
            with self.assertRaises(FileExistsError):
                run([path],out,scorer='rules',model='/no')


if __name__=='__main__':unittest.main()
