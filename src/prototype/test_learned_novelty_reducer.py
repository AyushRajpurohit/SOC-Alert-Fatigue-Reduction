"""Synthetic-only tests; results are not empirical SOC findings."""
import unittest
from datetime import datetime, timedelta, timezone
from correlation_engine import Event, Cluster
from evidence_reducer import Policy, choose, requirements
from learned_novelty_reducer import (context_features, train_predictor,
    novelty_scores, learned_cover, evaluate)

T = datetime(2024, 1, 1, tzinfo=timezone.utc)
def ev(i, rule='R', host='h', level=3, user='', source='', tech=()):
    return Event(str(i), T+timedelta(seconds=i), host, '', source, '', user, rule,
                 level, tuple(tech), (), '')

class NoveltyTests(unittest.TestCase):
    def test_no_current_rule_leak_into_features(self):
        a=[ev(0,'A'),ev(1,'B'),ev(2,'C')]
        b=[ev(0,'A'),ev(1,'Z'),ev(2,'C')]
        af,bf=context_features(a),context_features(b)
        self.assertEqual(af[1],bf[1])
        self.assertNotEqual(af[2]['prev_host_rule'],bf[2]['prev_host_rule'])

    def test_train_scores_finite_and_nonnegative(self):
        a=[ev(i,'A' if i%2 else 'B',host=f'h{i%3}') for i in range(60)]
        dv,m,f=train_predictor(a)
        scores=novelty_scores(a[-10:],dv,m,f)
        self.assertTrue(all(0<=x<=14 for x in scores))

    def test_mandatory_and_coverage_preserved(self):
        a=[ev(0,tech=('T1',)),ev(1,tech=('T2',)),ev(2,tech=('T1','T2',),level=11),ev(3,tech=('T1',))]
        c=Cluster(1,a[0].timestamp,a[-1].timestamp,a,[{'from_id':'0','to_id':'1'}])
        policy=Policy(critical_level=10)
        U,cov,mandatory=requirements(c,policy)
        for mode in ('novelty_tiebreak','novelty_weighted'):
            selected=learned_cover(c,[1,2,5,1],policy,mode)
            covered=set().union(*(cov[i] for i in selected))
            self.assertEqual(covered,U)
            self.assertTrue(mandatory.issubset(selected))
            self.assertIn(2,selected)

    def test_comparator_counts_and_extras(self):
        a=[ev(i,rule='R',user='u' if i<2 else 'v',tech=('T1',)) for i in range(5)]
        rows=evaluate(a,[1,2,3,4,5],Policy(critical_level=10))
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r['constraints_covered']==r['constraints_total'] for r in rows))
        self.assertTrue(all(r['extra_proxy_total']>=r['extra_proxy_retained'] for r in rows))

    def test_too_small_train_rejected(self):
        with self.assertRaises(ValueError):
            train_predictor([ev(i) for i in range(5)])

if __name__=='__main__': unittest.main()
