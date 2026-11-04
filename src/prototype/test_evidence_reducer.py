import unittest
from datetime import datetime, timedelta, timezone
from correlation_engine import Event, Cluster
from evidence_reducer import Policy, choose, evaluate

T = datetime(2024, 1, 1, tzinfo=timezone.utc)

def e(n, tech=(), level=3, host='a', rule='x'):
    return Event(str(n), T+timedelta(seconds=n), host, '', '', '', '', rule,
                 level, tuple(tech), (), '')

def c(events, links=()):
    return Cluster(1, events[0].timestamp, events[-1].timestamp, list(events), list(links))

class ReducerTests(unittest.TestCase):
    def test_real_preservation_and_critical(self):
        cc = c([e(0, ('T1001',)), e(1, ('T1001','T1002')), e(2, ('T1002',), level=13), e(3, ('T1001',))])
        for method in ('all','first_cover','greedy_cover'):
            indices, audit = choose(cc, method)
            self.assertTrue(audit['constraints_total']==audit['constraints_covered'])
            self.assertIn(2, indices)
            self.assertIn(0, indices)
            self.assertIn(3, indices)

    def test_greedy_not_worse_than_first_in_construction(self):
        cc=c([e(0, ('A',)), e(1, ('B',)), e(2, ('C',)), e(3, ('B','C')), e(4, ('A',))])
        policy=Policy(keep_anchors=False)
        first,_=choose(cc,'first_cover',policy)
        greedy,_=choose(cc,'greedy_cover',policy)
        self.assertEqual(len(first),3)
        self.assertEqual(len(greedy),2)

    def test_unknown_rule_not_silenced(self):
        cc=c([e(0),e(1,rule='y'),e(2,rule='x')])
        indices,_=choose(cc,'greedy_cover',Policy(keep_anchors=False))
        self.assertEqual(len(indices),2)
        self.assertIn(1,indices)

    def test_observed_link_endpoint_retained(self):
        cc=c([e(0),e(1),e(2)], [{'from_id':'0','to_id':'1'},{'from_id':'1','to_id':'2'}])
        selected,audit=choose(cc,'greedy_cover',Policy(keep_anchors=False))
        self.assertIn(1,selected)
        self.assertEqual(audit['constraints_covered'],audit['constraints_total'])

    def test_duplicate_uid_conservative(self):
        cc=c([e(0), e(0), e(2)], [{'from_id':'0','to_id':'2'}])
        selected,_=choose(cc,'greedy_cover',Policy(keep_anchors=False))
        self.assertEqual(selected,[0,1,2])

    def test_host_representation(self):
        cc=c([e(0,host='a'),e(1,host='b'),e(2,host='a')])
        selected,_=choose(cc,'greedy_cover',Policy(keep_anchors=False))
        self.assertEqual({cc.members[i].host for i in selected},{'a','b'})

    def test_empty_and_singleton(self):
        empty=Cluster(1,T,T,[])
        self.assertEqual(choose(empty)[0],[])
        self.assertEqual(choose(c([e(0)]))[0],[0])

    def test_comparison(self):
        cc=c([e(0,('T1',)), e(1,('T1',)),e(2,('T1',))])
        rows,details=evaluate([cc])
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['visible_alerts'],3)
        self.assertTrue(all(r['constraints_preserved'] for r in rows))
        self.assertEqual(len(details),3)

if __name__=='__main__':
    unittest.main()
