import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta

from correlation_engine import Event, Cluster
from evidence_reducer import Policy, requirements
from reducer_six_method_eval import METHODS, selections, audit_cluster, run

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def ev(i, level=5, tech=(), rule='11'):
    return Event(uid=str(i),timestamp=BASE+timedelta(seconds=i),host='h1',host_ip='',src_ip='',dst_ip='',
                 user='',rule=rule,level=level,techniques=tech,tactics=())


class SixMethodTests(unittest.TestCase):
    def test_six_methods(self):
        c=Cluster(cid=1,first=BASE,last=BASE+timedelta(seconds=2),members=[ev(0,tech=('T1',)),ev(1,tech=('T1','T2')),ev(2,tech=('T2',))])
        s=selections(c,Policy())
        self.assertEqual(set(s),set(METHODS))
        self.assertEqual(len(s['all']),3)
        self.assertEqual(len(s['severity_budget']),len(s['greedy_cover']))
        for m in ('greedy_cover','first_cover'):
            a=audit_cluster(c,s[m],Policy())
            self.assertEqual(a['constraints_total'],a['constraints_covered'])
            self.assertEqual(a['hard_required'],a['hard_retained'])

    def test_baselines_can_lose_constraints(self):
        c=Cluster(cid=1,first=BASE,last=BASE+timedelta(seconds=3),members=[
            ev(0,tech=('T1',),rule='1'), ev(1,tech=('T1',),rule='1'),
            ev(2,tech=('T2',),rule='2'),ev(3,tech=(),rule='unknown')])
        p=Policy()
        s=selections(c,p)
        self.assertTrue(audit_cluster(c,s['greedy_cover'],p)['cases_with_unmet_requirements']==0)
        self.assertGreater(audit_cluster(c,s['one_per_technique'],p)['cases_with_unmet_requirements'],0)

    def test_critical_retention_audited(self):
        c=Cluster(cid=1,first=BASE,last=BASE+timedelta(seconds=3),members=[
            ev(0),ev(1,level=15),ev(2),ev(3)])
        p=Policy(critical_level=12)
        a=audit_cluster(c,selections(c,p)['greedy_cover'],p)
        self.assertEqual(a['critical_total'],1)
        self.assertEqual(a['critical_retained'],1)

    def test_empty_cluster(self):
        c=Cluster(cid=1,first=BASE,last=BASE,members=[])
        for method,selected in selections(c,Policy()).items():
            self.assertEqual(selected,[],method)
            self.assertEqual(audit_cluster(c,selected,Policy())['constraints_total'],0)

    def test_lab_run_aggregate_only_and_no_overwrite(self):
        lab=Path(__file__).with_name('evidence_reducer_labdata')
        if not lab.is_dir():
            self.skipTest('Optional Wazuh archive not unpacked')
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'out'
            rows, _=run(lab,dest)
            self.assertEqual(len(rows),30)
            self.assertEqual((dest/'method_comparison.csv').exists(),True)
            self.assertTrue((dest/'manifest.json').exists())
            self.assertEqual(next(x for x in rows if x['scenario']=='TOTAL_SEPARATE_REPLAYS' and x['method']=='all')['alerts'],65)
            self.assertNotIn('src_ip', (dest/'method_comparison.csv').read_text())
            with self.assertRaises(FileExistsError):
                run(lab,dest)

if __name__=='__main__': unittest.main()
