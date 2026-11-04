"""Standalone unit tests for the evidence-linked correlation prototype."""
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from correlation_engine import Correlator, Event, parse_wazuh, read_json_lines, run

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)

def ev(n, sec=0, host='h1', host_ip='10.0.0.10', src='', dst='', user='', rule='1', tech=()):
    return Event(str(n), BASE+timedelta(seconds=sec), host, host_ip, src, dst,
                 user, rule, 5, tuple(tech), tuple())

class CorrelationTests(unittest.TestCase):
    def group(self, *items):
        c = Correlator()
        for e in items:
            c.feed(e)
        return c

    def test_same_host_and_source(self):
        c = self.group(ev(1, src='10.0.0.20'), ev(2, 30, src='10.0.0.20'))
        self.assertEqual(len(c.clusters), 1)
        self.assertIn('same_source_ip_on_host', c.clusters[0].links[0]['reasons'])

    def test_host_only_does_not_merge(self):
        c = self.group(ev(1), ev(2, 30))
        self.assertEqual(len(c.clusters), 2)

    def test_user_only_different_host_does_not_merge(self):
        c = self.group(ev(1, user='root'), ev(2, 20, host='h2',host_ip='10.0.0.11',user='root'))
        self.assertEqual(len(c.clusters), 2)

    def test_same_source_different_host_does_not_merge(self):
        c = self.group(ev(1, src='10.0.0.20'), ev(2, 20, host='h2',host_ip='10.0.0.11',src='10.0.0.20'))
        self.assertEqual(len(c.clusters), 2)

    def test_cross_host_pivot(self):
        c = self.group(ev(1,dst='10.0.0.11'),ev(2,25,host='h2',host_ip='10.0.0.11'))
        self.assertEqual(len(c.clusters), 1)
        self.assertIn('cross_host_endpoint_pivot',c.clusters[0].links[0]['reasons'])

    def test_gap_and_span(self):
        c = self.group(ev(1,src='10.0.0.20'),ev(2,121,src='10.0.0.20'))
        self.assertEqual(len(c.clusters),2)
        c = Correlator(gap_seconds=120,max_span_seconds=40)
        for e in [ev(1,src='10.0.0.20'),ev(2,30,src='10.0.0.20'),ev(3,60,src='10.0.0.20')]:c.feed(e)
        self.assertEqual(len(c.clusters),2)

    def test_out_of_order_rejected(self):
        c = Correlator()
        c.feed(ev(1,sec=30))
        with self.assertRaises(ValueError): c.feed(ev(2,sec=20))

    def test_missing_entities_and_raw_parsing(self):
        data={'timestamp':'2026-01-01T00:00:00+0000','rule':{'id':'42','mitre':{'id':['T1078.004']}},'agent':{'name':'Host'},'data':{}}
        a=parse_wazuh(data)
        self.assertEqual(a.techniques, ('T1078.004',))
        self.assertEqual(a.src_ip, '')
        self.assertEqual(a.host, 'host')

    def test_real_archive_format(self):
        p=Path(__file__).parent/'_corr_work'/'data'/'chain_alerts.json'
        if not p.exists(): self.skipTest('Optional review archive not unpacked')
        alerts=read_json_lines([p])
        self.assertEqual(len(alerts),16)
        self.assertTrue(any(e.src_ip for e in alerts))
        self.assertTrue(any(e.user for e in alerts))
        with tempfile.TemporaryDirectory() as d:
            summary=run([str(p)],str(Path(d)/'results'))
            self.assertEqual(summary['alert_count'],16)
            self.assertEqual(sum(json.loads(s)['n_alerts'] for s in (Path(d)/'results'/'incidents.jsonl').read_text().splitlines()),16)

if __name__=='__main__': unittest.main()
