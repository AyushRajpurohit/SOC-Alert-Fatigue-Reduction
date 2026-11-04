"""Conservative, auditable Wazuh alert association (research prototype).

Standalone Stage 3.1 / Stage 4.1 component. Does not modify socflow.py.
Uses real observed entities, never attack-stage labels. No trained weights or claims
of calibrated probabilities. Incoming events should be ordered by event timestamp.

Usage:
    python correlation_engine.py data/lab/alerts/chain_alerts.json --outdir results/correlation_baseline

Interpretation: fewer clusters is NOT automatically better. Run with labelled
multi-campaign/mixed-activity traffic before reporting false-merge performance.
"""
from __future__ import annotations
import argparse
import json
import ipaddress
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def stamp(value):
    value = str(value or '').strip()
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
    elif len(value) >= 5 and value[-5] in '+-' and value[-3] != ':':
        value = value[:-2] + ':' + value[-2:]
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError('Event timestamp must be timezone-aware')
    return dt.astimezone(timezone.utc)


def clean(value):
    return str(value).strip().lower() if value is not None else ''


def valid_ip(value):
    try:
        ip = ipaddress.ip_address(str(value))
        return str(ip) if not (ip.is_unspecified or ip.is_multicast or ip.is_loopback) else ''
    except ValueError:
        return ''


@dataclass(frozen=True)
class Event:
    uid: str
    timestamp: datetime
    host: str
    host_ip: str
    src_ip: str
    dst_ip: str
    user: str
    rule: str
    level: int
    techniques: tuple[str, ...]
    tactics: tuple[str, ...]
    source_file: str = ''

    def as_dict(self):
        d = vars(self).copy()
        d['timestamp'] = self.timestamp.isoformat()
        d['techniques'] = list(self.techniques)
        d['tactics'] = list(self.tactics)
        return d


def parse_wazuh(obj, source_file='', index=0):
    rule = obj.get('rule') or {}
    agent = obj.get('agent') or {}
    data = obj.get('data') or {}
    mitre = rule.get('mitre') or {}
    ids = mitre.get('id') or []
    if isinstance(ids, str): ids = [ids]
    tactics = mitre.get('tactic') or []
    if isinstance(tactics, str): tactics = [tactics]
    # Wazuh fields are not universal; missing entities are left empty.
    return Event(
        uid=str(obj.get('id') or f'{source_file}:{index}'),
        timestamp=stamp(obj['timestamp']),
        host=clean(agent.get('name') or agent.get('id')),
        host_ip=valid_ip(agent.get('ip')),
        src_ip=valid_ip(data.get('srcip') or data.get('src_ip')),
        dst_ip=valid_ip(data.get('dstip') or data.get('dst_ip')),
        user=clean(data.get('dstuser') or data.get('srcuser') or data.get('user')),
        rule=str(rule.get('id') or ''),
        level=int(rule.get('level') or 0),
        techniques=tuple(sorted({str(i).upper() for i in ids if i})),
        tactics=tuple(sorted({str(t) for t in tactics if t})),
        source_file=source_file,
    )


def read_json_lines(paths):
    """Accept JSONL, pretty-printed concatenated JSON objects, or a JSON array."""
    events = []
    decoder = json.JSONDecoder()
    for path in paths:
        content = Path(path).read_text(encoding='utf-8')
        pos, index = 0, 0
        while pos < len(content):
            while pos < len(content) and (content[pos].isspace() or content[pos] == ','):
                pos += 1
            if pos >= len(content): break
            if content[pos] in '[]':
                pos += 1
                continue
            obj, end = decoder.raw_decode(content, pos)
            if not isinstance(obj, dict):
                raise ValueError(f'Expected Wazuh JSON object in {path}')
            index += 1
            events.append(parse_wazuh(obj, Path(path).name, index))
            pos = end
    # Sorting is offline replay normalization, NOT out-of-order streaming support.
    return sorted(events, key=lambda e: (e.timestamp, e.source_file, e.uid))


@dataclass
class Cluster:
    cid: int
    first: datetime
    last: datetime
    members: list[Event] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    closed: bool = False

    def as_dict(self):
        return dict(id=self.cid, first=self.first.isoformat(), last=self.last.isoformat(),
                    n_alerts=len(self.members), alert_ids=[e.uid for e in self.members],
                    hosts=sorted({e.host for e in self.members if e.host}),
                    techniques=sorted({t for e in self.members for t in e.techniques}),
                    links=self.links)


class Correlator:
    """Bounded, streaming, direct-evidence alert attachment; no transitive auto-merge.

    Host-only, user-only, shared tactic-only or shared source IP across different
    hosts cannot cause a merge. Cross-host pivot needs an observed endpoint link.
    Match requires a recent direct member rather than arbitrary component union.
    """
    def __init__(self, gap_seconds=120, max_span_seconds=900, min_score=2.6,
                 max_active=512, max_members=1000):
        if gap_seconds <= 0 or max_span_seconds <= 0 or max_active <= 0 or max_members <= 0:
            raise ValueError('Time and capacity limits must be positive')
        self.gap = gap_seconds
        self.max_span = max_span_seconds
        self.min_score = min_score
        self.max_active = max_active
        self.max_members = max_members
        self.clusters: list[Cluster] = []
        self.active: list[Cluster] = []
        self.stats = Counter()
        self._max_seen: Optional[datetime] = None

    def _link(self, old: Event, new: Event):
        if old.timestamp > new.timestamp:
            return None
        seconds = (new.timestamp - old.timestamp).total_seconds()
        if seconds > self.gap:
            return None
        same_host = bool(old.host and new.host and old.host == new.host)
        same_src = bool(old.src_ip and new.src_ip and old.src_ip == new.src_ip)
        same_user = bool(old.user and new.user and old.user == new.user)
        # Pivot evidence is directional: a destination/source endpoint on one
        # alert equals the observed agent IP or endpoint on the other.
        pivot = bool((old.dst_ip and (old.dst_ip == new.host_ip or old.dst_ip == new.src_ip)) or
                     (new.src_ip and new.src_ip == old.host_ip) or
                     (new.dst_ip and new.dst_ip == old.host_ip))
        evidence = []
        score = 0.0
        if same_host:
            score += 1.4; evidence.append('same_host')
        if same_src and same_host:
            score += 2.0; evidence.append('same_source_ip_on_host')
        if same_user and same_host:
            score += 1.0; evidence.append('same_user_on_host')
        if pivot and not same_host:
            score += 3.0; evidence.append('cross_host_endpoint_pivot')
        if same_host and old.rule and old.rule == new.rule:
            score += 0.3; evidence.append('same_rule')
        if same_host and set(old.techniques) & set(new.techniques):
            score += 0.2; evidence.append('shared_technique_context')
        # Time is supporting evidence, never sufficient alone.
        if evidence:
            score += 0.4 * (1.0 - seconds / self.gap)
        # A cross-host relationship needs the pivot, not common source/user.
        if not same_host and not pivot:
            return None
        if score < self.min_score:
            return None
        return dict(score=round(score, 4), reasons=evidence, seconds=round(seconds, 3),
                    from_id=old.uid, to_id=new.uid)

    def feed(self, event: Event):
        if self._max_seen is not None and event.timestamp < self._max_seen:
            self.stats['out_of_order_rejected'] += 1
            raise ValueError('Out-of-order event rejected: sort offline or provide reorder buffer')
        self._max_seen = event.timestamp
        self.stats['alerts'] += 1
        keep = []
        for c in self.active:
            if (event.timestamp - c.last).total_seconds() > self.gap or \
               (event.timestamp - c.first).total_seconds() > self.max_span or \
               len(c.members) >= self.max_members:
                c.closed = True
            else:
                keep.append(c)
        self.active = keep
        candidates = []
        for c in self.active:
            if (event.timestamp - c.first).total_seconds() > self.max_span:
                continue
            # Direct edges to recent members; do not rely on transitive equivalence.
            best = None
            for member in reversed(c.members):
                if (event.timestamp - member.timestamp).total_seconds() > self.gap:
                    break
                link = self._link(member, event)
                if link and (best is None or link['score'] > best['score']):
                    best = link
            if best:
                candidates.append((best['score'], -c.cid, c, best))
        if candidates:
            _, _, c, link = max(candidates, key=lambda x: (x[0], x[1]))
            c.members.append(event)
            c.links.append(link)
            c.last = event.timestamp
            self.stats['linked_alerts'] += 1
            return c
        if len(self.active) >= self.max_active:
            oldest = min(self.active, key=lambda c: (c.last, c.cid))
            oldest.closed = True
            self.active.remove(oldest)
            self.stats['capacity_evictions'] += 1
        c = Cluster(cid=len(self.clusters) + 1, first=event.timestamp,
                    last=event.timestamp, members=[event])
        self.clusters.append(c)
        self.active.append(c)
        return c

    def finish(self):
        for c in self.active:
            c.closed = True
        self.active.clear()


def host_time_baseline(events, gap_seconds=120):
    """Grouping count baseline, matching socflow's host/time inactivity logic."""
    last = {}
    groups = []
    for event in events:
        host = event.host
        if host not in last or (event.timestamp - last[host][1]).total_seconds() > gap_seconds:
            groups.append([event])
            last[host] = (len(groups) - 1, event.timestamp)
        else:
            idx, _ = last[host]
            groups[idx].append(event)
            last[host] = (idx, event.timestamp)
    return groups


def run(paths, outdir, gap=120):
    events = read_json_lines(paths)
    engine = Correlator(gap_seconds=gap)
    for ev in events:
        engine.feed(ev)
    engine.finish()
    baseline = host_time_baseline(events, gap)
    out = Path(outdir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'incidents.jsonl', 'w') as f:
        for c in engine.clusters:
            f.write(json.dumps(c.as_dict()) + '\n')
    summary = dict(source='real_Wazuh_lab_alerts_if_input_is_lab_json',
                   alert_count=len(events), host_time_incidents=len(baseline),
                   evidence_linked_incidents=len(engine.clusters),
                   evidence_links=sum(len(c.links) for c in engine.clusters),
                   stats=dict(engine.stats),
                   limitations=[
                       'No ground-truth campaign evaluation in this script.',
                       'Incident count is NOT analyst workload reduction or correlation precision.',
                       'No training, no benign-aware weights, no live Wazuh deployment.',
                       'Cross-host links require observed network endpoint evidence.',
                       'Input timestamps sorted for retrospective replay; no reorder buffer.',
                       'One monitored host in supplied lab archive; cross-host gain untested.'
                   ])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('alerts', nargs='+', help='Raw Wazuh JSON-lines files')
    p.add_argument('--outdir', required=True)
    p.add_argument('--gap', type=int, default=120)
    args = p.parse_args()
    run(args.alerts, args.outdir, args.gap)


if __name__ == '__main__':
    main()
