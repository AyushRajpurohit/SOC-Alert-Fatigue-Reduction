"""Evidence-preserving analyst-view selection for already-correlated Wazuh alerts.

Stage 4 research prototype: the FULL alert set remains stored. Only the default
analyst-facing representative view is reduced. No learned weights or production
safety guarantees are claimed.

Inputs are the Event/Cluster objects from correlation_engine.py. Algorithms:
  all          full view / no reduction
  first_cover  simple chronological coverage baseline
  greedy_cover coverage-per-alert greedy heuristic (proposed)

Hard, auditable constraints (within each supplied correlation cluster):
  * all observed ATT&CK technique IDs represented, never inferred
  * at least one representative from each observed host
  * if technique mapping absent, retain an example of each (host,rule) type
  * all high-level alerts (>= critical_level) stay visible
  * for each observed correlation link, at least one endpoint stays visible
  * first and last alert retained as chronological anchors (configurable)

This is a weighted set-cover/vertex-cover-like heuristic, NOT guaranteed globally
minimal. Evidence-link coverage does not establish causal chain preservation.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from correlation_engine import Event, Cluster, Correlator, read_json_lines


@dataclass(frozen=True)
class Policy:
    critical_level: int = 12
    keep_anchors: bool = True
    protect_unknown_rule: bool = True
    protect_link_endpoints: bool = True


def requirements(cluster: Cluster, policy: Policy):
    """Produce token-to-supporters mapping and alert index -> covered token set."""
    events = cluster.members
    covers = [set() for _ in events]
    hard = set()
    for i, e in enumerate(events):
        for tech in e.techniques:
            covers[i].add(('technique', tech))
        if e.host:
            covers[i].add(('host', e.host))
        elif e.host_ip:
            covers[i].add(('host_ip', e.host_ip))
        if policy.protect_unknown_rule and not e.techniques:
            # Do not silently erase a possibly important unmapped Wazuh event type.
            covers[i].add(('unmapped_host_rule', e.host or e.host_ip or 'unknown', e.rule or 'unknown'))
        if e.level >= policy.critical_level:
            hard.add(i)
    if policy.keep_anchors and events:
        hard.update((0, len(events) - 1))
    if policy.protect_link_endpoints:
        # Real links come from the correlator and have from_id/to_id, but may
        # contain duplicate UIDs; IDs are resolved only when unambiguous.
        uid_indices = {}
        for i, e in enumerate(events):
            uid_indices.setdefault(e.uid, []).append(i)
        for j, link in enumerate(cluster.links):
            left = uid_indices.get(link.get('from_id'), [])
            right = uid_indices.get(link.get('to_id'), [])
            if len(left) != 1 or len(right) != 1:
                # Unverifiable links are not silently considered preserved.
                # Keep every member to avoid losing uncertain provenance.
                hard.update(range(len(events)))
                continue
            token = ('observed_link', j)
            covers[left[0]].add(token)
            covers[right[0]].add(token)
    universe = set().union(*covers) if covers else set()
    return universe, covers, hard


def choose(cluster: Cluster, method='greedy_cover', policy=None):
    """Return representative alert indices in original order and an audit dict."""
    if method not in ('all', 'first_cover', 'greedy_cover'):
        raise ValueError(f'Unknown reduction method: {method}')
    policy = policy or Policy()
    universe, covers, required = requirements(cluster, policy)
    events = cluster.members
    if method == 'all':
        selected = set(range(len(events)))
    else:
        selected = set(required)
        covered = set().union(*(covers[i] for i in selected)) if selected else set()
        remaining = universe - covered
        while remaining:
            candidates = [i for i in range(len(events)) if i not in selected and (covers[i] & remaining)]
            if not candidates:
                raise RuntimeError('Unsatisfiable evidence coverage requirements')
            if method == 'first_cover':
                pick = candidates[0]
            else:
                # Max marginal coverage; deterministic level and chronological
                # tie-breaking. No test label, no ground truth, no global tuning.
                pick = max(candidates, key=lambda i: (len(covers[i] & remaining), events[i].level, -i))
            selected.add(pick)
            remaining.difference_update(covers[pick])
    selected = sorted(selected)
    covered = set().union(*(covers[i] for i in selected)) if selected else set()
    missing = universe - covered
    if missing or not required.issubset(selected):
        raise AssertionError('Evidence preservation invariant broken')
    return selected, {
        'total_alerts': len(events),
        'visible_alerts': len(selected),
        'hidden_from_default_view': len(events) - len(selected),
        'selection_reduction_proxy': (1 - len(selected) / len(events)) if events else 0,
        'constraints_total': len(universe),
        'constraints_covered': len(covered),
        'critical_alerts_retained': sum(i in selected for i, e in enumerate(events) if e.level >= policy.critical_level),
        'critical_alerts_total': sum(e.level >= policy.critical_level for e in events),
        'missing_constraints': [list(t) for t in sorted(missing)],
    }


def evaluate(clusters: Sequence[Cluster], methods=('all', 'first_cover', 'greedy_cover'), policy=None):
    policy = policy or Policy()
    rows, details = [], []
    for method in methods:
        start = time.perf_counter()
        totals = dict(alerts=0, visible_alerts=0, constraints_total=0,
                      constraints_covered=0, critical_alerts_total=0, critical_alerts_retained=0)
        for c in clusters:
            selected, audit = choose(c, method, policy)
            for field in totals:
                totals[field] += audit[field if field != 'alerts' else 'total_alerts']
            details.append(dict(method=method, incident_id=c.cid,
                                visible_alert_indices=selected,
                                visible_alert_ids=[c.members[i].uid for i in selected],
                                hidden_alert_ids=[e.uid for i, e in enumerate(c.members) if i not in selected],
                                all_alerts_retained_in_backing_store=True, audit=audit))
        rows.append(dict(method=method, incidents=len(clusters), **totals,
                         hidden_from_default_view=totals['alerts']-totals['visible_alerts'],
                         view_reduction_proxy=round(1-totals['visible_alerts']/totals['alerts'], 6) if totals['alerts'] else 0,
                         constraints_preserved=(totals['constraints_total']==totals['constraints_covered']),
                         runtime_ms=round((time.perf_counter()-start)*1000, 3)))
    return rows, details


def run(paths, outdir, critical_level=12, no_anchors=False):
    events = read_json_lines(paths)
    engine = Correlator()
    for e in events:
        engine.feed(e)
    engine.finish()
    policy = Policy(critical_level=critical_level, keep_anchors=not no_anchors)
    rows, details = evaluate(engine.clusters, policy=policy)
    out = Path(outdir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'summary.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (out/'incident_views.jsonl').open('w') as fp:
        for item in details:
            fp.write(json.dumps(item, sort_keys=True)+'\n')
    manifest = dict(protocol='evidence_preserving_analyst_view_v1', input_files=[str(p) for p in paths],
                    source='real_Wazuh_lab_if_input_is_lab_JSON',
                    scope='read_only_retrospective_existing_incident_clusters',
                    constraints=dict(critical_level=critical_level, anchors=not no_anchors,
                                     technique_coverage=True, observed_host_coverage=True,
                                     unknown_host_rule_coverage=True, observed_link_endpoint_coverage=True),
                    limitations=[
                        'Alert view reduction is NOT measured analyst time savings or real-world fatigue reduction.',
                        'Constraints preserve OBSERVED metadata, not all real attack stages or causal edges.',
                        'Greedy heuristic not a proof of optimal minimum-cardinality cover.',
                        'Incident membership comes from evidence-linked Correlator; unverified merges remain possible.',
                        'Criticality threshold is an engineering policy, not validated threat severity.',
                        'Missing ATT&CK labels stay unknown; unmapped host/rule families are explicitly represented.',
                        'No independent benign/multi-campaign labels in supplied Wazuh set.',
                        'Preserving one endpoint per observed link does not preserve a visible connected attack path.',
                        'All full incident alerts must be retained and retrievable by downstream storage; this script writes views only.',
                        'No deployment, no model training, no altered GUIDE experiments.'
                    ])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(rows, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser(description='Evidence-preserving analyst view selection')
    ap.add_argument('alerts', nargs='+')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--critical-level', type=int, default=12)
    ap.add_argument('--no-anchors', action='store_true', help='Ablation: do not require earliest and latest event')
    args = ap.parse_args()
    run(args.alerts, args.outdir, args.critical_level, args.no_anchors)


if __name__ == '__main__':
    main()
