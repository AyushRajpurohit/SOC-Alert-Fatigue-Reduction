"""Read-only integrated SOC analyst-view replay.

Uses the existing Wazuh correlation, evidence reducer, and prototype scorer.
Does not modify socflow.py, run_engine.py or the existing dashboard.
Event replay is retrospective and sorted; this is not a live Wazuh deployment.
All event-level outputs are private; never publish institute state.json.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import _common_path  # provides src/lab modules
from correlation_engine import Cluster, Correlator, read_json_lines, host_time_baseline
from evidence_reducer import Policy, choose
from soc_common import Alert
from socflow import Engine, Incident, RuleScorer, token_of

HERE = Path(__file__).resolve().parent


def clusters_for(events, mode='evidence_linked', gap=120):
    if mode == 'evidence_linked':
        corr = Correlator(gap_seconds=gap)
        for e in events:
            corr.feed(e)
        corr.finish()
        return corr.clusters
    if mode == 'host_time':
        grouped = host_time_baseline(events, gap_seconds=gap)
        return [Cluster(cid=i, first=members[0].timestamp, last=members[-1].timestamp,
                        members=members, closed=True)
                for i, members in enumerate(grouped, 1)]
    raise ValueError('Invalid grouping mode')


def build_scorer(mode, model):
    if mode == 'learned':
        if not Path(model).is_file():
            raise FileNotFoundError(f'Learned model missing: {model}')
        from learned import LearnedScorer
        scorer, patterns = LearnedScorer.load(str(model))
        return scorer, patterns
    if mode == 'rules':
        # Reuse patterns from a local model if available; learned weights not used.
        if Path(model).is_file():
            data = json.loads(Path(model).read_text())
            return RuleScorer(), data.get('patterns', [])
        return RuleScorer(), []
    raise ValueError('Invalid scorer')


def score_cluster(cluster, engine, force_review_level):
    inc = Incident(cluster.cid, ', '.join(sorted({e.host for e in cluster.members if e.host})) or 'unknown',
                   cluster.first, cluster.last)
    for seq, event in enumerate(cluster.members):
        inc.seqs.append(seq)
        inc.rules[event.rule] += 1
        inc.max_level = max(inc.max_level, event.level)
        a = Alert(source='wazuh', timestamp=event.timestamp.isoformat(),
                  entity=event.host, detector_id=event.rule, level=event.level,
                  raw_tactics=list(event.tactics), techniques=list(event.techniques), seq=seq)
        tok = token_of(a)
        if tok and (not inc.tokens or inc.tokens[-1] != tok):
            inc.tokens.append(tok)
    engine._decide(inc)
    if inc.max_level >= force_review_level and inc.decision == 'DEMOTE':
        inc.decision = 'INVESTIGATE'
        inc.reasons.insert(0, f'experimental conservative review policy: level >= {force_review_level}')
    inc.closed = True
    return inc


def visible_event(event, is_visible, why):
    # These are already sanitized for institute data. DO NOT publish state files.
    return {'id': event.uid, 'timestamp': event.timestamp.isoformat(),
            'host': event.host, 'rule': event.rule, 'level': event.level,
            'techniques': list(event.techniques), 'visible': is_visible,
            'reason': why}


def make_view(events, correlation='evidence_linked', method='greedy_cover',
              critical_level=10, scorer_mode='learned', model=None, gap=120):
    if method not in ('greedy_cover', 'first_cover', 'all'):
        raise ValueError('Invalid reducer')
    if critical_level < 0:
        raise ValueError('critical level must be non-negative')
    model = str(model or HERE / 'model.json')
    scorer, patterns = build_scorer(scorer_mode, model)
    scoring = Engine(patterns, {'gap': gap}, scorer)
    clusters = clusters_for(events, correlation, gap)
    policy = Policy(critical_level=critical_level)
    incident_views = []
    counts = Counter()
    for cluster in clusters:
        selected, audit = choose(cluster, method, policy)
        selected_set = set(selected)
        inc = score_cluster(cluster, scoring, critical_level)
        all_rows = []
        for j, ev in enumerate(cluster.members):
            why = ('visible representative' if j in selected_set else
                   'suppressed from default view; full evidence retained')
            all_rows.append(visible_event(ev, j in selected_set, why))
        incident_views.append({
            'id': cluster.cid, 'first_ts': cluster.first.isoformat(),
            'last_ts': cluster.last.isoformat(), 'hosts': sorted({e.host for e in cluster.members if e.host}),
            'n_alerts': len(cluster.members), 'visible_alerts': len(selected),
            'hidden_alerts': len(cluster.members) - len(selected),
            'priority': round(float(inc.priority), 5), 'decision': inc.decision,
            'reasons': inc.reasons, 'tokens': inc.tokens,
            'best': inc.best, 'max_level': inc.max_level,
            'links': cluster.links, 'evidence_audit': audit,
            'alerts': all_rows
        })
        counts['alerts'] += len(cluster.members)
        counts['visible'] += len(selected)
        counts['requirements'] += audit['constraints_total']
        counts['covered'] += audit['constraints_covered']
        counts['protected'] += audit['critical_alerts_total']
        counts['protected_retained'] += audit['critical_alerts_retained']
        counts[inc.decision.lower()] += 1
    counts['incidents'] = len(clusters)
    counts['hidden'] = counts['alerts'] - counts['visible']
    return {
        'metadata': {'dataset': 'locally provided Wazuh replay', 'correlation': correlation,
                     'reducer': method, 'critical_level': critical_level,
                     'scorer': scorer_mode, 'no_live_deployment': True,
                     'no_independent_security_labels': True,
                     'disclaimer': ('Visible-alert reduction is not measured analyst hours. '
                       'Programmed evidence constraints are not independently validated. '
                       'The Wazuh scorer was trained on limited lab data and is not calibrated for institute records.')},
        'stats': {**counts, 'visible_reduction': round(counts['hidden'] / counts['alerts'], 6) if counts['alerts'] else 0},
        'incidents': incident_views,
    }


def run(paths, outdir, correlation='evidence_linked', method='greedy_cover',
        critical_level=10, scorer='learned', model=None, gap=120):
    out = Path(outdir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing to overwrite: {out}')
    events = read_json_lines(paths)
    view = make_view(events, correlation, method, critical_level, scorer, model, gap)
    # Only after successful parsing, evaluation and model loading.
    out.mkdir(parents=True, exist_ok=True)
    (out / 'state.json').write_text(json.dumps(view, indent=2, default=str))
    shutil.copyfile(HERE / 'integrated_dashboard.html', out / 'index.html')
    public_stats = {k: v for k, v in view['stats'].items()}
    (out / 'aggregate_summary.json').write_text(json.dumps({'policy': view['metadata'], 'stats': public_stats}, indent=2))
    print(json.dumps({'policy': view['metadata'], 'stats': public_stats}, indent=2))
    return view


def main():
    ap = argparse.ArgumentParser(description='Read-only integrated Wazuh analyst view replay')
    ap.add_argument('alerts', nargs='+')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--correlation', choices=('host_time', 'evidence_linked'), default='evidence_linked')
    ap.add_argument('--reducer', choices=('all', 'first_cover', 'greedy_cover'), default='greedy_cover')
    ap.add_argument('--critical-level', type=int, default=10, help='Exploratory policy, NOT institute policy')
    ap.add_argument('--scorer', choices=('learned', 'rules'), default='learned')
    ap.add_argument('--model', default=str(HERE / 'model.json'))
    ap.add_argument('--gap', type=int, default=120)
    args = ap.parse_args()
    run(args.alerts, args.outdir, args.correlation, args.reducer, args.critical_level,
        args.scorer, args.model, args.gap)


if __name__ == '__main__':
    main()
