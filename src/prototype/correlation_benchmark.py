"""Read-only Stage 6 correlation comparison on real Wazuh records and synthetic controls.

Methods: host/time, an AiSOC-*style* documented key/window approximation, and
our standalone evidence-linked Correlator. This is NOT a run of AiSOC itself.
No research-grade false-merge labels are inferred from attack-stage windows.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from correlation_engine import Event, Correlator, host_time_baseline, read_json_lines, stamp

DEFAULT_SCENARIOS = {
    'baseline': ('chain_alerts.json', 'ground_truth.csv'),
    'variant': ('chain_alerts_variant.json', 'ground_truth_variant.csv'),
    'exfil': ('chain_alerts_exfil.json', 'ground_truth_exfil.csv'),
    'exfil2': ('chain_alerts_exfil2.json', 'ground_truth_exfil.csv'),
}


def aisoc_style_key(event, tenant='lab'):
    """Documented AiSOC key *inspired* approximation, not upstream code.

    Wazuh host stands in for a normalized entity, first sorted tactic for
    tactic, and 'unknown' when absent. Actual AiSOC RawAlert entity selection
    and Redis window semantics have not been reproduced.
    """
    entity = event.host or event.host_ip or 'unknown'
    tactic = event.tactics[0].strip().lower() if event.tactics else 'unknown'
    return (tenant, entity, tactic)


def aisoc_style_groups(events, window_seconds=3600):
    current = {}
    groups = []
    for e in events:
        key = aisoc_style_key(e)
        if key not in current or (e.timestamp - current[key][1]).total_seconds() > window_seconds:
            groups.append([e]); current[key] = (len(groups)-1, e.timestamp)
        else:
            idx, _ = current[key]
            groups[idx].append(e)
    return groups


def evidence_groups(events, gap=120, span=900):
    eng = Correlator(gap_seconds=gap, max_span_seconds=span)
    for e in events: eng.feed(e)
    eng.finish()
    return [c.members for c in eng.clusters], eng


def stage_windows(csv_path):
    if not Path(csv_path).exists(): return []
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                start = stamp(row['start_time_utc']); end = stamp(row['end_time_utc'])
            except (KeyError, ValueError): continue
            if end >= start:
                rows.append((row.get('stage', 'unknown'), start, end))
    return rows


def stage_diagnostic(groups, windows):
    # Stage windows support *coverage and fragmentation diagnostics only*.
    # They do not identify which distinct incidents unrelated alerts belong to.
    membership = {id(e): gi for gi, g in enumerate(groups) for e in g}
    all_events = [e for g in groups for e in g]
    hit_windows, supported_stages, associated_stages, splits = 0, set(), set(), []
    for stage, lo, hi in windows:
        in_window = [e for e in all_events if lo <= e.timestamp <= hi]
        if not in_window: continue
        hit_windows += 1; supported_stages.add(stage)
        ids = {membership[id(e)] for e in in_window}
        splits.append(len(ids))
        if len(in_window) >= 2 and len(ids) == 1: associated_stages.add(stage)
    return dict(windows_with_alerts=hit_windows, distinct_stages_with_alerts=len(supported_stages),
                stages_with_multiple_alerts_in_one_group=len(associated_stages),
                mean_groups_per_populated_window=round(sum(splits)/len(splits),4) if splits else None)


def metrics(groups, n):
    multi = sum(len(g) for g in groups if len(g) >= 2)
    return dict(alerts=n, groups=len(groups), alerts_in_multi_alert_groups=multi,
                fraction_alerts_in_multi_alert_groups=round(multi/n, 6) if n else 0.0,
                grouping_reduction_proxy=round(1-len(groups)/n, 6) if n else 0.0,
                largest_group=max((len(g) for g in groups), default=0))


def all_methods(events, gap=120, span=900, aisoc_window=3600):
    return {
        'host_time': lambda: host_time_baseline(events, gap_seconds=gap),
        'aisoc_style_key_window': lambda: aisoc_style_groups(events, window_seconds=aisoc_window),
        'evidence_linked': lambda: evidence_groups(events, gap=gap, span=span)[0],
    }


def pair_metrics(groups, labels):
    """Pairwise precision/recall ONLY on deliberately synthetic true-membership labels."""
    tp = fp = fn = 0
    members = [e for g in groups for e in g]
    group_of = {e.uid: ix for ix, g in enumerate(groups) for e in g}
    for i, x in enumerate(members):
        for y in members[i+1:]:
            same_pred = group_of[x.uid] == group_of[y.uid]
            same_true = labels[x.uid] == labels[y.uid]
            if same_pred and same_true: tp += 1
            elif same_pred and not same_true: fp += 1
            elif not same_pred and same_true: fn += 1
    return dict(pairwise_tp=tp, pairwise_fp=fp, pairwise_fn=fn,
                pairwise_precision=tp/(tp+fp) if tp+fp else None,
                pairwise_recall=tp/(tp+fn) if tp+fn else None)


def fixture_events():
    """Controlled synthetic incident membership, NOT real lab or GUIDE evidence."""
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    def e(uid, sec, host, src, dst='', host_ip='', user='', tactic='Execution'):
        return Event(uid, t+timedelta(seconds=sec),host,host_ip,src,dst,user,
                     '100',5,('T1059',), (tactic,),source_file='synthetic')
    # Two independent investigations on one host, interleaved in time.
    # Different source IPs make them distinguishable for the evidence engine.
    case1 = [e('A1',0,'host1','10.0.0.101'), e('A2',30,'host1','10.0.0.101'),
             e('A3',70,'host1','10.0.0.101')]
    case2 = [e('B1',12,'host1','10.0.0.102'), e('B2',40,'host1','10.0.0.102'),
             e('B3',80,'host1','10.0.0.102')]
    # Cross-host pivot: an observed destination matches host2's endpoint.
    case3 = [e('C1',200,'host3','10.0.0.103',dst='10.0.0.204',host_ip='10.0.0.203'),
             e('C2',222,'host4','10.0.0.104',host_ip='10.0.0.204')]
    combined = sorted(case1 + case2 + case3, key=lambda x:(x.timestamp,x.uid))
    labels = {e.uid: c for c, seq in [('A',case1),('B',case2),('C',case3)] for e in seq}
    return combined, labels


def write_csv(path, rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(data_dir, outdir, gap=120, span=900, aisoc_window=3600):
    data_dir, outdir = Path(data_dir), Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output directory: {outdir}')
    outdir.mkdir(parents=True,exist_ok=True)
    results=[]; synthetic=[]; groups_output=[]; missing=[]
    for scenario,(filename,ground) in DEFAULT_SCENARIOS.items():
        filepath=data_dir/filename
        if not filepath.exists():
            missing.append(filename); continue
        events=read_json_lines([filepath]); windows=stage_windows(data_dir/ground)
        for method,make in all_methods(events,gap,span,aisoc_window).items():
            t0=time.perf_counter(); groups=make(); duration=time.perf_counter()-t0
            row=dict(scenario=scenario,origin='real_Wazuh_lab',method=method,**metrics(groups,len(events)),
                     runtime_ms=round(duration*1000,3),**stage_diagnostic(groups,windows))
            results.append(row)
            for idx,g in enumerate(groups,1):
                groups_output.append(dict(scenario=scenario,method=method,group_id=idx,
                                          alert_ids=[e.uid for e in g],n_alerts=len(g)))
    events, labels=fixture_events()
    for method,make in all_methods(events,gap,span,aisoc_window).items():
        groups=make()
        synthetic.append(dict(scenario='synthetic_interleaved_and_pivot',origin='SYNTHETIC_CONTROL',
                              method=method,**metrics(groups,len(events)),**pair_metrics(groups,labels)))
    write_csv(outdir/'lab_grouping.csv',results)
    write_csv(outdir/'synthetic_controls.csv',synthetic)
    with open(outdir/'lab_groups.jsonl','w') as f:
        for row in groups_output: f.write(json.dumps(row)+'\n')
    manifest=dict(protocol='read_only_correlation_benchmark_v1',
                  primary_method='evidence_linked',
                  baselines=['host_time','aisoc_style_key_window'],
                  baseline_caveat='AiSOC-style inspired approximation, NOT AiSOC source or full platform',
                  source_data='real_Wazuh_lab_alerts_for_lab_rows_only',
                  synthetic_fixture='8 synthetic alert events with artificial membership labels; never real performance',
                  settings=dict(gap_seconds=gap,max_span_seconds=span,aisoc_window_seconds=aisoc_window),
                  files_missing=missing,
                  limitations=[
                      'All available Wazuh lab alerts belong to one monitored host; no measured real cross-host benefit.',
                      'Attack-stage time windows do not provide independent incident membership or benign labels.',
                      'No real Wazuh pairwise precision, false-merge rate, attack recall, or analyst-hours saved can be inferred.',
                      'Grouping reduction proxy (1 - group_count/alert_count) is NOT validated analyst workload reduction.',
                      'AiSOC-style method approximates its public key/window semantics; does not invoke upstream implementation.',
                      'Synthetic controls are designed examples, not estimates of operational performance.',
                      'Wall-clock timings on small inputs are only smoke timings, not throughput benchmarks.',
                      'Scores/weights in standalone evidence engine are hand selected, not benign-aware learned weights.'
                  ])
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Wazuh lab comparisons:')
    for x in results: print(f"  {x['scenario']:9s} {x['method']:24s} {x['alerts']:2d} alerts -> {x['groups']:2d} groups")
    print('Synthetic controls (not real evidence):')
    for x in synthetic: print(f"  {x['method']:24s} pairwise precision={x['pairwise_precision']} recall={x['pairwise_recall']}")
    print('Outputs:',outdir)
    return results,synthetic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True,help='Directory with chain_alerts*.json and optional ground_truth*.csv')
    p.add_argument('--outdir',required=True)
    p.add_argument('--gap',type=int,default=120)
    p.add_argument('--span',type=int,default=900)
    p.add_argument('--aisoc-window',type=int,default=3600)
    a=p.parse_args()
    if min(a.gap,a.span,a.aisoc_window)<=0:p.error('All windows must be positive')
    run(a.data_dir,a.outdir,a.gap,a.span,a.aisoc_window)

if __name__=='__main__':main()
