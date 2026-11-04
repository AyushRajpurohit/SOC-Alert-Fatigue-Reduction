"""Read-only Stage 6 evaluation: six incident-level alert-view selection policies.

Runs on local Wazuh JSON events. Produces AGGREGATES ONLY (no raw entity IDs).
All original alerts remain in input files; visibility is not deletion.

Fair comparison notes:
  * `severity_budget` uses greedy's visible count as an *evaluation budget*;
    this is retrospective budget matching, NOT a stand-alone live policy.
  * `rule_dedup` and `one_per_technique` have different preservation goals.
    They may hide evidence: coverage and critical-retention are measured, not forced.
  * `first_cover` and `greedy_cover` optimize the EXACT SAME constraints.
  * Source Wazuh data without independent incident labels cannot establish
    false-merge precision, missed attacks or analyst time savings.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from correlation_engine import Correlator, read_json_lines
from evidence_reducer import Policy, choose, requirements

METHODS = ('all', 'severity_budget', 'rule_dedup', 'one_per_technique', 'first_cover', 'greedy_cover')


def _rule_key(e):
    return (e.host or e.host_ip or 'unknown', e.rule or 'unknown')


def _severity_budget(c, k):
    return sorted(sorted(range(len(c.members)), key=lambda i: (-c.members[i].level, i))[:k])


def _rule_dedup(c):
    selected = {}
    for i, e in enumerate(c.members):
        key = _rule_key(e)
        if key not in selected or (e.level, -i) > (c.members[selected[key]].level, -selected[key]):
            selected[key] = i
    return sorted(selected.values())


def _one_per_technique(c):
    # One highest-severity alert per observed technique; fall back to the
    # highest-severity alert in a cluster without any technique annotations.
    # This intentionally does not protect unknown-rule evidence or link endpoints.
    by_tech = {}
    for i, e in enumerate(c.members):
        for tech in e.techniques:
            if tech not in by_tech or (e.level, -i) > (c.members[by_tech[tech]].level, -by_tech[tech]):
                by_tech[tech] = i
    if by_tech:
        return sorted(set(by_tech.values()))
    if not c.members:
        return []
    return _severity_budget(c, 1)


def selections(c, policy):
    greedy, _ = choose(c, 'greedy_cover', policy)
    first, _ = choose(c, 'first_cover', policy)
    return {
        'all': list(range(len(c.members))),
        'severity_budget': _severity_budget(c, len(greedy)),
        'rule_dedup': _rule_dedup(c),
        'one_per_technique': _one_per_technique(c),
        'first_cover': first,
        'greedy_cover': greedy,
    }


def audit_cluster(c, selected, policy):
    selected = set(selected)
    universe, covers, hard = requirements(c, policy)
    covered = set().union(*(covers[i] for i in selected)) if selected else set()
    observed_tech = {t for e in c.members for t in e.techniques}
    retained_tech = {t for i in selected for t in c.members[i].techniques}
    critical = {i for i, e in enumerate(c.members) if e.level >= policy.critical_level}
    # Each observed link is satisfied by >=1 endpoint; not a visible path guarantee.
    uid_positions = {}
    for i, e in enumerate(c.members):
        uid_positions.setdefault(e.uid, []).append(i)
    link_valid = 0
    link_preserved = 0
    for link in c.links:
        a = uid_positions.get(link.get('from_id'), [])
        b = uid_positions.get(link.get('to_id'), [])
        if len(a) == 1 and len(b) == 1:
            link_valid += 1
            link_preserved += int(a[0] in selected or b[0] in selected)
    unmapped = { _rule_key(e) for e in c.members if not e.techniques }
    retained_unmapped = { _rule_key(c.members[i]) for i in selected if not c.members[i].techniques }
    return dict(
        alerts=len(c.members), visible=len(selected), incidents=1,
        constraints_total=len(universe), constraints_covered=len(covered),
        hard_required=len(hard), hard_retained=len(hard & selected),
        critical_total=len(critical), critical_retained=len(critical & selected),
        techniques_total=len(observed_tech), techniques_retained=len(observed_tech & retained_tech),
        unmapped_types_total=len(unmapped), unmapped_types_retained=len(unmapped & retained_unmapped),
        links_evaluable=link_valid, links_endpoints_retained=link_preserved,
        cases_with_unmet_requirements=int(bool(universe - covered or hard - selected)),
    )


def evaluate_scenario(name, paths, policy):
    events = read_json_lines(paths)
    engine = Correlator()
    for e in events:
        engine.feed(e)
    engine.finish()
    totals = {m: Counter() for m in METHODS}
    durations = Counter()
    # Time all policies over same cluster set. Very small runtimes are not throughput benchmarks.
    for c in engine.clusters:
        t = time.perf_counter()
        choices = selections(c, policy)
        durations['selection_all'] += (time.perf_counter() - t)
        for method, indices in choices.items():
            totals[method].update(audit_cluster(c, indices, policy))
    rows = []
    for m in METHODS:
        row = dict(scenario=name, data_origin='real_Wazuh_lab_if_user_lab_input', method=m, **totals[m])
        row['hidden'] = row['alerts'] - row['visible']
        row['visible_reduction_pct'] = round(100 * row['hidden'] / row['alerts'], 3) if row['alerts'] else 0.0
        row['constraint_coverage_pct'] = round(100 * row['constraints_covered'] / row['constraints_total'], 3) if row['constraints_total'] else 100.0
        row['constraints_fully_preserved'] = (row['cases_with_unmet_requirements'] == 0)
        rows.append(row)
    return rows, dict(scenario=name, alerts=len(events), incidents=len(engine.clusters), links=sum(len(c.links) for c in engine.clusters),
                      selection_and_evaluation_wall_ms=round(1000 * durations['selection_all'], 3))


def run(data_dir, outdir, critical_level=12, keep_anchors=True):
    directory = Path(data_dir)
    scenarios = {'baseline': 'chain_alerts.json', 'variant': 'chain_alerts_variant.json',
                 'exfil': 'chain_alerts_exfil.json', 'exfil2': 'chain_alerts_exfil2.json'}
    out = Path(outdir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing to overwrite: {out}')
    policy = Policy(critical_level=critical_level, keep_anchors=keep_anchors)
    all_rows, scenario_audits, unavailable = [], [], []
    for name, filename in scenarios.items():
        path = directory / filename
        if not path.is_file():
            unavailable.append(filename)
            continue
        rows, audit = evaluate_scenario(name, [path], policy)
        all_rows.extend(rows)
        scenario_audits.append(audit)
    if not all_rows:
        raise FileNotFoundError(f'No expected chain_alerts*.json found inside {directory}')
    out.mkdir(parents=True, exist_ok=True)
    totals = {}
    additive = ('alerts','visible','hidden','incidents','constraints_total','constraints_covered',
                'hard_required','hard_retained','critical_total','critical_retained','techniques_total',
                'techniques_retained','unmapped_types_total','unmapped_types_retained',
                'links_evaluable','links_endpoints_retained','cases_with_unmet_requirements')
    for m in METHODS:
        items = [r for r in all_rows if r['method'] == m]
        merged = {k:sum(x[k] for x in items) for k in additive}
        merged.update(scenario='TOTAL_SEPARATE_REPLAYS', data_origin='real_Wazuh_lab_if_user_lab_input',method=m)
        merged['visible_reduction_pct'] = round(100 * merged['hidden']/merged['alerts'],3) if merged['alerts'] else 0
        merged['constraint_coverage_pct'] = round(100*merged['constraints_covered']/merged['constraints_total'],3) if merged['constraints_total'] else 100
        merged['constraints_fully_preserved'] = (merged['cases_with_unmet_requirements'] == 0)
        totals[m] = merged
    all_rows.extend(totals.values())
    with (out/'method_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader(); writer.writerows(all_rows)
    manifest = dict(protocol='six_method_evidence_view_benchmark_v1', source='Wazuh lab input files',
                    scenario_files=scenarios, missing_files=unavailable,
                    method_definitions={
                        'all':'Show every alert',
                        'severity_budget':'Highest-severity k alerts per cluster; k matched to greedy for comparison only',
                        'rule_dedup':'Highest-severity alert for each (host, Wazuh rule) in cluster',
                        'one_per_technique':'Highest-severity representative per known technique, or one event if none tagged',
                        'first_cover':'Chronological first uncovered evidence requirement; same hard constraints as greedy',
                        'greedy_cover':'Max uncovered evidence constraints per selected alert; same constraints as first_cover'},
                    policy=dict(critical_level=critical_level, keep_anchors=keep_anchors, unknown_rule=True, observed_link_endpoint=True),
                    scenario_audits=scenario_audits,
                    privacy='Aggregate-only outputs: no event IDs, IPs, usernames, descriptions or raw records written',
                    limitations=[
                        'Four separate short real Wazuh lab attack replays; NOT independent benign or production observations.',
                        'One monitored host; no empirical multi-host correlation improvement.',
                        'Constraint satisfaction is defined from observed fields and does not independently validate attack-stage preservation.',
                        'The same original 65 lab alerts were seen in prior development; exploratory, not independent held-out evaluation.',
                        'Severity-budget uses proposed method selection count as budget; retrospective fair-size comparison only.',
                        'Rule dedup and one-per-technique do not share hard constraints with greedy; report their coverage losses.',
                        'Visible alert reduction is not verified analyst effort or real fatigue reduction.',
                        'Critical threshold is an engineering rule and may not reflect institute incident policy.',
                        'One endpoint per correlation link is retained but a complete visible evidence path is not guaranteed.',
                        'No changes to base reducer, correlation logic, GUIDE models, or historic results.'
                    ])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Scenario inputs:', ', '.join(x['scenario'] for x in scenario_audits))
    for m in METHODS:
        r=totals[m]
        print(f'{m:20s} visible {r["visible"]:2d}/{r["alerts"]:2d} ({r["visible_reduction_pct"]:5.1f}% hidden); evidence {r["constraint_coverage_pct"]:5.1f}%; critical {r["critical_retained"]}/{r["critical_total"]}; failures {r["cases_with_unmet_requirements"]}')
    print('Saved:', out)
    return all_rows, manifest


def main():
    ap=argparse.ArgumentParser(description='Read-only six-method evidence-preserving alert-view comparison')
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--critical-level',type=int,default=12)
    ap.add_argument('--no-anchors',action='store_true',help='Explicit ablation; new output dir required')
    args=ap.parse_args()
    run(args.data_dir,args.outdir,args.critical_level,not args.no_anchors)

if __name__=='__main__': main()
