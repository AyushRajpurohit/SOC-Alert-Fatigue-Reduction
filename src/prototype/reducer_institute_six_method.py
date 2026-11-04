"""Aggregate-only six-method evaluation on locally sanitized institute Wazuh samples.

This is a separate input adapter for reducer_six_method_eval.py; it does NOT
change the correlator, reducer, policy, or six selection methods. Each sample
is replayed independently. Event-level data stays local, and no raw paths,
identifiers, event IDs, or records are emitted in the report.

Usage:
  python src/prototype/reducer_institute_six_method.py \
    --input-dir /media/ayush/CSOC/alert_fatigue_eval/sanitized \
    --outdir /media/ayush/CSOC/alert_fatigue_eval/results/six_method_institute
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from correlation_engine import Correlator, read_json_lines
from evidence_reducer import Policy
from reducer_six_method_eval import METHODS, selections, audit_cluster

ADDITIVE = (
    'alerts', 'visible', 'hidden', 'incidents', 'constraints_total',
    'constraints_covered', 'hard_required', 'hard_retained',
    'critical_total', 'critical_retained', 'techniques_total',
    'techniques_retained', 'unmapped_types_total', 'unmapped_types_retained',
    'links_evaluable', 'links_endpoints_retained', 'cases_with_unmet_requirements',
    'tagged_alerts', 'tagged_visible', 'untagged_alerts', 'untagged_visible',
    'level10plus_alerts', 'level10plus_visible', 'selected_unmapped_alerts',
)


def inspect_sample(sample: str, path: Path, policy: Policy):
    events = read_json_lines([path])
    if not events:
        raise ValueError(f'Sample {sample} is empty')
    engine = Correlator()
    for event in events:
        engine.feed(event)
    engine.finish()
    totals = {method: Counter() for method in METHODS}
    for cluster in engine.clusters:
        decisions = selections(cluster, policy)
        for method, selected in decisions.items():
            a = audit_cluster(cluster, selected, policy)
            selected_set = set(selected)
            a['tagged_alerts'] = sum(bool(e.techniques) for e in cluster.members)
            a['tagged_visible'] = sum(bool(cluster.members[i].techniques) for i in selected_set)
            a['untagged_alerts'] = sum(not e.techniques for e in cluster.members)
            a['untagged_visible'] = sum(not cluster.members[i].techniques for i in selected_set)
            a['level10plus_alerts'] = sum(e.level >= 10 for e in cluster.members)
            a['level10plus_visible'] = sum(cluster.members[i].level >= 10 for i in selected_set)
            a['selected_unmapped_alerts'] = a['untagged_visible']
            totals[method].update(a)
    rows = [format_row(sample, method, totals[method]) for method in METHODS]
    return rows, {'sample': sample, 'alerts': len(events),
                  'reconstructed_groups': len(engine.clusters),
                  'observed_links': sum(len(c.links) for c in engine.clusters),
                  'tagged_alerts': sum(bool(e.techniques) for e in events),
                  'level10plus_alerts': sum(e.level >= 10 for e in events)}


def format_row(scenario, method, numbers):
    row = {'sample': scenario, 'method': method}
    row.update({k: int(numbers[k]) for k in ADDITIVE})
    row['hidden'] = row['alerts'] - row['visible']
    row['visible_reduction_pct'] = round(100 * row['hidden'] / row['alerts'], 3) if row['alerts'] else 0.0
    row['constraint_coverage_pct'] = round(100 * row['constraints_covered'] / row['constraints_total'], 3) if row['constraints_total'] else 100.0
    row['technique_coverage_pct'] = round(100 * row['techniques_retained'] / row['techniques_total'], 3) if row['techniques_total'] else 100.0
    row['unmapped_type_coverage_pct'] = round(100 * row['unmapped_types_retained'] / row['unmapped_types_total'], 3) if row['unmapped_types_total'] else 100.0
    row['link_endpoint_coverage_pct'] = round(100 * row['links_endpoints_retained'] / row['links_evaluable'], 3) if row['links_evaluable'] else 100.0
    row['all_constraints_preserved'] = row['cases_with_unmet_requirements'] == 0
    return row


def run(input_dir, outdir, names=('A','B','C')):
    directory, dest = Path(input_dir), Path(outdir)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f'Refusing to overwrite nonempty path: {dest}')
    if len(names) != len(set(names)) or not names:
        raise ValueError('Sample identifiers must be distinct and nonempty')
    if any(n not in ('A','B','C') for n in names):
        raise ValueError('Only predefined sample labels A, B and C are permitted')
    # Validate presence of all inputs before writing any output.
    paths = {n: directory / f'sample_{n}.jsonl' for n in names}
    for n, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f'Missing local sanitized sample {n}')
    policy = Policy(critical_level=12, keep_anchors=True)
    rows, summaries = [], []
    for n in names:
        sample_rows, stats = inspect_sample(n, paths[n], policy)
        rows.extend(sample_rows)
        summaries.append(stats)
    totals = {method: Counter() for method in METHODS}
    for row in rows:
        totals[row['method']].update({k: row[k] for k in ADDITIVE})
    rows.extend(format_row('TOTAL_SEPARATE_WINDOWS', m, totals[m]) for m in METHODS)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'method_comparison.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        'protocol': 'institute_Wazuh_sanitized_six_method_v1',
        'source': 'locally sanitized, authorized institute Wazuh fixed two-minute samples',
        'samples': list(names), 'sample_summaries': summaries,
        'methods': list(METHODS),
        'policy_frozen': {'critical_level':12,'keep_anchors':True,
                          'protect_unknown_rule':True,'protect_link_endpoints':True},
        'method_reference': 'reducer_six_method_eval.py selections() / audit_cluster()',
        'privacy': 'Only aggregate per-sample counts; no raw or pseudonymous entities, timestamps, file paths, UIDs or event bodies are written.',
        'limitations': [
            'Natural institutional alerts, but unlabelled: cannot estimate attack recall, false positives, analyst time saved or true incident grouping precision.',
            'One institute, three predefined two-minute windows; outcomes not independent SOC deployments.',
            'Sanitization omits original descriptions and some entity fields; only observed evidence requirements can be assessed.',
            'Coverage requirements come from the proposed method and are not independently validated analyst evidence.',
            'The same evidence-linked correlator and clusters are shared across methods; impact of alternative grouping not evaluated here.',
            'Severity-budget baseline matches candidate visible count retrospectively and is not an independently deployable threshold policy.',
            'Technique and unmapped-rule coverage are measured per reconstructed cluster, not global unique counts.',
            'Coverage of one endpoint per observed link does not guarantee a visible connected attack path.',
            'Severities >=10 are separately audited; >=12 alone is the predefined hard critical policy.',
            'No tuning, retraining, live deployment, or changes to existing code and outputs.',
        ],
    }
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Aggregate institute Wazuh six-method comparison (NOT analyst workload reduction):')
    for summary in summaries:
        print(f"  Sample {summary['sample']}: {summary['alerts']} alerts, {summary['reconstructed_groups']} reconstructed groups")
    for r in rows[-len(METHODS):]:
        print(f"  {r['method']:19s} visible {r['visible']:5d}/{r['alerts']:5d}, "
              f"hidden {r['visible_reduction_pct']:5.1f}%, "
              f"defined coverage {r['constraint_coverage_pct']:5.1f}%, "
              f"level>=10 retained {r['level10plus_visible']}/{r['level10plus_alerts']}, "
              f"failed groups {r['cases_with_unmet_requirements']}")
    print('Results:', dest)
    return rows, manifest


def main():
    ap = argparse.ArgumentParser(description='Local-only, aggregate six-method institute Wazuh benchmark')
    ap.add_argument('--input-dir', required=True, help='Private directory containing sample_A/B/C.jsonl')
    ap.add_argument('--outdir', required=True, help='New directory for aggregate-only results')
    args = ap.parse_args()
    run(args.input_dir, args.outdir)


if __name__ == '__main__':
    main()
