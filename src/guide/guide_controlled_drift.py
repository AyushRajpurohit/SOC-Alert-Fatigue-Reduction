"""Controlled case-mix shift on REAL GUIDE completed incidents (offline replay).

Purpose: validate whether a frozen score-distribution episode monitor detects a
known, deliberately constructed change in the prevalence of MITRE-tagged
incidents, versus a matched no-change control. Existing incidents are sampled
WITHOUT replacement within each replay; scores, grades, and alert counts are
never modified. Their event order is intentionally shuffled, so replay position
is NOT an actual GUIDE timestamp. This does not test concept drift or retraining.

Calibrate the monitor exclusively from the baseline CALIBRATION predictions.
The held-out TEST predictions supply examples to a controlled replay; their
labels are read ONLY after score-only detection decisions for evaluation.

Requires guide_temporal.py, guide_drift_monitor.py in the same folder.
Example:
 python src/guide/guide_controlled_drift.py \\
   --baseline-dir results/guide_temporal \\
   --cache data/processed/guide_incidents.pkl \\
   --outdir results/guide_controlled_drift

Outputs: manifest.json, comparison.csv, bucket_trace.csv. Synthetic replay
ORDER and case-mix intervention; real GUIDE records. No claimed production
performance or natural drift, no model retraining or promotion.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from guide_temporal import load_cache, prepare
from guide_drift_monitor import read_scores


def build_samples(pool, pre_n, post_n, pre_share, post_share, rng):
    """Sample unique actual incidents in a controlled mixture; no duplicates."""
    groups = {}
    for tag in (0, 1):
        arr = np.flatnonzero(pool.tagged.to_numpy() == tag)
        groups[tag] = rng.permutation(arr)
    selected = []
    for size, frac in ((pre_n, pre_share), (post_n, post_share)):
        tagged_n = int(round(size * frac))
        selected.append((tagged_n, size - tagged_n))
    totals = {1: sum(a for a, b in selected), 0: sum(b for a, b in selected)}
    for tag in (0, 1):
        if totals[tag] > len(groups[tag]):
            raise ValueError(f"Not enough unique {'tagged' if tag else 'untagged'} test incidents: "
                             f"need {totals[tag]}, available {len(groups[tag])}. "
                             "Reduce --buckets or --post-share.")
    cursor = {0: 0, 1: 0}
    chunks = []
    for tagged_n, untagged_n in selected:
        ids = []
        for tag, count in ((1, tagged_n), (0, untagged_n)):
            ids.extend(groups[tag][cursor[tag]:cursor[tag] + count].tolist())
            cursor[tag] += count
        ids = rng.permutation(ids)
        chunks.append(pool.iloc[ids].copy().reset_index(drop=True))
    assert not set(chunks[0].sid).intersection(chunks[1].sid)
    return pd.concat(chunks, ignore_index=True)


def calibrated_cutoff(cal, bucket_size, quantile, min_sd):
    """Fit an independent reference and tolerance on calibration scores only."""
    x = cal.sort_values(['t_last', 'sid'], kind='stable').reset_index(drop=True)
    half = len(x) // 2
    ref = x.iloc[:half]
    tune = x.iloc[half:]
    reference_mean = float(ref.score.mean())
    reference_sd = float(ref.score.std(ddof=0))
    full = len(tune) // bucket_size
    if full < 10:
        raise ValueError('Need at least 10 tuning buckets; reduce --bucket-size')
    a = tune.score.to_numpy()[:full * bucket_size].reshape(-1, bucket_size).mean(axis=1)
    empirical = float(np.quantile(np.abs(a - reference_mean), quantile))
    return reference_mean, reference_sd, max(empirical, min_sd * reference_sd), empirical


def monitor_replay(frame, scenario, bucket_size, reference_mean, cutoff, confirm):
    """Scores only for alarms; review TP labels strictly after detection."""
    n = len(frame)
    trace = []
    streak = 0
    episode = False
    first_confirmed = None
    first_post = n // (2 * bucket_size)
    for b in range(n // bucket_size):
        part = frame.iloc[b * bucket_size:(b + 1) * bucket_size]
        mean = float(part.score.mean())
        direction = int(np.sign(mean - reference_mean)) if abs(mean - reference_mean) > cutoff else 0
        if direction:
            streak += 1
        else:
            streak = 0
        new_alarm = bool(streak >= confirm and not episode)
        if new_alarm:
            episode = True
            if first_confirmed is None:
                first_confirmed = b
        if direction == 0:
            episode = False
        trace.append({'scenario': scenario, 'bucket': b, 'phase': 'post' if b >= first_post else 'pre',
                      'mean_score': mean, 'score_deviation': mean - reference_mean,
                      'beyond_cutoff': bool(direction), 'new_episode': new_alarm,
                      'tagged_fraction': float(part.tagged.mean())})
    return trace, first_confirmed


def period_metrics(frame, threshold):
    y = frame.is_true_positive.to_numpy(dtype=int)
    s = frame.score.to_numpy(dtype=float)
    demote = s <= threshold
    return {'n': int(len(frame)), 'tagged_share': float(frame.tagged.mean()),
            'tp_prevalence': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
            'average_precision': float(average_precision_score(y, s)) if y.sum() else None,
            'tp_recall': float(y[~demote].sum() / y.sum()) if y.sum() else None,
            'incident_demotion': float(demote.mean()),
            'alert_demotion': float(frame.n_alerts.to_numpy()[demote].sum() / frame.n_alerts.sum()),
            'tp_incidents_demoted': int(y[demote].sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline-dir', default='results/guide_temporal')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--outdir', default='results/guide_controlled_drift')
    ap.add_argument('--bucket-size', type=int, default=1000)
    ap.add_argument('--buckets-per-phase', type=int, default=10)
    ap.add_argument('--post-share', type=float, default=0.8,
                    help='Target MITRE-tagged share AFTER intervention; default 0.8')
    ap.add_argument('--quantile', type=float, default=0.99)
    ap.add_argument('--min-effect-sd', type=float, default=0.25)
    ap.add_argument('--confirm-buckets', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if not (args.bucket_size >= 10 and args.buckets_per_phase >= 3 and
            0 < args.post_share < 1 and 0.5 <= args.quantile < 1 and
            args.min_effect_sd >= 0 and args.confirm_buckets >= 1):
        ap.error('Invalid monitor/replay settings')
    root = Path(args.baseline_dir)
    output = Path(args.outdir)
    if output.exists() and any(output.iterdir()):
        ap.error('Refusing to overwrite a nonempty output directory')
    cal = read_scores(root / 'calibration_predictions.csv')
    test = read_scores(root / 'test_predictions.csv')
    metrics = json.loads((root / 'metrics.json').read_text())
    raw = prepare(load_cache(args.cache))
    if not raw.index.is_unique:
        raise ValueError('Incident cache has duplicate IDs')
    # Only derive the tagged cohort from independent recorded features. No labels in cohort selection.
    tag = raw.n_techs.gt(0).rename('tagged')
    test = test.merge(tag, left_on='sid', right_index=True, how='left', validate='one_to_one')
    if test.tagged.isna().any():
        raise ValueError('Test incident ID absent from cache')
    test['tagged'] = test.tagged.astype(int)
    test['is_true_positive'] = raw.loc[test.sid, 'grade'].eq('TruePositive').to_numpy(dtype=int)
    test['n_alerts'] = raw.loc[test.sid, 'n_alerts'].to_numpy(dtype=int)
    # Reference protocol and thresholds fixed using baseline calibration (no test labels).
    ref_mean, ref_sd, cutoff, empirical = calibrated_cutoff(
        cal, args.bucket_size, args.quantile, args.min_effect_sd)
    thresholds = {k.removeprefix('target_'): v['threshold'] for k, v in metrics.items()
                  if k.startswith('target_') and isinstance(v, dict)}
    if not thresholds or any(x is None for x in thresholds.values()):
        raise ValueError('Expected finite saved baseline thresholds')
    pre_share = float(test.tagged.mean())
    if abs(pre_share - args.post_share) < 0.02:
        raise ValueError('Post-share too close to reference share for a meaningful intervention')
    per = args.bucket_size * args.buckets_per_phase
    results, traces = [], []
    for scenario, post_share, offset in [('no_change', pre_share, 0),
                                          ('mixture_shift', args.post_share, 1)]:
        stream = build_samples(test, per, per, pre_share, post_share,
                               np.random.default_rng(args.seed + offset))
        trace, first = monitor_replay(stream, scenario, args.bucket_size,
                                      ref_mean, cutoff, args.confirm_buckets)
        traces.extend(trace)
        pre = stream.iloc[:per]
        post = stream.iloc[per:]
        for phase, frame in [('pre', pre), ('post', post)]:
            rec = {'scenario': scenario, 'phase': phase,
                   'confirmed_episodes_in_phase': sum(int(t['new_episode']) for t in trace
                                                      if t['phase'] == phase)}
            for name, thr in thresholds.items():
                for k, v in period_metrics(frame, float(thr)).items():
                    rec[f'target_{name}_{k}'] = v
            results.append(rec)
        print(f'{scenario}: episodes={sum(t["new_episode"] for t in trace)}, '
              f'first_confirmation_bucket={first}, '
              f'known_change_bucket={args.buckets_per_phase if scenario == "mixture_shift" else "none"}')
    out_manifest = {
        'protocol': 'controlled_tagged_case_mix_replay_from_real_GUIDE_incidents_v1',
        'baseline_dir': str(root), 'cache': str(args.cache),
        'seed': args.seed, 'bucket_size': args.bucket_size,
        'buckets_per_phase': args.buckets_per_phase,
        'known_change_bucket': args.buckets_per_phase,
        'reference_tagged_share': pre_share,
        'intervention_post_tagged_share': args.post_share,
        'calibration_reference_mean': ref_mean, 'calibration_reference_std': ref_sd,
        'empirical_score_cutoff': empirical, 'applied_score_cutoff': cutoff,
        'confirm_buckets': args.confirm_buckets, 'frozen_thresholds': thresholds,
        'scenario_results': {sc: {
            'confirmed_episode_buckets': [x['bucket'] for x in traces if x['scenario'] == sc and x['new_episode']],
            'false_episodes_before_change': sum(x['new_episode'] for x in traces if x['scenario'] == sc and x['phase'] == 'pre')
        } for sc in ('no_change', 'mixture_shift')},
        'limitations': [
            'CONTROLLED REPLAY: incident order and cohort mixture are deliberately constructed, not natural GUIDE chronology.',
            'Real GUIDE rows/labels/scores are unchanged; within each scenario, incidents are used without replacement.',
            'Cohort composition is changed by selecting MITRE-tagged incidents, NOT by creating attacks or changing grades.',
            'Baseline score detector only; no predictive model training, adaptive promotion, or measured adaptation benefit.',
            'Only one seed and one injected change are run by default; no false-alarm or detection-delay rate estimate.',
            'GUIDE completed-incident predictions, not live causal detection; privacy-noised timestamps.',
            'Retrospective labels evaluate impact only; no test labels are used by the monitor.',
            'Original chronological test set is reused for controlled replay; not a fresh independent benchmark.',
            'Score-mean shifts may reflect case mix rather than concept drift or changes in conditional labels.'
        ]}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(out_manifest, indent=2))
    pd.DataFrame(results).to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(traces).to_csv(output / 'bucket_trace.csv', index=False)
    print(f'Saved controlled replay outputs in {output}. No previous results overwritten.')


if __name__ == '__main__':
    main()
