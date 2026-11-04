"""Controlled score-vs-MITRE-tag-coverage shift detection on existing GUIDE incidents.

NO model training, NO adaptation, NO deployment, NO real-world drift claims.
All incident records and their model scores/grades are unchanged. The replay
changes the MIX of existing MITRE-tagged and untagged incidents at a known
synthetic change point. Each replay uses incidents without replacement; across
seeds/scenarios the same dataset can be reused. Results are controlled-replay
measurements, NOT natural temporal GUIDE drift or fresh external validation.

The score-only monitor replicates the frozen mean-score criterion from the
previous controlled-drift experiment. The MITRE monitor watches the share of
incidents carrying at least one MITRE technique. The OR-combined monitor
confirms a signal if either criterion persists. All thresholds are fixed from
calibration data, never from test intervention outcomes or test labels.

Run from project root:
  python src/guide/guide_dual_signal.py \\
    --baseline-dir results/guide_temporal \\
    --cache data/processed/guide_incidents.pkl \\
    --outdir results/guide_dual_signal

Requires: guide_temporal.py and guide_drift_monitor.py in the same directory.
Outputs: manifest.json, replay_metrics.csv, bucket_trace.csv, summary.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from guide_temporal import load_cache, prepare
from guide_drift_monitor import read_scores, validate_baseline


def make_reference(cal, tag_by_id, bucket_size, quantile, min_score_sd, min_tag_gap):
    """Calibrate using earlier calibration half for reference, later for tolerance."""
    x = cal.merge(tag_by_id, left_on='sid', right_index=True, how='left', validate='one_to_one')
    if x.tagged.isna().any():
        raise ValueError('Calibration incident missing from cache')
    x = x.sort_values(['t_last', 'sid'], kind='stable').reset_index(drop=True)
    half = len(x) // 2
    ref, tune = x.iloc[:half], x.iloc[half:]
    nr = len(ref) // bucket_size
    nt = len(tune) // bucket_size
    if min(nr, nt) < 10:
        raise ValueError('Need at least 10 full buckets in each calibration half')
    mean_score = float(ref.score.mean())
    mean_tag = float(ref.tagged.mean())
    score_sd = float(ref.score.std(ddof=0))
    # The frozen reference is from first-half calibration, but the tuning
    # distribution from second-half calibration may already have shifted.
    score_means = tune.score.to_numpy()[:nt * bucket_size].reshape(nt, bucket_size).mean(axis=1)
    tagged_means = tune.tagged.to_numpy()[:nt * bucket_size].reshape(nt, bucket_size).mean(axis=1)
    score_cut = max(float(np.quantile(abs(score_means - mean_score), quantile)), min_score_sd * score_sd)
    tag_cut = max(float(np.quantile(abs(tagged_means - mean_tag), quantile)), min_tag_gap)
    return dict(reference_score_mean=mean_score, reference_score_std=score_sd,
                reference_tagged_share=mean_tag, score_cutoff=score_cut, tagged_cutoff=tag_cut,
                reference_buckets=nr, tuning_buckets=nt, calibration_ref_rows=len(ref),
                calibration_tuning_rows=len(tune))


def sample_replay(pool, n_per_phase, pre_share, post_share, seed):
    """Draw disjoint incident records for the two phases of one replay."""
    rng = np.random.default_rng(seed)
    tags = pool.tagged.to_numpy(dtype=int)
    groups = {t: rng.permutation(np.flatnonzero(tags == t)) for t in (0, 1)}
    selections = []
    cursor = {0: 0, 1: 0}
    for share in (pre_share, post_share):
        n1 = int(round(share * n_per_phase))
        n0 = n_per_phase - n1
        ids = []
        for t, n in ((0, n0), (1, n1)):
            if cursor[t] + n > len(groups[t]):
                raise ValueError('Insufficient unique test incidents for replay; reduce size or shift strength')
            ids.extend(groups[t][cursor[t]:cursor[t] + n].tolist())
            cursor[t] += n
        selections.append(pool.iloc[rng.permutation(ids)].copy().reset_index(drop=True))
    if set(selections[0].sid) & set(selections[1].sid):
        raise AssertionError('Replay repeated an incident across phases')
    return pd.concat(selections, ignore_index=True)


def monitor_replay(frame, reference, bucket_size, buckets_per_phase, confirm, recovery):
    """Three independent policies; one episode until sustained recovery.

    No access to outcomes/labels. The combined monitor is an OR of two
    predeclared monitors, not a new optimized decision threshold.
    """
    policies = ['score_only', 'mitre_only', 'combined_or']
    state = {p: dict(active=False, streak=0, recovery=0, starts=[]) for p in policies}
    traces = []
    n_buckets = len(frame) // bucket_size
    for b in range(n_buckets):
        part = frame.iloc[b * bucket_size:(b + 1) * bucket_size]
        score_mean = float(part.score.mean())
        tag_share = float(part.tagged.mean())
        score_alarm = abs(score_mean - reference['reference_score_mean']) > reference['score_cutoff']
        mitre_alarm = abs(tag_share - reference['reference_tagged_share']) > reference['tagged_cutoff']
        checks = {'score_only': score_alarm, 'mitre_only': mitre_alarm,
                  'combined_or': score_alarm or mitre_alarm}
        row = dict(bucket=b, phase='pre' if b < buckets_per_phase else 'post',
                   score_mean=score_mean, tagged_share=tag_share,
                   score_outside_reference=score_alarm, mitre_outside_reference=mitre_alarm)
        for p, shifted in checks.items():
            s = state[p]
            fired = False
            if not s['active']:
                s['streak'] = s['streak'] + 1 if shifted else 0
                if s['streak'] >= confirm:
                    s['active'] = True
                    s['recovery'] = 0
                    s['starts'].append(b)
                    fired = True
            else:
                s['recovery'] = 0 if shifted else s['recovery'] + 1
                if s['recovery'] >= recovery:
                    s['active'] = False
                    s['streak'] = 0
                    s['recovery'] = 0
            row[p + '_confirmed'] = fired
            row[p + '_active'] = s['active']
        traces.append(row)
    return traces, state


def evaluate_one(frame, ref, args, scenario, seed, post_share):
    stream = sample_replay(frame, args.bucket_size * args.buckets_per_phase,
                           float(frame.tagged.mean()), post_share, seed)
    trace, states = monitor_replay(stream, ref, args.bucket_size, args.buckets_per_phase,
                                    args.confirm_buckets, args.recovery_buckets)
    rows = []
    for policy, st in states.items():
        alarms = st['starts']
        pre = [b for b in alarms if b < args.buckets_per_phase]
        post = [b for b in alarms if b >= args.buckets_per_phase]
        rows.append(dict(seed=seed, scenario=scenario, target_post_share=post_share,
                         policy=policy, change_bucket=args.buckets_per_phase if scenario != 'no_change' else '',
                         pre_false_episodes=len(pre), post_episodes=len(post),
                         detected_after_change=bool(post) if scenario != 'no_change' else None,
                         delay_buckets=(post[0] - args.buckets_per_phase) if post and scenario != 'no_change' else None,
                         first_post_confirmation=post[0] if post else None,
                         actual_pre_tagged_share=float(stream.iloc[:args.bucket_size*args.buckets_per_phase].tagged.mean()),
                         actual_post_tagged_share=float(stream.iloc[args.bucket_size*args.buckets_per_phase:].tagged.mean())))
    for row in trace:
        row.update(seed=seed, scenario=scenario, target_post_share=post_share)
    return rows, trace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline-dir', default='results/guide_temporal')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--outdir', default='results/guide_dual_signal')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--post-shares', type=float, nargs='+', default=[0.55, 0.65, 0.8])
    ap.add_argument('--bucket-size', type=int, default=1000)
    ap.add_argument('--buckets-per-phase', type=int, default=10)
    ap.add_argument('--confirm-buckets', type=int, default=3)
    ap.add_argument('--recovery-buckets', type=int, default=3)
    ap.add_argument('--quantile', type=float, default=0.99)
    ap.add_argument('--min-score-sd', type=float, default=0.25)
    ap.add_argument('--min-tag-gap', type=float, default=0.05)
    args = ap.parse_args()
    if args.seeds < 1 or args.bucket_size < 10 or args.buckets_per_phase < args.confirm_buckets or \
       args.confirm_buckets < 1 or args.recovery_buckets < 1 or not (0.5 <= args.quantile < 1) or \
       args.min_score_sd < 0 or args.min_tag_gap < 0 or not args.post_shares or \
       any(not (0 < x < 1) for x in args.post_shares):
        ap.error('Invalid monitoring configuration')
    out = Path(args.outdir)
    if out.exists() and any(out.iterdir()):
        ap.error('Refusing to overwrite a nonempty directory')
    base = Path(args.baseline_dir)
    cal = read_scores(base / 'calibration_predictions.csv')
    test = read_scores(base / 'test_predictions.csv')
    baseline_manifest = validate_baseline(base, cal, test)
    raw = prepare(load_cache(args.cache))
    if not raw.index.is_unique:
        raise ValueError('Cache incident IDs must be unique')
    # Only the MITRE-presence flag is used in the detector, NOT labels.
    tag = raw.n_techs.gt(0).astype(int).rename('tagged')
    ref = make_reference(cal, tag, args.bucket_size, args.quantile,
                         args.min_score_sd, args.min_tag_gap)
    test = test.merge(tag, left_on='sid', right_index=True, how='left', validate='one_to_one')
    if test.tagged.isna().any():
        raise ValueError('Test incident absent from cache')
    pool = test[['sid','score','tagged']].copy()
    pre_share = float(pool.tagged.mean())
    if any(abs(x - pre_share) < 0.02 for x in args.post_shares):
        ap.error('A post-share is too close to the held-out test tagged share')
    rows, traces = [], []
    scenarios = [('no_change', pre_share)] + [(f'mixture_{x:.2f}', x) for x in args.post_shares]
    for i in range(args.seeds):
        for j, (name, post_share) in enumerate(scenarios):
            records, trace = evaluate_one(pool, ref, args, name, args.seed + 1009*i + 17*j, post_share)
            rows.extend(records)
            traces.extend(trace)
    df = pd.DataFrame(rows)
    # Aggregate across SEEDS. Non-detection counts as missing delay; never impute zero.
    summary = df.groupby(['scenario','policy'], sort=False).agg(
        runs=('seed','count'), pre_false_episodes_total=('pre_false_episodes','sum'),
        post_episodes_total=('post_episodes','sum'),
        detection_rate=('detected_after_change', 'mean'),
        mean_detected_delay_buckets=('delay_buckets','mean'),
        median_detected_delay_buckets=('delay_buckets','median'),
        mean_post_tagged_share=('actual_post_tagged_share','mean')).reset_index()
    # No-change scenario has no known change, so post episodes are false alarms too.
    summary['false_episode_total'] = np.where(summary.scenario.eq('no_change'),
                                             summary.pre_false_episodes_total + summary.post_episodes_total,
                                             summary.pre_false_episodes_total)
    metadata = dict(protocol='controlled_GUIDE_tagged_case_mix_dual_signal_v1',
                    source='existing GUIDE completed-incident predictions, not new labels or synthetic alerts',
                    user_seed=args.seed, repetitions=args.seeds,
                    bucket_size=args.bucket_size, buckets_per_phase=args.buckets_per_phase,
                    known_change_bucket=args.buckets_per_phase,
                    requested_post_tagged_shares=args.post_shares,
                    reference_tagged_share_in_test=pre_share,
                    confirmation_buckets=args.confirm_buckets,
                    recovery_buckets=args.recovery_buckets,
                    calibration=ref,
                    original_baseline_test_cutoff=baseline_manifest['calibration_end_exclusive'],
                    original_test_reused=True,
                    limitations=[
                      'CONTROLLED case-mix reordering/selection of real GUIDE incident records, not natural drift.',
                      'No modified source incidents, predictions, or grades; no fabricated SOC performance numbers.',
                      'Thresholds tuned on historic calibration; test data only used for evaluation, not tuning.',
                      'Changing tagged prevalence is the intervention itself: MITRE monitor has direct access to the changed variable, which is a designed test not a generic performance guarantee.',
                      'Only binary presence of any MITRE technique is monitored; not yet technique identities, transition drift, or concept drift.',
                      'OR combination can increase false positives. Both sensitivity and control false alarms must be reported.',
                      'Replay draws without replacement within scenarios, but repeats reuse original held-out data; they are not independent datasets.',
                      'The original held-out test set has already been inspected in preceding experiments; confirmatory external validation remains outstanding.',
                      'No model update, safe promotion, improved recall, or overall research novelty is demonstrated by this monitor alone.',
                      'GUIDE timestamps are privacy-adjusted and case-mix replay order is artificial.'
                    ])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'replay_metrics.csv', index=False)
    pd.DataFrame(traces).to_csv(out / 'bucket_trace.csv', index=False)
    summary.to_csv(out / 'summary.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    print('Controlled GUIDE replay only; no training, no live deployment.')
    print(f'Repetitions: {args.seeds}; scenarios: {len(scenarios)}; bucket size: {args.bucket_size}')
    print(f'Calibration cutoffs: score={ref["score_cutoff"]:.5f}, tagged-share={ref["tagged_cutoff"]:.5f}')
    print(summary[['scenario','policy','runs','detection_rate','false_episode_total',
                   'mean_detected_delay_buckets']].to_string(index=False))
    print(f'Saved outputs to {out}. No existing results were overwritten.')


if __name__ == '__main__':
    main()
