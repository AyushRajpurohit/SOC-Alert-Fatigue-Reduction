"""GUIDE read-only, label-free drift monitor (bucketed, episode-based).

This is a MONITORING experiment, not concept-drift proof or automatic retraining.
Consumes guide_temporal.py's frozen calibration_predictions.csv and
 test_predictions.csv. Never reads prediction labels or trains a model.

Method:
  1. Divide calibration chronologically into reference (first 50%) and tuning
     (remaining 50%). All reference statistics/thresholds are fixed before test.
  2. Process test predictions in nonoverlapping completion-order buckets.
  3. Track the bucket score mean relative to a frozen reference. Mark a bucket
     as shifted only beyond BOTH an empirical tuning-quantile threshold and a
     minimum practical shift (in reference SD units).
  4. Optional statistical detector: Page-Hinkley on bucket means (implemented
     here), or river's ADWIN on bucket means (requires `pip install river`).
     These are corroborating diagnostics; they do not bypass the frozen-reference
     shift gate. Their flags do not decide whether to retrain automatically.
  5. Emit at most one event per shift episode. Require consecutive shifted
     buckets for confirmation and consecutive in-range buckets for recovery.

NOTE: Calibration tuning is observational, not a guarantee of false-alarm rate.
GUIDE timestamps are privacy-adjusted; incidents are completed, not live.
This is not a measured adaptation improvement. No labels are used for alarms.

Example:
  python src/guide/guide_drift_monitor.py \\
    --baseline-dir results/guide_temporal \\
    --outdir results/guide_drift_monitor \\
    --detector ph
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"sid", "OrgId", "t_first", "t_last", "score"}


def read_scores(path):
    frame = pd.read_csv(path, usecols=lambda c: c in REQUIRED, dtype={"sid": str, "OrgId": str})
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    if frame.sid.isna().any() or frame.sid.duplicated().any() or frame.OrgId.isna().any():
        raise ValueError(f"{path}: missing or duplicate incident IDs / missing org")
    frame["t_first"] = pd.to_datetime(frame.t_first, utc=True, errors="coerce")
    frame["t_last"] = pd.to_datetime(frame.t_last, utc=True, errors="coerce")
    frame["score"] = pd.to_numeric(frame.score, errors="coerce")
    if frame[["t_first", "t_last", "score"]].isna().any().any():
        raise ValueError(f"{path}: invalid score or timestamp")
    if (frame.t_first > frame.t_last).any() or not frame.score.between(0, 1).all():
        raise ValueError(f"{path}: invalid incident times or score outside [0,1]")
    return frame.sort_values(["t_last", "sid"], kind="stable").reset_index(drop=True)


def buckets(frame, n, allow_partial=False):
    out = []
    count = len(frame)
    for begin in range(0, count, n):
        end = min(begin + n, count)
        if end - begin != n and not allow_partial:
            break
        part = frame.iloc[begin:end]
        out.append({
            "index": len(out),
            "start_utc": part.t_last.iloc[0].isoformat(),
            "end_utc": part.t_last.iloc[-1].isoformat(),
            "incidents": len(part),
            "mean_score": float(part.score.mean()),
            "std_score": float(part.score.std(ddof=0)),
            "orgs": int(part.OrgId.nunique()),
        })
    return out


class PageHinkley:
    """Two-sided, running-mean PH applied once per nonoverlapping score bucket.

    Tuning values are in units of bucket-mean changes (raw probability points).
    Once a shift is detected, state is restarted; further alerts are suppressed
    by the separate episode policy, not by a fixed incident-count cooldown.
    """

    def __init__(self, delta, threshold):
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.up = 0.0
        self.down = 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.up = max(0.0, self.up + x - self.mean - self.delta)
        self.down = max(0.0, self.down + self.mean - x - self.delta)
        alarm = self.n > 1 and max(self.up, self.down) > self.threshold
        if alarm:
            self.reset()
        return alarm


def make_detector(kind, args):
    if kind == "none":
        return None
    if kind == "ph":
        return PageHinkley(args.ph_delta, args.ph_threshold)
    if kind == "adwin":
        try:
            from river import drift
        except ImportError as err:
            raise RuntimeError("ADWIN needs river: python -m pip install river") from err
        return drift.ADWIN(delta=args.adwin_delta, clock=1, min_window_length=5, grace_period=10)
    raise ValueError(kind)


def validate_baseline(base, cal, test):
    manifest = json.loads((base / "manifest.json").read_text())
    if len(cal) != manifest["splits"]["calibration"]["incidents"]:
        raise ValueError("Calibration predictions do not match manifest count")
    if len(test) != manifest["splits"]["test"]["incidents"]:
        raise ValueError("Test predictions do not match manifest count")
    if set(cal.sid) & set(test.sid):
        raise ValueError("Calibration and test overlap by incident ID")
    cutoff = pd.Timestamp(manifest["calibration_end_exclusive"])
    if cal.t_last.max() >= cutoff or test.t_first.min() < cutoff:
        raise ValueError("Chronological split contradicts manifest cutoff")
    return manifest


def monitor(cal, test, args):
    # Split the *earlier* calibration records; never tune on test observations.
    n_ref = len(cal) // 2
    ref = cal.iloc[:n_ref]
    tuning = cal.iloc[n_ref:]
    ref_mean = float(ref.score.mean())
    ref_std = float(ref.score.std(ddof=0))
    if ref_std < 1e-8:
        raise ValueError("Reference score variance too small for standardised shifts")
    ref_batches = buckets(ref, args.bucket_size)
    tuning_batches = buckets(tuning, args.bucket_size)
    if len(ref_batches) < args.min_tuning_buckets or len(tuning_batches) < args.min_tuning_buckets:
        raise ValueError("Not enough calibration buckets; reduce --bucket-size or --min-tuning-buckets")
    # Historical, sequential tuning variation relative to a FROZEN early reference.
    calibration_abs_shifts = np.array([abs(b["mean_score"] - ref_mean) for b in tuning_batches])
    empirical_cut = float(np.quantile(calibration_abs_shifts, args.reference_quantile))
    practical_cut = float(args.min_effect_sd * ref_std)
    shift_cut = max(empirical_cut, practical_cut)
    detector = make_detector(args.detector, args)
    # Prime adaptive change detectors on reference period bucket means. Do not
    # give them labels, and don't use test feedback to adjust their settings.
    if detector is not None:
        for b in ref_batches:
            if args.detector == "adwin":
                detector.update(b["mean_score"])
            else:
                detector.update(b["mean_score"])
    rows, events = [], []
    consecutive_shift = consecutive_recovery = 0
    episode_active = False
    episode_id = 0
    for b in buckets(test, args.bucket_size):
        x = b["mean_score"]
        deviation = x - ref_mean
        shifted = abs(deviation) > shift_cut
        if detector is None:
            detector_alarm = False
        elif args.detector == "adwin":
            detector.update(x)
            detector_alarm = bool(detector.drift_detected)
        else:
            detector_alarm = bool(detector.update(x))
        # Detector corroboration is recorded, not required for reference-shift
        # episode detection (ADWIN/PH may see a step and then stabilise).
        if not episode_active:
            consecutive_shift = consecutive_shift + 1 if shifted else 0
            if consecutive_shift >= args.confirm_buckets:
                episode_id += 1
                episode_active = True
                consecutive_recovery = 0
                events.append({"episode_id": episode_id, "start_bucket_index": b["index"] - args.confirm_buckets + 1,
                               "confirmed_bucket_index": b["index"], "confirmed_utc": b["end_utc"],
                               "direction": "higher_scores" if deviation > 0 else "lower_scores",
                               "mean_score": x, "reference_mean": ref_mean,
                               "deviation": deviation, "shift_cutoff": shift_cut,
                               "detector_alarm_on_confirmation": detector_alarm,
                               "status": "open"})
        else:
            consecutive_recovery = consecutive_recovery + 1 if not shifted else 0
            if consecutive_recovery >= args.recovery_buckets:
                events[-1]["status"] = "recovered"
                events[-1]["recovered_bucket_index"] = b["index"]
                events[-1]["recovered_utc"] = b["end_utc"]
                episode_active = False
                consecutive_shift = 0
                consecutive_recovery = 0
        rows.append({**b, "reference_mean": ref_mean, "deviation": deviation,
                     "shift_cutoff": shift_cut, "shifted": shifted,
                     "detector_alarm": detector_alarm, "episode_active": episode_active,
                     "episode_id": episode_id if episode_active else None})
    meta = {"calibration_reference_incidents": len(ref), "calibration_tuning_incidents": len(tuning),
            "reference_mean": ref_mean, "reference_std": ref_std,
            "calibration_reference_buckets": len(ref_batches),
            "calibration_tuning_buckets": len(tuning_batches),
            "empirical_threshold": empirical_cut, "practical_threshold": practical_cut,
            "applied_shift_threshold": shift_cut,
            "test_buckets_processed": len(rows), "test_incidents_monitored": len(rows) * args.bucket_size,
            "test_incomplete_tail_ignored": len(test) - len(rows) * args.bucket_size,
            "detector_flags": sum(r["detector_alarm"] for r in rows),
            "confirmed_shift_episodes": len(events),
            "active_episode_at_end": episode_active}
    return rows, events, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-dir", default="results/guide_temporal")
    p.add_argument("--outdir", default="results/guide_drift_monitor")
    p.add_argument("--bucket-size", type=int, default=1000)
    p.add_argument("--min-tuning-buckets", type=int, default=10)
    p.add_argument("--reference-quantile", type=float, default=0.99)
    p.add_argument("--min-effect-sd", type=float, default=0.25,
                   help="Minimum shift in units of reference incident-score SD")
    p.add_argument("--confirm-buckets", type=int, default=3)
    p.add_argument("--recovery-buckets", type=int, default=3)
    p.add_argument("--detector", choices=["ph", "adwin", "none"], default="ph")
    p.add_argument("--ph-delta", type=float, default=0.002,
                   help="PH insensitivity in raw bucket-mean score units")
    p.add_argument("--ph-threshold", type=float, default=0.08,
                   help="PH cumulative-deviation threshold, raw score units")
    p.add_argument("--adwin-delta", type=float, default=0.002)
    args = p.parse_args()
    if (args.bucket_size < 10 or args.min_tuning_buckets < 3 or
            not (0.5 < args.reference_quantile < 1) or args.min_effect_sd <= 0 or
            args.confirm_buckets < 1 or args.recovery_buckets < 1 or
            args.ph_delta < 0 or args.ph_threshold <= 0 or not (0 < args.adwin_delta < 1)):
        p.error("Invalid monitor parameters")
    base = Path(args.baseline_dir)
    out = Path(args.outdir)
    if out.resolve() == base.resolve():
        p.error("Output directory must be different from baseline directory")
    cal = read_scores(base / "calibration_predictions.csv")
    test = read_scores(base / "test_predictions.csv")
    validate_baseline(base, cal, test)
    rows, events, stats = monitor(cal, test, args)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "monitor_buckets.csv", index=False)
    pd.DataFrame(events, columns=["episode_id", "start_bucket_index", "confirmed_bucket_index",
        "confirmed_utc", "direction", "mean_score", "reference_mean", "deviation",
        "shift_cutoff", "detector_alarm_on_confirmation", "status", "recovered_bucket_index",
        "recovered_utc"]).to_csv(out / "shift_episodes.csv", index=False)
    summary = {"protocol": "read_only_completed_incident_GUIDE_shift_monitor_v1",
               "baseline_dir": str(base.resolve()), "settings": vars(args), **stats,
               "uses_test_labels": False, "trains_or_promotes_model": False,
               "limitations": [
                   "Unsupervised score-distribution episodes are not verified concept drift.",
                   "Calibration tuning quantiles do not guarantee a controlled false-alarm rate.",
                   "Scores depend on organisation mixture and true-positive prevalence.",
                   "Privacy-adjusted GUIDE timestamps limit temporal inference.",
                   "Completed-incident monitoring; not live alert ingestion.",
                   "PH/ADWIN flags are secondary diagnostics; episode policy is reference-based.",
                   "Final incomplete test bucket is ignored and its count is disclosed.",
                   "No retraining, shadow models, promotion, or rollback is implemented here."]}
    (out / "monitor_summary.json").write_text(json.dumps(summary, indent=2))
    print("GUIDE read-only shift monitor (no model training or modification)")
    print(f"  Calibration: {len(cal):,}; test: {len(test):,} completed incidents")
    print(f"  Buckets: {len(rows)} (size {args.bucket_size}); ignored tail: {stats['test_incomplete_tail_ignored']}")
    print(f"  Frozen score-reference mean: {stats['reference_mean']:.5f}; shift cutoff: {stats['applied_shift_threshold']:.5f}")
    print(f"  Detector flags ({args.detector}): {stats['detector_flags']}; confirmed shift episodes: {len(events)}")
    print(f"  Results: {out}")


if __name__ == "__main__":
    main()
