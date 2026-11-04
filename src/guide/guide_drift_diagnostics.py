"""Read-only GUIDE temporal diagnostics and label-free drift monitoring.

This script consumes the EXISTING guide_temporal.py outputs. It never fits or
updates a prediction model, and never changes the saved baseline files.

Inputs:
  * calibration_predictions.csv + test_predictions.csv + manifest.json
  * Optional GUIDE incident cache (.pkl/.parquet) for ATT&CK distribution shifts

Monitoring (prospective-in-replay, using NO test labels):
  * Page-Hinkley-style cumulative-change alarms on completed-incident scores.
    Reference is the earlier calibration scores; test examples are consumed in
    completion-time order, with a configurable alarm cooldown.
  * Reference-vs-current ATT&CK technique/category JS divergence per time bucket
    (only with --cache). This is descriptive, not an automatic retraining trigger.

Retrospective diagnostics (test labels used ONLY for reporting):
  * Daily/weekly incident volume, TP prevalence, ROC-AUC, average precision,
    recall at the saved baseline thresholds, and incident/alert demotion.
  * Eligible-org within-org AUC and organisation-level model performance.

IMPORTANT: GUIDE timestamps are privacy-adjusted; data are fully observed
COMPLETED incidents, not a live/causal online alert system. Drift alarms are
unsupervised distribution alerts, not verified concept drift or model failure.

Example:
  python src/guide/guide_drift_diagnostics.py \\
    --baseline-dir results/guide_temporal \\
    --cache data/processed/guide_incidents.pkl \\
    --outdir results/guide_drift_diagnostics

For score-only monitoring omit --cache. No external 'river' dependency needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import average_precision_score, roc_auc_score


def load_predictions(path, label):
    x = pd.read_csv(path, dtype={"sid": str, "OrgId": str})
    required = {"sid", "OrgId", "t_first", "t_last", "is_true_positive", "n_alerts", "score"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if x.sid.isna().any() or x.sid.duplicated().any():
        raise ValueError(f"{path}: missing or duplicate incident IDs")
    for col in ("t_first", "t_last"):
        x[col] = pd.to_datetime(x[col], utc=True, errors="coerce")
    for col in ("score", "n_alerts", "is_true_positive"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if x[["t_first", "t_last", "score", "n_alerts", "is_true_positive", "OrgId"]].isna().any().any():
        raise ValueError(f"{path}: missing/invalid values in required columns")
    if not x.is_true_positive.isin([0, 1]).all() or not x.score.between(0, 1).all():
        raise ValueError(f"{path}: labels must be 0/1 and scores must be in [0,1]")
    if (x.n_alerts <= 0).any() or (x.t_last < x.t_first).any():
        raise ValueError(f"{path}: invalid alert count or incident timestamps")
    x["is_true_positive"] = x.is_true_positive.astype(int)
    x["n_alerts"] = x.n_alerts.astype(int)
    x["split"] = label
    return x.sort_values(["t_last", "sid"], kind="stable").reset_index(drop=True)


def read_thresholds(metrics):
    out = {}
    for k, v in metrics.items():
        if k.startswith("target_") and isinstance(v, dict):
            t = v.get("threshold")
            if t is not None and not 0 <= float(t) <= 1:
                raise ValueError(f"Invalid threshold for {k}")
            out[k] = float("-inf") if t is None else float(t)
    if not out:
        raise ValueError("No target_* threshold entries in baseline metrics.json")
    return out


def auc_safe(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def ap_safe(y, s):
    return float(average_precision_score(y, s)) if np.sum(y) > 0 else None


def measurements(part, thresholds, min_auc_n=30):
    y = part.is_true_positive.to_numpy(dtype=int)
    s = part.score.to_numpy(dtype=float)
    counts = part.n_alerts.to_numpy(dtype=int)
    result = {
        "incidents": int(len(part)),
        "alerts": int(counts.sum()),
        "tp_incidents": int(y.sum()),
        "tp_rate": float(y.mean()) if len(y) else None,
        "mean_score": float(s.mean()) if len(s) else None,
        "roc_auc": auc_safe(y, s) if len(y) >= min_auc_n else None,
        "average_precision": ap_safe(y, s) if len(y) >= min_auc_n else None,
    }
    for name, threshold in thresholds.items():
        d = s <= threshold
        result[f"{name}_recall"] = float(y[~d].sum() / y.sum()) if y.sum() else None
        result[f"{name}_incidents_demoted"] = int(d.sum())
        result[f"{name}_incident_demotion"] = float(d.mean()) if len(d) else None
        result[f"{name}_alerts_demoted"] = int(counts[d].sum())
        result[f"{name}_alert_demotion"] = float(counts[d].sum() / counts.sum()) if counts.sum() else None
        result[f"{name}_tp_demoted"] = int(y[d].sum())
    return result


def within_org_results(test, thresholds, min_n=30, min_class=5):
    rows = []
    for org, part in test.groupby("OrgId", sort=False):
        row = {"OrgId": org, **measurements(part, thresholds, min_auc_n=min_n)}
        p = int(part.is_true_positive.sum())
        qualifies = len(part) >= min_n and p >= min_class and (len(part) - p) >= min_class
        row["auc_eligible"] = bool(qualifies)
        if not qualifies:
            row["roc_auc"] = None
            row["average_precision"] = None
        rows.append(row)
    result = pd.DataFrame(rows)
    eligible = result[result.auc_eligible]
    return result, (float(eligible.roc_auc.mean()) if len(eligible) else None), int(len(eligible))


def score_monitor(calibration, test, *, min_ref=100, delta=0.01, threshold=30.0,
                  warmup=100, cooldown=500):
    """A transparent Page-Hinkley-style *score distribution* detector.

    Calibration mean and SD define a frozen historical reference. Test values
    are z-scored against that reference, clipped to +/-5, then a two-sided
    cumulative-deviation detector is run in completion-time order. After an
    alarm the cumulative statistics are reset. This is an intentionally simple
    PH-style monitor, NOT a statistical guarantee or verified concept-drift
    test. Alarm parameters must be selected independently of test labels.
    """
    ref = calibration.score.to_numpy(dtype=float)
    if len(ref) < min_ref:
        raise ValueError(f"Need at least {min_ref} calibration scores")
    ref_mean = float(ref.mean())
    ref_std = max(float(ref.std(ddof=0)), 1e-6)
    positives = negatives = 0.0
    n_since_alarm = 0
    last_alarm_idx = -cooldown - 1
    events = []
    for i, row in enumerate(test.itertuples(index=False)):
        z = float(np.clip((float(row.score) - ref_mean) / ref_std, -5, 5))
        positives = max(0.0, positives + z - delta)
        negatives = max(0.0, negatives - z - delta)
        n_since_alarm += 1
        if n_since_alarm >= warmup and (i - last_alarm_idx) > cooldown:
            if max(positives, negatives) >= threshold:
                direction = "higher_scores" if positives >= negatives else "lower_scores"
                events.append({
                    "index_in_test": i,
                    "timestamp_utc": row.t_last.isoformat(),
                    "signal": "score_distribution",
                    "method": "reference_zscore_page_hinkley_style",
                    "direction": direction,
                    "statistic": float(max(positives, negatives)),
                    "delta": delta,
                    "threshold": threshold,
                    "incidents_since_reset": n_since_alarm,
                })
                positives = negatives = 0.0
                n_since_alarm = 0
                last_alarm_idx = i
    return events, {"calibration_mean": ref_mean, "calibration_std": ref_std,
                    "delta": delta, "threshold": threshold,
                    "warmup": warmup, "cooldown": cooldown, "test_observations": len(test)}


def read_cache(path):
    p = Path(path)
    if p.suffix == ".parquet":
        raw = pd.read_parquet(p, columns=["techs", "cats", "t_first", "t_last"])
    elif p.suffix in (".pkl", ".pickle"):
        raw = pd.read_pickle(p)[["techs", "cats", "t_first", "t_last"]]
    else:
        raise ValueError("--cache must be a parquet or pickle file")
    raw.index = raw.index.astype(str)
    if raw.index.duplicated().any():
        raise ValueError("Cache contains duplicate incident IDs")
    return raw


def token_counts(series):
    # Counts incidents that contain each token, once per incident. No labels.
    counts = {}
    for val in series.fillna("").astype(str):
        for token in set(v.strip() for v in val.split(";") if v.strip()):
            counts[token] = counts.get(token, 0) + 1
    return counts


def js_distance(ref, current, smoothing=1e-6):
    keys = sorted(set(ref) | set(current))
    if not keys:
        return None
    a = np.array([ref.get(k, 0) for k in keys], dtype=float) + smoothing
    b = np.array([current.get(k, 0) for k in keys], dtype=float) + smoothing
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2))


def distribution_by_bucket(cache, cal, test, freq, min_bucket_n):
    # A fixed calibration-period reference for all subsequent test buckets.
    missing = set(cal.sid) | set(test.sid)
    missing -= set(cache.index)
    if missing:
        raise ValueError(f"Cache is missing {len(missing)} saved prediction incident IDs")
    cal_data = cache.loc[cal.sid.to_numpy()]
    baseline = {name: token_counts(cal_data[name]) for name in ("techs", "cats")}
    out = []
    for bucket, grp in test.groupby(pd.Grouper(key="t_last", freq=freq)):
        if len(grp) < min_bucket_n:
            continue
        current = cache.loc[grp.sid.to_numpy()]
        item = {"bucket_utc": bucket.isoformat(), "incidents": int(len(grp))}
        for name in ("techs", "cats"):
            item[f"{name}_js_distance"] = js_distance(baseline[name], token_counts(current[name]))
        out.append(item)
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", default="results/guide_temporal")
    parser.add_argument("--cache", default=None, help="Optional GUIDE cache for technique/category JS distances")
    parser.add_argument("--outdir", default="results/guide_drift_diagnostics")
    parser.add_argument("--bucket", choices=("D", "W"), default="D", help="Calendar-day or weekly test buckets")
    parser.add_argument("--min-bucket-n", type=int, default=50)
    parser.add_argument("--min-org-n", type=int, default=30)
    parser.add_argument("--min-org-class", type=int, default=5)
    parser.add_argument("--ph-delta", type=float, default=0.01)
    parser.add_argument("--ph-threshold", type=float, default=30.0)
    parser.add_argument("--ph-warmup", type=int, default=100)
    parser.add_argument("--ph-cooldown", type=int, default=500)
    args = parser.parse_args()
    if (args.min_bucket_n < 1 or args.min_org_n < 2 or args.min_org_class < 1
            or args.ph_delta < 0 or args.ph_threshold <= 0
            or args.ph_warmup < 1 or args.ph_cooldown < 0):
        parser.error("Invalid minimum sizes or Page-Hinkley parameters")

    base = Path(args.baseline_dir)
    manifest = json.loads((base / "manifest.json").read_text())
    metrics = json.loads((base / "metrics.json").read_text())
    cal = load_predictions(base / "calibration_predictions.csv", "calibration")
    test = load_predictions(base / "test_predictions.csv", "test")
    if set(cal.sid) & set(test.sid):
        raise ValueError("Calibration/test prediction sets overlap")
    cut = pd.Timestamp(manifest["calibration_end_exclusive"])
    if cal.t_last.max() >= cut or test.t_first.min() < cut:
        raise ValueError("Prediction timestamps contradict baseline calibration/test cutoff")
    if (len(cal) != manifest["splits"]["calibration"]["incidents"] or
            len(test) != manifest["splits"]["test"]["incidents"]):
        raise ValueError("Prediction counts do not match baseline manifest")
    thresholds = read_thresholds(metrics)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    bucket_rows = []
    for timestamp, part in test.groupby(pd.Grouper(key="t_last", freq=args.bucket)):
        if len(part) < args.min_bucket_n:
            continue
        bucket_rows.append({"bucket_utc": timestamp.isoformat(), **measurements(part, thresholds)})
    buckets = pd.DataFrame(bucket_rows)
    buckets.to_csv(out / "test_time_buckets.csv", index=False)

    orgs, within_auc, n_orgs = within_org_results(test, thresholds, args.min_org_n, args.min_org_class)
    orgs.to_csv(out / "test_organisations.csv", index=False)

    events, monitor_meta = score_monitor(cal, test, delta=args.ph_delta,
        threshold=args.ph_threshold, warmup=args.ph_warmup, cooldown=args.ph_cooldown)
    pd.DataFrame(events, columns=["index_in_test", "timestamp_utc", "signal", "method",
        "direction", "statistic", "delta", "threshold", "incidents_since_reset"]).to_csv(
        out / "score_drift_events.csv", index=False)

    distributions = None
    if args.cache:
        cache = read_cache(args.cache)
        distributions = distribution_by_bucket(cache, cal, test, args.bucket, args.min_bucket_n)
        distributions.to_csv(out / "mitre_distribution_shift.csv", index=False)

    summary = {
        "protocol": "read_only_completed_incident_GUIDE_drift_diagnostics",
        "baseline_dir": str(base.resolve()),
        "cache_supplied": bool(args.cache),
        "calibration_incidents": len(cal),
        "test_incidents": len(test),
        "test_buckets": len(buckets),
        "test_eligible_orgs": n_orgs,
        "mean_eligible_org_auc": within_auc,
        "baseline_test_auc": metrics.get("test_roc_auc"),
        "baseline_test_ap": metrics.get("test_average_precision"),
        "score_monitor": monitor_meta,
        "score_distribution_alarms": len(events),
        "mitre_distribution_buckets": len(distributions) if distributions is not None else None,
        "limitations": [
            "No model training, retraining, promotion, or evaluation-split changes performed.",
            "Uses completed-incident scores; not a live/causal SOC monitor.",
            "GUIDE timestamps are privacy-adjusted; temporal inferences are limited.",
            "Test labels are used solely for retrospective reporting, never for drift alarms.",
            "Score-distribution alarms are not evidence of verified concept drift or performance failure.",
            "Alarm thresholds are illustrative and not tuned or calibrated for false trigger rate.",
            "Bucket AUC/AP missing when sample size or class variation is inadequate.",
            "Score-reference period is calibration, and may differ in prevalence or organisation composition.",
            "MITRE JS divergence compares each test bucket with the calibration distribution; not a statistical test.",
            "No per-org personalised scoring or feedback loop is implemented.",
        ],
    }
    (out / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2))
    print("GUIDE completed-incident diagnostics (no training or model update)")
    print(f"  Read: {len(cal):,} calibration; {len(test):,} test incidents")
    print(f"  Test buckets ({args.bucket}): {len(buckets)}; eligible orgs: {n_orgs}")
    print(f"  Reported baseline AUC/AP: {summary['baseline_test_auc']:.4f} / {summary['baseline_test_ap']:.4f}")
    print(f"  Score-distribution alarms (UNSUPERVISED, unvalidated): {len(events)}")
    print(f"  MITRE distribution check: {'enabled' if args.cache else 'not requested (no cache)'}")
    print(f"  Outputs: {out}")


if __name__ == "__main__":
    main()
