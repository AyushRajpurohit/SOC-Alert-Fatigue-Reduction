"""GUIDE chronological baseline (standalone; does not change reduction_curve.py).

Input: incident cache made by build_guide_cache.py (parquet or pickle).
Task: rank *completed* GUIDE incidents as TruePositive vs BenignPositive/FalsePositive.
This is NOT causal online alert scoring: full incident aggregates are used only after
an incident's last observed alert. GUIDE timestamps have privacy noise.

Protocol:
  * Pick global chronological boundaries from incident completion times.
  * Train / calibrate decision thresholds / test on separate time periods.
  * Exclude incidents overlapping a boundary (no cross-period aggregation).
  * Learn feature vocabulary on TRAIN ONLY (unknown future tokens ignored).
  * Choose thresholds using calibration labels, then evaluate once on test.
  * Store predictions, metrics, split manifest and a model bundle for later drift work.

Example:
  python src/guide/guide_temporal.py \\
    --cache data/processed/guide_incidents.pkl \\
    --outdir results/guide_temporal

Do not compare these temporal numbers directly with organisation-held-out
reduction_curve.py: the population, cap and split protocols are different.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_TARGETS = (0.99, 0.95)
REQUIRED = {"OrgId", "grade", "t_first", "t_last", "n_alerts", "n_techs", "techs", "cats"}


def load_cache(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError("Cache must be a .parquet or .pkl file")


def prepare(raw):
    missing = REQUIRED.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing cache columns: {sorted(missing)}")
    d = raw.copy()
    d.index = d.index.astype(str)
    if not d.index.is_unique:
        raise ValueError("Incident IDs (index) must be unique")
    # Only explicitly graded examples are used; unknown/ungraded is not a negative.
    d = d[d["grade"].isin(["TruePositive", "BenignPositive", "FalsePositive"])].copy()
    d["t_first"] = pd.to_datetime(d["t_first"], utc=True, errors="coerce")
    d["t_last"] = pd.to_datetime(d["t_last"], utc=True, errors="coerce")
    d["n_alerts"] = pd.to_numeric(d["n_alerts"], errors="coerce")
    d["n_techs"] = pd.to_numeric(d["n_techs"], errors="coerce")
    d = d.dropna(subset=["t_first", "t_last", "n_alerts", "n_techs", "OrgId"])
    d = d[(d.t_last >= d.t_first) & (d.n_alerts > 0) & (d.n_techs >= 0)].copy()
    d["techs"] = d["techs"].fillna("").astype(str)
    d["cats"] = d["cats"].fillna("").astype(str)
    return d


def chronological_split(d, train_frac, cal_frac):
    if not (0 < train_frac < 1 and 0 < cal_frac < 1 and train_frac + cal_frac < 1):
        raise ValueError("Fractions must be positive, and train + calibration < 1")
    times = d.t_last.sort_values(kind="stable")
    if len(times) < 12:
        raise ValueError("Not enough graded incidents for temporal splitting")
    cut1 = times.iloc[int(len(times) * train_frac)]
    cut2 = times.iloc[int(len(times) * (train_frac + cal_frac))]
    if cut1 >= cut2:
        raise ValueError("Temporal cutoffs coincide; cannot form chronological periods")
    # Half-open windows. Instances straddling a cutoff are excluded completely.
    train = d[d.t_last < cut1].copy()
    cal = d[(d.t_first >= cut1) & (d.t_last < cut2)].copy()
    test = d[d.t_first >= cut2].copy()
    used = len(train) + len(cal) + len(test)
    dropped = len(d) - used
    assert train.index.intersection(cal.index).empty
    assert train.index.intersection(test.index).empty
    assert cal.index.intersection(test.index).empty
    assert train.t_last.max() < cut1 and cal.t_first.min() >= cut1
    assert cal.t_last.max() < cut2 and test.t_first.min() >= cut2
    for name, split in (("train", train), ("calibration", cal), ("test", test)):
        if split.empty or split.grade.eq("TruePositive").nunique() < 2:
            raise ValueError(f"{name} split is empty or has only one class")
    return {"train": train, "calibration": cal, "test": test}, cut1, cut2, dropped


def make_vectorizer(train, field, min_df):
    vec = CountVectorizer(binary=True, min_df=min_df, lowercase=False, token_pattern=r"[^;]+")
    try:
        vec.fit(train[field])
    except ValueError as exc:
        raise ValueError(f"No {field} vocabulary meets min_df={min_df}; lower --min-df") from exc
    return vec


def features(df, tv, cv):
    tagged = (df.n_techs.to_numpy(dtype=float) > 0).astype(float)
    numeric = sparse.csr_matrix(np.column_stack([
        np.log1p(df.n_alerts.to_numpy(dtype=float)),
        tagged,
        np.log1p(df.n_techs.to_numpy(dtype=float)),
    ]))
    return sparse.hstack((tv.transform(df.techs), cv.transform(df.cats), numeric), format="csr")


def labels(df):
    return df.grade.eq("TruePositive").to_numpy(dtype=int)


def threshold_for_recall(scores, y, target):
    """Largest tie-safe 'score <= threshold' demotion with >= target calibration recall.

    Strictly return -inf (demote none) if no whole score-tie group can be removed.
    This is a *calibration-set* guarantee only, not a promise on future test data.
    """
    order = np.argsort(scores, kind="stable")
    s, yy = np.asarray(scores)[order], np.asarray(y)[order]
    if yy.sum() == 0:
        raise ValueError("Calibration set has no TruePositive incidents")
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    recall = 1.0 - np.cumsum(yy)[ends] / yy.sum()
    allowed = np.flatnonzero(recall >= target)
    return float(s[ends[allowed[-1]]]) if allowed.size else float("-inf")


def operating_metrics(scores, y, alert_counts, threshold):
    demote = scores <= threshold
    return {
        "recall": float(y[~demote].sum() / y.sum()),
        "incidents_demoted": int(demote.sum()),
        "incidents_total": int(len(demote)),
        "incident_demotion_rate": float(demote.mean()),
        "alerts_demoted": int(alert_counts[demote].sum()),
        "alerts_total": int(alert_counts.sum()),
        "alert_demotion_rate": float(alert_counts[demote].sum() / alert_counts.sum()),
        "tp_incidents_demoted": int(y[demote].sum()),
    }


def within_org_auc(scores, y, orgs, min_n=30, min_class=5):
    # Same qualifying-org criteria as reduction_curve.py, without importing it.
    fr = pd.DataFrame({"org": orgs, "score": scores, "y": y})
    aucs = []
    for _, sub in fr.groupby("org", sort=False):
        pos = int(sub.y.sum())
        if len(sub) >= min_n and pos >= min_class and (len(sub) - pos) >= min_class:
            aucs.append(roc_auc_score(sub.y, sub.score))
    return (float(np.mean(aucs)) if aucs else None), len(aucs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="GUIDE incident .parquet or .pkl from build_guide_cache.py")
    ap.add_argument("--outdir", default="results/guide_temporal")
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--cal-frac", type=float, default=0.20)
    ap.add_argument("--min-df", type=int, default=20, help="Min TRAIN incidents per technique/category token")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    args = ap.parse_args()
    if args.min_df < 1 or args.C <= 0 or any(not 0 < t <= 1 for t in args.targets):
        ap.error("--min-df >= 1, --C > 0 and targets in (0, 1] are required")

    raw = load_cache(args.cache)
    d = prepare(raw)
    splits, cut1, cut2, excluded = chronological_split(d, args.train_frac, args.cal_frac)
    tr, ca, te = splits["train"], splits["calibration"], splits["test"]
    tv = make_vectorizer(tr, "techs", args.min_df)
    cv = make_vectorizer(tr, "cats", args.min_df)
    Xt, Xc, Xe = (features(df, tv, cv) for df in (tr, ca, te))
    yt, yc, ye = (labels(df) for df in (tr, ca, te))
    model = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed)
    model.fit(Xt, yt)
    sc = model.predict_proba(Xc)[:, 1]
    se = model.predict_proba(Xe)[:, 1]
    wa, wn = within_org_auc(se, ye, te.OrgId.to_numpy())
    metrics = {
        "test_roc_auc": float(roc_auc_score(ye, se)),
        "test_average_precision": float(average_precision_score(ye, se)),
        "test_within_org_auc": wa,
        "test_within_org_count": wn,
        "test_tp_rate": float(ye.mean()),
    }
    thresholds = {}
    for target in args.targets:
        thr = threshold_for_recall(sc, yc, target)
        label = f"{target:.4f}"
        thresholds[label] = None if np.isneginf(thr) else thr
        metrics[f"target_{label}"] = {
            "threshold": thresholds[label],
            "calibration": operating_metrics(sc, yc, ca.n_alerts.to_numpy(), thr),
            "test": operating_metrics(se, ye, te.n_alerts.to_numpy(), thr),
        }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df, scores in (("calibration", ca, sc), ("test", te, se)):
        result = pd.DataFrame({
            "sid": df.index,
            "OrgId": df.OrgId.to_numpy(),
            "t_first": df.t_first.to_numpy(),
            "t_last": df.t_last.to_numpy(),
            "grade": df.grade.to_numpy(),
            "is_true_positive": labels(df),
            "n_alerts": df.n_alerts.to_numpy(),
            "score": scores,
        })
        for target, thr in thresholds.items():
            result[f"demote_target_{target}"] = scores <= (float("-inf") if thr is None else thr)
        result.to_csv(outdir / f"{name}_predictions.csv", index=False)

    manifest = {
        "protocol": "completed_incident_global_chronological_60_20_20_by_default",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_cache": str(Path(args.cache).resolve()),
        "source_incidents": len(raw),
        "eligible_graded_valid_incidents": len(d),
        "excluded_boundary_crossing_incidents": excluded,
        "train_end_exclusive": cut1.isoformat(),
        "calibration_end_exclusive": cut2.isoformat(),
        "splits": {
            name: {
                "incidents": len(df), "orgs": int(df.OrgId.nunique()),
                "tp_rate": float(labels(df).mean()),
                "earliest_start": df.t_first.min().isoformat(),
                "latest_end": df.t_last.max().isoformat(),
            } for name, df in splits.items()
        },
        "hyperparameters": {
            "train_frac": args.train_frac, "cal_frac": args.cal_frac,
            "min_df": args.min_df, "C": args.C, "seed": args.seed,
            "recall_targets": args.targets,
        },
        "feature_dimensions": {"techniques": len(tv.vocabulary_), "categories": len(cv.vocabulary_), "numeric": 3},
        "caveats": [
            "GUIDE incident-level analyst grades; not alert-level safety recall.",
            "Complete incident features are available only after its last observed alert.",
            "GUIDE timestamps are privacy-noised; chronological drift is a proxy.",
            "No organisation-disjoint holdout: this is temporal evaluation, not zero-shot portability.",
            "Score is logistic output, not independently probability-calibrated.",
            "Calibration threshold meeting a target does not guarantee test recall.",
            "No adaptive model or drift detector is used in this baseline.",
        ],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"model": model, "tech_vectorizer": tv, "cat_vectorizer": cv,
                 "thresholds": thresholds, "manifest": manifest}, outdir / "static_model.joblib")

    print("Chronological completed-incident baseline (GUIDE; not a live/causal evaluation)")
    print(f"  Cutoffs UTC: {cut1.isoformat()} | {cut2.isoformat()}")
    print(f"  Train/cal/test: {len(tr):,} / {len(ca):,} / {len(te):,}; excluded crossing: {excluded:,}")
    print(f"  Test AUC: {metrics['test_roc_auc']:.4f}; AP: {metrics['test_average_precision']:.4f}")
    print(f"  Within-org AUC: {wa} over {wn} eligible orgs")
    for target in args.targets:
        v = metrics[f"target_{target:.4f}"]["test"]
        print(f"  Target {target:.1%}: test recall {v['recall']:.3%}; "
              f"incident demotion {v['incident_demotion_rate']:.3%}; "
              f"alert demotion {v['alert_demotion_rate']:.3%}")
    print(f"  Outputs: {outdir}")


if __name__ == "__main__":
    main()
