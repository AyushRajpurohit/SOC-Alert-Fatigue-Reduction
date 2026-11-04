"""GUIDE retrospective scheduled shadow update (simulation only).

Reads GUIDE's completed-incident cache and the FROZEN output of guide_temporal.py.
Does not edit either one. No confirmed natural drift trigger was observed, so this
simulates ONE scheduled update at the beginning of the existing test period.

Chronology:
  original train | early calibration (shadow fitting) | middle calibration
  (shadow threshold selection) | late calibration (one-way promotion gate) |
  untouched original test (evaluation only).
All sub-periods use incident-completion cutoffs; boundary-spanning incidents
are discarded. Features of a completed incident are not available earlier.

The frozen baseline was originally threshold-calibrated on ALL calibration data,
including the gate slice. Thus the static comparator's gate metrics are not an
independent evaluation. This experiment is exploratory and cannot establish
unbiased adaptive superiority. Shadow gate labels are unseen during shadow
training/threshold tuning; test labels are never used to promote or roll back.

Artifacts: shadow_model.joblib, simulated_active_model.joblib,
previous_model.joblib, rollback_restored_model.joblib (rollback rehearsal),
gate_results.json, test_results.json, lifecycle.json and test_comparison.csv.
NO model is deployed to the SIEM; no old file is modified.

Example:
 python src/guide/guide_shadow_update.py \\
   --cache data/processed/guide_incidents.pkl \\
   --baseline-dir results/guide_temporal \\
   --outdir results/guide_shadow_update
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# Reuse existing preprocessing/metrics, rather than changing established scripts.
from guide_temporal import (load_cache, prepare, make_vectorizer, features, labels,
                            threshold_for_recall, operating_metrics, within_org_auc)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_frozen(path):
    d = pd.read_csv(path, dtype={"sid": str, "OrgId": str})
    required = {"sid", "OrgId", "t_first", "t_last", "is_true_positive", "n_alerts", "score"}
    if required - set(d.columns):
        raise ValueError(f"Missing columns from {path}: {sorted(required - set(d.columns))}")
    if d.sid.isna().any() or d.sid.duplicated().any():
        raise ValueError(f"Duplicate/missing frozen incident IDs: {path}")
    d["t_first"] = pd.to_datetime(d.t_first, utc=True)
    d["t_last"] = pd.to_datetime(d.t_last, utc=True)
    return d.set_index("sid", drop=False)


def split_calibration(cal, fit_frac, threshold_frac):
    if not (0 < fit_frac < 1 and 0 < threshold_frac < 1 and fit_frac + threshold_frac < 1):
        raise ValueError("Require positive calibration fractions with sum < 1")
    ordered = cal.t_last.sort_values(kind="stable")
    t1 = ordered.iloc[int(len(ordered) * fit_frac)]
    t2 = ordered.iloc[int(len(ordered) * (fit_frac + threshold_frac))]
    if t1 >= t2:
        raise ValueError("Calibration cutoffs coincide")
    fit = cal[cal.t_last < t1].copy()
    tune = cal[(cal.t_first >= t1) & (cal.t_last < t2)].copy()
    gate = cal[cal.t_first >= t2].copy()
    if min(map(len, (fit, tune, gate))) < 20:
        raise ValueError("Insufficient observations in a calibration sub-period")
    for name, frame in (("fit", fit), ("tune", tune), ("gate", gate)):
        if labels(frame).min() == labels(frame).max():
            raise ValueError(f"{name} sub-period has only one class")
    if not (fit.t_last.max() < t1 <= tune.t_first.min() and
            tune.t_last.max() < t2 <= gate.t_first.min()):
        raise ValueError("Temporal separation failed")
    return fit, tune, gate, t1, t2, len(cal) - len(fit) - len(tune) - len(gate)


def get_frozen_scores(frame, frozen):
    if not frame.index.isin(frozen.index).all():
        raise ValueError("Frozen predictions missing cache incidents; baseline may not match cache")
    f = frozen.loc[frame.index]
    if not np.array_equal(labels(frame), f.is_true_positive.to_numpy(dtype=int)):
        raise ValueError("Frozen labels disagree with GUIDE cache")
    if not np.array_equal(frame.n_alerts.to_numpy(dtype=int), f.n_alerts.to_numpy(dtype=int)):
        raise ValueError("Frozen alert counts disagree with GUIDE cache")
    if not np.array_equal(frame.OrgId.astype(str).to_numpy(), f.OrgId.astype(str).to_numpy()):
        raise ValueError("Frozen organisation IDs disagree with cache")
    if not np.array_equal(frame.t_last.to_numpy(), f.t_last.to_numpy()):
        raise ValueError("Frozen incident times disagree with cache")
    return f.score.to_numpy(dtype=float)


def model_summary(scores, frame, thresholds):
    y = labels(frame)
    orgauc, norg = within_org_auc(scores, y, frame.OrgId.to_numpy())
    result = {"n": len(frame), "tp_rate": float(y.mean()),
              "auc": float(roc_auc_score(y, scores)),
              "average_precision": float(average_precision_score(y, scores)),
              "within_org_auc": orgauc, "within_org_count": norg,
              "operating_points": {}}
    for target, threshold in thresholds.items():
        result["operating_points"][target] = operating_metrics(
            scores, y, frame.n_alerts.to_numpy(), threshold)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--baseline-dir", default="results/guide_temporal")
    p.add_argument("--outdir", default="results/guide_shadow_update")
    p.add_argument("--fit-frac", type=float, default=0.40,
                   help="Early calibration share added to the original training set")
    p.add_argument("--threshold-frac", type=float, default=0.30,
                   help="Middle calibration share for shadow score thresholds")
    p.add_argument("--min-df", type=int, default=20)
    p.add_argument("--C", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-recall-drop", type=float, default=0.005,
                   help="Largest gate recall drop versus frozen baseline")
    p.add_argument("--min-demotion-gain", type=float, default=0.0,
                   help="Minimum gate incident-demotion improvement at every target")
    p.add_argument("--max-auc-drop", type=float, default=0.01)
    args = p.parse_args()
    if (args.min_df < 1 or args.C <= 0 or not (0 <= args.max_recall_drop < 1)
            or not (0 <= args.min_demotion_gain < 1) or not (0 <= args.max_auc_drop < 1)):
        p.error("Invalid model or promotion-gate parameters")
    base = Path(args.baseline_dir)
    out = Path(args.outdir)
    if out.resolve() == base.resolve() or out.resolve() == Path(args.cache).resolve():
        p.error("Output must not overwrite the baseline or source cache")
    frozen_path = base / "static_model.joblib"
    original_hash = digest(frozen_path)
    existing_bundle = joblib.load(frozen_path)
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest.get("protocol") != "completed_incident_global_chronological_60_20_20_by_default":
        raise ValueError("Unexpected baseline split protocol")
    if len(manifest["hyperparameters"]["recall_targets"]) == 0:
        raise ValueError("Baseline must contain operating points")
    targets = [float(t) for t in manifest["hyperparameters"]["recall_targets"]]
    frozen_thresholds = {f"{t:.4f}": (float("-inf") if existing_bundle["thresholds"][f"{t:.4f}"] is None
                                      else float(existing_bundle["thresholds"][f"{t:.4f}"])) for t in targets}
    data = prepare(load_cache(args.cache))
    cutoff1 = pd.Timestamp(manifest["train_end_exclusive"])
    cutoff2 = pd.Timestamp(manifest["calibration_end_exclusive"])
    tr = data[data.t_last < cutoff1].copy()
    cal = data[(data.t_first >= cutoff1) & (data.t_last < cutoff2)].copy()
    test = data[data.t_first >= cutoff2].copy()
    for name, frame in (("train", tr), ("calibration", cal), ("test", test)):
        if len(frame) != manifest["splits"][name]["incidents"]:
            raise ValueError(f"{name} split does not match frozen baseline manifest")
    fcal = read_frozen(base / "calibration_predictions.csv")
    ftest = read_frozen(base / "test_predictions.csv")
    # Frozen test scores are used ONLY after the model selection gate is fixed.
    if set(fcal.index) != set(cal.index) or set(ftest.index) != set(test.index):
        raise ValueError("Frozen prediction IDs do not match cache partitions")
    fit, tune, gate, t1, t2, crossing = split_calibration(cal, args.fit_frac, args.threshold_frac)
    shadow_training = pd.concat((tr, fit), axis=0)
    if shadow_training.index.duplicated().any():
        raise ValueError("Shadow-training incident IDs are duplicated")
    tv = make_vectorizer(shadow_training, "techs", args.min_df)
    cv = make_vectorizer(shadow_training, "cats", args.min_df)
    shadow = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed)
    shadow.fit(features(shadow_training, tv, cv), labels(shadow_training))
    tune_scores = shadow.predict_proba(features(tune, tv, cv))[:, 1]
    shadow_thresholds = {f"{t:.4f}": threshold_for_recall(tune_scores, labels(tune), t)
                         for t in targets}
    # Only now score the independent gate slice. No threshold tuning on gate labels.
    gate_shadow_scores = shadow.predict_proba(features(gate, tv, cv))[:, 1]
    gate_static_scores = get_frozen_scores(gate, fcal)
    static_gate = model_summary(gate_static_scores, gate, frozen_thresholds)
    shadow_gate = model_summary(gate_shadow_scores, gate, shadow_thresholds)
    reasons = []
    if shadow_gate["auc"] < static_gate["auc"] - args.max_auc_drop:
        reasons.append("AUC noninferiority gate failed")
    for t in targets:
        key = f"{t:.4f}"
        s = shadow_gate["operating_points"][key]
        b = static_gate["operating_points"][key]
        if s["recall"] < t:
            reasons.append(f"{key}: minimum recall target failed")
        if s["recall"] < b["recall"] - args.max_recall_drop:
            reasons.append(f"{key}: recall regression relative to static baseline")
        if s["incident_demotion_rate"] < b["incident_demotion_rate"] + args.min_demotion_gain:
            reasons.append(f"{key}: incident-demotion improvement gate failed")
    accepted = not reasons
    decision = "ELIGIBLE_FOR_SIMULATED_PROMOTION" if accepted else "REJECTED_KEEP_STATIC"

    out.mkdir(parents=True, exist_ok=True)
    # Never overwrite existing experiment artifacts (including previous runs here).
    guarded = ["shadow_model.joblib", "simulated_active_model.joblib", "previous_model.joblib",
               "rollback_restored_model.joblib", "gate_results.json", "test_results.json",
               "lifecycle.json", "test_comparison.csv"]
    if any((out / x).exists() for x in guarded):
        raise FileExistsError("Output files already exist: choose a new --outdir to preserve prior runs")
    candidate_bundle = {"model": shadow, "tech_vectorizer": tv, "cat_vectorizer": cv,
                        "thresholds": {k: None if np.isneginf(v) else v for k, v in shadow_thresholds.items()},
                        "type": "shadow_completed_incident_GUIDE", "gate_decision": decision}
    joblib.dump(candidate_bundle, out / "shadow_model.joblib")
    shutil.copy2(frozen_path, out / "previous_model.joblib")
    if accepted:
        shutil.copy2(out / "shadow_model.joblib", out / "simulated_active_model.joblib")
    else:
        shutil.copy2(frozen_path, out / "simulated_active_model.joblib")
    # Rehearse rollback locally: restore the untouched previous model to a new file.
    shutil.copy2(out / "previous_model.joblib", out / "rollback_restored_model.joblib")
    rollback_verified = digest(out / "rollback_restored_model.joblib") == original_hash
    if not rollback_verified or digest(frozen_path) != original_hash:
        raise RuntimeError("Model rollback/original integrity verification failed")

    # One held-out test pass AFTER gate decision, no test-guided selection.
    test_static_scores = get_frozen_scores(test, ftest)
    test_shadow_scores = shadow.predict_proba(features(test, tv, cv))[:, 1]
    static_test = model_summary(test_static_scores, test, frozen_thresholds)
    shadow_test = model_summary(test_shadow_scores, test, shadow_thresholds)
    test_metrics = {"static": static_test, "shadow": shadow_test,
                    "simulated_active": "shadow" if accepted else "static",
                    "decision_was_locked_before_test_scoring": True}
    pd.DataFrame({"sid": test.index, "t_last": test.t_last.to_numpy(),
                  "OrgId": test.OrgId.to_numpy(), "is_true_positive": labels(test),
                  "n_alerts": test.n_alerts.to_numpy(), "static_score": test_static_scores,
                  "shadow_score": test_shadow_scores,
                  **{f"static_demote_{k}": test_static_scores <= v for k, v in frozen_thresholds.items()},
                  **{f"shadow_demote_{k}": test_shadow_scores <= v for k, v in shadow_thresholds.items()}
                  }).to_csv(out / "test_comparison.csv", index=False)
    summary = {
        "protocol": "one_scheduled_pretest_shadow_update_completed_incidents",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_dir": str(base.resolve()), "cache": str(Path(args.cache).resolve()),
        "frozen_model_sha256": original_hash,
        "calibration_fit_end_exclusive": t1.isoformat(),
        "calibration_tune_end_exclusive": t2.isoformat(),
        "original_test_start": cutoff2.isoformat(),
        "sizes": {"original_train": len(tr), "shadow_additional_fit": len(fit),
                  "shadow_total_fit": len(shadow_training), "threshold_tuning": len(tune),
                  "validation_gate": len(gate), "gate_boundary_crossing_excluded": crossing,
                  "held_out_test": len(test)},
        "shadow_thresholds": candidate_bundle["thresholds"],
        "static_thresholds": {k: None if np.isneginf(v) else v for k, v in frozen_thresholds.items()},
        "gate_policy": {"recall_targets": targets, "max_recall_drop": args.max_recall_drop,
                        "min_demotion_gain": args.min_demotion_gain,
                        "max_auc_drop": args.max_auc_drop,
                        "require_improvement_at_all_targets": True},
        "gate_decision": decision, "gate_reasons": reasons,
        "rollback_rehearsal_integrity_verified": rollback_verified,
        "no_real_deployment": True,
        "caveats": [
            "Scheduled update ONLY; no naturally confirmed drift event triggered training.",
            "Baseline thresholds were historically tuned on all calibration incidents, including gate; static gate comparison is not independent.",
            "Shadow training/tuning and its gate are chronologically separate; no test labels used for selection.",
            "Completed-incident features and privacy-adjusted GUIDE timestamps; not online causal detection.",
            "Shadow-training reuses original training data plus early calibration; no unbiased claim of adaptive gain.",
            "No live model swap: simulated active copy and rollback rehearsal only.",
            "Test labels used only once for retrospective comparison; test is not a second promotion gate.",
        ]
    }
    (out / "gate_results.json").write_text(json.dumps({"decision": decision, "reasons": reasons,
        "static": static_gate, "shadow": shadow_gate}, indent=2))
    (out / "test_results.json").write_text(json.dumps(test_metrics, indent=2))
    (out / "lifecycle.json").write_text(json.dumps(summary, indent=2))
    print("GUIDE scheduled shadow-model update simulation (NOT drift-triggered / NOT deployed)")
    print(f"  Training: original {len(tr):,} + recent {len(fit):,}; "
          f"tune {len(tune):,}; gate {len(gate):,}; test {len(test):,}")
    print(f"  Gate decision: {decision}" + (f"; reasons: {'; '.join(reasons)}" if reasons else ""))
    print(f"  Gate AUC static/shadow: {static_gate['auc']:.4f} / {shadow_gate['auc']:.4f}")
    print(f"  Held-out test AUC static/shadow: {static_test['auc']:.4f} / {shadow_test['auc']:.4f}")
    for t in targets:
        k = f"{t:.4f}"
        b, s = static_test["operating_points"][k], shadow_test["operating_points"][k]
        print(f"  Target {t:.0%} static/shadow test: recall {b['recall']:.3%}/{s['recall']:.3%}; "
              f"incident demotion {b['incident_demotion_rate']:.3%}/{s['incident_demotion_rate']:.3%}")
    print(f"  Rollback rehearsal verified: {rollback_verified}; files: {out}")

if __name__ == "__main__":
    main()
