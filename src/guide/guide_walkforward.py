"""Walk-forward GUIDE completed-incident prioritization, scheduled updates (SIMULATION).

This is a NEW script; it does not change the previous GUIDE baseline or experiments.
The original fixed model is trained on an initial chronological period and its
thresholds are tuned on the next period. Each later scheduled candidate uses only
past completed incidents for fitting, a later period for threshold tuning, and a
still later *separate* period for its acceptance gate. A subsequent test period
is evaluated after the gate decision has been locked.

Two comparators: a frozen static model and a gated scheduled-update policy.
Also report the candidate before gating to explain rejections, never to select it.
All reported recall is of TruePositive INCIDENTS, not attack alerts.

Important limitations: GUIDE timestamps are privacy-noised, features are complete
incident aggregates, organisation mixes can change, and accumulating training data
assumes past analyst labels become available for later updates. Consecutive fold
windows are nonoverlapping for TEST; earlier test labels may become past training
labels in subsequent folds. This is a retrospective walk-forward simulation, not
an unbiased single-use prospective trial. No drift-triggered training is done.

Example:
 python src/guide/guide_walkforward.py --cache data/processed/guide_incidents.pkl \\
      --outdir results/guide_walkforward

Requires guide_temporal.py in the same directory (imports preprocessing functions).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from guide_temporal import (load_cache, prepare, make_vectorizer, features,
                            labels, threshold_for_recall, operating_metrics,
                            within_org_auc)


def period_starting(d, lo, hi):
    """Incident must start inside the window and complete before its end."""
    return d[(d.t_first >= lo) & (d.t_last < hi)].copy()


def completed_before(d, hi):
    """Past completed incidents only; no partial aggregates crossing the cutoff."""
    return d[d.t_last < hi].copy()


def fit_bundle(train, tune, args):
    if min(len(train), len(tune)) < args.min_rows:
        raise ValueError("Insufficient fit/tune incidents")
    for name, f in (("train", train), ("tune", tune)):
        if labels(f).min() == labels(f).max():
            raise ValueError(f"{name} has only one grade class")
    tv = make_vectorizer(train, "techs", args.min_df)
    cv = make_vectorizer(train, "cats", args.min_df)
    model = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed)
    model.fit(features(train, tv, cv), labels(train))
    tune_scores = model.predict_proba(features(tune, tv, cv))[:, 1]
    thresholds = {f"{t:.4f}": threshold_for_recall(tune_scores, labels(tune), t)
                  for t in args.targets}
    return {"model": model, "tech_vectorizer": tv, "cat_vectorizer": cv,
            "thresholds": thresholds, "trained_incidents": len(train),
            "tuning_incidents": len(tune)}


def predict(bundle, frame):
    if len(frame) == 0:
        return np.array([], dtype=float)
    return bundle["model"].predict_proba(features(
        frame, bundle["tech_vectorizer"], bundle["cat_vectorizer"]))[:, 1]


def evaluate(scores, frame, thresholds):
    y = labels(frame)
    if len(frame) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Evaluation slice empty or one-class")
    wa, eligible = within_org_auc(scores, y, frame.OrgId.to_numpy())
    return {
        "incidents": int(len(frame)), "tp_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "within_org_auc": wa, "within_org_count": eligible,
        "operating_points": {name: operating_metrics(scores, y, frame.n_alerts.to_numpy(), thr)
                             for name, thr in thresholds.items()}
    }


def gate(candidate_metrics, active_metrics, args):
    """Gate uses validation slice only. No selection on test results."""
    reasons = []
    if candidate_metrics["roc_auc"] < active_metrics["roc_auc"] - args.max_auc_drop:
        reasons.append("AUC regression")
    for target in args.targets:
        key = f"{target:.4f}"
        c = candidate_metrics["operating_points"][key]
        a = active_metrics["operating_points"][key]
        if c["recall"] < target:
            reasons.append(f"{key}: target recall missed")
        if c["recall"] < a["recall"] - args.max_recall_drop:
            reasons.append(f"{key}: recall regression")
        if c["incident_demotion_rate"] < a["incident_demotion_rate"] + args.min_demotion_gain:
            reasons.append(f"{key}: incident demotion improvement failed")
    return not reasons, reasons


def timestamp_cuts(d, initial_frac, step_frac, folds):
    # Uses completion times only (without grades or predictor values).
    end_fraction = initial_frac + (folds + 3) * step_frac
    if end_fraction > 1.0 + 1e-10:
        raise ValueError("Not enough chronological mass: initial_frac + (folds+3)*step_frac > 1")
    t = d.t_last.sort_values(kind="stable").reset_index(drop=True)
    n = len(t)
    cuts = []
    for i in range(folds + 4):
        f = initial_frac + i * step_frac
        idx = min(int(n * f), n - 1)
        cuts.append(pd.Timestamp(t.iloc[idx]))
    if any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValueError("Repeated timestamp cutoffs; smaller step or different data required")
    return cuts


def score_rows(frame, fold, role, scores, thresholds):
    out = pd.DataFrame({
        "sid": frame.index.to_numpy(), "fold": fold, "model_role": role,
        "OrgId": frame.OrgId.astype(str).to_numpy(),
        "t_last": frame.t_last.to_numpy(), "y": labels(frame),
        "n_alerts": frame.n_alerts.to_numpy(), "score": scores})
    for key, threshold in thresholds.items():
        out[f"demote_{key}"] = scores <= threshold
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--outdir", default="results/guide_walkforward")
    p.add_argument("--initial-frac", type=float, default=0.35,
                   help="Fraction for initial fixed-model fitting")
    p.add_argument("--step-frac", type=float, default=0.10,
                   help="Fraction per time block; default supports 2 folds")
    p.add_argument("--folds", type=int, default=2,
                   help="Number of scheduled update/test checkpoints")
    p.add_argument("--min-df", type=int, default=20)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--C", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--targets", type=float, nargs="+", default=[0.99, 0.95])
    p.add_argument("--max-auc-drop", type=float, default=0.01)
    p.add_argument("--max-recall-drop", type=float, default=0.005)
    p.add_argument("--min-demotion-gain", type=float, default=0.0)
    args = p.parse_args()
    if not (0 < args.initial_frac < 1 and 0 < args.step_frac < 1 and args.folds >= 1
            and args.min_df >= 1 and args.min_rows >= 2 and args.C > 0
            and len(args.targets) == len(set(args.targets))
            and all(0 < t <= 1 for t in args.targets)
            and args.max_auc_drop >= 0 and args.max_recall_drop >= 0
            and args.min_demotion_gain >= 0):
        p.error("Invalid fractions, targets, or gate/model settings")
    if args.initial_frac + (args.folds + 3) * args.step_frac > 1.0 + 1e-10:
        p.error("Need initial-frac + (folds+4)*step-frac <= 1; lower step or folds")
    src = Path(args.cache)
    out = Path(args.outdir)
    if out.resolve() == src.resolve():
        p.error("--outdir must differ from --cache")
    # Protect prior experiment and read-only inputs. This script never overwrites outputs.
    if out.exists() and any(out.iterdir()):
        p.error("Output directory is not empty; choose a fresh --outdir")
    raw = load_cache(src)
    d = prepare(raw)
    cuts = timestamp_cuts(d, args.initial_frac, args.step_frac, args.folds)
    # Global initial training and independent static threshold tuning.
    initial_train = completed_before(d, cuts[0])
    static_tune = period_starting(d, cuts[0], cuts[1])
    static_bundle = fit_bundle(initial_train, static_tune, args)
    active_bundle = static_bundle
    results, decisions, predictions, skipped_boundary = [], [], [], []
    for i in range(args.folds):
        # Scheduled checkpoint i: all fitting data ends before the update's tune window.
        # Windows (i = 0): fit before cut[1]; tune [1,2); gate [2,3); test [3,4).
        # Next checkpoint advances everything by one block.
        fit_end, tune_end, gate_end, test_end = cuts[i + 1:i + 5]
        candidate_fit = completed_before(d, fit_end)
        tune = period_starting(d, fit_end, tune_end)
        validation = period_starting(d, tune_end, gate_end)
        test = period_starting(d, gate_end, test_end)
        for name, f in (("fit", candidate_fit), ("tune", tune),
                        ("gate", validation), ("test", test)):
            if len(f) < args.min_rows or labels(f).min() == labels(f).max():
                raise ValueError(f"Fold {i}: {name} too small or one-class; adjust parameters")
        candidate = fit_bundle(candidate_fit, tune, args)
        gate_active = evaluate(predict(active_bundle, validation), validation, active_bundle["thresholds"])
        gate_candidate = evaluate(predict(candidate, validation), validation, candidate["thresholds"])
        accepted, reasons = gate(gate_candidate, gate_active, args)
        # Decision is locked now. Test data and labels have not yet been accessed for scoring.
        active_bundle = candidate if accepted else active_bundle
        decision = "PROMOTE_SIMULATED" if accepted else "REJECT_KEEP_PREVIOUS"
        decisions.append({"fold": i + 1, "decision": decision, "reasons": reasons,
                          "fit_end_exclusive": fit_end.isoformat(),
                          "tune_end_exclusive": tune_end.isoformat(),
                          "gate_end_exclusive": gate_end.isoformat(),
                          "test_end_exclusive": test_end.isoformat(),
                          "n_fit": len(candidate_fit), "n_tune": len(tune),
                          "n_gate": len(validation), "n_test": len(test),
                          "gate_active": gate_active, "gate_candidate": gate_candidate})
        model_bundles = {"static": static_bundle, "candidate": candidate, "selected": active_bundle}
        fold_result = {"fold": i + 1, "decision": decision, "test_start": gate_end.isoformat(),
                       "test_end": test_end.isoformat(), "test_incidents": len(test), "models": {}}
        for role, bundle in model_bundles.items():
            scores = predict(bundle, test)
            fold_result["models"][role] = evaluate(scores, test, bundle["thresholds"])
            predictions.append(score_rows(test, i + 1, role, scores, bundle["thresholds"]))
        results.append(fold_result)
        skipped_boundary.append({"fold": i + 1,
            "tune_crossers": int(((d.t_first < fit_end) & (d.t_last >= fit_end) & (d.t_last < tune_end)).sum()),
            "gate_crossers": int(((d.t_first < tune_end) & (d.t_last >= tune_end) & (d.t_last < gate_end)).sum()),
            "test_crossers": int(((d.t_first < gate_end) & (d.t_last >= gate_end) & (d.t_last < test_end)).sum())})
        print(f"Fold {i+1}: {decision}; test static AUC={fold_result['models']['static']['roc_auc']:.4f}, "
              f"selected AUC={fold_result['models']['selected']['roc_auc']:.4f}")
    out.mkdir(parents=True, exist_ok=True)
    # Manifest records the full protocol and known limitations.
    manifest = {
        "protocol": "scheduled_walkforward_GUIDE_completed_incidents_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(src.resolve()), "source_incidents": len(raw),
        "eligible_incidents": len(d), "cutoffs_utc": [x.isoformat() for x in cuts],
        "settings": vars(args), "boundary_crossing_by_fold": skipped_boundary,
        "policy": "candidate fit on past; tune next; gate after tune; test after locked gate; "
                  "static fit/tune frozen from initial periods; accepted model becomes active only in simulation",
        "limitations": [
            "GUIDE time is privacy-noised, not exact production event time.",
            "Complete-incident aggregates are only known after completion; not live causal scoring.",
            "No organisation-disjoint splitting; organisation mix can change substantially.",
            "Rolling later training can use previously tested incidents after labels become available; "
            "a retrospective prequential assumption, not single-use prospective holdout.",
            "Walk-forward evaluation is not drift-triggered; checkpoint times are scheduled.",
            "Promotion is simulated, no real deployment or automatic rollback.",
            "Recall is TruePositive incident recall, not malicious alert recall.",
            "Logistic outputs are not independently probability calibrated.",
            "No test labels are used for model selection *at that checkpoint*; test performance "
            "may inform future development and needs external validation for final claims.",
        ]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "gate_decisions.json").write_text(json.dumps(decisions, indent=2))
    (out / "fold_metrics.json").write_text(json.dumps(results, indent=2))
    pd.concat(predictions, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    joblib.dump(static_bundle, out / "static_model.joblib")
    joblib.dump(active_bundle, out / "simulated_selected_model.joblib")
    print(f"Saved read-only simulation outputs to {out}; no existing experiment files were modified")


if __name__ == "__main__":
    main()
