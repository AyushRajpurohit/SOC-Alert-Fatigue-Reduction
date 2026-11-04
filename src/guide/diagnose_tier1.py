"""
diagnose_tier1.py -- WHY did the first reduction_curve run give AUC < 0.5?

Tests four hypotheses (see printed section titles):
  A. fold composition / organisation heterogeneity   (does GroupKFold produce lopsided folds?)
  B. size-vs-TP relationship differs by technique coverage   (sign flip between strata?)
  C. pooled vs per-fold AUC                          (is my pooled metric misleading?)
  D. model misspecification: linear vs tree model vs separate models per stratum

Usage:
    python diagnose_tier1.py --cache guide_incidents.parquet --frac-orgs 0.2
Needs reduction_curve.py in the same folder (reuses its helpers).
"""
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from reduction_curve import load, bag


def cv_eval(make_model, X, y, groups, mask=None, folds=5):
    """Per-fold AUC (the protocol of our earlier ladder) AND pooled out-of-fold AUC."""
    idx = np.arange(len(y)) if mask is None else np.nonzero(mask)[0]
    Xs, ys, gs = X[idx], y[idx], groups[idx]
    oof = np.zeros(len(idx)); per = []
    for tr, te in GroupKFold(folds).split(Xs, ys, gs):
        m = make_model().fit(Xs[tr], ys[tr])
        p = m.predict_proba(Xs[te])[:, 1]
        oof[te] = p
        if 0 < ys[te].sum() < len(te):
            per.append(roc_auc_score(ys[te], p))
    return np.mean(per), np.std(per), roc_auc_score(ys, oof), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--frac-orgs", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = load(args.cache)
    d = d[d.grade.notna()].copy()
    if args.frac_orgs < 1.0:
        ou = d.OrgId.unique()
        d = d[d.OrgId.isin(rng.choice(ou, int(len(ou) * args.frac_orgs), replace=False))].copy()
    y = (d.grade == "TruePositive").astype(int).values
    g = d.OrgId.values
    tagged = (d.n_techs.values > 0)
    print(f"incidents {len(d):,}  orgs {len(np.unique(g)):,}  TP rate {y.mean():.3f}  tagged {tagged.mean():.1%}")

    # ---------------- A. organisations and folds ----------------
    print("\n=== A. ORGANISATION HETEROGENEITY AND FOLD COMPOSITION ===")
    o = d.assign(tp=y).groupby("OrgId").agg(n=("tp", "size"), tp=("tp", "mean"))
    print("incidents per org  quantiles (50/90/99/max):",
          o.n.quantile([.5, .9, .99, 1.0]).round(0).tolist(),
          "  top-5 orgs hold", f"{o.n.sort_values(ascending=False).head(5).sum() / o.n.sum():.1%}")
    big = o[o.n >= 50]
    print(f"orgs with >=50 incidents: {len(big)}   their TP-rate quantiles (10/50/90):",
          big.tp.quantile([.1, .5, .9]).round(3).tolist())
    print(f"   share with TP rate <5%: {(big.tp < .05).mean():.1%}   >50%: {(big.tp > .5).mean():.1%}")
    for i, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, g)):
        top = pd.Series(g[te]).value_counts(normalize=True).iloc[0]
        print(f"  fold {i}: n={len(te):>7,}  orgs={len(np.unique(g[te])):>4}  TP rate={y[te].mean():.3f}  "
              f"tagged={tagged[te].mean():.1%}  biggest org={top:.1%}")

    # ---------------- B. size vs TP by coverage ----------------
    print("\n=== B. TP RATE BY INCIDENT SIZE, TAGGED vs UNTAGGED ===")
    bins = [0, 1, 2, 5, 20, 100, 1e9]; labels = ["1", "2", "3-5", "6-20", "21-100", ">100"]
    sz = pd.cut(d.n_alerts, bins=bins, labels=labels)
    tab = (pd.DataFrame({"size": sz.values, "tagged": np.where(tagged, "tagged", "untagged"), "tp": y})
             .groupby(["tagged", "size"], observed=True).tp.agg(n="size", tp_rate="mean").round(3))
    print(tab.to_string())

    # ---------------- features ----------------
    X_tech, _ = bag(d.techs, 20); X_cat, cv_cat = bag(d.cats, 20); X_det, _ = bag(d.dets, 20)
    size_d = np.c_[np.log1p(d.n_alerts.values), tagged.astype(float), np.log1p(d.n_techs.values)]
    size_s = sparse.csr_matrix(size_d)
    H = lambda *xs: sparse.hstack(xs).tocsr()
    LR = lambda: LogisticRegression(C=0.1, max_iter=2000)

    # ---------------- C. pooled vs per-fold ----------------
    print("\n=== C. PER-FOLD vs POOLED AUC (linear models, whole population) ===")
    print(f"{'model':<34}{'per-fold mean':>14}{'std':>7}{'pooled':>9}")
    for name, X in [("size only", size_s), ("techniques+size", H(X_tech, size_s)),
                    ("techniques+category+size", H(X_tech, X_cat, size_s)),
                    ("detectors+size (REFERENCE)", H(X_det, size_s))]:
        m, s, p, _ = cv_eval(LR, X, y, g)
        print(f"{name:<34}{m:>14.3f}{s:>7.3f}{p:>9.3f}")

    # ---------------- D. misspecification ----------------
    print("\n=== D. MISSPECIFICATION: tree model, and separate models per stratum ===")
    top_t = CountVectorizer(analyzer=lambda s: s.split(";") if s else [], binary=True, max_features=60)
    Xd = np.c_[size_d, (d.n_mitre_alerts / d.n_alerts).values,
               top_t.fit_transform(d.techs.values).toarray(),
               cv_cat.transform(d.cats.values).toarray()]
    HGB = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0)
    m, s, p, _ = cv_eval(HGB, Xd, y, g)
    print(f"{'tree model, portable features':<34}{m:>14.3f}{s:>7.3f}{p:>9.3f}   (whole population)")
    for lab, mask in [("TAGGED only", tagged), ("UNTAGGED only", ~tagged)]:
        for name, X, fac in [("linear tech+cat+size", H(X_tech, X_cat, size_s), LR),
                             ("tree portable", Xd, HGB),
                             ("linear detectors+size (REF)", H(X_det, size_s), LR)]:
            m, s, p, n = cv_eval(fac, X, y, g, mask=mask)
            print(f"  {lab:<14}{name:<30}{m:>7.3f}{s:>7.3f}{p:>9.3f}   n={n:,}")
    print("\nHow to read: B shows whether size means different things in the two strata; "
          "C whether pooling inflates the problem; D whether a flexible/stratified model fixes it.")


if __name__ == "__main__":
    main()
