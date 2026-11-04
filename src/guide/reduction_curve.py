"""
reduction_curve.py (v2) -- reduction vs safety recall, evaluated so organisation
heterogeneity cannot fool us.

WHAT v1 GOT WRONG (found with diagnose_tier1.py)
  * It pooled out-of-fold scores from 5 different models whose test folds had very
    different base rates (TP rate 0.08 .. 0.43; one org = 60% of a fold). Pooled AUCs
    fell below 0.5 and the curve/table were not meaningful.
  * GUIDE organisations grade very differently (about half of the orgs with >=50
    incidents have TP rate < 5%, a quarter > 50%).

WHAT v2 DOES
  1. Caps each organisation at --cap-per-org incidents (random), so no org dominates.
  2. Repeats --repeats times: random split of ORGANISATIONS into
        train 50% | calibration 25% | test 25%
     Fit on train; choose the demotion threshold on calibration orgs; measure on test orgs.
     One model per repeat => scores are comparable. The spread across repeats is the safety story.
  3. Metrics per model: test AUC, WITHIN-ORG AUC (rank incidents inside each org, then
     average over orgs: removes org base-rate effects), and for each recall target the
     achieved recall + reduction (mean, sd, 10th percentile of recall).
  4. Models: size only | category+size | techniques+size | techniques+category+size |
     detectors+size (REFERENCE, not portable).

Usage:
    python reduction_curve.py --cache guide_incidents.parquet --frac-orgs 0.2 --repeats 5   (quick)
    python reduction_curve.py --cache guide_incidents.parquet                               (full)
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = (0.99, 0.95, 0.90)


def load(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_pickle(path)


def bag(series, min_df):
    cv = CountVectorizer(analyzer=lambda s: s.split(";") if s else [],
                         binary=True, min_df=min_df)
    return cv.fit_transform(series.values).tocsr(), cv


def boundaries(score, y, w):
    """Demote lowest scores first; stats at tie-group boundaries (starting from 'demote nothing')."""
    order = np.argsort(score, kind="stable")
    s, yy, ww = score[order], y[order], w[order]
    ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    thr = np.r_[-np.inf, s[ends]]
    recall = np.r_[1.0, 1 - np.cumsum(yy)[ends] / yy.sum()]
    red_inc = np.r_[0.0, (ends + 1) / len(s)]
    red_alert = np.r_[0.0, np.cumsum(ww)[ends] / ww.sum()]
    return thr, recall, red_inc, red_alert


def pick_threshold(score, y, w, target):
    thr, recall, _, _ = boundaries(score, y, w)
    return thr[np.nonzero(recall >= target)[0].max()]


def apply_threshold(score, y, w, thr):
    sup = score <= thr
    return y[~sup].sum() / y.sum(), sup.mean(), w[sup].sum() / w.sum()


def within_org_auc(s, y, g, min_n=30, min_class=5):
    df = pd.DataFrame({"g": g, "y": y, "s": s})
    aucs = []
    for _, sub in df.groupby("g"):
        pos = sub.y.sum()
        if len(sub) >= min_n and pos >= min_class and len(sub) - pos >= min_class:
            aucs.append(roc_auc_score(sub.y, sub.s))
    return (np.mean(aucs) if aucs else np.nan), len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--frac-orgs", type=float, default=1.0)
    ap.add_argument("--cap-per-org", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = load(args.cache)
    d = d[d.grade.notna()].copy()
    if args.frac_orgs < 1.0:
        ou = d.OrgId.unique()
        d = d[d.OrgId.isin(rng.choice(ou, int(len(ou) * args.frac_orgs), replace=False))].copy()
    n_before = len(d)
    d = d.sample(frac=1.0, random_state=args.seed)               # shuffle, then keep first N per org
    d = d[d.groupby("OrgId").cumcount() < args.cap_per_org].reset_index()
    y = (d.grade == "TruePositive").astype(int).values
    w = d.n_alerts.values.astype(float)
    g = d.OrgId.values
    tagged = d.n_techs.values > 0
    orgs = np.unique(g)
    print(f"graded incidents before cap: {n_before:,} -> after cap {len(d):,}   orgs: {len(orgs):,}   "
          f"TP rate: {y.mean():.3f}   tagged: {tagged.mean():.1%}")
    otp = d.assign(y=y).groupby("OrgId").y.agg(["size", "mean"])
    big = otp[otp["size"] >= 50]
    print(f"orgs with >=50 incidents (after cap): {len(big)}   TP-rate quantiles 10/50/90: "
          f"{big['mean'].quantile([.1, .5, .9]).round(3).tolist()}")

    # ---- Category sanity table: how portable / how informative is it? ----
    ex = d.assign(y=y)[["cats", "y", "OrgId"]]
    ex = ex.assign(cat=ex.cats.str.split(";")).explode("cat")
    ex = ex[ex.cat.notna() & (ex.cat != "")]
    ct = ex.groupby("cat").agg(incidents=("y", "size"), orgs=("OrgId", "nunique"), tp_rate=("y", "mean"))
    print("\nCATEGORY TABLE (top 15 by incidents; an incident counts in each of its categories)")
    print(ct.sort_values("incidents", ascending=False).head(15).round(3).to_string())

    # ---- features ----
    size = sparse.csr_matrix(np.c_[np.log1p(d.n_alerts.values), tagged.astype(float),
                                   np.log1p(d.n_techs.values)])
    X_t, _ = bag(d.techs, 20); X_c, _ = bag(d.cats, 20); X_d, _ = bag(d.dets, 20)
    H = lambda *xs: sparse.hstack(xs).tocsr()
    feats = {
        "size only": size,
        "category+size": H(X_c, size),
        "techniques+size": H(X_t, size),
        "techniques+category+size": H(X_t, X_c, size),
        "detectors+size (REFERENCE)": H(X_d, size),
    }

    grid = np.linspace(0, 1, 101)
    curves = {m: {"inc": [], "alert": []} for m in feats}
    rows = []
    for r in range(args.repeats):
        perm = rng.permutation(orgs); n = len(perm)
        tr_o, ca_o, te_o = perm[: n // 2], perm[n // 2: 3 * n // 4], perm[3 * n // 4:]
        tr, ca, te = (np.isin(g, o) for o in (tr_o, ca_o, te_o))
        for name, X in feats.items():
            m = LogisticRegression(C=args.C, max_iter=2000).fit(X[tr], y[tr])
            s_ca, s_te = m.predict_proba(X[ca])[:, 1], m.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], s_te)
            wauc, n_w = within_org_auc(s_te, y[te], g[te])
            b = boundaries(s_te, y[te], w[te])
            curves[name]["inc"].append(np.interp(grid, b[2], b[1]))
            curves[name]["alert"].append(np.interp(grid, b[3], b[1]))
            for t in TARGETS:
                thr = pick_threshold(s_ca, y[ca], w[ca], t)
                rec, ri, ra = apply_threshold(s_te, y[te], w[te], thr)
                rows.append(dict(repeat=r, model=name, target=t, auc=auc, within_org_auc=wauc,
                                 n_within_orgs=n_w, recall=rec, red_inc=ri, red_alert=ra))
        print(f"repeat {r + 1}/{args.repeats} done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.outdir, "tier1_all_repeats.csv"), index=False)
    pd.set_option("display.width", 250)

    # ---- summary ----
    auc_t = (R.drop_duplicates(["repeat", "model"]).groupby("model", sort=False)
               .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                    within_org_auc_mean=("within_org_auc", "mean"),
                    within_org_auc_sd=("within_org_auc", "std"),
                    orgs_used=("n_within_orgs", "mean")))
    print("\nAUC ON UNSEEN ORGS (mean/sd over repeats). within_org_auc ranks incidents INSIDE each org.")
    print(auc_t.round(3).to_string())
    auc_t.round(4).to_csv(os.path.join(args.outdir, "tier1_auc.csv"))

    summ = (R.groupby(["model", "target"], sort=False)
              .agg(recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
                   recall_p10=("recall", lambda v: np.percentile(v, 10)),
                   red_inc_mean=("red_inc", "mean"), red_inc_sd=("red_inc", "std"),
                   red_alert_mean=("red_alert", "mean"), red_alert_sd=("red_alert", "std")))
    print("\nOPERATING POINTS (threshold from calibration orgs, measured on unseen test orgs)")
    print(summ.round(3).to_string())
    summ.round(4).to_csv(os.path.join(args.outdir, "tier1_operating_points.csv"))

    # ---- plot: mean curve with +/- 1 sd band over repeats ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, key, title in [(axes[0], "inc", "share of INCIDENTS demoted"),
                           (axes[1], "alert", "share of ALERTS demoted")]:
        for name in feats:
            arr = np.array(curves[name][key]); mu, sd = arr.mean(0), arr.std(0)
            ls = "--" if "REFERENCE" in name else "-"
            ax.plot(grid, mu, ls, label=name)
            ax.fill_between(grid, mu - sd, mu + sd, alpha=0.12)
        ax.plot([0, 1], [1, 0], ":", color="grey", label="random demotion")
        ax.axhline(0.95, color="red", lw=0.6)
        ax.set_xlabel(f"Reduction: {title}")
        ax.set_ylabel("Safety recall (TruePositive incidents still surfaced)")
        ax.set_ylim(0.5, 1.005); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"GUIDE: reduction vs safety recall on unseen organisations "
                 f"(cap {args.cap_per_org}/org, {args.repeats} random org splits, band = 1 sd)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "reduction_vs_safety.png"), dpi=150)
    print(f"\nwrote results to {args.outdir}/")


if __name__ == "__main__":
    main()
