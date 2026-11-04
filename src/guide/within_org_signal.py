"""
within_org_signal.py -- WHERE does the within-organisation signal come from?

Same population and protocol as reduction_curve.py (cap per org, random org splits),
so numbers are directly comparable to your full run.

  PART 1  Model within-org AUC by subset:  all | tagged only | untagged only | size==1 | size>=2
          (is the signal in incidents that carry techniques, or in the tagged/untagged flag?)
  PART 2  Single-feature within-org AUC: for every frequent technique / category / size feature,
          rank incidents INSIDE each org by that one feature alone.
          AUC > 0.5 => incidents WITH the feature are more often TruePositive inside an org.
  PART 3  Concentration: refit techniques+size with the 5 strongest techniques removed.
          If within-org AUC collapses, the signal rests on a few techniques; if it barely moves,
          it is spread over many (note: those 5 were picked on the same data -> slightly
          optimistic drop, treat as a diagnostic, not a result).

Usage:
    python within_org_signal.py --cache guide_incidents.parquet --outdir results_signal
Options: --cap-per-org 300  --repeats 5  --min-orgs-feature 30
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from reduction_curve import load, bag, within_org_auc


def single_feature_table(F, names, kinds, y, org_idx, min_orgs, min_side=5):
    """Within-org AUC of every binary column of F (csr). An org counts for a feature only if the feature is
    present AND absent >= min_side times there. Each org is densified once (fast)."""
    aucs = {j: [] for j in range(len(names))}
    for idx in org_idx:
        D = F[idx].toarray()
        yo = y[idx]
        k = D.sum(0)
        ok = (k >= min_side) & (len(idx) - k >= min_side)
        for j in np.nonzero(ok)[0]:
            aucs[j].append(roc_auc_score(yo, D[:, j]))
    rows = []
    for j, nm in enumerate(names):
        if len(aucs[j]) >= min_orgs:
            a = np.array(aucs[j])
            present = (F[:, j].toarray().ravel() > 0)
            rows.append(dict(kind=kinds[j], feature=nm, orgs_used=len(a), auc_mean=a.mean(),
                             auc_se=a.std(ddof=1) / np.sqrt(len(a)), incidents=int(present.sum()),
                             tp_rate_with=y[present].mean(), tp_rate_without=y[~present].mean()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", default="results_signal")
    ap.add_argument("--cap-per-org", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--min-orgs-feature", type=int, default=30)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = load(args.cache)
    d = d[d.grade.notna()].copy()
    d = d.sample(frac=1.0, random_state=args.seed)
    d = d[d.groupby("OrgId").cumcount() < args.cap_per_org].reset_index()
    y = (d.grade == "TruePositive").astype(int).values
    g = d.OrgId.values
    tagged = d.n_techs.values > 0
    big1 = d.n_alerts.values >= 2
    orgs = np.unique(g)
    print(f"incidents {len(d):,}  orgs {len(orgs):,}  TP rate {y.mean():.3f}  tagged {tagged.mean():.1%}  "
          f"size>=2 {big1.mean():.1%}")

    size = sparse.csr_matrix(np.c_[np.log1p(d.n_alerts.values), tagged.astype(float),
                                   np.log1p(d.n_techs.values)])
    X_t, cv_t = bag(d.techs, 20); X_c, cv_c = bag(d.cats, 20)
    t_names = list(cv_t.get_feature_names_out()); c_names = list(cv_c.get_feature_names_out())
    H = lambda *xs: sparse.hstack(xs).tocsr()
    models = {"size only": size, "techniques+size": H(X_t, size),
              "category+size": H(X_c, size), "techniques+category+size": H(X_t, X_c, size)}
    flag = sparse.csr_matrix(tagged.astype(float)[:, None])
    models = {"tagged-flag only": flag, **models}

    subsets = {"all": np.ones(len(d), bool), "tagged only": tagged, "untagged only": ~tagged,
               "size==1": ~big1, "size>=2": big1}

    # ---------------- PART 1 ----------------
    print("\n=== PART 1: within-org AUC by subset (mean over repeats; orgs_used in brackets) ===")
    res = {m: {s: [] for s in subsets} for m in models}; used = {m: {s: [] for s in subsets} for m in models}
    for r in range(args.repeats):
        perm = rng.permutation(orgs); n = len(perm)
        tr = np.isin(g, perm[: n // 2]); te = np.isin(g, perm[3 * n // 4:])
        for m, X in models.items():
            mdl = LogisticRegression(C=args.C, max_iter=2000).fit(X[tr], y[tr])
            s_te = mdl.predict_proba(X[te])[:, 1]
            for sname, mask in subsets.items():
                mm = mask[te]
                a, k = within_org_auc(s_te[mm], y[te][mm], g[te][mm])
                res[m][sname].append(a); used[m][sname].append(k)
        print(f"  repeat {r + 1}/{args.repeats}", flush=True)
    tab = pd.DataFrame({s: [f"{np.nanmean(res[m][s]):.3f} ({np.mean(used[m][s]):.0f})" for m in models]
                        for s in subsets}, index=list(models))
    print(tab.to_string())
    tab.to_csv(os.path.join(args.outdir, "part1_within_org_by_subset.csv"))
    print("\nIf 'tagged-flag only' is already high, part of the signal is just 'has techniques vs not'.")

    # ---------------- PART 2 ----------------
    print("\n=== PART 2: single-feature within-org AUC (AUC>0.5: feature => more TP inside an org) ===")
    sizecols = sparse.csr_matrix(np.c_[(d.n_alerts.values >= 2), (d.n_alerts.values >= 6),
                                       tagged, (d.n_techs.values >= 2)].astype(float))
    s_names = ["size>=2 alerts", "size>=6 alerts", "has any technique", ">=2 techniques"]
    # organisations eligible for within-org ranking
    df = pd.DataFrame({"g": g, "y": y})
    elig = df.groupby("g").y.agg(["size", "sum"])
    elig = elig[(elig["size"] >= 50) & (elig["sum"] >= 5) & (elig["size"] - elig["sum"] >= 5)].index
    order = np.argsort(g, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(g[order]))[0] + 1, len(g)]
    org_idx = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1) if g[order[bounds[i]]] in set(elig)]
    print(f"eligible orgs: {len(org_idx)}")
    top_t = np.argsort(-np.asarray(X_t.sum(0)).ravel())[:80]          # 80 most frequent techniques
    F_all = H(sizecols, X_c, X_t[:, top_t])
    names = s_names + c_names + [t_names[j] for j in top_t]
    kinds = ["structure"] * len(s_names) + ["category"] * len(c_names) + ["technique"] * len(top_t)
    rows = single_feature_table(F_all, names, kinds, y, org_idx, args.min_orgs_feature)
    F = pd.DataFrame(rows)
    F["dev"] = (F.auc_mean - 0.5).abs()
    F.sort_values("dev", ascending=False).drop(columns="dev").round(3).to_csv(
        os.path.join(args.outdir, "part2_single_feature_auc.csv"), index=False)
    show = ["kind", "feature", "orgs_used", "auc_mean", "auc_se", "incidents", "tp_rate_with", "tp_rate_without"]
    pd.set_option("display.width", 200)
    print("\nstrongest MORE-TP features:"); print(F.sort_values("auc_mean", ascending=False).head(12)[show].round(3).to_string(index=False))
    print("\nstrongest LESS-TP features:"); print(F.sort_values("auc_mean").head(12)[show].round(3).to_string(index=False))
    tech_only = F[F.kind == "technique"]
    print(f"\ntechniques tested: {len(tech_only)}   share with |AUC-0.5| > 2*se: "
          f"{(tech_only.dev > 2 * tech_only.auc_se).mean():.0%}   median |AUC-0.5|: {tech_only.dev.median():.3f}")

    # ---------------- PART 3 ----------------
    print("\n=== PART 3: concentration -- drop the 5 strongest techniques, refit techniques+size ===")
    strongest = tech_only.sort_values("dev", ascending=False).head(5).feature.tolist()
    print("dropped:", strongest)
    keep = [i for i, t in enumerate(t_names) if t not in strongest]
    X_drop = H(X_t[:, keep], size)
    full_a, drop_a = [], []
    rng = np.random.default_rng(args.seed + 1)
    for r in range(args.repeats):
        perm = rng.permutation(orgs); n = len(perm)
        tr = np.isin(g, perm[: n // 2]); te = np.isin(g, perm[3 * n // 4:])
        for X, store in [(models["techniques+size"], full_a), (X_drop, drop_a)]:
            mdl = LogisticRegression(C=args.C, max_iter=2000).fit(X[tr], y[tr])
            store.append(within_org_auc(mdl.predict_proba(X[te])[:, 1], y[te], g[te])[0])
    print(f"within-org AUC  all techniques: {np.mean(full_a):.3f} (sd {np.std(full_a):.3f})   "
          f"without top-5: {np.mean(drop_a):.3f} (sd {np.std(drop_a):.3f})")


if __name__ == "__main__":
    main()
