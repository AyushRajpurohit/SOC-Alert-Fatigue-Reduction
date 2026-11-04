"""
build_guide_cache.py (v2) -- run ONCE. Evidence-level GUIDE_Train.csv -> one row per incident.

WHAT CHANGED FROM v1 AND WHY
  v1 collapsed evidence rows to alerts with drop_duplicates(), which keeps only the
  FIRST evidence row per alert. If an alert's evidence rows disagree on technique,
  category, detector or timestamp, v1 silently dropped information. (Only ~1/3 of
  incidents carry a MITRE technique, so a blank-first-row / tagged-later-row case
  would directly shrink our signal.)

  v2 never depends on row order:
    * techniques, categories, detectors = UNION over ALL evidence rows of the incident
    * an alert counts as "has MITRE" if ANY of its evidence rows has a technique
    * timestamps = min / max over ALL evidence rows
  and it first MEASURES how much the evidence rows of one alert actually disagree,
  and how many incidents v1's first-row logic would have lost techniques for.

Usage:
    python build_guide_cache.py --csv GUIDE_Train.csv --out guide_incidents.parquet
    (add --no-diagnose to skip the diagnostic block; ~1-2 extra minutes)
"""
import argparse
import pandas as pd

COLS = ["OrgId", "IncidentId", "AlertId", "Timestamp", "DetectorId",
        "Category", "MitreTechniques", "IncidentGrade"]
AKEY = ["OrgId", "IncidentId", "AlertId"]
IKEY = ["OrgId", "IncidentId"]


def diagnose(df):
    """How much do the evidence rows of ONE alert disagree? (NaN counts as a value.)"""
    n_alerts = len(df.drop_duplicates(AKEY))
    print("\nDIAGNOSTIC: do evidence rows of the same alert disagree?")
    print(f"  evidence rows: {len(df):,}   alerts: {n_alerts:,}   "
          f"avg evidence rows/alert: {len(df) / n_alerts:.2f}")
    for c in ["MitreTechniques", "Category", "DetectorId", "Timestamp", "IncidentGrade"]:
        n_pairs = len(df[AKEY + [c]].drop_duplicates())
        extra = n_pairs - n_alerts            # alerts with k distinct values contribute k-1
        share = extra / n_alerts
        print(f"  {c:<16} extra distinct values beyond 1/alert: {extra:>10,}  ({share:.2%} of alerts, lower bound)")


def pair_union(df, col, sep=";", split_parents=False):
    """Union of `col` values per incident over ALL evidence rows -> Series (index OrgId-IncidentId)."""
    p = df.loc[df[col].notna(), IKEY + [col]].drop_duplicates()   # tiny vs 9.5M rows
    if split_parents:
        p[col] = p[col].astype(str).str.split(";")
        p = p.explode(col)
        p[col] = p[col].str.split(".").str[0]
        p = p[p[col].notna() & (p[col] != "")].drop_duplicates()
    else:
        p[col] = p[col].astype(str)
    p["sid"] = p.OrgId.astype(str) + "-" + p.IncidentId.astype(str)
    return p.groupby("sid")[col].agg(lambda s: sep.join(sorted(set(s))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="guide_incidents.parquet")
    ap.add_argument("--nrows", type=int, default=None, help="debug only")
    ap.add_argument("--no-diagnose", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=COLS, nrows=args.nrows)
    print(f"evidence rows: {len(df):,}")
    if not args.no_diagnose:
        diagnose(df)

    df["has_m"] = df.MitreTechniques.notna() & (df.MitreTechniques.astype(str).str.len() > 0)

    # ---- alert level: order-independent aggregation over ALL evidence rows ----
    al = df.groupby(AKEY, sort=False).agg(
        has_mitre=("has_m", "max"),                 # ANY evidence row carries a technique
        t_min=("Timestamp", "min"), t_max=("Timestamp", "max"),
        grade=("IncidentGrade", "first")).reset_index()
    print(f"\nalerts: {len(al):,}")

    # ---- incident level ----
    g = al.groupby(IKEY, sort=False)
    inc = pd.DataFrame({
        "n_alerts": g.size(),
        "n_mitre_alerts": g.has_mitre.sum(),
        "grade": g.grade.first(),
        "grade_nunique": g.grade.nunique(),
        "t_first": g.t_min.min(),
        "t_last": g.t_max.max(),
    }).reset_index()
    inc["sid"] = inc.OrgId.astype(str) + "-" + inc.IncidentId.astype(str)
    inc = inc.set_index("sid").drop(columns="IncidentId")

    inc["techs"] = pair_union(df, "MitreTechniques", split_parents=True).reindex(inc.index).fillna("")
    inc["cats"] = pair_union(df, "Category").reindex(inc.index).fillna("")
    inc["dets"] = pair_union(df, "DetectorId").reindex(inc.index).fillna("")
    inc["n_techs"] = inc.techs.map(lambda s: len(s.split(";")) if s else 0)
    inc["t_first"] = pd.to_datetime(inc.t_first, utc=True)
    inc["t_last"] = pd.to_datetime(inc.t_last, utc=True)

    # ---- how much would v1's first-row logic have lost? ----
    first_rows = df.drop_duplicates(AKEY)
    old_has = first_rows.assign(h=first_rows.MitreTechniques.notna()).groupby(IKEY).h.any()
    old_has.index = [f"{o}-{i}" for o, i in old_has.index]
    old_share = old_has.reindex(inc.index).fillna(False).astype(bool).mean()
    print("\nFIRST-ROW (v1) vs UNION (v2)")
    print(f"  incidents with >=1 technique, v1 first-row logic: {old_share:.2%}")
    print(f"  incidents with >=1 technique, v2 union logic:     {(inc.n_techs > 0).mean():.2%}")
    print(f"  alerts with a technique, v1: {first_rows.MitreTechniques.notna().mean():.2%}   "
          f"v2: {al.has_mitre.mean():.2%}")

    print("\nSANITY CHECKS (earlier v1 run: 730,778 incidents, 1,666,227 alerts)")
    print(f"  incidents: {len(inc):,}   alerts: {int(inc.n_alerts.sum()):,}")
    print(f"  incidents whose alerts carry >1 distinct grade: {(inc.grade_nunique > 1).sum():,}")
    print("  grade mix:\n", inc.grade.value_counts(dropna=False, normalize=True).round(3).to_string())
    print(f"  timestamp range: {inc.t_first.min()}  ->  {inc.t_last.max()}")

    out = args.out
    try:
        inc.to_parquet(out)
    except Exception as e:
        out = out.rsplit(".", 1)[0] + ".pkl"
        print(f"\n(parquet unavailable: {e.__class__.__name__}; falling back to pickle)")
        inc.to_pickle(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
