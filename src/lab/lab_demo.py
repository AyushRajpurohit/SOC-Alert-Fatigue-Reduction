"""
lab_demo.py -- lab tier: known chains, exact ground truth.

WHAT IT SHOWS (the mid-evaluation demo)
  1. Learn a BACKBONE (longest subsequence common to all baseline runs) with your existing
     sessionize_and_mine.longest_common_backbone.
  2. Score any other session against it:
        match    = LCS / len(session)    "how much of this session is part of the known attack"
        coverage = LCS / len(backbone)   "how much of the known attack has been seen"
  3. Compare with the static Wazuh rule: did rule 100103 ("full chain detected") fire?
  4. --replay: stream one run's alerts in time order; the incident view updates after every alert.
  5. --check: reproduce the numbers in your week-1 log (needs --mode raw) and say PASS/FAIL.

EXAMPLES
  python lab_demo.py --baseline joined_run1.csv joined_run2.csv joined_run3.csv joined_run4.csv joined_run5.csv \
        --variant joined_variant.csv --other chain2=joined_chain2.csv --mode canonical
  python lab_demo.py ...same files... --mode raw --check          # reproduce week-1 numbers
  python lab_demo.py ...same files... --replay joined_variant.csv --delay 0.4   # live demo
"""
import argparse
import csv
import os
import time

from sessionize_and_mine import lcs2, longest_common_backbone
from soc_common import wazuh_joined_adapter, tokens
from canonical_tactics import unmapped_rule_ids

STATIC_FULL_CHAIN = "100103"                       # custom rule: full chain detected
STATIC_LINKS = ["100100", "100101", "100102", "100103"]


def describe(session, seq, backbone):
    lcs = lcs2(seq, backbone) if seq and backbone else []
    fired = {r for r in STATIC_LINKS if any(a.detector_id == r for a in session.alerts)}
    m = len(lcs) / len(seq) if seq else 0.0
    c = len(lcs) / len(backbone) if backbone else 0.0
    status = "FIRED" if STATIC_FULL_CHAIN in fired else ("MISSED" if fired else "n/a")
    return {
        "static_status": status,
        "f1": (2 * m * c / (m + c)) if (m + c) else 0.0,
        "session": session.sid,
        "raw_alerts": len(session.alerts),
        "tokens": len(seq),
        "lcs": len(lcs),
        "match": len(lcs) / len(seq) if seq else float("nan"),
        "coverage": len(lcs) / len(backbone) if backbone else float("nan"),
        "static_links_fired": ",".join(sorted(fired)) or "-",
        "static_full_chain": STATIC_FULL_CHAIN in fired,
        "seq": " -> ".join(seq),
    }


def replay(session, backbone, mode, delay):
    """Raw alert stream on the left, incident view (growing token sequence + score) on the right."""
    seq = []
    print(f"\nREPLAY of {session.sid}   (backbone = {len(backbone)} tokens)\n")
    for i, a in enumerate(session.alerts, 1):
        before = len(seq)
        one = type(session)(session.source, session.sid, [a])
        for t in tokens(one, mode, dedup=False):
            if not seq or seq[-1] != t:
                seq.append(t)
        lcs = len(lcs2(seq, backbone)) if seq else 0
        print(f"[{i:>3}] {a.timestamp[11:19]}  rule {a.detector_id:<7} L{a.level if a.level is not None else '-'}  "
              f"{(a.description or '')[:58]}")
        if len(seq) != before:
            shown = ' -> '.join(seq) if len(seq) <= 6 else '... -> ' + ' -> '.join(seq[-5:])
            print(f"      INCIDENT VIEW: {len(seq)} step(s): {shown}"
                  f"   | match {lcs / len(seq):.0%}  | chain progress {lcs / len(backbone):.0%}")
        if delay:
            time.sleep(delay)
    ids = {a.detector_id for a in session.alerts}
    status = "FIRED" if STATIC_FULL_CHAIN in ids else ("MISSED (earlier links fired, 100103 did not)" if ids & set(STATIC_LINKS) else "n/a (no custom rules in this log)")
    print(f"\nstatic rule {STATIC_FULL_CHAIN} on this run: {status}")


def check_week1(rows, backbone, base_seqs, variant, other):
    """Reproduce what the week-1 log (sections 7-8, 11) reports. Only meaningful with --mode raw."""
    print("\nREGRESSION CHECK against week-1 log  (run with --mode raw for the first block)")
    ok = True

    def line(name, got, want, tol=0.0):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got:.3f}  expected {want:.3f}")

    line("backbone length (5 baseline runs, raw tactics)", len(backbone), 31)
    if variant:
        line("variant sequence length", variant["tokens"], 23)
        line("variant match (every variant token inside backbone, in order)", variant["match"], 1.0, 0.005)
        line("variant coverage of backbone (23/31)", variant["coverage"], 0.74, 0.01)
        fired = set(variant["static_links_fired"].split(","))
        for rid, want in [("100100", True), ("100101", True), ("100102", False), ("100103", False)]:
            got = rid in fired
            ok &= (got == want)
            print(f"  [{'PASS' if got == want else 'FAIL'}] static rule {rid} on variant: fired={got} expected={want}")
    print("  (chain 2 '60% overlap' is NOT asserted: its exact definition must be confirmed -- see printed numbers.)")
    print(f"\nOVERALL: {'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--variant")
    ap.add_argument("--other", nargs="*", default=[], help="name=path pairs, e.g. chain2=joined_chain2.csv")
    ap.add_argument("--mode", choices=["canonical", "raw"], default="canonical")
    ap.add_argument("--out", default="lab_results")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--replay")
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = [wazuh_joined_adapter(p, sid=os.path.basename(p)) for p in args.baseline]
    base_seqs = [tokens(s, args.mode) for s in base]
    backbone = longest_common_backbone(base_seqs)
    print(f"mode: {args.mode}   baseline runs: {len(base)}   backbone ({len(backbone)} tokens):")
    print("  " + " -> ".join(backbone))

    sessions = [(s, "baseline (in-sample)") for s in base]
    if args.variant:
        sessions.append((wazuh_joined_adapter(args.variant, sid=os.path.basename(args.variant)), "variant"))
    for kv in args.other:
        name, path = kv.split("=", 1)
        sessions.append((wazuh_joined_adapter(path, sid=name), "other chain"))

    rows = []
    for s, kind in sessions:
        d = describe(s, tokens(s, args.mode), backbone)
        d["kind"] = kind
        rows.append(d)

    print("\nRESULTS")
    print(f"{'session':<24}{'kind':<22}{'alerts':>7}{'tokens':>8}{'match':>8}{'coverage':>10}{'f1':>6}  static rule 100103")
    for r in rows:
        print(f"{r['session']:<24}{r['kind']:<22}{r['raw_alerts']:>7}{r['tokens']:>8}{r['match']:>8.0%}"
              f"{r['coverage']:>10.0%}{r['f1']:>6.2f}  {r['static_status']:<7} [{r['static_links_fired']}]")
    print("static rule: FIRED = 100103 seen; MISSED = earlier links (100100-100102) fired but 100103 did not; "
          "n/a = no custom-rule alerts in this log at all (rules were not deployed when it was recorded).")
    print("\nbaseline rows are in-sample: the backbone is a subsequence of every baseline run, so their COVERAGE is 100% "
          "by construction (their match = len(backbone)/len(session)). Only the variant / other-chain rows are a real test.")
    if unmapped_rule_ids() and args.mode == "canonical":
        print(f"NOTE: unmapped rule_ids used the raw-tactic fallback: {unmapped_rule_ids()}")

    with open(os.path.join(args.out, "lab_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = [r["session"] for r in rows]
    x = range(len(rows))
    ax[0].bar([i - 0.2 for i in x], [r["raw_alerts"] for r in rows], 0.4, label="raw alerts")
    ax[0].bar([i + 0.2 for i in x], [r["tokens"] for r in rows], 0.4, label="steps in incident view")
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax[0].set_title("Alert flood -> incident view"); ax[0].legend(fontsize=8)
    ax[1].bar([i - 0.2 for i in x], [r["match"] for r in rows], 0.4, label="match (share of session inside known attack)")
    ax[1].bar([i + 0.2 for i in x], [r["coverage"] for r in rows], 0.4, label="coverage (share of known attack seen)")
    col = {"FIRED": "green", "MISSED": "red", "n/a": "gray"}
    lab_txt = {"FIRED": "static rule FIRED", "MISSED": "static rule MISSED", "n/a": "no static-rule data"}
    for i, r in enumerate(rows):
        ax[1].text(i, max(r["match"], r["coverage"]) + 0.03, lab_txt[r["static_status"]],
                   ha="center", fontsize=6, color=col[r["static_status"]], rotation=90)
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax[1].set_ylim(0, 1.75); ax[1].set_title(f"Similarity to learned backbone ({args.mode} tokens)")
    ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lab_summary.png"), dpi=150)
    print(f"\nwrote {args.out}/lab_results.csv and lab_summary.png")

    if args.check:
        var = next((r for r in rows if r["kind"] == "variant"), None)
        check_week1(rows, backbone, base_seqs, var, [r for r in rows if r["kind"] == "other chain"])
        for r in rows:
            if r["kind"] == "other chain":
                print(f"  info: {r['session']}: {r['tokens']} tokens, LCS with backbone {r['lcs']} -> "
                      f"LCS/len(session) {r['match']:.0%}, LCS/len(backbone) {r['coverage']:.0%}")
                print(f"        sequence: {r['seq']}")

    if args.replay:
        s = wazuh_joined_adapter(args.replay, sid=os.path.basename(args.replay))
        replay(s, backbone, args.mode, args.delay)


if __name__ == "__main__":
    main()
