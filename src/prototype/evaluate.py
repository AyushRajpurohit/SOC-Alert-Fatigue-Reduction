"""
evaluate.py -- measure the prototype on a scenario (background noise + injected attacks with known truth).

METRICS (the same idea as the GUIDE curve: how much work is removed vs how much real attack is still surfaced)
  analyst items      incidents surfaced (ESCALATE + INVESTIGATE)             <- what an analyst actually has to open
  reduction          1 - analyst items / raw alerts
  safety recall      share of INJECTED attack alerts that sit inside a surfaced incident
  stage recall       share of injected (run, ground-truth stage) pairs with at least one alert surfaced
  false-alarm items  surfaced incidents that contain no injected alert
BASELINE: a plain severity filter (show every alert with Wazuh level >= L), the common practice. Swept over L.
SWEEP:    the 'investigate' threshold of the engine, to trace reduction vs safety recall.

Usage:
  python evaluate.py --scenario data/scenarios/synthetic.jsonl --patterns src/prototype/patterns.json --out results/proto
"""
import argparse
import csv
import os

import _common_path  # noqa: F401
from soc_common import load_stream
from socflow import Engine, load_patterns, SURFACED


def run(alerts, patterns, cfg):
    eng = Engine(patterns, cfg)
    for a in alerts:
        eng.feed(a)
    eng.finish()
    return eng


def metrics(eng, alerts):
    seq2inc = {s: i.iid for i in eng.incidents.values() for s in i.seqs}
    surf = {i.iid for i in eng.incidents.values() if i.decision in SURFACED}
    inj = [a for a in alerts if a.truth]
    inj_surf = [a for a in inj if seq2inc[a.seq] in surf]
    stages = {(a.truth, a.stage) for a in inj if a.stage}
    stages_ok = {(a.truth, a.stage) for a in inj_surf if a.stage}
    has_inj = {seq2inc[a.seq] for a in inj}
    runs = {}
    for lab in sorted({a.truth for a in inj}):
        mine = [a for a in inj if a.truth == lab]
        ok = [a for a in mine if seq2inc[a.seq] in surf]
        dec = sorted({eng.incidents[seq2inc[a.seq]].decision for a in mine})
        runs[lab] = (len(ok), len(mine), dec)
    n = len(alerts)
    return {"alerts": n, "items": len(surf), "reduction": 1 - len(surf) / n,
            "inj_alerts": len(inj), "safety_recall": len(inj_surf) / len(inj) if inj else float("nan"),
            "stage_recall": len(stages_ok) / len(stages) if stages else float("nan"),
            "false_alarm_items": len(surf - has_inj), "runs": runs,
            "escalate": sum(1 for i in eng.incidents.values() if i.decision == "ESCALATE"),
            "investigate": sum(1 for i in eng.incidents.values() if i.decision == "INVESTIGATE")}


def severity_baseline(alerts):
    inj = [a for a in alerts if a.truth]
    rows = []
    for L in range(0, 16):
        shown = [a for a in alerts if int(a.level or 0) >= L]
        rows.append({"level>=": L, "items": len(shown), "reduction": 1 - len(shown) / len(alerts),
                     "safety_recall": sum(1 for a in inj if int(a.level or 0) >= L) / len(inj) if inj else float("nan")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--out", default="results/proto")
    ap.add_argument("--sweep-steps", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    alerts = load_stream(args.scenario)
    patterns = load_patterns(args.patterns)
    m = metrics(run(alerts, patterns, {}), alerts)
    print(f"SCENARIO: {m['alerts']} alerts, {m['inj_alerts']} injected attack alerts\n")
    print("DEFAULT SETTINGS")
    print(f"  analyst items: {m['items']} (escalate {m['escalate']}, investigate {m['investigate']}) "
          f"-> {m['reduction']:.1%} fewer than raw alerts")
    print(f"  safety recall (injected alerts surfaced): {m['safety_recall']:.1%}   "
          f"stage recall: {'n/a' if m['stage_recall'] != m['stage_recall'] else format(m['stage_recall'], '.1%')}   "
          f"false-alarm items: {m['false_alarm_items']}")
    for lab, (ok, tot, dec) in m["runs"].items():
        print(f"  injected run '{lab}': {ok}/{tot} alerts surfaced, incident decision(s): {', '.join(dec)}")

    sev = severity_baseline(alerts)
    full = [r for r in sev if r["safety_recall"] >= 0.999]
    if full:
        b = max(full, key=lambda r: r["level>="])
        print(f"\nBASELINE severity filter: to keep ALL injected alerts you must show level >= {b['level>=']}: "
              f"{b['items']} items = {b['reduction']:.1%} reduction")

    rows = []
    for i in range(args.sweep_steps):
        thr = i / (args.sweep_steps - 1)
        mm = metrics(run(alerts, patterns, {"investigate": thr, "escalate": max(0.6, thr)}), alerts)
        rows.append({"investigate_threshold": round(thr, 2), "items": mm["items"], "reduction": mm["reduction"],
                     "safety_recall": mm["safety_recall"], "stage_recall": mm["stage_recall"],
                     "false_alarm_items": mm["false_alarm_items"]})
    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "severity_baseline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sev[0].keys())); w.writeheader(); w.writerows(sev)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([r["reduction"] for r in rows], [r["safety_recall"] for r in rows], "o-", label="engine (investigate threshold sweep)")
    ax.plot([r["reduction"] for r in sev], [r["safety_recall"] for r in sev], "s--", color="gray", label="severity filter (level >= L)")
    ax.plot([m["reduction"]], [m["safety_recall"]], "r*", ms=14, label="engine default settings")
    ax.set_xlabel("Reduction (1 - analyst items / raw alerts)"); ax.set_ylabel("Safety recall (injected attack alerts surfaced)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Prototype on scenario ({m['alerts']} alerts, {m['inj_alerts']} injected)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "prototype_curve.png"), dpi=150)
    print(f"\nwrote {args.out}/sweep.csv, severity_baseline.csv, prototype_curve.png")
    print("\nSWEEP")
    pct = lambda v: "n/a" if v != v else f"{v:.0%}"
    for r in rows:
        print(f"  investigate>={r['investigate_threshold']:.1f}: items {r['items']:>4}  reduction {r['reduction']:.1%}  "
              f"safety recall {pct(r['safety_recall'])}  stage recall {pct(r['stage_recall'])}  false alarms {r['false_alarm_items']}")

if __name__ == "__main__":
    main()
