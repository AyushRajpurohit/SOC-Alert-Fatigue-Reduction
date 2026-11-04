"""
experiment.py -- does LEARNING beat grouping and hand-set rules? (the comparison the project needs)

PROTOCOL (no peeking)
  train attacks : baseline runs (original chain)                   -> the learned scorer's only attack knowledge
  test attacks  : variant and second chain (exfil2), never trained on, injected at random places in background C
  backgrounds   : A = benign transition counts, B = classifier training negatives, C = test (non-overlapping)
  approaches    : grouping only | severity filter (best case for it) | hand-set rules | LEARNED scorer
  repeats       : the test attacks are re-placed at random R times; we report mean +/- sd
Default operating points use thresholds fixed BEFORE testing (rules: hand-set; learned: derived on training data).
The sweeps show the whole trade-off curve (thresholds varied on the test data = optimistic, shown for shape only).

  python experiment.py --baseline joined_run{1..5}.csv --test variant=joined_variant.csv --test exfil2=joined_exfil2.csv \
      --background-synthetic 3000 --repeats 10 --out results/proto_experiment
SYNTHETIC background => numbers show mechanics only. Use --background-file with real alerts for real evidence.
"""
import argparse
import csv
import os
import random

import numpy as np

import _common_path  # noqa: F401
from soc_common import wazuh_joined_adapter
from make_scenario import get_backgrounds, place_runs, fresh_synthetic
from learned import train
from socflow import Engine, AllSurface, RuleScorer
from evaluate import metrics, severity_baseline


def run(alerts, patterns, scorer, cfg=None):
    eng = Engine(patterns, cfg or {}, scorer)
    for a in alerts:
        eng.feed(a)
    eng.finish()
    return metrics(eng, alerts)


def ms(vals, pct=False):
    v = np.array(vals, float)
    f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.1f}")
    return f"{f(v.mean())} +/- {f(v.std())}" if pct else f"{f(v.mean())} +/- {f(v.std())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--test", action="append", required=True, help="name=path (joined csv), repeat")
    ap.add_argument("--background-synthetic", type=int)
    ap.add_argument("--background-file")
    ap.add_argument("--hard-noise", action="store_true", help="synthetic background with benign cron/sudo look-alikes")
    ap.add_argument("--mix-hosts", action="store_true", help="attacks land on busy background hosts (recommended)")
    ap.add_argument("--inv-fp", type=float, default=0.01, help="false-alarm budget for INVESTIGATE (share of normal incidents)")
    ap.add_argument("--esc-fp", type=float, default=0.001, help="false-alarm budget for ESCALATE")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/proto_experiment")
    args = ap.parse_args()
    if not (args.background_synthetic or args.background_file):
        ap.error("need --background-synthetic or --background-file")
    os.makedirs(args.out, exist_ok=True)

    train_runs = {f"run{i + 1}": wazuh_joined_adapter(p).alerts for i, p in enumerate(args.baseline)}
    tests = {}
    for kv in args.test:
        name, path = kv.split("=", 1)
        tests[name] = wazuh_joined_adapter(path).alerts
    bgA, bgB, bgC = get_backgrounds(args.background_synthetic, args.background_file, args.seed, hard=args.hard_noise)
    print(f"background sizes (A counts / B train / C test): {len(bgA[0])} / {len(bgB[0])} / {len(bgC[0])} alerts"
          f"{'   [SYNTHETIC: mechanics only]' if args.background_synthetic else ''}\n")
    scorer, patterns, report = train(train_runs, bgA, bgB, seed=args.seed, mix_hosts=args.mix_hosts,
                                     inv_fp=args.inv_fp, esc_fp=args.esc_fp)

    NO_GUARD = {"late_stage": ()}                       # ablation: switch the late-stage guard rail OFF
    approaches = {"grouping only": (AllSurface(), {}), "hand-set rules": (RuleScorer(), {}),
                  "LEARNED scorer": (scorer, {}),
                  "rules, no guard rail": (RuleScorer(), NO_GUARD), "LEARNED, no guard rail": (scorer, NO_GUARD)}
    inv_logit, esc_logit = report["investigate_logit"], report["escalate_logit"]
    grid_l = [1 / (1 + np.exp(-z)) for z in np.linspace(inv_logit - 12, inv_logit + 6, 25)]
    grid_r = [i / 20 for i in range(21)]

    default, sweep_l, sweep_r, sev_rows = {k: [] for k in approaches}, [], [], []
    for r in range(args.repeats):
        if args.background_synthetic:                    # a NEW background every repeat => genuinely independent repeats
            bg_r = fresh_synthetic(args.background_synthetic, args.seed * 7919 + 100 + r, hard=args.hard_noise)
        else:                                            # real file: one fixed test window, only the placement varies
            bg_r = bgC
        scen = place_runs(bg_r[0], tests, bg_r[1], bg_r[2], random.Random(args.seed * 7919 + r), args.mix_hosts)
        for name, (sc, extra) in approaches.items():
            default[name].append(run(scen, patterns, sc, extra))
        sweep_r.append([run(scen, patterns, RuleScorer(), {"investigate": t, "escalate": max(0.6, t)}) for t in grid_r])
        sweep_l.append([run(scen, patterns, scorer, {"investigate": t, "escalate": max(t, scorer.escalate)}) for t in grid_l])
        sev_rows.append(severity_baseline(scen))
        print(f"repeat {r + 1}/{args.repeats} done", flush=True)

    # ---------------- table at default operating points ----------------
    n_alerts = default["grouping only"][0]["alerts"]
    print(f"\nTEST STREAM: {n_alerts} alerts, {default['grouping only'][0]['inj_alerts']} injected attack alerts "
          f"({', '.join(tests)}), mean +/- sd over {args.repeats} placements\n")
    hdr = f"{'approach':<23}{'analyst items':>20}{'reduction':>20}{'safety recall':>18}{'false-alarm items':>22}"
    print(hdr)
    rows = []
    for name in approaches:
        M = default[name]
        print(f"{name:<23}{ms([m['items'] for m in M]):>20}{ms([m['reduction'] for m in M], True):>20}"
              f"{ms([m['safety_recall'] for m in M], True):>18}{ms([m['false_alarm_items'] for m in M]):>22}")
        rows.append({"approach": name, "items": np.mean([m["items"] for m in M]),
                     "reduction": np.mean([m["reduction"] for m in M]),
                     "safety_recall": np.mean([m["safety_recall"] for m in M]),
                     "false_alarm_items": np.mean([m["false_alarm_items"] for m in M])})
    # severity filter: best case = the strictest level that still keeps ALL injected alerts (oracle)
    best = []
    for sv in sev_rows:
        ok = [x for x in sv if x["safety_recall"] >= 0.999]
        best.append(max(ok, key=lambda x: x["reduction"]) if ok else sv[0])
    print(f"{'severity filter*':<23}{ms([b['items'] for b in best]):>20}{ms([b['reduction'] for b in best], True):>20}"
          f"{ms([b['safety_recall'] for b in best], True):>18}{'n/a (alerts, not incidents)':>22}")
    print("  * strictest level that keeps every injected alert -- chosen with the answer in hand, so favourable to the filter")
    rows.append({"approach": "severity filter (oracle level)", "items": np.mean([b["items"] for b in best]),
                 "reduction": np.mean([b["reduction"] for b in best]),
                 "safety_recall": np.mean([b["safety_recall"] for b in best]), "false_alarm_items": float("nan")})

    print("\nPER-ATTACK: share of each test attack's alerts that end up in a surfaced incident (mean over placements)")
    for name in approaches:
        parts = []
        for t in tests:
            parts.append(f"{t} {np.mean([m['runs'][t][0] / m['runs'][t][1] for m in default[name]]):.0%}")
        print(f"  {name:<23}" + "   ".join(parts))

    print("\nSWEEP SUMMARY: best reduction among thresholds whose MEAN safety recall reaches the target")
    print("  (thresholds picked with the test answer in hand => optimistic for both; shows what each approach CAN do)")
    for lab, sw in (("hand-set rules", sweep_r), ("LEARNED scorer", sweep_l)):
        parts = []
        for target in (0.99, 0.90, 0.75):
            best = None
            for j in range(len(sw[0])):
                rec = np.mean([sw[r][j]["safety_recall"] for r in range(args.repeats)])
                if rec >= target:
                    its = np.mean([sw[r][j]["items"] for r in range(args.repeats)])
                    red = np.mean([sw[r][j]["reduction"] for r in range(args.repeats)])
                    if best is None or its < best[0]:
                        best = (its, red, rec)
            parts.append(f"recall>={target:.0%}: " + (f"{best[0]:.0f} items ({best[1]:.1%} reduction)" if best else "not reached"))
        print(f"  {lab:<16}" + "   ".join(parts))

    with open(os.path.join(args.out, "default_points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---------------- curves ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, sw, mk in (("hand-set rules (sweep)", sweep_r, "s-"), ("LEARNED scorer (sweep)", sweep_l, "o-")):
        xs = [np.mean([sw[r][j]["reduction"] for r in range(args.repeats)]) for j in range(len(sw[0]))]
        ys = [np.mean([sw[r][j]["safety_recall"] for r in range(args.repeats)]) for j in range(len(sw[0]))]
        ax.plot(xs, ys, mk, ms=4, label=lab)
    sx = [np.mean([sev_rows[r][L]["reduction"] for r in range(args.repeats)]) for L in range(16)]
    sy = [np.mean([sev_rows[r][L]["safety_recall"] for r in range(args.repeats)]) for L in range(16)]
    ax.plot(sx, sy, "x--", color="gray", label="severity filter (level >= L)")
    for name, mk in (("grouping only", "D"), ("hand-set rules", "s"), ("LEARNED scorer", "*"), ("LEARNED, no guard rail", "v")):
        ax.plot([np.mean([m["reduction"] for m in default[name]])], [np.mean([m["safety_recall"] for m in default[name]])],
                mk, ms=13 if mk == "*" else 8, mec="k", label=f"{name} (default)")
    ax.set_xlabel("Reduction (1 - analyst items / raw alerts)"); ax.set_ylabel("Safety recall (injected attack alerts surfaced)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"Unseen attacks in background C ({args.repeats} placements)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "learned_vs_baselines.png"), dpi=150)
    print(f"\nwrote {args.out}/default_points.csv and learned_vs_baselines.png")


if __name__ == "__main__":
    main()
