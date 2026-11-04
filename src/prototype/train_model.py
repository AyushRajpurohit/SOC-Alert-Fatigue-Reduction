"""
train_model.py -- train the LEARNED scorer and save it as model.json (used by run_engine.py by default).

  python src/prototype/train_model.py --baseline data/lab/joined/joined_run{1,2,3,4,5}.csv \
      --background-synthetic 3000 --out src/prototype/model.json
  (real background:  --background-file <institute alerts.json>  -> split by TIME into A/B/C, see make_scenario.get_backgrounds)

Only the baseline runs are used for training. Keep the variant and exfil2 OUT of training; they are the unseen test.
"""
import argparse
import _common_path  # noqa: F401
from soc_common import wazuh_joined_adapter
from make_scenario import get_backgrounds
from learned import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--background-synthetic", type=int)
    ap.add_argument("--background-file")
    ap.add_argument("--hard-noise", action="store_true", help="synthetic background with benign cron/sudo look-alikes")
    ap.add_argument("--mix-hosts", action="store_true", help="training attacks land on busy background hosts")
    ap.add_argument("--inv-fp", type=float, default=0.01, help="false-alarm budget for INVESTIGATE")
    ap.add_argument("--esc-fp", type=float, default=0.001, help="false-alarm budget for ESCALATE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not (args.background_synthetic or args.background_file):
        ap.error("need --background-synthetic or --background-file")

    runs = {f"run{i + 1}": wazuh_joined_adapter(p).alerts for i, p in enumerate(args.baseline)}
    bgs = get_backgrounds(args.background_synthetic, args.background_file, args.seed, hard=args.hard_noise)
    scorer, patterns, _ = train(runs, bgs[0], bgs[1], seed=args.seed, mix_hosts=args.mix_hosts,
                                  inv_fp=args.inv_fp, esc_fp=args.esc_fp)
    scorer.save(args.out, patterns)
    print(f"pattern library: {' -> '.join(patterns[0]['tokens'])}\nwrote {args.out}")


if __name__ == "__main__":
    main()
