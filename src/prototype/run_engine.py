"""
run_engine.py -- run the prototype on an alert stream and (optionally) feed the live dashboard.

INPUT (pick one)
  --input FILE     joined_*.csv, internal scenario .jsonl, or a raw Wazuh alerts.json  (replayed)
  --stdin          raw Wazuh JSON lines on standard input  (LIVE, see below)

LIVE DEMO (Wazuh VM -> your Mac), the engine reads whatever the VM appends to alerts.json:
  ssh <user>@<wazuh-vm> "sudo tail -n0 -F /var/ossec/logs/alerts/alerts.json" | python run_engine.py --stdin ...
  (path is the standard Wazuh location; check it on your install.)  Then start the attack from Kali.

DASHBOARD:   cd src/prototype/dashboard && python -m http.server 8000   ->   http://localhost:8000
The engine writes dashboard/state/queue.json; the page polls it every second.

REPLAY SPEED: --speed 20 plays event time 20x faster than real time (0 = as fast as possible).
"""
import argparse
import os
import sys
import time

import _common_path  # noqa: F401
from loader import parse_ts
from soc_common import load_stream, parse_line
from socflow import Engine, load_patterns, RuleScorer
from learned import LearnedScorer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input")
    g.add_argument("--stdin", action="store_true")
    ap.add_argument("--model", default=os.path.join(HERE, "model.json"),
                    help="learned scorer (default). Train it with train_model.py")
    ap.add_argument("--rules", action="store_true", help="use the HAND-SET rule scorer instead of the learned model")
    ap.add_argument("--patterns", default=os.path.join(HERE, "patterns.json"), help="pattern library for --rules")
    ap.add_argument("--state", default=os.path.join(HERE, "dashboard", "state", "queue.json"))
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--gap", type=int, default=120)
    ap.add_argument("--escalate", type=float, default=None, help="override the scorer's escalate threshold")
    ap.add_argument("--investigate", type=float, default=None, help="override the scorer's investigate threshold")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.rules and os.path.exists(args.model):
        scorer, patterns = LearnedScorer.load(args.model)
        print(f"scorer: LEARNED ({args.model})")
    else:
        if not args.rules:
            print(f"WARNING: {args.model} not found -> using the hand-set rules. Train with train_model.py.")
        if os.path.exists(args.patterns):
            patterns = load_patterns(args.patterns)
        elif os.path.exists(args.model):
            patterns = LearnedScorer.load(args.model)[1]          # reuse the pattern library stored with the model
        else:
            sys.exit("no pattern library: run train_model.py (or learn_patterns.py) first")
        scorer = RuleScorer()
        print("scorer: hand-set rules")
    cfg = {"gap": args.gap}
    if args.escalate is not None:
        cfg["escalate"] = args.escalate
    if args.investigate is not None:
        cfg["investigate"] = args.investigate
    eng = Engine(patterns, cfg, scorer)
    eng.write_state(args.state)
    if args.stdin:
        def gen():
            n = 0
            for line in sys.stdin:
                a = parse_line(line)
                if a is not None:
                    a.seq = n; n += 1
                    yield a
        stream = gen()
    else:
        stream = iter(load_stream(args.input))

    last_wall, prev_ts = 0.0, None
    for a in stream:
        ts = parse_ts(a.timestamp)
        if args.speed > 0 and prev_ts is not None and not args.stdin:
            time.sleep(min(2.0, max(0.0, (ts - prev_ts).total_seconds() / args.speed)))
        prev_ts = ts
        inc, prev = eng.feed(a)
        if not args.quiet and (prev is None or prev != inc.decision):
            print(f"{a.timestamp[11:19]}  {inc.entity:<10} incident #{inc.iid:<3} -> {inc.decision:<11} "
                  f"priority {inc.priority:.2f}  [{' > '.join(inc.tokens) or 'no tactics'}]  ({'; '.join(inc.reasons)})",
                  flush=True)
        if time.time() - last_wall > 0.25:
            eng.write_state(args.state); last_wall = time.time()
    eng.finish()
    eng.write_state(args.state)
    s = eng.stats()
    print(f"\nDONE: {s['alerts_seen']} alerts -> {s['incidents']} incidents | escalate {s['escalate']}, "
          f"investigate {s['investigate']}, demoted {s['demoted']} ({s['demoted_alerts']} alerts) | "
          f"analyst items {s['escalate'] + s['investigate']} = {s['reduction']:.1%} fewer than raw alerts")


if __name__ == "__main__":
    main()
