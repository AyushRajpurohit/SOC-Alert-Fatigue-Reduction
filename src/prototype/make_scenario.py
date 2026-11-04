"""
make_scenario.py -- build a realistic TEST STREAM: background noise + injected lab attack runs with known truth.

WHY: your lab runs are clean (nearly every alert belongs to the attack), so they cannot show noise reduction.
Standard practice is attack INJECTION: take real attack alerts, shift them in time, and hide them in a busy stream.
We then know exactly which alerts are the attack, so we can measure how much noise the engine removes (reduction)
and whether it still surfaces every injected attack alert (safety recall).

BACKGROUND (pick one)
  --background-file F      real alerts (e.g. your institute alerts.json); injected runs are placed inside its time span
  --background-synthetic N N synthetic benign-looking alerts over --agents hosts. ILLUSTRATIVE ONLY: its rule mix is my
                           assumption, so numbers on synthetic noise show the mechanics, not real-world performance.

INJECTION:  --inject label=path/to/joined_run.csv  (repeat), placed at random times; their agent is kept (e.g. 'victim').

Usage:
  python make_scenario.py --inject variant=data/lab/joined/joined_variant.csv --inject exfil2=data/lab/joined/joined_exfil2.csv \
      --background-synthetic 3000 --out data/scenarios/synthetic.jsonl
"""
import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import _common_path  # noqa: F401
from loader import parse_ts
from soc_common import Alert, load_stream, wazuh_joined_adapter

# (rule_id, level, description, raw tactics) -- approximate Wazuh rules; levels are an ASSUMPTION for illustration
POOL = {
    "5501": (3, "PAM: Login session opened.", "Initial Access;Persistence;Privilege Escalation;Defense Evasion"),
    "5502": (3, "PAM: Login session closed.", ""),
    "5402": (3, "Successful sudo to ROOT executed.", "Privilege Escalation;Defense Evasion"),
    "5503": (5, "PAM: User login failed.", "Credential Access"),
    "5760": (5, "sshd: authentication failed.", "Credential Access;Lateral Movement"),
    "5710": (5, "sshd: Attempt to login using a non-existent user", "Credential Access"),
    "5715": (3, "sshd: authentication success.", "Initial Access;Persistence;Privilege Escalation;Defense Evasion"),
    "2502": (10, "syslog: User missed the password more than one time", "Credential Access"),
    "2832": (3, "Crontab entry changed.", "Persistence"),
    "5405": (5, "Unauthorized user attempted to use sudo.", "Privilege Escalation;Defense Evasion"),
}
EPISODES = [  # (weight, rule sequence, seconds between alerts)
    (40, ["5501", "5502"], (1, 30)),
    (12, ["5402"], (1, 2)),
    (18, ["5503"], (1, 2)),
    (10, ["5760", "5760"], (2, 10)),
    (6,  ["5715", "5501", "5502"], (1, 20)),
    (5,  ["5760"] * 8, (1, 4)),                    # benign-looking scanner noise
    (2,  ["5710"] * 5 + ["2502"], (1, 4)),         # noisy but benign-ish bursts, includes a level-10 rule
]


# HARD NOISE: normal admin activity that shares tactics with the attack chain (Persistence, Defense Evasion), incl.
# look-alikes of its first steps. Without these the easy background can never produce those tactics, so any incident
# containing them is trivially an attack -- an artefact of my generator, not a property of the method.
HARD_EXTRA = [
    (6, ["2832"], (1, 2)),                                   # admin edits a crontab
    (6, ["5405"], (1, 2)),                                   # a failed sudo
    (5, ["5715", "5501", "2832", "5502"], (1, 20)),          # login, cron edit, logout
    (4, ["5503", "5715", "5405"], (2, 30)),                  # typo, login, failed sudo  (look-alike of the attack start)
    (3, ["5760", "5760", "5715", "5501", "2832", "5405"], (1, 15)),   # worst case: long benign look-alike
]


def synthetic_background(n, agents, start, span_s, rng, hard=False):
    episodes = EPISODES + (HARD_EXTRA if hard else [])
    out, weights = [], [e[0] for e in episodes]
    while len(out) < n:
        _, rules, gap = rng.choices(episodes, weights)[0]
        t = start + timedelta(seconds=rng.uniform(0, span_s))
        agent = f"host{rng.randint(1, agents):02d}"
        for r in rules:
            lvl, desc, tac = POOL[r]
            out.append(Alert("wazuh", t.isoformat(), agent, r, desc, lvl, [x for x in tac.split(";") if x], []))
            t += timedelta(seconds=rng.uniform(*gap))
    return out[:n]


def place_runs(bg, runs, t0, span_s, rng, mix_hosts=False):
    """Inject attack runs (dict label -> list of Alert) at random times inside [t0, t0+span]. The runs are COPIED, so
    the same run can be placed again with another seed. Returns the merged, time-sorted stream with .seq set."""
    import copy
    alerts = list(bg)
    hosts = sorted({a.entity for a in bg}) if mix_hosts else []
    for label, run in runs.items():
        host = rng.choice(hosts) if hosts else None            # mix_hosts: the attacked host is a busy background host
        r0 = parse_ts(run[0].timestamp)
        dur = (parse_ts(run[-1].timestamp) - r0).total_seconds()
        start = t0 + timedelta(seconds=rng.uniform(0, max(1.0, span_s - dur - 60)))
        for a in run:
            b = copy.copy(a)
            b.timestamp = (start + (parse_ts(a.timestamp) - r0)).isoformat()
            b.truth = label
            if host:
                b.entity = host
            alerts.append(b)
    alerts.sort(key=lambda a: parse_ts(a.timestamp))
    for i, a in enumerate(alerts):
        a.seq = i
    return alerts


def fresh_synthetic(n, seed, agents=8, hours=6.0, hard=False):
    """One new synthetic background (alerts, t0, span), time-sorted with .seq set (for independent test repeats)."""
    rng = random.Random(seed)
    t0 = datetime(2026, 10, 20, 8, 0, 0, tzinfo=timezone.utc)
    bg = synthetic_background(n, agents, t0, hours * 3600, rng, hard)
    bg.sort(key=lambda a: parse_ts(a.timestamp))
    for j, a in enumerate(bg):
        a.seq = j
    return bg, t0, hours * 3600


def get_backgrounds(synthetic_n=None, file=None, seed=0, agents=8, hours=6.0, hard=False):
    """Three NON-OVERLAPPING backgrounds [(alerts, t0, span_s), ...] for: A = benign statistics (counts),
    B = training the classifier, C = testing. Real file: split by TIME (50% / 25% / 25%). Synthetic: three seeds."""
    if file:
        bg = load_stream(file)
        cuts = [0, len(bg) // 2, 3 * len(bg) // 4, len(bg)]
        out = []
        for i in range(3):
            part = bg[cuts[i]:cuts[i + 1]]
            t0 = parse_ts(part[0].timestamp)
            out.append((part, t0, (parse_ts(part[-1].timestamp) - t0).total_seconds()))
        return out
    out = []
    for i in range(3):
        rng = random.Random(seed * 100 + i)
        t0 = datetime(2026, 10, 10 + i, 8, 0, 0, tzinfo=timezone.utc)
        bg = synthetic_background(synthetic_n, agents, t0, hours * 3600, rng, hard)
        bg.sort(key=lambda a: parse_ts(a.timestamp))           # the engine needs TIME-ORDERED input
        for j, a in enumerate(bg):
            a.seq = j
        out.append((bg, t0, hours * 3600))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", action="append", default=[], help="label=path (joined csv)")
    ap.add_argument("--background-file")
    ap.add_argument("--background-synthetic", type=int)
    ap.add_argument("--hard-noise", action="store_true",
                    help="synthetic only: add benign cron edits / failed sudo / attack look-alikes (a harder, fairer background)")
    ap.add_argument("--mix-hosts", action="store_true",
                    help="put each injected attack on a random BACKGROUND host so benign noise mixes into its incident")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--hours", type=float, default=6.0, help="synthetic span; use e.g. 1 for a short live-demo stream")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.background_file:
        bg = load_stream(args.background_file)
        t0 = parse_ts(bg[0].timestamp)
        span = (parse_ts(bg[-1].timestamp) - t0).total_seconds()
    elif args.background_synthetic:
        bg, t0, span = fresh_synthetic(args.background_synthetic, args.seed, args.agents, args.hours, args.hard_noise)
    else:
        ap.error("need --background-file or --background-synthetic")

    runs = {}
    for kv in args.inject:
        label, path = kv.split("=", 1)
        runs[label] = wazuh_joined_adapter(path).alerts
    alerts = place_runs(bg, runs, t0, span, rng, args.mix_hosts)
    for label in runs:
        mine = [a for a in alerts if a.truth == label]
        print(f"injected {label}: {len(mine)} alerts starting {mine[0].timestamp[11:19]} on host {mine[0].entity}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for a in alerts:
            f.write(json.dumps(asdict(a)) + "\n")
    inj = sum(1 for a in alerts if a.truth)
    print(f"wrote {args.out}: {len(alerts)} alerts ({inj} injected attack alerts, {len(alerts) - inj} background)")


if __name__ == "__main__":
    main()
