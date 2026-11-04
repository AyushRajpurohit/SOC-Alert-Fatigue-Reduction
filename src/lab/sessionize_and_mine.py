"""
sessionize_and_mine.py — turns joined_runN.csv files (alert + assigned
ground-truth stage) into per-session tactic sequences, then mines
frequent sequential patterns (a small, dependency-free PrefixSpan-style
miner — no pip install needed).

Usage:
    python3 sessionize_and_mine.py joined_run1.csv joined_run2.csv ... \
        --min-support 0.4 --max-len 4

Session definition: each input file IS one session (one attack-chain run
against the victim). This is the simplest possible sessionization and
matches your lab setup (one entity = one agent = one chain per run).
When you move to real/noisy data (institute data, or several concurrent
attackers), sessionize by (agent, time-gap) instead — see
`sessionize_by_gap()` below, included but unused for now.

Mining approach: enumerates candidate order-preserving subsequences (not
necessarily contiguous) up to --max-len tokens from each session, then
counts how many sessions each candidate occurs in as a subsequence. This
is equivalent in spirit to PrefixSpan for short sequences over a small
alphabet (a handful of MITRE tactics) — fine for lab-scale data. If you
move to much longer real sessions, swap in the `prefixspan` PyPI package
instead (same output shape: support count + pattern).
"""

import argparse
import csv
from itertools import combinations
from datetime import datetime, timedelta

from canonical_tactics import get_canonical_tactic, unmapped_rule_ids


def load_joined(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def rows_to_tactic_sequence(rows, dedup_consecutive=True, use_canonical=False):
    """
    use_canonical=False (default, unchanged behavior): expands each alert
    to its full raw mitre_tactics list, as Wazuh reports it. Vulnerable to
    the T1078-style auto-expansion artifact.

    use_canonical=True: maps each alert's rule_id to exactly ONE tactic via
    canonical_tactics.RULE_TO_TACTIC, excluding pure session-bookkeeping
    alerts entirely. This is the corrected approach — use it for any real
    comparison between sessions (scoring, backbone-matching, mining).
    """
    seq = []
    for r in rows:
        if use_canonical:
            tactic = get_canonical_tactic(r["rule_id"], r["mitre_tactics"])
            tactics = [tactic] if tactic else []
        else:
            tactics = [t for t in r["mitre_tactics"].split(";") if t]
        for t in tactics:
            if dedup_consecutive and seq and seq[-1] == t:
                continue
            seq.append(t)
    return seq


def sessionize_by_gap(alert_rows, gap_seconds=120):
    """Alternative sessionizer for noisy/real data: group consecutive
    alerts from the same agent into a session whenever the time gap
    exceeds `gap_seconds`. Not used on the lab data (each run is already
    one clean session) but ready for institute/multi-host data."""
    def parse_ts(ts):
        ts = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        return datetime.fromisoformat(ts)

    sessions, current, last_ts = [], [], None
    for r in sorted(alert_rows, key=lambda r: r["timestamp"]):
        ts = parse_ts(r["timestamp"])
        if last_ts is not None and (ts - last_ts) > timedelta(seconds=gap_seconds):
            if current:
                sessions.append(current)
            current = []
        current.append(r)
        last_ts = ts
    if current:
        sessions.append(current)
    return sessions


def is_subsequence(pattern, seq):
    """True if `pattern` occurs in `seq` in order (not necessarily
    contiguous). Classic iterator-consuming idiom."""
    it = iter(seq)
    return all(tok in it for tok in pattern)


def lcs2(a, b):
    """Longest common subsequence of two token lists (standard DP)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # backtrack
    i, j, out = n, m, []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(out))


def longest_common_backbone(sessions):
    """Longest subsequence common to ALL sessions, via pairwise LCS
    reduction. Efficient (polynomial), unlike enumerating every candidate
    — use this to answer 'what pattern holds across every run', rather
    than mine_patterns() with a large --max-len (combinatorial blowup)."""
    backbone = sessions[0]
    for seq in sessions[1:]:
        backbone = lcs2(backbone, seq)
    return backbone


def mine_patterns(sessions, max_len, min_support_count):
    """Enumerate candidate subsequences (length 1..max_len) from every
    session, then count true support (subsequence containment) across
    all sessions. Returns [(support_count, pattern_tuple), ...]."""
    candidates = set()
    for seq in sessions:
        n = len(seq)
        for length in range(1, min(max_len, n) + 1):
            for idx in combinations(range(n), length):
                candidates.add(tuple(seq[i] for i in idx))

    results = []
    for pat in candidates:
        support = sum(1 for seq in sessions if is_subsequence(pat, seq))
        if support >= min_support_count:
            results.append((support, pat))

    results.sort(key=lambda x: (-x[0], -len(x[1]), x[1]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("joined_files", nargs="+", help="joined_runN.csv files, one per session/run")
    ap.add_argument("--min-support", type=float, default=0.4,
                     help="Minimum fraction of sessions a pattern must appear in (default 0.4)")
    ap.add_argument("--max-len", type=int, default=4,
                     help="Maximum pattern length to search (default 4)")
    ap.add_argument("--top-k", type=int, default=15,
                     help="Show top K most frequent patterns (default 15)")
    ap.add_argument("--backbone", action="store_true",
                     help="Also compute the single longest subsequence common "
                          "to ALL sessions (efficient; use instead of a large "
                          "--max-len, which blows up combinatorially)")
    ap.add_argument("--raw-tactics", action="store_true",
                     help="Use Wazuh's raw (auto-expanded) mitre_tactics field "
                          "instead of the canonical rule_id lookup. Mainly for "
                          "reproducing/comparing against the pre-fix behavior; "
                          "canonical mapping is the default and the correct "
                          "choice for real comparisons.")
    args = ap.parse_args()
    use_canonical = not args.raw_tactics

    sessions = []
    print(f"Loading {len(args.joined_files)} session(s) "
          f"(using {'canonical rule->tactic mapping' if use_canonical else 'RAW Wazuh tactic expansion'}):\n")
    for path in args.joined_files:
        rows = load_joined(path)
        seq = rows_to_tactic_sequence(rows, use_canonical=use_canonical)
        sessions.append(seq)
        print(f"  {path}")
        print(f"    {len(rows)} alerts -> {len(seq)}-tactic sequence: {' -> '.join(seq)}")
    if use_canonical and unmapped_rule_ids():
        print(f"\n  NOTE: unmapped rule_ids seen (used raw-tactic fallback): "
              f"{unmapped_rule_ids()}")
        print("  Add these to canonical_tactics.RULE_TO_TACTIC once you know "
              "their intended tactic.")
    print()

    if len(sessions) < 2:
        print("Need at least 2 sessions to find meaningful patterns. "
              "Run the attack chain a few more times.")
        return

    min_support_count = max(2, int(args.min_support * len(sessions)))
    print(f"Mining sequential patterns (min support = {min_support_count} of "
          f"{len(sessions)} sessions, i.e. >= {args.min_support:.0%}, max length {args.max_len})\n")

    patterns = mine_patterns(sessions, args.max_len, min_support_count)

    print(f"Top {min(args.top_k, len(patterns))} frequent tactic sub-sequences:\n")
    for support, pattern in patterns[:args.top_k]:
        pct = support / len(sessions)
        print(f"  support={support}/{len(sessions)} ({pct:.0%})  {' -> '.join(pattern)}")

    if not patterns:
        print("  (none found at this support threshold — try --min-support lower, "
              "e.g. 0.2, or collect more runs)")

    print(f"\nTotal distinct patterns found: {len(patterns)}")

    if args.backbone:
        backbone = longest_common_backbone(sessions)
        print(f"\nLongest common backbone across all {len(sessions)} sessions "
              f"({len(backbone)} tactics):")
        print(f"  {' -> '.join(backbone)}")

    print("\nNext: this is your baseline evidence that sequential mining recovers "
          "the intended attack-chain structure from raw alert noise. Compare this "
          "against a single-rule if_matched_sid baseline on a chain variant to "
          "show the learned model catches what the static rule misses.")


if __name__ == "__main__":
    main()
