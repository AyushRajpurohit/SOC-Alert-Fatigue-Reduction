"""
loader.py — normalizes Wazuh alerts.json into the project's common schema,
and joins them against a ground_truth.csv attack-chain log.

Usage:
    python3 loader.py --alerts chain_alerts.json --ground-truth ground_truth.csv --out joined.csv

Matching strategy:
    Strict containment (alert timestamp falls inside [start,end] of a stage)
    is tried first. Any alert that doesn't land inside any stage window is
    instead assigned to its NEAREST stage by time, with the gap recorded in
    seconds — this makes the join robust to clock drift between the
    attacker and victim machines, which is common with VMs that have been
    suspended/resumed. Always check the 'match_type' and 'gap_seconds'
    columns in the output before trusting a label.
"""

import argparse
import json
import csv
from datetime import datetime, timezone


def parse_ts(ts: str) -> datetime:
    """Parse ISO8601 timestamps from either Wazuh (+0000) or the attack
    script (Z) into timezone-aware datetimes."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    else:
        # Wazuh style: 2026-10-04T15:54:12.032+0000 -> +00:00
        if len(ts) >= 5 and ts[-5] in "+-" and ts[-3] != ":":
            ts = ts[:-2] + ":" + ts[-2:]
    return datetime.fromisoformat(ts)


def load_alerts(path):
    """Flatten each Wazuh alert JSON line into the project's common schema."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            rule = a.get("rule", {})
            agent = a.get("agent", {})
            data = a.get("data", {})
            mitre = rule.get("mitre", {})
            records.append({
                "timestamp": a.get("timestamp"),
                "ts_dt": parse_ts(a["timestamp"]),
                "alert_id": a.get("id"),
                "agent_id": agent.get("id"),
                "agent_name": agent.get("name"),
                "rule_id": rule.get("id"),
                "rule_level": rule.get("level"),
                "rule_description": rule.get("description"),
                "mitre_ids": ";".join(mitre.get("id", [])),
                "mitre_tactics": ";".join(mitre.get("tactic", [])),
                "mitre_techniques": ";".join(mitre.get("technique", [])),
                "srcip": data.get("srcip", ""),
                "dstuser": data.get("dstuser", data.get("srcuser", "")),
                "full_log": a.get("full_log", ""),
            })
    records.sort(key=lambda r: r["ts_dt"])
    return records


def load_ground_truth(path):
    stages = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            stages.append({
                "stage": row["stage"],
                "technique_id": row["technique_id"],
                "tactic": row["tactic"],
                "technique": row["technique"],
                "start_dt": parse_ts(row["start_time_utc"]),
                "end_dt": parse_ts(row["end_time_utc"]),
                "notes": row.get("notes", ""),
            })
    stages.sort(key=lambda s: s["start_dt"])
    return stages


def assign_stage(alert_dt, stages):
    """Strict containment first; otherwise nearest stage by time gap."""
    for s in stages:
        if s["start_dt"] <= alert_dt <= s["end_dt"]:
            return s["stage"], "contained", 0.0

    best, best_gap = None, None
    for s in stages:
        if alert_dt < s["start_dt"]:
            gap = (s["start_dt"] - alert_dt).total_seconds()
        elif alert_dt > s["end_dt"]:
            gap = (alert_dt - s["end_dt"]).total_seconds()
        else:
            gap = 0.0
        if best_gap is None or gap < best_gap:
            best, best_gap = s["stage"], gap
    return best, "nearest", round(best_gap, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", default="joined.csv")
    args = ap.parse_args()

    alerts = load_alerts(args.alerts)
    stages = load_ground_truth(args.ground_truth)

    print(f"Loaded {len(alerts)} alerts, {len(stages)} ground-truth stages.\n")

    fieldnames = [
        "timestamp", "agent_name", "rule_id", "rule_level", "rule_description",
        "mitre_tactics", "mitre_techniques", "srcip", "dstuser",
        "assigned_stage", "match_type", "gap_seconds",
    ]
    rows_out = []
    stage_hit_count = {s["stage"]: 0 for s in stages}

    for a in alerts:
        stage, match_type, gap = assign_stage(a["ts_dt"], stages)
        stage_hit_count[stage] = stage_hit_count.get(stage, 0) + 1
        rows_out.append({
            "timestamp": a["timestamp"],
            "agent_name": a["agent_name"],
            "rule_id": a["rule_id"],
            "rule_level": a["rule_level"],
            "rule_description": a["rule_description"],
            "mitre_tactics": a["mitre_tactics"],
            "mitre_techniques": a["mitre_techniques"],
            "srcip": a["srcip"],
            "dstuser": a["dstuser"],
            "assigned_stage": stage,
            "match_type": match_type,
            "gap_seconds": gap,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    # --- Report ---
    print("Alerts per stage:")
    for s in stages:
        n = stage_hit_count.get(s["stage"], 0)
        flag = "  <-- NO ALERTS" if n == 0 else ""
        print(f"  {s['stage']:<16} {n:>3} alert(s){flag}")

    nearest_matches = [r for r in rows_out if r["match_type"] == "nearest"]
    if nearest_matches:
        print(f"\n{len(nearest_matches)} alert(s) fell OUTSIDE their assigned "
              f"stage window and were matched by nearest time instead:")
        for r in nearest_matches:
            print(f"  [{r['timestamp']}] rule {r['rule_id']} -> "
                  f"{r['assigned_stage']} (gap {r['gap_seconds']}s)")
        print("\nThis usually means clock drift between the attacker and "
              "victim machines, or a stage taking longer than expected. "
              "Sync clocks (ntpdate / chrony) before your next run.")

    print(f"\nTactic sequence (mined-ready, in time order):")
    seq = [t for r in rows_out for t in r["mitre_tactics"].split(";") if t]
    # de-duplicate consecutive repeats for a cleaner session view
    dedup = []
    for t in seq:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    print(f"  {' -> '.join(dedup)}")

    print(f"\nWrote joined dataset to: {args.out}")


if __name__ == "__main__":
    main()
