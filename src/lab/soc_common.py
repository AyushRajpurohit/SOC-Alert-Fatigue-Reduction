"""
soc_common.py -- one common record for every alert source, plus small adapters.

WHY THIS EXISTS
  Your pipeline (loader.py -> sessionize_and_mine.py) reads Wazuh-shaped CSV columns.
  To say "SIEM-agnostic" honestly, every source must be translated into the SAME small record
  before any mining/scoring happens. Downstream code then never sees source-specific fields.

  Alert    = one alert from any source (who, when, which detector, which techniques/tactics)
  Session  = alerts that belong together (a lab chain run, a GUIDE incident)

DESIGN CHOICES (all deliberately small)
  * Tokens are produced by a RESOLVER, not stored in the Alert, because the right token depends on the
    question: canonical tactic (your lab table), raw tactic expansion (to reproduce week-1 numbers),
    or parent technique ID (GUIDE-style).
  * Wazuh keeps using canonical_tactics.get_canonical_tactic, so lab behaviour is unchanged.
"""
import csv
from dataclasses import dataclass, field
from typing import List, Optional

from canonical_tactics import get_canonical_tactic


@dataclass
class Alert:
    source: str                      # "wazuh" | "guide" | ...
    timestamp: str
    entity: str                      # agent name (Wazuh) / org (GUIDE)
    detector_id: str                 # rule_id (Wazuh) / DetectorId (GUIDE)
    description: str = ""
    level: Optional[int] = None
    raw_tactics: List[str] = field(default_factory=list)     # as the source reports them
    techniques: List[str] = field(default_factory=list)      # PARENT technique IDs, e.g. "T1078"
    stage: Optional[str] = None                              # lab ground-truth stage, if known
    seq: int = -1                                            # position in the loaded stream
    truth: Optional[str] = None                              # label of an INJECTED attack run (scenarios only)


@dataclass
class Session:
    source: str
    sid: str
    alerts: List[Alert]
    ordered: bool = True             # False for sources without reliable order (e.g. GUIDE incidents as sets)
    label: Optional[str] = None      # analyst grade / attack label, if any


# --------------------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------------------
def wazuh_joined_adapter(path, sid=None) -> Session:
    """Reads a joined_runN.csv produced by loader.py (alert + assigned ground-truth stage)."""
    alerts = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            alerts.append(Alert(
                source="wazuh", timestamp=r["timestamp"], entity=r.get("agent_name", ""),
                detector_id=str(r["rule_id"]), description=r.get("rule_description", ""),
                level=int(r["rule_level"]) if r.get("rule_level") else None,
                raw_tactics=[t for t in r.get("mitre_tactics", "").split(";") if t],
                techniques=[],                       # joined CSV keeps technique NAMES only, not IDs
                stage=r.get("assigned_stage")))
    alerts.sort(key=lambda a: a.timestamp)
    return Session("wazuh", sid or path, alerts)


def wazuh_json_adapter(path, sid=None) -> Session:
    """Reads raw Wazuh alerts.json (one JSON per line) via loader.load_alerts; keeps technique IDs
    (T1078.004 -> T1078) so lab sessions can also be tokenised at technique level, like GUIDE."""
    from loader import load_alerts
    alerts = []
    for r in load_alerts(path):
        ids = [i.split(".")[0] for i in r["mitre_ids"].split(";") if i]
        alerts.append(Alert(
            source="wazuh", timestamp=r["timestamp"], entity=r["agent_name"] or "",
            detector_id=str(r["rule_id"]), description=r["rule_description"] or "",
            level=int(r["rule_level"]) if r["rule_level"] is not None else None,
            raw_tactics=[t for t in r["mitre_tactics"].split(";") if t],
            techniques=sorted(set(ids))))
    return Session("wazuh", sid or path, alerts)


def guide_incident_adapter(sid, techs: str, grade=None) -> Session:
    """One row of guide_incidents.parquet -> Session. GUIDE incidents are technique SETS here
    (order is not used, see addendum 22), so ordered=False."""
    ts = [t for t in techs.split(";") if t]
    alerts = [Alert(source="guide", timestamp="", entity=str(sid).split("-")[0],
                    detector_id="", techniques=[t]) for t in ts]
    return Session("guide", sid, alerts, ordered=False, label=grade)


# --------------------------------------------------------------------------------------
# Resolvers: Session -> token sequence
# --------------------------------------------------------------------------------------
def tokens(session: Session, mode="canonical", dedup=True) -> List[str]:
    """mode:
         canonical : ONE tactic per Wazuh rule_id (canonical_tactics.py; excludes session bookkeeping)
         raw       : every tactic Wazuh lists, expanded -- reproduces the week-1 31-token numbers
         technique : parent technique IDs (needs wazuh_json_adapter or GUIDE)
    """
    seq = []
    for a in session.alerts:
        if mode == "canonical":
            t = get_canonical_tactic(a.detector_id, ";".join(a.raw_tactics))
            toks = [t] if t else []
        elif mode == "raw":
            toks = list(a.raw_tactics)
        elif mode == "technique":
            toks = list(a.techniques)
        else:
            raise ValueError(mode)
        for t in toks:
            if dedup and seq and seq[-1] == t:
                continue
            seq.append(t)
    return seq


# --------------------------------------------------------------------------------------
# Stream loading (used by the prototype engine)
# --------------------------------------------------------------------------------------
def wazuh_line_to_alert(obj) -> Alert:
    """One raw Wazuh alerts.json object -> Alert (keeps parent technique IDs)."""
    rule, agent = obj.get("rule", {}), obj.get("agent", {})
    mitre = rule.get("mitre", {})
    ids = sorted({i.split(".")[0] for i in mitre.get("id", [])})
    return Alert(source="wazuh", timestamp=obj.get("timestamp", ""), entity=agent.get("name", "") or "",
                 detector_id=str(rule.get("id", "")), description=rule.get("description", "") or "",
                 level=rule.get("level"), raw_tactics=list(mitre.get("tactic", [])), techniques=ids)


def parse_line(line):
    """A JSON line -> Alert, for raw Wazuh objects ('rule' key) or the internal format ('detector_id' key)."""
    import json
    line = line.strip()
    if not line:
        return None
    obj = json.loads(line)
    if "rule" in obj:
        return wazuh_line_to_alert(obj)
    if "detector_id" in obj:
        return Alert(**obj)
    return None


def load_stream(path):
    """Whole file -> list of Alerts sorted by time, with .seq set. Accepts joined_*.csv, internal .jsonl,
    or a raw Wazuh alerts.json (one JSON object per line)."""
    from loader import parse_ts
    if path.endswith(".csv"):
        alerts = wazuh_joined_adapter(path).alerts
    else:
        alerts = []
        with open(path) as f:
            for line in f:
                a = parse_line(line)
                if a is not None:
                    alerts.append(a)
    alerts.sort(key=lambda a: parse_ts(a.timestamp))
    for i, a in enumerate(alerts):
        a.seq = i
    return alerts
