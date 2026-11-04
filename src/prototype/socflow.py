"""
socflow.py -- the prototype engine (SIEM-agnostic once alerts are in the common record).

PIPELINE, per incoming alert (scorer = RuleScorer by default, or the LEARNED scorer in learned.py):
    Alert (from any adapter)
      -> token            canonical tactic (lab table; unmapped rule ids fall back to the source's first tactic)
      -> incident         alerts from the same entity (agent) stay in one incident until it goes quiet for `gap` seconds
      -> pattern match    LCS of the incident's token sequence against each KNOWN attack pattern
      -> priority + decision   (explainable rules below, NOT a black box)

DECISIONS ("demote, never delete": a demoted incident is still kept, counted and shown in the digest)
    ESCALATE     analyst should look now
    INVESTIGATE  worth a look
    DEMOTE       folded into the low-priority digest

RULES (all thresholds live in DEFAULT_CFG and are swept in evaluate.py; they are design assumptions, not learned):
    sev   = how serious the highest-level alert is:  0 for level <= sev_floor, rising to 1 over `sev_span` levels
    chain = pattern.coverage * pattern.match         (how much of a known attack is present AND how pure the incident is)
    priority = max(sev, chain)
    ESCALATE if priority >= escalate;  INVESTIGATE if >= investigate;  else DEMOTE
    GUARD RAIL: an incident containing a late-stage tactic (Collection / Exfiltration / Impact) is always ESCALATE.
"""
import contextlib
import io
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from canonical_tactics import get_canonical_tactic, unmapped_rule_ids
from sessionize_and_mine import lcs2
from loader import parse_ts

DEFAULT_CFG = dict(
    gap=120,                  # seconds of silence that close an incident
    sev_floor=6,              # (rule scorer) Wazuh levels <= this are treated as routine
    sev_span=6,               # (rule scorer) levels floor+1 .. floor+span map to sev 1/span .. 1
    escalate=None,            # None = use the scorer's own thresholds; a number overrides them (used by sweeps)
    investigate=None,
    late_stage=("Collection", "Exfiltration", "Impact"),   # GUARD RAIL (hard rule, applies to every scorer)
    sticky=True,              # once an incident has been surfaced it is never withdrawn (analysts saw it)
)
SURFACED = ("ESCALATE", "INVESTIGATE")
RANK = {"DEMOTE": 0, "INVESTIGATE": 1, "ESCALATE": 2}


def load_patterns(path):
    with open(path) as f:
        return json.load(f)


def save_patterns(path, patterns):
    with open(path, "w") as f:
        json.dump(patterns, f, indent=2)


def token_of(alert):
    """One canonical tactic or None (None = bookkeeping, e.g. 'session closed')."""
    with contextlib.redirect_stdout(io.StringIO()):          # silence 'UNMAPPED rule_id' prints while streaming
        return get_canonical_tactic(alert.detector_id, ";".join(alert.raw_tactics))


class RuleScorer:
    """HAND-SET rules (the first prototype). Kept as a baseline to compare the learned scorer against."""
    name = "rules"
    escalate = 0.6
    investigate = 0.3

    def score(self, inc, eng):
        c = eng.cfg
        sev = min(1.0, max(0.0, (inc.max_level - c["sev_floor"]) / c["sev_span"]))
        chain = inc.best["match"] * inc.best["coverage"]
        reasons = []
        if sev > 0:
            reasons.append(f"max rule level {inc.max_level}")
        if chain > 0 and inc.best["name"] != "-":
            reasons.append(f"{inc.best['coverage']:.0%} of '{inc.best['name']}' seen, {inc.best['match']:.0%} of incident fits it")
        return max(sev, chain), reasons


class AllSurface:
    """Grouping only: every incident is shown (baseline B2 'similarity-only grouping', no filtering)."""
    name = "grouping only"
    escalate = 2.0
    investigate = 0.0

    def score(self, inc, eng):
        return 1.0, ["grouping only (no filtering)"]


@dataclass
class Incident:
    iid: int
    entity: str
    first_ts: datetime
    last_ts: datetime
    seqs: list = field(default_factory=list)                 # alert positions in the stream
    rules: Counter = field(default_factory=Counter)
    tokens: list = field(default_factory=list)
    max_level: int = 0
    best: dict = field(default_factory=lambda: {"name": "-", "match": 0.0, "coverage": 0.0})
    priority: float = 0.0
    decision: str = "DEMOTE"
    reasons: list = field(default_factory=list)
    peak: str = "DEMOTE"                                      # highest decision this incident has ever had
    closed: bool = False

    def to_json(self):
        return {"id": self.iid, "entity": self.entity, "first_ts": self.first_ts.isoformat(),
                "last_ts": self.last_ts.isoformat(), "n_alerts": len(self.seqs), "tokens": self.tokens,
                "rules": dict(self.rules.most_common(8)), "max_level": self.max_level,
                "priority": round(self.priority, 3), "decision": self.decision, "reasons": self.reasons,
                "best": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.best.items()},
                "closed": self.closed}


class Engine:
    def __init__(self, patterns, cfg=None, scorer=None):
        self.patterns = patterns
        self.scorer = scorer or RuleScorer()
        self.cfg = {**DEFAULT_CFG, **{k: v for k, v in (cfg or {}).items()}}
        self.incidents = {}                                   # iid -> Incident (all, open and closed)
        self._open = {}                                       # entity -> Incident
        self._next = 1
        self.alerts_seen = 0
        self.rule_desc = {}
        self._last_ts = None
        self.out_of_order = 0                                 # alerts older than one already processed (should be 0)

    # ---------- streaming ----------
    def feed(self, alert):
        """Process one alert. Returns (incident, previous_decision or None if the incident is new)."""
        ts = parse_ts(alert.timestamp)
        if self._last_ts is not None and ts < self._last_ts:
            self.out_of_order += 1
        self._last_ts = ts if self._last_ts is None else max(self._last_ts, ts)
        self.alerts_seen += 1
        for ent, inc in list(self._open.items()):             # close incidents that went quiet
            if (ts - inc.last_ts).total_seconds() > self.cfg["gap"]:
                inc.closed = True
                del self._open[ent]
        inc = self._open.get(alert.entity)
        prev = None if inc is None else inc.decision
        if inc is None:
            inc = Incident(self._next, alert.entity, ts, ts)
            self._next += 1
            self.incidents[inc.iid] = inc
            self._open[alert.entity] = inc
        inc.last_ts = ts
        inc.seqs.append(alert.seq)
        inc.rules[alert.detector_id] += 1
        self.rule_desc.setdefault(alert.detector_id, alert.description)
        inc.max_level = max(inc.max_level, int(alert.level or 0))
        t = token_of(alert)
        if t and (not inc.tokens or inc.tokens[-1] != t):
            inc.tokens.append(t)
        self._decide(inc)
        return inc, prev

    def finish(self):
        for inc in self.incidents.values():
            inc.closed = True
        self._open.clear()

    # ---------- decision ----------
    def _best_pattern(self, tokens):
        best = {"name": "-", "match": 0.0, "coverage": 0.0}
        for p in self.patterns:
            if not tokens or not p["tokens"]:
                continue
            n = len(lcs2(tokens, p["tokens"]))
            m, c = n / len(tokens), n / len(p["tokens"])
            if m * c > best["match"] * best["coverage"]:
                best = {"name": p["name"], "match": m, "coverage": c}
        return best

    def thresholds(self):
        c = self.cfg
        return (c["escalate"] if c["escalate"] is not None else self.scorer.escalate,
                c["investigate"] if c["investigate"] is not None else self.scorer.investigate)

    def _decide(self, inc):
        c = self.cfg
        inc.best = self._best_pattern(inc.tokens)
        inc.priority, reasons = self.scorer.score(inc, self)
        esc, inv = self.thresholds()
        late = [t for t in inc.tokens if t in c["late_stage"]]
        if late:                                              # GUARD RAIL: never demote late-stage activity
            inc.decision = "ESCALATE"
            reasons = [f"guard rail: late-stage tactic {', '.join(sorted(set(late)))}"] + reasons
        elif inc.priority >= esc:
            inc.decision = "ESCALATE"
        elif inc.priority >= inv:
            inc.decision = "INVESTIGATE"
        else:
            inc.decision = "DEMOTE"
        if RANK[inc.decision] > RANK[inc.peak]:
            inc.peak = inc.decision
        if c["sticky"] and RANK[inc.decision] < RANK[inc.peak]:
            reasons = [f"kept at {inc.peak}: it was already surfaced to the analyst"] + reasons
            inc.decision = inc.peak
        inc.reasons = reasons or ["routine alerts, no known-attack pattern"]

    # ---------- reporting ----------
    def stats(self):
        incs = list(self.incidents.values())
        by = Counter(i.decision for i in incs)
        surfaced = by["ESCALATE"] + by["INVESTIGATE"]
        return {"alerts_seen": self.alerts_seen, "out_of_order": self.out_of_order, "incidents": len(incs), "escalate": by["ESCALATE"],
                "investigate": by["INVESTIGATE"], "demoted": by["DEMOTE"],
                "demoted_alerts": sum(len(i.seqs) for i in incs if i.decision == "DEMOTE"),
                "reduction": (1 - surfaced / self.alerts_seen) if self.alerts_seen else 0.0}

    def snapshot(self, limit=200):
        incs = list(self.incidents.values())
        surf = [i for i in incs if i.decision in SURFACED]
        dem = [i for i in incs if i.decision == "DEMOTE"]
        surf.sort(key=lambda i: (i.decision != "ESCALATE", -i.priority, -i.last_ts.timestamp()))
        digest = Counter()
        for i in dem:
            digest.update(i.rules)
        esc, inv = self.thresholds()
        return {"stats": self.stats(), "scorer": self.scorer.name,
                "cfg": {"gap": self.cfg["gap"], "escalate": round(esc, 4), "investigate": round(inv, 4)},
                "patterns": [{"name": p["name"], "tokens": p["tokens"]} for p in self.patterns],
                "surfaced": [i.to_json() for i in surf[:limit]],
                "digest": {"incidents": len(dem),
                           "top_rules": [{"rule": r, "count": n, "description": self.rule_desc.get(r, "")}
                                         for r, n in digest.most_common(10)]},
                "unmapped_rules": unmapped_rule_ids()}

    def write_state(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self.snapshot(), f)
        os.replace(tmp, path)                                # atomic: the dashboard never reads half a file
