"""
learned.py -- the LEARNED scorer (replaces the hand-set rules; see the project guide, sections 3.3 and 4.3).

TWO LEARNED PARTS
  1. TransitionModel (guide 3.3): counts how a tactic follows the previous one or two (skip-gram, so a skipped step
     is tolerated), separately in ATTACK runs and in BENIGN background. For an incident's tactic sequence it
     returns a log-likelihood ratio (LLR):
         LLR = sum over steps of  log P_attack(next | recent tactics) - log P_benign(next | recent tactics)
     (why skip-gram: a first-order model had never seen 'Initial Access -> Defense Evasion' in an attack and scored
      a variant that skipped Persistence as benign; found in the first experiment)
     > 0 means "this sequence looks more like our attacks than like normal traffic".  Laplace-smoothed counts.
  2. Logistic priority (guide 4.3): learns how to combine seven incident features into one attack-likeness score:
         llr_mean, llr_total, pattern coverage, pattern match, depth (distinct tactics), max rule level, log #alerts.

NOT LEARNED (stated plainly): the late-stage GUARD RAIL (a safety rule) and the two false-alarm budgets.

HONEST TRAINING (the part that keeps the evaluation clean)
  * Attack counts and the pattern library for run L never include run L when scoring run L's training rows
    (leave-one-run-out cross-fitting), so training attacks look as "unseen" as test attacks will.
  * Benign counts come from background A; the classifier's training negatives come from a DIFFERENT background B;
    testing uses background C.
  * Thresholds come from a FALSE-ALARM BUDGET on normal incidents (a deployment policy, set by the analyst team):
        INVESTIGATE at the PEAK score only 1% of normal incidents ever reach,  ESCALATE at the one only 0.1% reach
    (peak, because decisions are sticky: an incident that spiked once stays surfaced).
    (First attempt: thresholds from the weakest training attack. Rejected: with a few identical complete chains the
     scores saturate and any shorter variant falls below the threshold. Whether unseen attacks clear a budget-based
     threshold is then a RESULT to be measured, not something fitted in.)
    The out-of-fold scores of the training attacks are still reported, to show how far above the thresholds they sit.
"""
import json
import math
import random
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

import _common_path  # noqa: F401
from sessionize_and_mine import longest_common_backbone
from soc_common import tokens as session_tokens, Session
from socflow import Engine, AllSurface
from make_scenario import place_runs

START = "<s>"
FEATS = ["llr_mean", "llr_total", "coverage", "match", "depth", "max_level", "log_alerts"]


class TransitionModel:
    """Skip-gram transition model. A tactic is explained by the previous `skip` tactics (distance 1..skip), so an
    attacker who SKIPS a step (A -> C instead of A -> B -> C) is still recognised, like a subsequence match in LCS.
    Each position contributes the AVERAGE log-ratio over its contexts, so longer sequences are not inflated."""

    def __init__(self, alpha=1.0, skip=2):
        self.alpha, self.skip = alpha, skip
        self.ca, self.cb = Counter(), Counter()          # (context, current) counts: attack / benign
        self.ta, self.tb = Counter(), Counter()          # totals per context
        self.vocab = set()

    def _contexts(self, seq, i):
        ctx = [seq[i - d] for d in range(1, self.skip + 1) if i - d >= 0]
        if i < self.skip:                                # the true start of the incident is also a context
            ctx.append(START)
        return ctx

    def fit(self, attack_seqs, benign_seqs):
        for seqs, c, tot in ((attack_seqs, self.ca, self.ta), (benign_seqs, self.cb, self.tb)):
            for seq in seqs:
                for i, t in enumerate(seq):
                    self.vocab.add(t)
                    for ctx in self._contexts(seq, i):
                        c[(ctx, t)] += 1
                        tot[ctx] += 1
        return self

    def _lp(self, c, tot, ctx, t):
        v = len(self.vocab) + 1                           # +1 for an unseen tactic
        return math.log((c.get((ctx, t), 0) + self.alpha) / (tot.get(ctx, 0) + self.alpha * v))

    def llr(self, seq):
        total = 0.0
        for i, t in enumerate(seq):
            ctxs = self._contexts(seq, i)
            total += sum(self._lp(self.ca, self.ta, c, t) - self._lp(self.cb, self.tb, c, t) for c in ctxs) / len(ctxs)
        return total, len(seq)

    def to_dict(self):
        pack = lambda c: [[p, t, n] for (p, t), n in c.items()]
        return {"alpha": self.alpha, "skip": self.skip, "attack": pack(self.ca), "benign": pack(self.cb),
                "vocab": sorted(self.vocab)}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["alpha"], d.get("skip", 1))
        for key, c, tot in (("attack", m.ca, m.ta), ("benign", m.cb, m.tb)):
            for p, t, n in d[key]:
                c[(p, t)] = n
                tot[p] += n
        m.vocab = set(d["vocab"])
        return m


def feature_vector(inc, trans):
    llr, n = trans.llr(inc.tokens)
    return [max(-6.0, min(6.0, llr / n)) if n else 0.0, max(-15.0, min(15.0, llr)),
            inc.best["coverage"], inc.best["match"], float(len(set(inc.tokens))),
            inc.max_level / 15.0, math.log1p(len(inc.seqs))]


PRETTY = {
    "llr_mean": lambda v: f"tactic transitions look attack-like (LLR {v:+.1f}/step)",
    "llr_total": lambda v: f"sequence LLR {v:+.1f}",
    "coverage": lambda v: f"{v:.0%} of a known attack pattern seen",
    "match": lambda v: f"{v:.0%} of the incident fits a known pattern",
    "depth": lambda v: f"{int(v)} distinct attack tactics",
    "max_level": lambda v: f"max rule level {v * 15:.0f}",
    "log_alerts": lambda v: f"{math.expm1(v):.0f} alerts in the incident",
}


class FeatureRecorder:
    """Scorer used only while TRAINING: records the feature vector after every alert (prefix snapshots)."""
    name = "recorder"
    escalate = 2.0
    investigate = 2.0

    def __init__(self, trans):
        self.trans, self.rows = trans, []

    def score(self, inc, eng):
        self.rows.append((inc.iid, feature_vector(inc, self.trans)))
        return 0.0, []


class LearnedScorer:
    name = "learned"

    def __init__(self, trans, mean, std, coef, intercept, escalate, investigate, meta=None):
        self.trans = trans
        self.mean, self.std = np.asarray(mean, float), np.maximum(np.asarray(std, float), 1e-6)
        self.coef, self.intercept = np.asarray(coef, float), float(intercept)
        self.escalate, self.investigate, self.meta = float(escalate), float(investigate), meta or {}

    def score(self, inc, eng):
        x = np.array(feature_vector(inc, self.trans))
        z = (x - self.mean) / self.std
        p = 1.0 / (1.0 + math.exp(-(self.intercept + float(z @ self.coef))))
        contrib = self.coef * z
        reasons = []
        for i in np.argsort(-contrib):                      # biggest push towards 'attack-like' first
            if contrib[i] <= 0.25 or len(reasons) == 2:
                break
            if FEATS[i] in ("llr_mean", "llr_total") and x[i] <= 0:
                continue                                    # never call a sequence attack-like when its LLR is not positive
            reasons.append(PRETTY[FEATS[i]](x[i]))
        return p, reasons

    def save(self, path, patterns):
        with open(path, "w") as f:
            json.dump({"scorer": "learned", "patterns": patterns, "trans": self.trans.to_dict(),
                       "mean": self.mean.tolist(), "std": self.std.tolist(), "coef": self.coef.tolist(),
                       "intercept": self.intercept, "escalate": self.escalate, "investigate": self.investigate,
                       "features": FEATS, "meta": self.meta}, f, indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(TransitionModel.from_dict(d["trans"]), d["mean"], d["std"], d["coef"], d["intercept"],
                   d["escalate"], d["investigate"], d.get("meta")), d["patterns"]


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def _seqs(session_alerts, mode="canonical"):
    return session_tokens(Session("wazuh", "x", session_alerts), mode)


def benign_sequences(bg_alerts, cfg=None):
    """Replay a background stream and return the tactic sequence of every incident it forms."""
    eng = Engine([], cfg, scorer=AllSurface())
    for a in bg_alerts:
        eng.feed(a)
    eng.finish()
    assert eng.out_of_order == 0, f"background not time-ordered ({eng.out_of_order} alerts out of order)"
    return [i.tokens for i in eng.incidents.values() if i.tokens]


def _logit_fit(X, y, C=1.0):
    mean, std = X.mean(0), np.maximum(X.std(0), 1e-6)
    lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit((X - mean) / std, y)
    return mean, std, lr.coef_[0], lr.intercept_[0]


def _logits(X, mean, std, coef, b):
    return np.clip(((X - mean) / std) @ coef + b, -30.0, 30.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def train(attack_runs, bg_a, bg_b, seed=0, cfg=None, inv_fp=0.01, esc_fp=0.001, C=0.1, verbose=True,
          mix_hosts=False):
    """attack_runs: dict label -> list of Alerts (training attacks). bg_a / bg_b: (alerts, t0, span) tuples.
    Returns (LearnedScorer, patterns, report)."""
    labels = list(attack_runs)
    if len(labels) < 3:
        raise ValueError("need at least 3 training attack runs for leave-one-run-out cross-fitting")
    benign = benign_sequences(bg_a[0], cfg)
    atk_tokens = {l: _seqs(r) for l, r in attack_runs.items()}
    rows_X, rows_y, rows_fold, pos_final = [], [], [], []       # pos_final: (fold index, row index) of final positive snapshots

    for k, L in enumerate(labels):                              # ---- leave-one-run-out feature building
        others = [l for l in labels if l != L]
        lib = {"name": "known_attack", "tokens": longest_common_backbone([atk_tokens[o] for o in others]),
               "mode": "canonical"}
        trans = TransitionModel().fit([atk_tokens[o] for o in others], benign)
        scen = place_runs(bg_b[0], {L: attack_runs[L]}, bg_b[1], bg_b[2], random.Random(seed * 1000 + k), mix_hosts)
        rec = FeatureRecorder(trans)
        eng = Engine([lib], cfg, scorer=rec)
        for a in scen:
            eng.feed(a)
        eng.finish()
        truth_of = {}
        for inc in eng.incidents.values():
            truth_of[inc.iid] = any(scen[s].truth == L for s in inc.seqs)
        last = {}
        for iid, vec in rec.rows:
            idx = len(rows_X)
            rows_X.append(vec); rows_y.append(1 if truth_of[iid] else 0); rows_fold.append(k)
            last[iid] = idx
        pos_final += [(k, idx) for iid, idx in last.items() if truth_of[iid]]

    X, y, fold = np.array(rows_X), np.array(rows_y), np.array(rows_fold)
    # ---- out-of-fold scores of the training attacks (final snapshot of each positive incident)
    oof = {}
    for k, idx in pos_final:
        tr = fold != k
        m, s, c, b = _logit_fit(X[tr], y[tr], C)
        oof.setdefault(k, []).append(float(_logits(X[idx:idx + 1], m, s, c, b)[0]))
    per_run = {labels[k]: max(v) for k, v in oof.items()}      # best incident of that run, in logits

    # ---- final model: attack statistics + pattern library from ALL training runs; classifier on all rows
    patterns = [{"name": "known_attack", "tokens": longest_common_backbone(list(atk_tokens.values())),
                 "mode": "canonical", "built_from": labels}]
    trans_all = TransitionModel().fit(list(atk_tokens.values()), benign)
    mean, std, coef, b = _logit_fit(X, y, C)
    # ---- thresholds from the false-alarm budget: score NORMAL incidents (background B, PEAK score of each)
    rec = FeatureRecorder(trans_all)
    eng = Engine(patterns, cfg, scorer=rec)
    for a in bg_b[0]:
        eng.feed(a)
    eng.finish()
    assert eng.out_of_order == 0, f"background B not time-ordered ({eng.out_of_order} alerts out of order)"
    allrows = np.array([vec for _, vec in rec.rows])
    row_logits = _logits(allrows, mean, std, coef, b)
    peak = {}
    for (iid, _), z in zip(rec.rows, row_logits):            # decisions are sticky, so the budget applies to each
        peak[iid] = max(peak.get(iid, -1e9), float(z))       # normal incident's PEAK score, not its final one
    normal = np.array(list(peak.values()))
    inv_logit = float(np.quantile(normal, 1 - inv_fp))
    esc_logit = max(inv_logit, float(np.quantile(normal, 1 - esc_fp)))
    investigate, escalate = float(_sigmoid(inv_logit)), float(_sigmoid(esc_logit))
    report = {"rows": int(len(y)), "positive_rows": int(y.sum()), "negative_rows": int((y == 0).sum()),
              "oof_best_incident_logit_per_run": {k: round(v, 2) for k, v in per_run.items()},
              "normal_incidents_scored": int(len(normal)), "inv_fp": inv_fp, "esc_fp": esc_fp,
              "investigate_logit": round(inv_logit, 2), "escalate_logit": round(esc_logit, 2),
              "investigate": investigate, "escalate": escalate,
              "coef": dict(zip(FEATS, [round(float(c), 3) for c in coef])),
              "benign_incident_sequences": len(benign)}
    meta = {"trained_on": labels, "inv_fp": inv_fp, "esc_fp": esc_fp, "report": report}
    scorer = LearnedScorer(trans_all, mean, std, coef, b, escalate, investigate, meta)
    if verbose:
        print(f"trained on {len(labels)} attack runs, {report['rows']} snapshots "
              f"({report['positive_rows']} attack / {report['negative_rows']} benign)")
        print("out-of-fold LOGIT of each training attack (best incident):", report["oof_best_incident_logit_per_run"])
        print(f"thresholds (logit) from the false-alarm budget over {len(normal)} normal incidents: "
              f"INVESTIGATE >= {inv_logit:.2f} (top {inv_fp:.1%} of normal), ESCALATE >= {esc_logit:.2f} (top {esc_fp:.1%})")
        print("learned weights (standardised features):", report["coef"])
    return scorer, patterns, report
