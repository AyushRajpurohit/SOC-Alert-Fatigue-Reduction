"""Experimental learned alert novelty for SOC evidence selection (offline only).

Learns P(rule_id | past alert context) using train data only. Novelty is
out-of-sample surprise, NOT maliciousness, importance or permission to suppress.

The existing greedy reducer is untouched. Learned selection obeys exactly the
same evidence requirements and mandatory-retention policy.

Model artifacts can contain pseudonymous entity tokens: keep outdir PRIVATE.
Only aggregate JSON/CSV should be shared after local privacy review.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from correlation_engine import Correlator, read_json_lines
from evidence_reducer import Policy, requirements, choose
from reducer_six_method_eval import audit_cluster


def context_features(events):
    """Features use preceding alerts only; current rule/techniques excluded."""
    last_by_host = {}
    last_global = None
    prior_counts = Counter()
    feats = []
    for e in events:
        previous = last_by_host.get(e.host)
        prev_rule = previous.rule if previous else '<START>'
        if previous is not None:
            delta = max(0.0, (e.timestamp - previous.timestamp).total_seconds())
            gap_bin = ('<1s' if delta < 1 else '<10s' if delta < 10 else
                       '<60s' if delta < 60 else '<5m' if delta < 300 else '>=5m')
        else:
            gap_bin = 'new_host'
        # Host pseudonyms will be embedded in the private vectorizer artifact.
        feats.append({
            'host': e.host or '<MISSING>',
            'prev_host_rule': prev_rule or '<EMPTY>',
            'prev_global_rule': (last_global.rule or '<EMPTY>') if last_global else '<START>',
            'gap_bin': gap_bin,
            'host_rule_count_bin': str(min(prior_counts[(e.host, prev_rule)], 5)),
        })
        last_by_host[e.host] = e
        last_global = e
        prior_counts[(e.host, e.rule)] += 1
    return feats


def train_predictor(events, min_rule_count=2):
    if len(events) < 20:
        raise ValueError('At least 20 training records required')
    counts = Counter(e.rule for e in events)
    # Collapsing singleton classes is a train-only vocabulary decision.
    frequent = {rule for rule, n in counts.items() if n >= min_rule_count}
    labels = [e.rule if e.rule in frequent else '<OTHER>' for e in events]
    if len(set(labels)) < 2:
        raise ValueError('Need at least two training rule classes')
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(context_features(events))
    model = LogisticRegression(max_iter=400, C=0.3, random_state=42)
    model.fit(X, labels)
    return dv, model, frequent


def novelty_scores(events, vectorizer, model, frequent):
    X = vectorizer.transform(context_features(events))
    prob = model.predict_proba(X)
    lookup = {str(label): j for j, label in enumerate(model.classes_)}
    scores = []
    for i, e in enumerate(events):
        label = e.rule if e.rule in frequent else '<OTHER>'
        p = float(prob[i, lookup[label]]) if label in lookup else 1e-6
        # Truncated surprise, not calibrated security relevance.
        scores.append(min(14.0, -math.log(max(p, 1e-6))))
    return scores


def learned_cover(cluster, novelties, policy, mode='novelty_tiebreak'):
    """Same hard/coverage constraints as original greedy; learned tie or blend."""
    universe, covers, mandatory = requirements(cluster, policy)
    selected = set(mandatory)
    covered = set().union(*(covers[i] for i in selected)) if selected else set()
    remaining = universe - covered
    while remaining:
        candidates = [i for i in range(len(covers)) if i not in selected and covers[i] & remaining]
        if not candidates:
            raise RuntimeError('Unable to cover required evidence')
        if mode == 'novelty_tiebreak':
            pick = max(candidates, key=lambda i: (
                len(covers[i] & remaining), novelties[i], cluster.members[i].level, -i))
        elif mode == 'novelty_weighted':
            pick = max(candidates, key=lambda i: (
                len(covers[i] & remaining) * (1 + 0.5 * min(novelties[i], 10) / 10),
                len(covers[i] & remaining), cluster.members[i].level, -i))
        else:
            raise ValueError(mode)
        selected.add(pick)
        remaining.difference_update(covers[pick])
    if not mandatory.issubset(selected) or universe - set().union(*(covers[i] for i in selected)):
        raise AssertionError('Evidence-preservation constraint violated')
    return sorted(selected)


def extra_facts(cluster, indices):
    """Exploratory proxy NOT included in hard selection constraints.

    Counts observed rule-user and rule-source-IP combinations, NOT analyst facts.
    """
    facts = set()
    for i in indices:
        e = cluster.members[i]
        if e.user:
            facts.add(('rule_user', e.rule, e.user))
        if e.src_ip:
            facts.add(('rule_source', e.rule, e.src_ip))
    return facts


def evaluate(events, novelties, policy):
    engine = Correlator()
    for e in events:
        engine.feed(e)
    engine.finish()
    # Use object identity, not UID strings (UIDs can be nonunique).
    by_obj = {id(e): n for e, n in zip(events, novelties)}
    methods = ('greedy_cover', 'novelty_tiebreak', 'novelty_weighted')
    totals = {m: Counter() for m in methods}
    for c in engine.clusters:
        n = [by_obj[id(e)] for e in c.members]
        candidates = {
            'greedy_cover': choose(c, 'greedy_cover', policy)[0],
            'novelty_tiebreak': learned_cover(c, n, policy, 'novelty_tiebreak'),
            'novelty_weighted': learned_cover(c, n, policy, 'novelty_weighted'),
        }
        all_facts = extra_facts(c, range(len(c.members)))
        for method, ids in candidates.items():
            audit = audit_cluster(c, ids, policy)
            totals[method].update(audit)
            totals[method]['external_proxy_total'] += len(all_facts)
            totals[method]['external_proxy_retained'] += len(extra_facts(c, ids) & all_facts)
    rows = []
    for method in methods:
        r = totals[method]
        n_alerts = r['alerts']
        rows.append({
            'method': method,
            'alerts': n_alerts,
            'groups': r['incidents'],
            'visible': r['visible'],
            'hidden_percent': round(100 * (n_alerts-r['visible'])/n_alerts, 3) if n_alerts else 0,
            'constraints_covered': r['constraints_covered'],
            'constraints_total': r['constraints_total'],
            'constraint_failures': r['cases_with_unmet_requirements'],
            'critical_retained': r['critical_retained'],
            'critical_total': r['critical_total'],
            'extra_proxy_retained': r['external_proxy_retained'],
            'extra_proxy_total': r['external_proxy_total'],
            'extra_proxy_coverage_percent': round(100*r['external_proxy_retained']/r['external_proxy_total'], 3)
                if r['external_proxy_total'] else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description='Offline self-supervised alert novelty experimental comparator')
    ap.add_argument('--train', nargs='+', required=True, help='Earlier sanitized JSONL windows (training only)')
    ap.add_argument('--eval', required=True, help='Later sanitized JSONL window; NOT used for fitting')
    ap.add_argument('--outdir', required=True, help='PRIVATE local directory (model embeds pseudonyms)')
    ap.add_argument('--critical-level', type=int, default=10)
    args = ap.parse_args()
    train_paths = [Path(p) for p in args.train]
    eval_path = Path(args.eval)
    out = Path(args.outdir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing overwrite: {out}')
    if eval_path.resolve() in {p.resolve() for p in train_paths}:
        raise ValueError('Training and evaluation files must be disjoint')
    training = read_json_lines(train_paths)
    testing = read_json_lines([eval_path])
    if not training or not testing:
        raise ValueError('Training and evaluation input must be nonempty')
    if max(e.timestamp for e in training) >= min(e.timestamp for e in testing):
        raise ValueError('Chronology violation: training must strictly precede evaluation')
    started = time.perf_counter()
    dv, model, frequent = train_predictor(training)
    scores = novelty_scores(testing, dv, model, frequent)
    rows = evaluate(testing, scores, Policy(critical_level=args.critical_level))
    pred = model.predict(dv.transform(context_features(testing)))
    true = [e.rule if e.rule in frequent else '<OTHER>' for e in testing]
    prediction_accuracy = round(float(np.mean(np.array(pred) == np.array(true))), 5)
    common_rule = Counter(e.rule if e.rule in frequent else '<OTHER>' for e in training).most_common(1)[0][0]
    majority_accuracy = round(sum(v == common_rule for v in true) / len(true), 5)
    elapsed = round((time.perf_counter()-started)*1000, 2)
    out.mkdir(parents=True, exist_ok=True)
    # PRIVATE MODEL: train labels, vectorizer features can embed pseudonymous names.
    joblib.dump(dict(vectorizer=dv, model=model, frequent=frequent), out/'private_model.joblib')
    with (out/'aggregate_comparison.csv').open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest = {
        'protocol':'exploratory_context_rule_prediction_v1',
        'train_records':len(training),'eval_records':len(testing),
        'training_rule_classes':len(model.classes_),
        'heldout_next_rule_accuracy':prediction_accuracy,
        'heldout_train_majority_rule_accuracy':majority_accuracy,
        'train_rule_predictor_accuracy_not_measured':True,
        'critical_level':args.critical_level,
        'wall_ms_including_training_and_selection':elapsed,
        'method':'Logistic regression predicting current rule from preceding contextual events; surprisal used as selection tie-break / small weighted bonus',
        'privacy':'PRIVATE model and original inputs must not be uploaded; only reviewed aggregate results',
        'caveats':[
            'No real security importance, maliciousness, benignness or suppression labels.',
            'Selection preserves only existing programmed constraints; not independently validated analyst evidence.',
            'Extra proxy is observed (rule,user) and (rule,source IP) pairs, not true security ground truth.',
            'Training/eval filenames and event records intentionally excluded from shared outputs.',
            'Comparison on previously inspected Sample C is exploratory, not independent confirmatory testing.',
            'Full reduction not analyst time saved; no live deployment, no retraining of the existing Wazuh scorer.',
            'Model joblib contains training-derived pseudonymous feature names; keep in private storage.',
        ],
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for r in rows:
        print(f"{r['method']:19s} visible={r['visible']}/{r['alerts']} "+
              f"reduction={r['hidden_percent']:.1f}% constraints={r['constraints_covered']}/{r['constraints_total']} "+
              f"extra_proxy={r['extra_proxy_retained']}/{r['extra_proxy_total']}")
    print('Next-rule prediction accuracy (not security accuracy):', prediction_accuracy,
          'train-majority baseline:', majority_accuracy)
    print('Saved PRIVATE model and aggregate outputs:', out)

if __name__ == '__main__':
    main()
