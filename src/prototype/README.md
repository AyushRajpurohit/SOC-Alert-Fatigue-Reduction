# Prototype: streaming alert-fatigue engine with a LEARNED scorer

alert stream (any source) -> common record -> canonical tactic tokens -> incidents (per host, closed after `gap` s quiet)
-> known-attack pattern match -> LEARNED score (skip-gram transition LLR + logistic priority) -> ESCALATE / INVESTIGATE / DEMOTE
-> dashboard + evaluation.  Demoted incidents are kept in a digest, never deleted.  Decisions are sticky (once surfaced,
an incident stays surfaced) and a late-stage guard rail (Collection / Exfiltration / Impact => always ESCALATE) applies to
every scorer.  The hand-set rule scorer is kept as a BASELINE (--rules).

Run everything from the project root (~/Desktop/SSE) with the venv active.

1) Train the learned scorer on the 5 baseline runs (+ a background for benign statistics)
   python src/prototype/train_model.py --baseline data/lab/joined/joined_run{1,2,3,4,5}.csv \
       --background-synthetic 3000 --out src/prototype/model.json
   (real background: --background-file <institute alerts.json>; it is split by TIME into A/B/C, see make_scenario.py)

2) Compare grouping only / severity filter / hand-set rules / LEARNED on attacks the model never saw
   python src/prototype/experiment.py --baseline data/lab/joined/joined_run{1,2,3,4,5}.csv \
       --test variant=data/lab/joined/joined_variant.csv --test exfil2=data/lab/joined/joined_exfil2.csv \
       --background-synthetic 3000 --repeats 10 --out results/proto_experiment

3) Live dashboard
   terminal A:  cd src/prototype/dashboard && python -m http.server 8000      (open http://localhost:8000)
   terminal B:  python src/prototype/make_scenario.py --inject variant=data/lab/joined/joined_variant.csv \
                  --inject exfil2=data/lab/joined/joined_exfil2.csv --background-synthetic 3000 --out data/scenarios/synthetic.jsonl
                python src/prototype/run_engine.py --input data/scenarios/synthetic.jsonl --speed 30
                (add --rules to run the hand-set rules instead)

4) LIVE from the Wazuh VM (check the alerts.json path on your install)
   ssh <user>@<wazuh-vm> "sudo tail -n0 -F /var/ossec/logs/alerts/alerts.json" | python src/prototype/run_engine.py --stdin

Other tools: learn_patterns.py (pattern library for --rules), evaluate.py (rule-engine sweep on one scenario file).

SHORT LIVE DEMO (attacks appear within ~2 minutes; the 6-hour stream needs --speed 300 to reach them)
   python src/prototype/make_scenario.py --inject variant=data/lab/joined/joined_variant.csv \
       --inject exfil2=data/lab/joined/joined_exfil2.csv --background-synthetic 500 --hours 1 --agents 6 --mix-hosts \
       --out data/scenarios/demo.jsonl
   python src/prototype/run_engine.py --input data/scenarios/demo.jsonl --speed 20     (dashboard open at localhost:8000)

STRESS TEST (recommended before you quote any number): --hard-noise adds benign cron edits / failed sudo / attack
look-alikes, --mix-hosts puts each attack on a busy host. Synthetic repeats now use a NEW background each time.
   python src/prototype/experiment.py --baseline data/lab/joined/joined_run{1,2,3,4,5}.csv \
       --test variant=data/lab/joined/joined_variant.csv --test exfil2=data/lab/joined/joined_exfil2.csv \
       --background-synthetic 3000 --repeats 10 --hard-noise --mix-hosts --out results/proto_stress
   Budgets: --inv-fp 0.03 (share of normal incidents allowed to be surfaced at INVESTIGATE; default 0.01).
   The hard background contains 'typo > login > failed sudo' = the variant's tactic sequence ON PURPOSE (worst case):
   no tactic-sequence method can separate those two, so expect lower recall there. Use it as a limit test.

HONESTY NOTES
- SYNTHETIC background numbers show mechanics only (my noise mix is an assumption; the default one is easy to separate).
- Percentages on the dashboard use one decimal (1 item out of 245 alerts = 99.6%, not 100%).
- In your real files the BASELINE runs predate the custom rules 100100-100103 while the variant / exfil2 runs contain
  them, so training and test attacks differ in which rules fired. The learned scorer sees canonical tactics + levels,
  so check `max_level` weight and the per-attack results before trusting a result.
- Thresholds come from a false-alarm budget on normal incidents (1% INVESTIGATE, 0.1% ESCALATE, on each incident's
  PEAK score). Scores cluster, so the realised false-alarm rate can exceed the budget.
- Only 5 training attack runs and 2 test attacks: a proof of concept, not a statistical result.
