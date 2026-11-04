#!/usr/bin/env bash
# demo_run.sh -- guided, repeatable demo of the WORKING prototype, designed to be screen-recorded.
#
# Run from anywhere (it moves to the project root itself), with the venv active:
#     bash demo/demo_run.sh
# Each step prints an on-screen caption, then runs the REAL commands. Press Enter to move on, so you control the pace
# while you narrate.   Options (environment variables):
#     AUTO=1            no pauses (timed instead; AUTO_PAUSE=4 seconds)      DEMO_SPEED=40   dashboard replay speed
#     DEMO_PORT=8000    dashboard port                                      PYTHON=python   interpreter to use
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"
SPEED="${DEMO_SPEED:-40}"
PORT="${DEMO_PORT:-8000}"
J=data/lab/joined
SERVER_PID=""

C1=$'\033[1;36m'; C2=$'\033[1;33m'; C3=$'\033[0;32m'; DIM=$'\033[2m'; OFF=$'\033[0m'

banner() { [ "${AUTO:-0}" = "1" ] || clear
           printf "\n%s============================================================================%s\n" "$C1" "$OFF"
           printf "%s  %s%s\n" "$C1" "$1" "$OFF"
           printf "%s============================================================================%s\n" "$C1" "$OFF"; }
say()    { printf "%s  > %s%s\n" "$C2" "$1" "$OFF"; }
note()   { printf "%s    %s%s\n" "$DIM" "$1" "$OFF"; }
pause()  { if [ "${AUTO:-0}" = "1" ]; then sleep "${AUTO_PAUSE:-3}"; else printf "\n%s   [press Enter to continue]%s " "$C3" "$OFF"; read -r _; fi; }
run()    { printf "%s\$ %s%s\n" "$C3" "$*" "$OFF"; sleep 1; eval "$*"; }
cleanup(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

# Prints whether the two injected attacks are in the analyst queue. Reads the state file the engine just wrote.
# (variant = a surfaced incident containing Defense Evasion; exfil2 = one containing Collection/Exfiltration;
#  valid for the default demo scenario.)  The caption is therefore computed from the REAL result, never assumed.
check_attacks() {
  $PY - <<'PYEOF'
import json
d = json.load(open("src/prototype/dashboard/state/queue.json"))
tok = [set(i["tokens"]) for i in d["surfaced"]]
var = any("Defense Evasion" in t for t in tok)
exf = any(("Exfiltration" in t or "Collection" in t) for t in tok)
print("  ==> items in the analyst queue: %d   |   variant attack surfaced: %s   |   exfiltration attack surfaced: %s"
      % (len(d["surfaced"]), "YES" if var else "NO  (missed!)", "YES" if exf else "NO  (missed!)"))
PYEOF
}

# ---------------------------------------------------------------- pre-flight
missing=0
for f in $J/joined_run1.csv $J/joined_run2.csv $J/joined_run3.csv $J/joined_run4.csv $J/joined_run5.csv \
         $J/joined_variant.csv $J/joined_exfil2.csv src/prototype/model.json src/prototype/run_engine.py \
         src/prototype/experiment.py src/prototype/make_scenario.py src/lab/lab_demo.py src/prototype/dashboard/index.html; do
  [ -e "$f" ] || { echo "MISSING: $f"; missing=1; }
done
if [ "$missing" = 1 ]; then
  echo; echo "Fix the missing files first (model.json: run train_model.py, see README). Nothing was run."; exit 1
fi
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use (an old dashboard server?)."
  echo "Stop it (Ctrl+C in its terminal), or run:  DEMO_PORT=8001 bash demo/demo_run.sh"; exit 1
fi
mkdir -p results data/scenarios

# ---------------------------------------------------------------- 0. intro
clear
banner "SOC ALERT-FATIGUE REDUCTION  -  working prototype demo"
say "Problem: a SOC drowns in alerts; most are routine, a few are real attacks."
say "Idea: group alerts into incidents, learn what an attack SEQUENCE looks like, surface only what matters."
say "Nothing is deleted: low-priority incidents go to a digest."
pause

# ---------------------------------------------------------------- 1. lab proof
banner "STEP 1 / 5   Lab proof: learn a known attack, then test a variant and a different chain"
say "Five baseline runs of the same attack define the known pattern (a sequence of MITRE tactics)."
say "Then two UNSEEN runs are scored: a variant (persistence skipped) and a different chain (exfiltration)."
run "$PY src/lab/lab_demo.py --baseline $J/joined_run1.csv $J/joined_run2.csv $J/joined_run3.csv $J/joined_run4.csv $J/joined_run5.csv --variant $J/joined_variant.csv --other exfil2=$J/joined_exfil2.csv --mode canonical --replay $J/joined_exfil2.csv --delay 0.3 --out results/lab 2>&1 | sed -n '/^mode:/,\$p'"
say "variant: 100% match, 60% coverage  -> the known attack with a step missing; the static rule MISSED it."
say "exfil2:  match falls to 50% during the replay  -> a different attack; the learned pattern tells them apart."
pause

# ---------------------------------------------------------------- 2. noisy stream
banner "STEP 2 / 5   Build a realistic test stream: real attack alerts hidden in background noise"
say "500 synthetic background alerts over 1 hour on 6 hosts + the 33 real lab attack alerts, placed on busy hosts."
run "$PY src/prototype/make_scenario.py --inject variant=$J/joined_variant.csv --inject exfil2=$J/joined_exfil2.csv --background-synthetic 500 --hours 1 --agents 6 --mix-hosts --out data/scenarios/demo.jsonl"
note "We know exactly which alerts are the attack, so we can measure what the system keeps and what it removes."
pause

# ---------------------------------------------------------------- 3. live dashboard
banner "STEP 3 / 5   The prototype live: engine + dashboard (learned scorer)"
say "Switch to the browser: the analyst queue starts empty and fills as alerts stream in."
( cd src/prototype/dashboard && exec $PY -m http.server "$PORT" >/dev/null 2>&1 ) &
SERVER_PID=$!
sleep 1
if command -v open >/dev/null 2>&1; then open "http://localhost:$PORT"; else note "open http://localhost:$PORT in your browser"; fi
pause
run "$PY src/prototype/run_engine.py --input data/scenarios/demo.jsonl --speed $SPEED"
check_attacks
say "Raw alerts -> a handful of incidents. Look at the reasons column and the digest on the right (demoted, not deleted)."
note "Surfaced items include a few look-alike false alarms: benign login-failure/login-success sequences. The two real attacks are surfaced."
pause

# ---------------------------------------------------------------- 4. baseline comparison
banner "STEP 4 / 5   Same stream, hand-set rules instead of the learned scorer"
say "Same grouping, but decisions come from fixed severity/pattern rules I wrote by hand (the baseline)."
run "$PY src/prototype/run_engine.py --input data/scenarios/demo.jsonl --speed 200 --rules --quiet"
say "Compare the KPI cards: far more items reach the analyst. The learned scorer cuts them much further."
pause
if [ -e src/prototype/model_hard.json ]; then
  banner "STEP 4b / 5   The trade-off: a STRICTER model (trained against harder normal traffic)"
  say "Same stream, same grouping, but the scorer was trained with benign admin activity that looks like attacks."
  run "$PY src/prototype/run_engine.py --input data/scenarios/demo.jsonl --speed 200 --model src/prototype/model_hard.json --quiet"
  check_attacks
  say "Fewer items, but if a real attack shows NO above, it was silently dropped. This is the limit the stress test measures."
  pause
else
  note "(optional step skipped: src/prototype/model_hard.json not found; see NARRATION.md to create it)"
fi
say "Back to the learned scorer (fast replay) so the dashboard ends on it."
run "$PY src/prototype/run_engine.py --input data/scenarios/demo.jsonl --speed 400 --quiet"
pause

# ---------------------------------------------------------------- 5. evaluation
banner "STEP 5 / 5   Measured: items shown to the analyst vs attacks still surfaced (attacks never trained on)"
say "EASY background (attacks use tactics normal traffic does not): quick run, 3 repeats."
run "$PY src/prototype/experiment.py --baseline $J/joined_run1.csv $J/joined_run2.csv $J/joined_run3.csv $J/joined_run4.csv $J/joined_run5.csv --test variant=$J/joined_variant.csv --test exfil2=$J/joined_exfil2.csv --background-synthetic 3000 --repeats 3 --mix-hosts --out results/demo_easy 2>&1 | sed -n '/TEST STREAM/,/severity filter/p'"
pause
say "STRESS background (benign admin activity that shares the attack's tactics): the honest limit."
run "$PY src/prototype/experiment.py --baseline $J/joined_run1.csv $J/joined_run2.csv $J/joined_run3.csv $J/joined_run4.csv $J/joined_run5.csv --test variant=$J/joined_variant.csv --test exfil2=$J/joined_exfil2.csv --background-synthetic 3000 --repeats 3 --hard-noise --mix-hosts --out results/demo_stress 2>&1 | sed -n '/TEST STREAM/,/severity filter/p'"
pause

# ---------------------------------------------------------------- closing
banner "SUMMARY"
say "Works end to end: alerts -> incidents -> learned score -> live dashboard, with a guard rail and a digest."
say "Easy background: about 10x fewer items than hand-set rules, every injected attack still surfaced."
say "Stress background: the learned scorer loses recall. Cause: training attacks were recorded before the custom rules existed."
say "Next: re-record training attacks with the rules active, add context features (IP, user), test on real institute data."
note "Background noise is synthetic; attack runs are real lab alerts. Numbers show mechanics, not production performance."
echo
