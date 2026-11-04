# Mid-evaluation video: working prototype demo

**What this is.** A recording of your REAL system running on your REAL lab alerts, so you can play it at the
evaluation instead of running everything live. The script `demo_run.sh` runs the real commands in the right order and
prints a caption before each step; you only press Enter to move on and talk over it.

**Be upfront about two things on screen or in your voice-over** (examiners respect this, and it protects you):
1. The attack alerts are real (your Wazuh lab runs); the *background noise* is synthetic.
2. The numbers show that the mechanism works, not production performance.

Total time to make the video: about 45-60 minutes. Final video: about 7-8 minutes with step 4b (trim the waits).

---

## 1. Prepare (do this once, without recording)

```bash
cd ~/Desktop/SSE
source venv/bin/activate
unzip -o ~/Downloads/demo_kit.zip -d ~/Desktop/SSE      # creates demo/demo_run.sh and demo/NARRATION.md
ls src/prototype/model.json                              # must exist (train_model.py if not, see README)
```

**Optional step 4b needs the stricter model.** If `src/prototype/model_hard.json` does not exist, the script skips that
step and says so. To create it (a few seconds):

```bash
python src/prototype/train_model.py --baseline data/lab/joined/joined_run{1,2,3,4,5}.csv \
  --background-synthetic 3000 --hard-noise --mix-hosts --out src/prototype/model_hard.json
```

**Dry run, no recording, no pauses** (about 2-3 minutes). It must end with SUMMARY and no error:

```bash
AUTO=1 AUTO_PAUSE=1 DEMO_SPEED=400 bash demo/demo_run.sh
```

If it prints `MISSING: ...` it tells you which file is absent and runs nothing. If it says the port is in use, stop
the old dashboard server (Ctrl+C in its terminal) or use `DEMO_PORT=8001`.

**Check that the captions match your screen.** The script's captions say: variant = 100% match / 60% coverage and
"static rule MISSED"; exfil2 match falls to 50%. Those are your real week-1/week-2 numbers. If the table on your screen
differs, edit the caption text in `demo/demo_run.sh` (search for `variant: 100% match`).

## 2. Set up the screen

- **Terminal:** font size 16-18 (Terminal > Settings > Profiles > Text). Window on the left half. Run `clear`.
- **Browser:** right half, zoom 100%, hide the bookmarks bar, one tab. Don't open `localhost:8000` yourself; the script
  opens it at step 3 (drag the new window to the right half the first time, or leave a browser window there already).
- Turn on **Do Not Disturb / Focus** so notifications don't appear. Close Slack, mail, anything with personal content.
- Dark or light mode: either, but keep it the same for the whole recording.

## 3. Record

macOS: press **Cmd+Shift+5** > *Record Entire Screen* > **Options** > choose your **Microphone** > *Record*.
(Or QuickTime Player > File > New Screen Recording.) Stop with the stop button in the menu bar; the file lands on the Desktop.

Then in the terminal:

```bash
bash demo/demo_run.sh
```

Press **Enter** after you finish saying each part. If you stumble, just keep going; you can trim in QuickTime
(Edit > Trim) or iMovie. Recording in two or three chunks is fine.

**No microphone or not confident talking over it?** Record silent, and narrate live at the evaluation while the video
plays. The captions on screen already carry the message.

---

## 4. Shot list and voice-over (about 7 minutes)

Read these in your own words. Numbers: say "about", and let the screen show the exact value.

### 0. Intro (about 40 s) - the title screen
> Security teams receive thousands of alerts a day, and most are routine. Our project reduces that load without losing
> real attacks. The alerts come from Wazuh, which we don't change. We group alerts into incidents, compare each
> incident's sequence of MITRE tactics with attack sequences learned from our own lab runs, and surface only what looks
> like an attack. Nothing is deleted: low-priority incidents go to a digest.

### 1. Lab proof (about 80 s) - the results table, then the replay
> First, a lab proof. We ran one attack five times and learned its pattern: Credential Access, Initial Access,
> Persistence, Initial Access, Defense Evasion. Now two runs the system has never seen.
> The variant skips persistence. The table shows 100% match, meaning everything in it fits the known attack, and 60%
> coverage, meaning three of the five steps. Wazuh's own rule for the full chain missed it.
> The second run is a different attack, exfiltration. Watch the replay: match starts at 100% and drops to 50% when
> Collection and Exfiltration appear, because those steps are not part of the known pattern.

### 2. The test stream (about 30 s)
> To test at scale, we hide the real attack alerts in noise: 500 synthetic background alerts plus 33 real lab alerts,
> placed on busy hosts. Because we know which alerts are the attack, we can measure what the system keeps.

### 3. Live dashboard (about 2 min) - say this while it streams
> Here is the analyst queue filling as alerts stream in. Each row is an incident, not an alert. The chain shows the
> attack steps seen so far, the bar shows how much of the known attack is present, and the reason says why it was
> surfaced. The digest on the right holds everything that was demoted; nothing is deleted.
> At the end, about 530 alerts became a handful of items, around 99% fewer. Both real attacks are in the queue. The
> exfiltration run is escalated by a safety rule that always escalates late-stage tactics. A few items are false
> alarms: failed logins followed by a success look like the start of an attack.

### 4. Baseline comparison (about 40 s)
> Same stream, but decisions come from the hand-set rules we built first. Many more items reach the analyst. The
> learned scorer does the same job with a fraction of the workload.

### 4b. The trade-off (about 45 s) - optional, strongest if the line says "NO (missed!)"
> This is the same stream, scored by a stricter model that was trained against harder normal traffic, the kind of
> administrator activity that looks like an attack. It shows even fewer items. Now look at the line the script prints:
> it checks whether each real attack is still in the queue. [Read what it says.] If it says NO, a real attack was
> silently dropped. Showing fewer alerts is easy; showing fewer alerts without losing attacks is the hard part, and this
> is the trade-off we measure in the next step.

*Say what the screen says.* The check line is computed from the engine's actual output, so if it shows YES/YES for your
stricter model, say that both attacks are still surfaced and skip the "silently dropped" sentence.

### 5. Evaluation (about 2 min) - the two tables
> Now a measurement on attacks the model never trained on, repeated with different placements.
> On this background, grouping alone leaves about 500 items, the hand-set rules about 120, and the learned scorer
> about 12, with every attack still surfaced.
> Then a stress test, where normal admin activity uses the same tactics as the attack. Here the learned scorer loses
> attacks, mostly the variant, while the rules keep them but show about 200 items. We traced the cause: our training
> attacks were recorded before our custom detection rules existed, so the model cannot learn that signal.

### 6. Closing (about 30 s)
> Next we will re-record the training attacks with the rules active, add context such as source address and user,
> and test on real institute data. One caveat: the background noise here is synthetic; the attack alerts are real.

**Optional extra slide (30 s) if you have time:** on Microsoft's public GUIDE dataset, which has real analyst labels,
technique- and category-level features rank incidents within an organization with AUC about 0.65, matching
vendor-specific detector IDs; across unseen organizations the reduction is modest, about 9% of incidents at about
96% recall, because analysts' grading policies differ between organizations.

---

## 5. If something goes wrong

| Problem | Fix |
|---|---|
| `MISSING: ...` at start | Put that file in place (see README); nothing was run |
| Port already in use | Stop the old server (Ctrl+C) or `DEMO_PORT=8001 bash demo/demo_run.sh` (open that port in the browser) |
| Browser says "no state file yet" | Engine hasn't started; press Enter at step 3 and wait a few seconds |
| Browser didn't open | Open `http://localhost:8000` yourself |
| `NO (missed!)` appears for the DEFAULT model | Don't hide it: it is a real result. Tell me and we will look at it before you record |
| A step is slow | It is fine; trim the wait in the editor. Replay speed: `DEMO_SPEED=80 bash demo/demo_run.sh` |
| You want a retake of one step only | Ctrl+C, then run that step's command from `demo_run.sh` by hand and record only that |

## 6. After recording

1. Play it back once with sound: terminal text readable? numbers visible? no notifications?
2. Trim waits (QuickTime: Edit > Trim; iMovie for cuts). Export as **MP4 (1080p)**.
3. Keep **two copies** (laptop and a USB stick or cloud), and test the video on the evaluation computer if possible.
4. Rehearse the voice-over twice before recording; it should sound like explaining, not reading.

## 7. Questions you may get (answers from your own results)

- **How do you decide what to show?** Wazuh's rules generate the alerts. We group them per host, map each to a MITRE
  tactic with a lookup table, then score each incident by how closely its tactic sequence matches attack sequences
  learned from lab runs compared with normal traffic. A logistic model over sequence features gives the score; its
  threshold comes from a false-alarm budget. A few fixed safety rules sit on top.
- **Why synthetic background?** We have not yet been able to evaluate on real background traffic; the attack alerts
  themselves are real lab alerts. The next step is to repeat the evaluation on real institute data.
- **Isn't the learned part tiny?** Yes: five training runs, count tables and seven weights. It is a proof of concept;
  the stress test shows its limit and we know the fix.
- **What about false alarms?** In the demo, 4 of 6 surfaced items were false alarms (login failure then success).
  The easy test shows about 10 false-alarm items out of about 490 incidents; the stress test is harder.
- **Does it work with other SIEMs?** The engine works on a common alert record. The tactic lookup table is
  Wazuh-specific, so another SIEM needs its own mapping; separate experiments on Microsoft's GUIDE data show the
  technique-level features carry over, but the learned weights do not.
