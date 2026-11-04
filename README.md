# SOC Alert Fatigue Reduction

A research prototype for reducing default-visible SOC security alerts while preserving explicitly defined investigation evidence.

## 1. Project Overview

The project investigates two complementary approaches:

1. Incident prioritization using the Microsoft GUIDE dataset.
2. Evidence-preserving alert reduction using Wazuh security alerts.

The implementation includes correlation, alert reduction, incident ranking, drift diagnostics, safe model-update simulations, and an interactive investigation dashboard.

## 2. Project Structure

- `src/guide/` — GUIDE data processing, temporal evaluation, drift detection, and model updates.
- `src/lab/` — Laboratory alert processing and attack-pattern mining.
- `src/prototype/` — Entity-aware correlation, evidence reduction, scoring, and dashboard.
- `demo/` — Demonstration materials.
- `docs/` — Technical documentation.
- `results/public/` — Reviewed aggregate evaluation outputs.

## 3. Microsoft GUIDE Evaluation

A chronological completed-incident evaluation was conducted on 139,696 test incidents.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.7141 |
| Average precision | 0.3387 |
| Recall at 99% target | 99.525% |
| Incident demotion at 99% target | 3.184% |
| Recall at 95% target | 95.980% |
| Incident demotion at 95% target | 14.673% |

These results evaluate retrospective incident prioritization, not live security detection.

## 4. Institutional Wazuh Evaluation

The evidence-reduction algorithm was evaluated on 1,780 locally pseudonymized institutional Wazuh alerts.

| Method | Visible alerts | View reduction | Defined evidence coverage |
|---|---:|---:|---:|
| All alerts | 1,780 | 0% | 100% |
| Severity-matched | 1,186 | 33.37% | 77.05% |
| Greedy-cover | 1,186 | 33.37% | 100% |

Greedy-cover preserved 2,392 out of 2,392 explicitly defined evidence requirements.

Under the original critical-level-12 policy, it retained only 10 of 20 alerts with severity level 10 or higher.

The reduction measures default-visible alerts, not analyst time saved or validated false-positive suppression.

## 5. Main Implemented Components

- Entity-aware Wazuh alert correlation.
- Evidence-constrained greedy alert selection.
- Six-method comparative evaluation.
- Retrospective incident-prioritization model.
- Temporal drift diagnostics and monitoring.
- Shadow-update and rollback simulations.
- Walk-forward evaluation.
- Integrated investigation dashboard.
- Experimental learned alert-novelty model.

## 6. Running Tests

Install dependencies:

    pip install -r requirements.txt

Run the prototype tests:

    python -m unittest discover -s src/prototype -p 'test_*.py' -v

Some tests require optional laboratory archives and are skipped when those archives are unavailable.

## 7. Research Limitations

- No live production SOC deployment.
- No measured analyst-time savings.
- Institutional Wazuh records do not have independent malicious/benign labels.
- Programmed evidence coverage has not been independently validated by SOC analysts.
- GUIDE experiments use completed-incident features.
- Learned-novelty selection did not improve upon the greedy baseline in the exploratory pilot.

## 8. Datasets and Privacy

See `DATASETS.md` for dataset descriptions and acquisition information.

Raw institutional Wazuh records, pseudonymized event-level data, sensitive identifiers, and generated event-level dashboard states are not included.
