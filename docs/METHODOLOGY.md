# Methodology

## 1. Research Objective

The project studies two complementary SOC alert-fatigue problems:

1. Retrospective prioritization of analyst-graded security incidents.
2. Reduction of default-visible alerts while preserving explicitly defined evidence constraints.

The two evaluations use different datasets and must not be interpreted as one end-to-end experiment.

## 2. GUIDE Incident Prioritization

The Microsoft GUIDE data was converted from evidence-level records into completed-incident features.

A global chronological train/calibration/test protocol was used:

| Split | Incidents |
|---|---:|
| Training | 424,264 |
| Calibration | 139,344 |
| Test | 139,696 |

The baseline uses a logistic model trained on incident-level features, including MITRE technique and category information.

Calibration thresholds were selected for target true-positive recall levels of 99% and 95%.

The test ROC-AUC was 0.7141, with average precision 0.3387.

The evaluation uses completed incidents and privacy-adjusted timestamps. It does not establish real-time classification performance.

## 3. Entity-Aware Correlation

Wazuh alerts are grouped using observed temporal and entity relationships, rather than host identity alone.

The correlation engine conservatively uses observed host, source, and user relationships. Missing entity information is not fabricated.

Reconstructed groups are evaluation units, not independently verified ground-truth incidents.

## 4. Evidence-Constrained Reduction

Each reconstructed group contains a set of evidence requirements derived from its observed structured fields.

The greedy algorithm selects representative alerts that cover uncovered requirements.

Requirements include selected combinations of:
- Incident anchors and protected events
- Observed MITRE techniques
- Unknown or unmapped rule representation
- Observed correlation-link endpoints

The original alerts remain available as supporting evidence.

The method is a heuristic for a set-cover-style objective. Its coverage guarantees apply to the programmed requirements, not to all possible security-relevant information.

## 5. Six-Method Comparison

The institutional Wazuh evaluation compares:

- All alerts
- Severity-based matched-budget selection
- Rule deduplication
- One representative per technique
- First-cover selection
- Greedy-cover selection

The three fixed evaluation samples contain 535, 605, and 640 alerts.

The original benchmark uses critical level 12.

Greedy-cover selected 1,186 of 1,780 alerts for the default view, giving 33.37% reduction and 2,392/2,392 programmed requirement coverage.

Severity-based selection at the same visibility budget covered 1,843/2,392 requirements.

However, greedy-cover retained only 10/20 alerts with severity >=10; the severity comparator retained 20/20. This is an important trade-off.

## 6. Drift and Update Evaluation

Additional GUIDE experiments include:

- Temporal drift diagnostics
- Score-distribution monitoring
- MITRE-distribution monitoring
- Controlled drift replay
- Shadow-model training and promotion gates
- Rollback simulation
- Walk-forward evaluation

The reported shadow candidate did not pass the promotion gate. The active static baseline was therefore retained in the simulation.

These experiments were offline and did not deploy adaptive models into an operational SOC.

## 7. Learned-Novelty Pilot

An experimental logistic model was trained to predict Wazuh rule IDs from preceding context.

Prediction surprise was used as a candidate alert-selection signal.

On the exploratory Sample C evaluation, learned novelty achieved 73.75% next-rule prediction accuracy but reduced fewer default-visible alerts than the existing greedy method.

Prediction accuracy is not security-detection accuracy, and this result does not establish that novelty identifies safe-to-hide alerts.

## 8. Limitations

- Institutional Wazuh samples have no independently verified incident labels.
- Evidence coverage is measured against algorithm-defined requirements.
- No actual analyst-time reduction has been measured.
- No production deployment has been performed.
- Institutional evaluation covers one deployment and three short collection windows.
- GUIDE incident features are available only after the incident completes.
