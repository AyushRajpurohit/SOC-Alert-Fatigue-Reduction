# Results from the semester

This page records completed experiments. The numbers below come from saved evaluations, not estimates of live SOC performance.

## GUIDE temporal ranking

| Measure | Result |
|---|---:|
| Training incidents | 424,264 |
| Calibration incidents | 139,344 |
| Test incidents | 139,696 |
| Test organizations | 2,971 |
| TruePositive test incidents | 25,474 |
| Alerts within test incidents | 184,464 |
| ROC-AUC | 0.7141 |
| Average precision | 0.3387 |

| Recall target | Actual test recall | Incidents demoted | Incident demotion | Underlying alert demotion |
|---|---:|---:|---:|---:|
| 99% | 99.525% | 4,448 | 3.184% | 3.899% |
| 95% | 95.980% | 20,497 | 14.673% | 13.122% |

## GUIDE monitoring and update studies

The read only shift monitor flagged three candidate events but confirmed no shift episodes under its policy. A separately trained shadow candidate failed its promotion gate, so the static baseline was retained. The held out ROC-AUC was about 0.7141 for static and 0.7135 for shadow. Two walk forward folds also rejected replacement candidates. Controlled replay tested response to constructed distribution changes. These are retrospective and simulated evaluations.

## Institute Wazuh correlation

| Sample | Input alerts | Reconstructed groups | Observed links |
|---|---:|---:|---:|
| A | 535 | 204 | 331 |
| B | 605 | 160 | 445 |
| C | 640 | 209 | 431 |
| Total | 1,780 | 573 | 1,207 |

## Six method comparison

| Method | Initially visible | Reduction | Defined requirement coverage |
|---|---:|---:|---:|
| All alerts | 1,780 | 0.00% | 100% |
| Severity budget | 1,186 | 33.37% | 77.05% |
| Rule deduplication | 612 | 65.62% | 53.47% |
| One per technique | 589 | 66.91% | 51.67% |
| First cover | 1,729 | 2.87% | 100% |
| Greedy cover | 1,186 | 33.37% | 100% |

The greedy method selected 1,186 alerts for the initial view and kept another 594 as supporting alerts. The selected alerts covered 2,392 of 2,392 defined requirements. At the same visibility budget, severity selection covered 1,843 requirements, or 77.05%.

The frozen greedy policy protected severity 12 and above. It left 10 of 20 alerts at severity 10 or above in the initial view. This is an important safety limitation.

The saved comparison was independently rerun and all 24 result rows matched the original.

## Sample B dashboard

The separate dashboard replay used severity threshold 10. It processed 605 alerts in 160 groups, kept 391 in the initial view and 214 as supporting alerts. The reduction was 35.37% with 779 of 779 requirements covered and 10 of 10 severity 10 or higher alerts retained. Its scorer marked one group ESCALATE and 159 DEMOTE. These are experimental decisions, not verified attack labels.

## Lab controls and novelty pilot

On a synthetic correlation fixture, the evidence linked approach achieved pairwise precision and recall of 1.0. That is a result on constructed test data, not institute ground truth.

An exploratory next rule prediction model achieved 73.75% accuracy on its test sample versus 14.69% for a majority rule baseline. Its alert selection retained more alerts than greedy cover under the tested setup, so it was not adopted as the final reducer. Next rule accuracy does not measure attack detection.

## Regression tests

The final verified suite ran 40 tests. Thirty eight passed and two optional archive tests were skipped.

## Interpretation

The primary measured outcome is a reduction in the number of alerts shown initially while preserving every requirement our implementation defines. We have not established equal reductions in analyst working time or guaranteed preservation of all attack evidence.
