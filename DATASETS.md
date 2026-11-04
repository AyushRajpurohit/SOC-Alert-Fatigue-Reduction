# Datasets

## Microsoft GUIDE

Public dataset: Microsoft Security Incident Prediction
https://www.kaggle.com/competitions/microsoft-security-incident-prediction

The project uses the publicly available GUIDE training data.

The original evidence-level records are processed into an
incident-level cache before chronological evaluation.

Chronological evaluation:
- Source incidents: 730,778
- Eligible graded incidents: 707,108
- Training: 424,264
- Calibration: 139,344
- Testing: 139,696
- Boundary-crossing incidents excluded: 3,804

The original dataset files and derived incident-level
prediction exports are not included in this repository.

## Institutional Wazuh

The institutional evaluation used 1,780 actual Wazuh alerts
from three fixed two-minute windows.

- Sample A: 535 alerts
- Sample B: 605 alerts
- Sample C: 640 alerts

The records were locally pseudonymized before evaluation.

Raw records, pseudonymized event-level records, identifiers,
pseudonymization keys, and generated event-level dashboard
states are not distributed.

The institutional records have no independently verified
malicious/benign labels.

The reported 33.37% reduction refers to default-visible
alert records, not measured analyst time or false positives.

## Controlled Laboratory and Synthetic Data

Controlled attack-chain recordings were used to develop and
test correlation and evidence-reduction behavior.

Synthetic fixtures were used for regression tests.

Synthetic results must not be interpreted as real-world
attack-detection performance.
