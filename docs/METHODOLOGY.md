# Methodology

## What we studied

We investigated two parts of the alert fatigue problem. The GUIDE experiments ask whether completed security incidents can be ranked for review. The Wazuh experiments ask whether a smaller set of alerts can represent the evidence our system is designed to preserve. The results come from separate evaluations.

## GUIDE ranking model

We processed the public GUIDE data into incident records and removed cases without the information needed for evaluation. We split eligible incidents by time. The training set contains 424,264 incidents, the calibration set has 139,344 and the later test set has 139,696. Incidents spanning the split boundaries were excluded.

Our baseline is logistic regression. It uses binary MITRE technique and detection category indicators together with three numeric features: log alert count, presence of at least one technique and log technique count. We learned feature vocabulary from the training data only.

The model ranks completed incidents by its logistic score. We selected thresholds on calibration incidents to target 99% and 95% recall of incidents graded TruePositive. We then applied those fixed thresholds to the test data. The scores are not independently calibrated probabilities.

## Wazuh correlation

We processed three short samples of approved and locally sanitized Wazuh alerts. The correlation engine uses time and observed entity relationships to reconstruct groups. It does not invent missing host, user or source values. These groups are working investigation units, not verified attack incidents.

## Evidence requirements and selection

For each group, the reducer builds requirements from MITRE techniques, observed hosts, host IP fallbacks, unmapped host and rule combinations and observed correlation links.

The first and last alert in each group are mandatory. So is every alert at or above the configured critical severity threshold. In the original experiment this threshold was 12.

An unmapped rule contributes a requirement but is not automatically mandatory on its own. For a link with uniquely identified endpoints, either endpoint can cover its requirement. When a link cannot be resolved safely, all alerts in the group are retained.

Greedy selection starts with the mandatory alerts and then adds whichever alert covers the largest number of remaining requirements. Ties favor higher severity and then earlier alert order. It stops when all defined requirements are represented. The remaining alerts are still available as supporting information.

## Comparison methods

We compared six approaches: showing all alerts, severity based selection at a matched visibility budget, rule deduplication, one representative per MITRE technique, first cover and greedy cover.

Greedy selection used the frozen policy with critical severity 12, mandatory anchors and conservative link and unmapped rule handling. We compared methods using alert visibility, defined requirement coverage and severity retention.

## Additional experiments

We also ran GUIDE drift diagnostics, read only shift monitoring, controlled distribution changes, guarded shadow update tests and walk forward checks. The shadow candidate failed the promotion gate and was not deployed. A separate next rule prediction pilot was tested but did not improve alert selection compared with greedy cover.

## What the measurements mean

GUIDE recall describes how many graded TruePositive incidents stayed above the demotion threshold. Wazuh coverage describes how many requirements defined by our program remain represented. Neither value guarantees that every attack was detected. We did not measure analyst time savings or deploy automatic suppression in a live SOC.
