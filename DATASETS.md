# Datasets

## Microsoft GUIDE

We used the public Microsoft Security Incident Prediction data:
https://www.kaggle.com/competitions/microsoft-security-incident-prediction

The original data contains evidence records linked to alerts and incidents. We generated an incident level cache before running chronological experiments. The source contained 730,778 incidents. Of these, 707,108 passed the grade and data checks. The evaluated split contains 424,264 training incidents, 139,344 calibration incidents and 139,696 test incidents. Another 3,804 incidents were left out because they crossed time boundaries.

The original GUIDE records, processed cache, model file and prediction exports are not uploaded here. Reviewed aggregate metrics and the split manifest are included under `results/public/`.

## Institute Wazuh

We evaluated three authorized two minute windows of institute Wazuh data after local sanitization.

| Sample | Alerts |
|---|---:|
| A | 535 |
| B | 605 |
| C | 640 |
| Total | 1,780 |

These records have no independently checked malicious or benign labels. Even locally sanitized event data may be sensitive, so raw records, sanitized event records, identifying details, private models and dashboard state files are excluded.

The 33.37% result counts alerts omitted from the initial view. It is not a measurement of analyst hours saved.

## Lab and synthetic fixtures

We used controlled lab observations to understand Wazuh rules and attack chain behavior. Unit tests use synthetic fixtures. Synthetic test outcomes should not be presented as production detection results.
