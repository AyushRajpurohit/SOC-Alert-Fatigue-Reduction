# SOC Alert Fatigue Reduction

This project looks at ways to make security alerts easier to review. We studied incident ranking with the public Microsoft GUIDE dataset and alert selection with Wazuh records. The two parts were tested separately.

Our main Wazuh result is a **33.37% reduction in alerts shown in the initial view** while covering every evidence requirement defined by our reducer. This does not mean that 33.37% of analyst work was saved or that every real attack was detected.

## What we built

- A logistic regression baseline for ranking completed GUIDE incidents
- GUIDE experiments for drift monitoring, controlled changes, shadow updates and walk forward testing
- A Wazuh correlation engine that groups alerts using observed time and entity information
- A greedy reducer that chooses representative alerts while keeping required evidence visible
- A six method comparison on three short, authorized institute Wazuh samples
- A local dashboard where analysts can expand supporting alerts
- Tests and an exploratory next rule prediction model

## GUIDE results

We trained on 424,264 earlier incidents, selected score thresholds with 139,344 calibration incidents and tested on 139,696 later incidents.

| Measure | Test result |
|---|---:|
| ROC-AUC | 0.7141 |
| Average precision | 0.3387 |
| Actual recall at 99% target | 99.525% |
| Incident demotion at 99% target | 3.184% |
| Actual recall at 95% target | 95.980% |
| Incident demotion at 95% target | 14.673% |

These results describe retrospective ranking of completed incidents. They are not measurements of live SOC detection.

## Wazuh results

Three two minute windows contained 1,780 alerts. Our correlation engine reconstructed 573 groups and recorded 1,207 links. The groups and links are not independently verified attack chains.

| Selection method | Initially visible alerts | Reduction | Defined evidence coverage |
|---|---:|---:|---:|
| Show everything | 1,780 | 0% | 100% |
| Severity selection at a matched budget | 1,186 | 33.37% | 77.05% |
| Greedy evidence selection | 1,186 | 33.37% | 100% |

Greedy selection kept 1,186 alerts in the initial view and retained 594 as supporting alerts. It covered all 2,392 requirements created by the program. With the original severity threshold of 12, only 10 of 20 alerts at severity 10 or higher remained initially visible. This is a limitation of the frozen setup.

The separate Sample B dashboard uses a severity threshold of 10. It shows 391 of 605 alerts and has a 35.37% initial view reduction. The dashboard result and the three sample benchmark use different settings.

## Repository layout

- `src/guide/` contains the GUIDE model and drift experiments.
- `src/lab/` contains laboratory alert processing tools.
- `src/prototype/` contains correlation, reduction, evaluation, the dashboard and tests.
- `docs/` explains the approach, results and limitations.
- `results/public/` contains reviewed GUIDE aggregate outputs.
- `demo/demo_run.sh` is the laboratory demo script.

## Running the tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s src/prototype -p 'test_*.py' -v
```

The last verified run completed 40 tests with 38 passing and two skipped because optional archives were not available.

## Documentation

- [Datasets](DATASETS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Methodology](docs/METHODOLOGY.md)
- [Results](docs/RESULTS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Limitations](docs/LIMITATIONS.md)
- [Work planned after the semester](docs/FUTURE_WORK.md)

## Data access and privacy

Raw Wazuh records, locally sanitized event records, private models and event level dashboard states are not part of this repository. The institute evaluation cannot be reproduced without approved access to the original samples. See [Datasets](DATASETS.md).

## Project status

This repository documents the semester implementation and its verified experiments. More research is planned after the semester, particularly stronger baselines, better safety requirements and evaluation with independent ground truth. None of that planned work is reported as completed here.
