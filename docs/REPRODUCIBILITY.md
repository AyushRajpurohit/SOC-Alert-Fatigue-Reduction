# Reproducibility

## Set up the environment

Use a supported Python 3 environment and install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the tests

```bash
python -m unittest discover -s src/prototype -p 'test_*.py' -v
```

Our last verified run completed 40 tests with 38 passing and two skipped because optional laboratory archives were absent.

## GUIDE

Get the public Microsoft Security Incident Prediction dataset from the link in `DATASETS.md`. The processing and evaluation scripts are in `src/guide/`.

The repository includes the reviewed aggregate metrics and split manifest in `results/public/`. It does not include the large source data, incident cache, model artifact or prediction files. Reproducing the full experiment needs the original data and sufficient local resources.

## Wazuh

Relevant files include `correlation_engine.py`, `evidence_reducer.py`, `reducer_six_method_eval.py` and `reducer_institute_six_method.py` in `src/prototype/`.

The institute benchmark uses three authorized local sample files and writes aggregate comparisons to a new output directory. Those sample files cannot be published here. The exact benchmark therefore requires approved access to the private data. The public tests use synthetic fixtures to check the algorithm's behavior.

## Dashboard replay

`src/prototype/integrated_replay.py` produces a local HTML dashboard and a state file. Read the command line help for its current arguments:

```bash
python src/prototype/integrated_replay.py --help
```

Do not upload a dashboard state file made from institutional alerts. Use synthetic or explicitly approved inputs for a distributable demonstration.

## Interpreting the output

GUIDE recall concerns graded TruePositive incidents. Wazuh reduction concerns the number of alerts initially displayed. Wazuh evidence coverage concerns only the requirements defined in the reducer. None of these are measured analyst time savings or proof of production attack detection.
