# Reproducibility

## 1. Environment

Use Python 3 with dependencies installed from the repository requirements file.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## 2. Unit Tests

Run the correlation, reducer, integration, and evaluation tests:

    python -m unittest discover -s src/prototype -p 'test_*.py' -v

Some tests requiring optional archives will be skipped when those archives are unavailable.

## 3. GUIDE Dataset

Obtain the public Microsoft GUIDE dataset using the source information in `DATASETS.md`.

The original data and generated incident cache are intentionally excluded from this repository.

The main pipeline is implemented under `src/guide/`.

Published aggregate results:

- `results/public/guide_temporal_manifest.json`
- `results/public/guide_temporal_metrics.json`

The manifest documents the chronological split and model settings.

Regenerating the results requires the original GUIDE data and sufficient local compute/storage.

## 4. Wazuh Evidence Reduction

The algorithm and evaluator are implemented in:

- `src/prototype/correlation_engine.py`
- `src/prototype/evidence_reducer.py`
- `src/prototype/reducer_six_method_eval.py`
- `src/prototype/reducer_institute_six_method.py`

The institutional samples are not publicly distributed.

The exact institutional benchmark cannot be independently rerun without approved access to those samples.

The public unit tests exercise correlation, evidence preservation, and evaluation logic using generated fixtures.

## 5. Integrated Replay

The retrospective integrated replay is implemented in:

    src/prototype/integrated_replay.py

For available options:

    python src/prototype/integrated_replay.py --help

The original institutional dashboard state is not distributed.

For a local demonstration, use explicitly synthetic or otherwise approved alert files.

## 6. Evaluation Interpretation

- GUIDE recall measures retention of analyst-graded true-positive incidents.
- Wazuh reduction measures default-visible alert records.
- Wazuh coverage measures preservation of programmed evidence requirements.
- Neither experiment demonstrates measured analyst-time savings.
- Institutional results are not proof of correct malicious-incident detection.
