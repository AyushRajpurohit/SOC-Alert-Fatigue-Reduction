# Architecture

The semester project has two separate evaluation pipelines.

## GUIDE pipeline

```text
Public GUIDE records
    -> completed incident features and grades
    -> chronological train, calibration and test splits
    -> logistic regression ranking model
    -> calibration thresholds
    -> test ranking and demotion metrics
```

The GUIDE experiments also include offline drift diagnostics, shift monitoring, controlled replay, shadow update checks and walk forward evaluation. These checks did not deploy an adaptive model.

## Wazuh pipeline

```text
Authorized local Wazuh alert samples
    -> structured alert parsing
    -> time and entity based correlation
    -> reconstructed groups and observed links
    -> evidence requirements and mandatory alert rules
    -> greedy representative selection
    -> visible alerts and supporting alerts
    -> local analyst dashboard
```

The correlation engine reconstructs groups from the available alert fields. The reducer works within each group. It keeps the first and last alerts, protects alerts meeting the critical severity threshold and selects additional alerts until all of its defined evidence requirements are covered.

An alert outside the initial view is not deleted. The local dashboard lets the analyst inspect supporting alerts when needed.

## Relevant source files

- `src/guide/guide_temporal.py`: chronological training, calibration and testing
- `src/guide/guide_drift_monitor.py`: read only shift monitoring
- `src/guide/guide_shadow_update.py`: candidate update and gate simulation
- `src/guide/guide_walkforward.py`: walk forward evaluation
- `src/prototype/correlation_engine.py`: correlation and links
- `src/prototype/evidence_reducer.py`: requirement construction and greedy selection
- `src/prototype/reducer_institute_six_method.py`: three sample benchmark
- `src/prototype/integrated_replay.py`: local dashboard replay

The GUIDE ranking model was not used to produce the Wazuh alert selection results. They are complementary experiments, not one validated live pipeline.
