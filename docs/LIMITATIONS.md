# Limitations

## We measured what is visible, not analyst time

The reported 33.37% reduction means that 594 of 1,780 alerts are outside the initial view. Analysts can still open those alerts. We did not measure how long investigations take or whether fatigue improves.

## Coverage depends on our definitions

The reducer covers requirements derived from selected fields and correlation links. It does not represent every possible security detail. For instance, two alerts with the same host and technique can still have different source IPs. Covering the shared tokens does not guarantee that the difference is visible immediately.

## We do not have independent attack labels for the institute samples

The 573 reconstructed groups are not confirmed attacks. We cannot measure missed attacks or attack detection recall using these records.

## The institute evaluation is small

The three samples cover short windows from one environment. We have not shown that the same reduction would hold over weeks of activity or at other SOCs.

## Correlation can make mistakes

Missing fields and weak relationships can lead to incorrect grouping. An observed link is a correlation decision, not proof of a causal attack step.

## Severity retention was not complete under the frozen policy

The original reducer protected severity 12 or higher. Only 10 of the 20 alerts at severity 10 or above stayed in the initial view. A stricter policy needs its own controlled evaluation.

## Greedy selection is not always the smallest possible solution

The method chooses the alert covering the most new requirements at each step. It meets its programmed constraints but may use more alerts than an exact minimum selection.

## GUIDE is retrospective

GUIDE features are complete only after an incident ends. Timestamps are privacy adjusted, organizations are not held out independently and the logistic scores are not independently probability calibrated. Drift and shadow update experiments were not live deployments.

## Production behavior is untested

The prototype has not been validated for automatic suppression or response. Analyst review, independent labels, stronger baselines and longer evaluation periods are still needed.
