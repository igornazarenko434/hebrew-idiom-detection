# Confidence Calibration Summary

- CLS confidence uses stored prediction probabilities
- SPAN confidence recomputed from logits per example

Outputs:
- experiments/results/analysis/calibration/cls/<model>/<split>/confidence_by_error.csv
- experiments/results/analysis/calibration/span/<model>/<split>/confidence_by_error.csv
- paper/figures/calibration/<task>/<model>/<split>/confidence_by_error.png