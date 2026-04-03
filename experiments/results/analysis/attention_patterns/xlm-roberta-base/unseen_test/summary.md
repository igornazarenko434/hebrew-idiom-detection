# Attention Pattern Analysis

- Model: xlm-roberta-base
- Seed (best on seen): 456
- Seen F1 (seed): 0.9931
- Examples analyzed: 480
- Token length mismatches: 0

## Key Comparison
- PERFECT mean attention: 0.1778
- PARTIAL_END mean attention: 0.2558
- Relative drop: -43.9%

## Outputs
- `attention_by_error.csv`
- `attention_example_scores.csv`
- `experiments/results/analysis/attention_patterns/xlm-roberta-base/unseen_test/attention_ratio_correct_vs_partial_end.png`
- `experiments/results/analysis/attention_patterns/xlm-roberta-base/unseen_test/attention_ratio_by_error.png`
