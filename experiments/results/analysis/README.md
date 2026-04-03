# Analysis Results

Aggregated analysis outputs from all fine-tuning experiments (6 models x 2 tasks x 3 seeds = 36 runs).

## Key Files

| File | Description |
|------|-------------|
| `finetuning_summary.md` | Main results table with mean F1, std, and 95% bootstrap CIs |
| `finetuning_summary.csv` | Same data in CSV format |
| `statistical_significance.txt` | Summary of significance tests |

## Analysis Directories

| Directory | Analysis | Key Finding |
|-----------|----------|-------------|
| `error_analysis/` | Error categorization and distribution | Partial-end truncation dominates (1,761 vs 1 partial-start) |
| `generalization/` | Seen vs unseen performance gap | CLS drops 0.5-3.5 F1 points; SPAN drops 23-41 points |
| `per_idiom_f1/` | Per-idiom difficulty ranking | Long unseen idioms (5+ tokens) show 70-100% error rates |
| `statistical_tests/` | Paired t-tests with Bonferroni correction | 60 pairwise comparisons across all conditions |
| `token_importance/` | Integrated Gradients + attention weights | Models attend to idiom-internal tokens for seen, context for unseen |
| `attention_patterns/` | Attention head visualization | Specific heads specialize in boundary detection |
| `embedding_space/` | t-SNE visualization (seen vs unseen) | Unseen idioms cluster differently in embedding space |
| `calibration/` | Confidence calibration analysis | Models are overconfident on unseen span predictions |
| `consistency/` | CLS vs SPAN cross-task agreement | High CLS-SPAN agreement on seen, divergence on unseen |
| `error_factors/` | Error by length, position, directionality | Idiom length is strongest predictor of unseen errors |
| `learning_curves/` | Training loss/F1 trajectories | Models converge within 3-5 epochs |
| `interpretability/` | Morphology + boundary heatmaps | Hebrew morphological variants affect boundary detection |
| `qualitative_errors/` | Example error cases with analysis | Curated examples for paper appendix |
| `contextual_distractors/` | Context token dominance analysis | Models over-rely on context words for unseen idioms |
| `morphology_verification/` | Hebrew morphological variant statistics | Verified up to 35 variants per idiom |

## Reproducing

All analysis can be regenerated from the evaluation predictions:

```bash
python src/analyze_finetuning_results.py      # Summary tables
python src/analyze_generalization.py           # Generalization gap
python src/analyze_error_distribution.py       # Error analysis
python scripts/statistical_tests.py            # Statistical tests
python scripts/analyze_per_idiom_f1.py         # Per-idiom analysis
```

See [docs/REPRODUCING.md](../../../docs/REPRODUCING.md) for the full pipeline.
