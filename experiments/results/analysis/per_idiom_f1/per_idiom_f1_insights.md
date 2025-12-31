# Per-Idiom Insights (Unseen Set Deep Dive)
**Generated:** 2025-12-31 14:19:28

## Executive Summary
- This report focuses on **SPAN task performance on the Unseen test set** (6 held-out idioms).
- Difficulty is defined as lower average F1 across all models and seeds.
- Error categories are drawn from the shared taxonomy (PERFECT, PARTIAL_END, MISS, etc.).

## Unseen Idioms Ranked by Difficulty (SPAN)
|   idiom_id | base_pie             |   f1_mean |
|-----------:|:---------------------|----------:|
|         49 | רץ אחרי הזנב של עצמו |    0.0008 |
|         19 | חצה קו אדום          |    0.6837 |
|         33 | נשאר מאחור           |    0.7277 |
|         55 | שבר שתיקה            |    0.8221 |
|         20 | חתך פינה             |    0.8980 |
|          2 | איבד את הראש         |    0.9971 |

## Mini Interpretability Block: Why Idiom 49 Is Hard
- Idiom: **רץ אחרי הזנב של עצמו** (id=49)
- Mean F1 across models: **0.000823**
- Dominant error types (all models/seeds):
| error_category   |   count |
|:-----------------|--------:|
| PARTIAL_END      |    1119 |
| MISS             |      42 |
| MULTI_SPAN       |      27 |
| PARTIAL_BOTH     |       7 |
| WRONG_SPAN       |       4 |
| PERFECT          |       1 |
- Interpretation: The overwhelming dominance of **PARTIAL_END** suggests models detect the idiom but truncate the ending boundary, which points to a boundary alignment issue rather than a failure to detect the idiom at all.
- Hypotheses to test:
  - Surface form variability at the end of the idiom (suffixes, clitics, punctuation).
  - Tokenization alignment issues for the last token(s).
  - Idiom length or syntactic attachment causing boundary drift.

## Deep Dive by Unseen Idiom (SPAN)
### Idiom 2: איבד את הראש
- Mean F1 (across models): **0.9971**
- Error category distribution (all models/seeds):
| error_category   |   count |
|:-----------------|--------:|
| PERFECT          |    1194 |
| MULTI_SPAN       |       4 |
| WRONG_SPAN       |       1 |
| MISS             |       1 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 19: חצה קו אדום
- Mean F1 (across models): **0.6837**
- Error category distribution (all models/seeds):
| error_category   |   count |
|:-----------------|--------:|
| PERFECT          |     801 |
| PARTIAL_END      |     280 |
| MISS             |      72 |
| MULTI_SPAN       |      22 |
| WRONG_SPAN       |      19 |
| PARTIAL_START    |       5 |
| PARTIAL_BOTH     |       1 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 20: חתך פינה
- Mean F1 (across models): **0.8980**
- Error category distribution (all models/seeds):
| error_category   |   count |
|:-----------------|--------:|
| PERFECT          |    1051 |
| MISS             |      52 |
| WRONG_SPAN       |      24 |
| MULTI_SPAN       |      22 |
| PARTIAL_END      |      22 |
| EXTEND_END       |      21 |
| SHIFT            |       6 |
| PARTIAL_START    |       2 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 33: נשאר מאחור
- Mean F1 (across models): **0.7277**
- Error category distribution (all models/seeds):
| error_category   |   count |
|:-----------------|--------:|
| PERFECT          |     782 |
| MISS             |     279 |
| PARTIAL_END      |      66 |
| WRONG_SPAN       |      58 |
| MULTI_SPAN       |      13 |
| EXTEND_END       |       2 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 49: רץ אחרי הזנב של עצמו
- Mean F1 (across models): **0.0008**
- Error category distribution (all models/seeds):
| error_category   |   count |
|:-----------------|--------:|
| PARTIAL_END      |    1119 |
| MISS             |      42 |
| MULTI_SPAN       |      27 |
| PARTIAL_BOTH     |       7 |
| WRONG_SPAN       |       4 |
| PERFECT          |       1 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 55: שבר שתיקה
- Mean F1 (across models): **0.8221**
- Error category distribution (all models/seeds):
| error_category   |   count |
|:-----------------|--------:|
| PERFECT          |     952 |
| MISS             |     102 |
| PARTIAL_END      |      71 |
| WRONG_SPAN       |      46 |
| MULTI_SPAN       |      22 |
| SHIFT            |       4 |
| EXTEND_END       |       3 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

## Next Verification Steps
- Inspect 10–15 examples per idiom to confirm boundary failure patterns.
- Compare tokenization (`tokens`) with predicted spans to spot systematic offset patterns.
- Cross-reference with `create_prediction_report.py` for qualitative error inspection.
