# Per-Idiom Insights (Unseen Set Deep Dive)
**Generated:** 2026-01-02 23:32:11

## Executive Summary
- This report focuses on **SPAN task performance on the Unseen test set** (6 held-out idioms).
- Difficulty is defined as lower average F1 across all models and seeds.
- Error categories are drawn from the shared taxonomy (PERFECT, PARTIAL_END, MISS, etc.).

## Unseen Idioms Ranked by Difficulty (SPAN)
|   idiom_id | base_pie             |   f1_mean |
|-----------:|:---------------------|----------:|
|         49 | רץ אחרי הזנב של עצמו |    0.0232 |
|         33 | נשאר מאחור           |    0.4901 |
|         19 | חצה קו אדום          |    0.6455 |
|         55 | שבר שתיקה            |    0.9187 |
|         20 | חתך פינה             |    0.9451 |
|          2 | איבד את הראש         |    0.9911 |

## Mini Interpretability Block: Why Idiom 49 Is Hard
- Idiom: **רץ אחרי הזנב של עצמו** (id=49)
- Mean F1 across models: **0.023167**
- Dominant error types (all models/seeds):
|              |   count |
|:-------------|--------:|
| PARTIAL_END  |    1319 |
| PERFECT      |      31 |
| MISS         |      26 |
| MULTI_SPAN   |      26 |
| PARTIAL_BOTH |      23 |
| WRONG_SPAN   |      15 |
- Interpretation: The dominant category suggests boundary vs detection issues; verify with qualitative samples.

**Impact on overall SPAN Unseen F1:**
- With idiom 49: 0.6839
- Without idiom 49: 0.8247
- Delta: 0.1408

## Deep Dive by Unseen Idiom (SPAN)
### Idiom 49: רץ אחרי הזנב של עצמו
- Mean F1 (across models): **0.0232**
- Error category distribution (all models/seeds):
|              |   count |
|:-------------|--------:|
| PARTIAL_END  |    1319 |
| PERFECT      |      31 |
| MISS         |      26 |
| MULTI_SPAN   |      26 |
| PARTIAL_BOTH |      23 |
| WRONG_SPAN   |      15 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 33: נשאר מאחור
- Mean F1 (across models): **0.4901**
- Error category distribution (all models/seeds):
|             |   count |
|:------------|--------:|
| MISS        |     712 |
| PERFECT     |     570 |
| WRONG_SPAN  |      96 |
| PARTIAL_END |      45 |
| MULTI_SPAN  |      15 |
| EXTEND_END  |       2 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 19: חצה קו אדום
- Mean F1 (across models): **0.6455**
- Error category distribution (all models/seeds):
|               |   count |
|:--------------|--------:|
| PERFECT       |     901 |
| PARTIAL_END   |     347 |
| MISS          |     123 |
| WRONG_SPAN    |      46 |
| MULTI_SPAN    |      22 |
| PARTIAL_START |       1 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 55: שבר שתיקה
- Mean F1 (across models): **0.9187**
- Error category distribution (all models/seeds):
|             |   count |
|:------------|--------:|
| PERFECT     |    1296 |
| MISS        |      59 |
| PARTIAL_END |      35 |
| WRONG_SPAN  |      31 |
| MULTI_SPAN  |      17 |
| EXTEND_END  |       1 |
| SHIFT       |       1 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 20: חתך פינה
- Mean F1 (across models): **0.9451**
- Error category distribution (all models/seeds):
|              |   count |
|:-------------|--------:|
| PERFECT      |    1338 |
| MISS         |      36 |
| WRONG_SPAN   |      19 |
| MULTI_SPAN   |      14 |
| PARTIAL_END  |      14 |
| EXTEND_END   |      13 |
| SHIFT        |       4 |
| EXTEND_START |       1 |
| EXTEND_BOTH  |       1 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

### Idiom 2: איבד את הראש
- Mean F1 (across models): **0.9911**
- Error category distribution (all models/seeds):
|             |   count |
|:------------|--------:|
| PERFECT     |    1418 |
| MISS        |      14 |
| MULTI_SPAN  |       5 |
| WRONG_SPAN  |       2 |
| PARTIAL_END |       1 |
- Interpretability hypotheses:
  - Boundary sensitivity (partial/extend errors) vs detection failures (miss/false positive).
  - Morphological variation in `pie_span` vs canonical idiom form.
  - Idiom position effects (start/middle/end) and punctuation adjacency.

## Next Verification Steps
- Inspect 10–15 examples per idiom to confirm boundary failure patterns.
- Compare tokenization (`tokens`) with predicted spans to spot systematic offset patterns.
- Cross-reference with `create_prediction_report.py` for qualitative error inspection.