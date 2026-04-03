# Per-Idiom F1 Summary Report
**Generated:** 2026-01-02 23:30:11

## Scope
- Models: 6
- Tasks: cls, span
- Splits: seen_test, unseen_test
- Idioms (Seen): 54
- Idioms (Unseen): 6

## Methodology
- Per-idiom F1 computed from `eval_predictions.json`.
- CLS uses macro F1; SPAN uses exact span F1.
- Aggregation: mean ± std across seeds (42, 123, 456).
- Difficulty ordering: average F1 across all models (lower = harder).

## How To Read The Heatmaps
- Columns are **idiom IDs ordered by difficulty** (left = hardest).
- Rows are models ordered by average performance.
- Color reflects **mean F1 across seeds** for each model–idiom pair.

## Difficulty Rankings (Per Task/Split)
### CLS - Seen Test
**Hardest 5 idioms (lowest F1):**
|   idiom_id | base_pie         |   f1_mean |
|-----------:|:-----------------|----------:|
|         40 | עשה סצנה         |    0.6193 |
|         25 | ירה לכל הכיוונים |    0.6603 |
|         46 | קיפל את הזנב     |    0.7714 |
|          6 | החזיק אצבעות     |    0.7978 |
|         44 | קבר את עצמו      |    0.8295 |

**Easiest 5 idioms (highest F1):**
|   idiom_id | base_pie      |   f1_mean |
|-----------:|:--------------|----------:|
|         24 | ירד מהפסים    |    1.0000 |
|         17 | חטף מכה       |    1.0000 |
|         58 | שם עליו פס    |    1.0000 |
|          8 | הלך בין טיפות |    1.0000 |
|         37 | נתן יד        |    1.0000 |

### CLS - Unseen Test
**Hardest 5 idioms (lowest F1):**
|   idiom_id | base_pie     |   f1_mean |
|-----------:|:-------------|----------:|
|         55 | שבר שתיקה    |    0.7663 |
|         20 | חתך פינה     |    0.9049 |
|         33 | נשאר מאחור   |    0.9051 |
|          2 | איבד את הראש |    0.9393 |
|         19 | חצה קו אדום  |    0.9645 |

**Easiest 5 idioms (highest F1):**
|   idiom_id | base_pie             |   f1_mean |
|-----------:|:---------------------|----------:|
|         20 | חתך פינה             |    0.9049 |
|         33 | נשאר מאחור           |    0.9051 |
|          2 | איבד את הראש         |    0.9393 |
|         19 | חצה קו אדום          |    0.9645 |
|         49 | רץ אחרי הזנב של עצמו |    0.9791 |

### SPAN - Seen Test
**Hardest 5 idioms (lowest F1):**
|   idiom_id | base_pie          |   f1_mean |
|-----------:|:------------------|----------:|
|         11 | הרים את הראש      |    0.8840 |
|         22 | ירד לו האסימון    |    0.9338 |
|         10 | הניף דגל לבן      |    0.9861 |
|         36 | נתן גז            |    0.9869 |
|         30 | נכנס מתחת לאלונקה |    0.9889 |

**Easiest 5 idioms (highest F1):**
|   idiom_id | base_pie        |   f1_mean |
|-----------:|:----------------|----------:|
|         31 | נפל בין הכיסאות |    1.0000 |
|         32 | נפל מהכיסא      |    1.0000 |
|         34 | נשבר מבפנים     |    1.0000 |
|         37 | נתן יד          |    1.0000 |
|         60 | שפכה אור        |    1.0000 |

### SPAN - Unseen Test
**Hardest 5 idioms (lowest F1):**
|   idiom_id | base_pie             |   f1_mean |
|-----------:|:---------------------|----------:|
|         49 | רץ אחרי הזנב של עצמו |    0.0232 |
|         33 | נשאר מאחור           |    0.4901 |
|         19 | חצה קו אדום          |    0.6455 |
|         55 | שבר שתיקה            |    0.9187 |
|         20 | חתך פינה             |    0.9451 |

**Easiest 5 idioms (highest F1):**
|   idiom_id | base_pie     |   f1_mean |
|-----------:|:-------------|----------:|
|         33 | נשאר מאחור   |    0.4901 |
|         19 | חצה קו אדום  |    0.6455 |
|         55 | שבר שתיקה    |    0.9187 |
|         20 | חתך פינה     |    0.9451 |
|          2 | איבד את הראש |    0.9911 |

## Idiom 49 Deep Check (SPAN Unseen)
- Idiom 49 average F1 across models: 0.023167
- Error category distribution (all models, all seeds):
|              |   count |
|:-------------|--------:|
| PARTIAL_END  |    1319 |
| PERFECT      |      31 |
| MULTI_SPAN   |      26 |
| MISS         |      26 |
| PARTIAL_BOTH |      23 |
| WRONG_SPAN   |      15 |

**Impact on overall SPAN Unseen F1:**
- With idiom 49: 0.6839
- Without idiom 49: 0.8247
- Delta: 0.1408

## Figures
- `paper/figures/per_idiom/per_idiom_heatmap_cls_seen_test.png`
- `paper/figures/per_idiom/per_idiom_heatmap_cls_unseen_test.png`
- `paper/figures/per_idiom/per_idiom_heatmap_span_seen_test.png`
- `paper/figures/per_idiom/per_idiom_heatmap_span_unseen_test.png`

## Output Files
- `experiments/results/analysis/per_idiom_f1/per_idiom_f1_raw.csv`
- `experiments/results/analysis/per_idiom_f1/per_idiom_f1_summary.csv`
- `experiments/results/analysis/per_idiom_f1/idiom_metadata.csv`
- `experiments/results/analysis/per_idiom_f1/idiom_difficulty_ranking_{task}_{split}.csv`
