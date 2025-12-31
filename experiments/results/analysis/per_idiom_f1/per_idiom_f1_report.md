# Per-Idiom F1 Summary Report
**Generated:** 2025-12-31 14:07:30

## Scope
- Models: 5
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
|         40 | עשה סצנה         |    0.5790 |
|         25 | ירה לכל הכיוונים |    0.7014 |
|          6 | החזיק אצבעות     |    0.7844 |
|         46 | קיפל את הזנב     |    0.8072 |
|          5 | הוריד פרופיל     |    0.8298 |

**Easiest 5 idioms (highest F1):**
|   idiom_id | base_pie            |   f1_mean |
|-----------:|:--------------------|----------:|
|         34 | נשבר מבפנים         |    1.0000 |
|         17 | חטף מכה             |    1.0000 |
|         54 | שבר את תקרת הזכוכית |    1.0000 |
|         58 | שם עליו פס          |    1.0000 |
|          8 | הלך בין טיפות       |    1.0000 |

### CLS - Unseen Test
**Hardest 5 idioms (lowest F1):**
|   idiom_id | base_pie     |   f1_mean |
|-----------:|:-------------|----------:|
|         55 | שבר שתיקה    |    0.7407 |
|         20 | חתך פינה     |    0.8885 |
|         33 | נשאר מאחור   |    0.9011 |
|          2 | איבד את הראש |    0.9431 |
|         19 | חצה קו אדום  |    0.9591 |

**Easiest 5 idioms (highest F1):**
|   idiom_id | base_pie             |   f1_mean |
|-----------:|:---------------------|----------:|
|         20 | חתך פינה             |    0.8885 |
|         33 | נשאר מאחור           |    0.9011 |
|          2 | איבד את הראש         |    0.9431 |
|         19 | חצה קו אדום          |    0.9591 |
|         49 | רץ אחרי הזנב של עצמו |    0.9858 |

### SPAN - Seen Test
**Hardest 5 idioms (lowest F1):**
|   idiom_id | base_pie          |   f1_mean |
|-----------:|:------------------|----------:|
|         11 | הרים את הראש      |    0.8735 |
|         22 | ירד לו האסימון    |    0.9368 |
|         30 | נכנס מתחת לאלונקה |    0.9822 |
|         35 | נשך שפתיים        |    0.9833 |
|          7 | הייתה בעננים      |    0.9877 |

**Easiest 5 idioms (highest F1):**
|   idiom_id | base_pie      |   f1_mean |
|-----------:|:--------------|----------:|
|         27 | לב זהב        |    1.0000 |
|         28 | לקח צעד אחורה |    1.0000 |
|         59 | שם רגליים     |    1.0000 |
|         16 | חטף חום       |    1.0000 |
|         60 | שפכה אור      |    1.0000 |

### SPAN - Unseen Test
**Hardest 5 idioms (lowest F1):**
|   idiom_id | base_pie             |   f1_mean |
|-----------:|:---------------------|----------:|
|         49 | רץ אחרי הזנב של עצמו |    0.0008 |
|         19 | חצה קו אדום          |    0.6837 |
|         33 | נשאר מאחור           |    0.7277 |
|         55 | שבר שתיקה            |    0.8221 |
|         20 | חתך פינה             |    0.8980 |

**Easiest 5 idioms (highest F1):**
|   idiom_id | base_pie     |   f1_mean |
|-----------:|:-------------|----------:|
|         19 | חצה קו אדום  |    0.6837 |
|         33 | נשאר מאחור   |    0.7277 |
|         55 | שבר שתיקה    |    0.8221 |
|         20 | חתך פינה     |    0.8980 |
|          2 | איבד את הראש |    0.9971 |

## Idiom 49 Deep Check (SPAN Unseen)
- Idiom 49 average F1 across models: 0.000823
- Error category distribution (all models, all seeds):
|              |   count |
|:-------------|--------:|
| PARTIAL_END  |    1119 |
| MISS         |      42 |
| MULTI_SPAN   |      27 |
| PARTIAL_BOTH |       7 |
| WRONG_SPAN   |       4 |
| PERFECT      |       1 |

**Impact on overall SPAN Unseen F1:**
- With idiom 49: 0.6914
- Without idiom 49: 0.8337
- Delta: 0.1423

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
