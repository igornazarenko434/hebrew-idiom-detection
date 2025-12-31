# Error Analysis Summary Report
**Generated:** 2025-12-31 13:36:37
**Total Predictions Analyzed:** 27,360
**Scope:** 5 models × 2 tasks × 3 seeds × 2 splits × variable samples

---

## Methodology

### Data Aggregation Process

**1. Error Categorization (Step 1)**
- **Tool:** `scripts/categorize_all_errors.py`
- **Input:** 60 evaluation files (5 models × 2 tasks × 3 seeds × 2 splits)
- **Process:** Applied standardized error taxonomy to all 27,360 predictions
- **Output:** Added `error_category` field to all `eval_predictions.json` files
- **Taxonomy Source:** `src/utils/error_analysis.py` (categorize_span_error, categorize_cls_error)

**2. Aggregation Across Seeds (Step 2)**
- **Tool:** `src/analyze_error_distribution.py`
- **Aggregation Method:** Pooled all predictions across 3 seeds (42, 123, 456)
- **Rationale:** Provides robust error statistics by combining all runs
- **Total Samples per Model/Task/Split:**
  - CLS Seen: ~1,296 predictions per model (across 3 seeds)
  - CLS Unseen: ~1,440 predictions per model
  - SPAN Seen: ~1,296 predictions per model
  - SPAN Unseen: ~1,440 predictions per model

**3. Cross-Model Aggregation (Step 3)**
- **Method:** Averaged percentages across all 5 models
- **Purpose:** Report overall error distribution patterns
- **Models Included:**
  - alephbert-base
  - alephbertgimmel-base
  - bert-base-multilingual-cased
  - dictabert
  - xlm-roberta-base

---

## Error Taxonomy

### CLS Task Categories (3 categories)
| Category | Description |
|----------|-------------|
| **CORRECT** | Predicted label matches ground truth (TP + TN) |
| **FALSE_POSITIVE** | Predicted Figurative, actually Literal |
| **FALSE_NEGATIVE** | Predicted Literal, actually Figurative |

### SPAN Task Categories (12 categories → 4 groups)

#### Group 1: PERFECT
**Exact Match**
- **PERFECT**: Predicted span boundaries match ground truth exactly

#### Group 2: BOUNDARY_ERRORS (6 categories)
**Span detected but boundaries incorrect**
- **PARTIAL_START**: Missing beginning token(s) of idiom
- **PARTIAL_END**: Missing ending token(s) of idiom
- **PARTIAL_BOTH**: Truncated on both start and end
- **EXTEND_START**: Extra token(s) at start of span
- **EXTEND_END**: Extra token(s) at end of span
- **EXTEND_BOTH**: Extended on both start and end

**Grouping Logic:**
```python
BOUNDARY_ERRORS = ['PARTIAL_START', 'PARTIAL_END', 'PARTIAL_BOTH',
                   'EXTEND_START', 'EXTEND_END', 'EXTEND_BOTH']
```

#### Group 3: DETECTION_ERRORS (2 categories)
**Failed to detect idiom or hallucinated non-existent idiom**
- **MISS**: No span predicted when ground truth has idiom
- **FALSE_POSITIVE**: Span predicted when no idiom exists

**Grouping Logic:**
```python
DETECTION_ERRORS = ['MISS', 'FALSE_POSITIVE']
```

#### Group 4: POSITION_ERRORS (3 categories)
**Span at wrong location or fragmented**
- **SHIFT**: Span overlaps but boundaries misaligned
- **WRONG_SPAN**: Completely different phrase tagged as idiom
- **MULTI_SPAN**: Multiple spans predicted (hallucination)

**Grouping Logic:**
```python
POSITION_ERRORS = ['SHIFT', 'WRONG_SPAN', 'MULTI_SPAN']
```

---

## CLS Task Error Distribution

| error_category   |   Seen |   Unseen |
|:-----------------|-------:|---------:|
| CORRECT          |  92.61 |    90.62 |
| FALSE_POSITIVE   |   3.16 |     6.01 |
| FALSE_NEGATIVE   |   4.23 |     3.36 |

**Interpretation:**
- Models maintain high accuracy (~93-91%) on both seen and unseen idioms
- False Positives increase on unseen idioms (3.16% → 6.01%), suggesting models over-predict figurative meaning for novel idioms
- False Negatives decrease on unseen idioms (4.23% → 3.36%)

---

## SPAN Task Error Distribution (Grouped)

| category_group   |   Seen |   Unseen |
|:-----------------|-------:|---------:|
| PERFECT          |  98.94 |    66.4  |
| BOUNDARY_ERRORS  |   0.15 |     8.54 |
| DETECTION_ERRORS |   0.26 |     7.61 |
| POSITION_ERRORS  |   0.55 |     1.57 |

**Category Grouping Breakdown:**
- **PERFECT**: 1 category (exact matches)
- **BOUNDARY_ERRORS**: 6 categories (partial/extended spans)
- **DETECTION_ERRORS**: 2 categories (missed or hallucinated)
- **POSITION_ERRORS**: 3 categories (wrong location)

**Interpretation:**
- **Dramatic Generalization Gap:** Perfect matches drop from 98.9% (seen) to 66.4% (unseen)
- **Boundary Errors Dominate Unseen:** 8.54% boundary errors on unseen idioms vs 0.15% on seen
  - Models can detect idioms but struggle with exact boundaries for novel expressions
- **Detection Failures:** 7.61% detection errors on unseen idioms
  - Models miss some unseen idioms entirely or hallucinate non-existent ones
- **Position Errors Rare:** Only 1.57% on unseen idioms
  - When models detect idioms, they usually find the correct region

---

## Key Findings

### CLS Task Performance
- **Seen Test Accuracy:** 92.6% (averaged across 5 models, 3 seeds each)
- **Unseen Test Accuracy:** 90.6%
- **Generalization Gap:** 2.0 percentage points
- **Dominant Error (Unseen):** 90.6% (CORRECT)

### SPAN Task Performance
- **Seen Test Perfect Matches:** 98.9%
- **Unseen Test Perfect Matches:** 66.4%
- **Generalization Gap:** 32.5 percentage points
- **Dominant Errors (Unseen):**
  1. **BOUNDARY ERRORS:** 8.5% (['PARTIAL_START', 'PARTIAL_END', 'PARTIAL_BOTH', 'EXTEND_START', 'EXTEND_END', 'EXTEND_BOTH'])
  2. **DETECTION ERRORS:** 7.6% (['MISS', 'FALSE_POSITIVE'])
  3. **POSITION ERRORS:** 1.6% (['SHIFT', 'WRONG_SPAN', 'MULTI_SPAN'])

### Critical Insights
1. **CLS generalizes well:** Only 2.0% performance drop on unseen idioms
2. **SPAN struggles with generalization:** 32.5% drop indicates exact boundary detection is harder for novel idioms
3. **Boundary detection is the bottleneck:** Models can often detect idioms but fail on precise token boundaries
4. **Seen idioms nearly perfect:** 98.9% perfect matches shows models learn seen idiom boundaries very well

---

## Visualizations Generated

1. **error_distribution_cls.png** - Stacked bar chart of CLS errors (CORRECT/FALSE_POSITIVE/FALSE_NEGATIVE)
2. **error_distribution_span_aggregated.png** - Grouped bar chart of 4 SPAN error groups
3. **error_heatmap_span.png** - Heatmap showing all 12 SPAN categories across models
4. **seen_unseen_comparison.png** - Comparison showing error shift from seen to unseen
5. **model_error_profiles.png** - Radar chart showing distinctive error patterns per model

All figures saved to: `paper/figures/error_analysis/` (300 DPI, publication-ready)

## Figures (Embedded)

![CLS Error Distribution](paper/figures/error_analysis/error_distribution_cls.png)
![SPAN Error Distribution (Grouped)](paper/figures/error_analysis/error_distribution_span_aggregated.png)
![SPAN Error Heatmap (All Categories)](paper/figures/error_analysis/error_heatmap_span.png)
![Seen vs Unseen Shift](paper/figures/error_analysis/seen_unseen_comparison.png)
![Model Error Profiles](paper/figures/error_analysis/model_error_profiles.png)
