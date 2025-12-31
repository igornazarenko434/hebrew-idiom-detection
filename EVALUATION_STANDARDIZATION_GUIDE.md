# Unified Evaluation & Analysis Guide
# Hebrew Idiom Detection Project

**Version:** 4.0 (Comprehensive + Best Practices)
**Last Updated:** December 31, 2025
**Purpose:** Complete reference for evaluation, analysis, and comparison across fine-tuning and prompting methods following NLP research best practices.

---

## 0. How To Use This Guide

### 0.1 Document Purpose
- **Single source of truth** for all evaluation and analysis procedures
- Enables **direct comparison** between fine-tuning and prompting results
- Follows **NLP research best practices** (CoNLL 2003, Dodge et al. 2020, Dror et al. 2018)
- **Both partners must follow exactly** to ensure mergeable results

### 0.2 Quick Navigation
- **Sections 1-3:** Project reality (data, models, metrics) - **READ FIRST**
- **Sections 4-5:** Error taxonomy and output formats - **SHARED BY BOTH**
- **Sections 6-12:** Standard procedures and naming - **REQUIRED**
- **Sections 13-16:** Detailed protocols with code examples - **IMPLEMENTATION**
- **Sections 17-24:** Method-specific analyses - **FINE-TUNING ONLY**
- **Sections 25-27:** Prompting-specific analyses - **PROMPTING ONLY**
- **Sections 28-30:** Joint analyses and reproducibility - **BOTH PARTNERS**

---

## 1. Project Reality (Exact Data + Tasks)

### 1.1 Tasks
- **Task 1 (CLS):** Sentence-level binary classification
  - Label 0: Literal interpretation
  - Label 1: Figurative interpretation (idiom)
  - **Primary metric:** Macro F1

- **Task 2 (SPAN):** Idiom span detection using IOB2 tagging
  - Tags: `O` (outside), `B-IDIOM` (beginning), `I-IDIOM` (inside)
  - **Primary metric:** Exact Span F1 (both boundaries must match)
  - **Critical:** Exact match required, partial matches = incorrect

### 1.2 Dataset Files (Actual Repo Paths)
```
data/
├── expressions_data_tagged.csv          # Main dataset (4,800 sentences, 60 idioms)
└── splits/
    ├── train.csv                        # Training set
    ├── validation.csv                   # Validation set
    ├── test.csv                         # Seen test (contains idioms from training)
    └── unseen_idiom_test.csv           # Unseen test (new idioms)
```

### 1.3 Required Dataset Columns (Exact Names)
```python
{
  "id": str,                  # Format: "{idiom_id}_{lit|fig}_{sample_num}"
  "sentence": str,            # Original Hebrew sentence
  "tokens": list[str],        # Pre-tokenized, punctuation separated
  "iob_tags": list[str],      # IOB2 tags aligned to tokens
  "label": int,               # 0=literal, 1=figurative
  "base_pie": str,            # Base idiom form (for grouping)
  "pie_span": str,            # Specific idiom variant
  "split": str                # "train", "validation", "test", "unseen_idiom_test"
}
```

### 1.4 Critical Data Properties
- **Expression-based split:** Same idiom never appears in both train and test
- **No data leakage:** Test idioms are completely held out during training
- **Balanced splits:** Approximately 50% literal, 50% figurative in each split
- **Token alignment:** `len(tokens) == len(iob_tags)` always holds

**CRITICAL FOR SPAN TASK:**
- Always use dataset-provided `tokens` and `iob_tags`
- Tokenize with `is_split_into_words=True` to preserve alignment
- Never re-tokenize from scratch (breaks IOB alignment)

---

## 2. Models In Scope

### 2.1 Fine-Tuning (Encoder Models)
```python
FINE_TUNING_MODELS = {
    # Hebrew-specific models
    "onlplab/alephbert-base": "AlephBERT",           # 12-layer BERT trained on Hebrew
    "dicta-il/alephbertgimmel-base": "AlephBERTGimmel",  # Enhanced AlephBERT
    "dicta-il/dictabert": "DictaBERT",               # Modern Hebrew BERT
    "dicta-il/neodictabert": "NeoDictaBERT",         # Latest Dicta model

    # Multilingual baselines
    "bert-base-multilingual-cased": "mBERT",         # Multilingual BERT
    "xlm-roberta-base": "XLM-R"                      # Cross-lingual RoBERTa
}
```

### 2.2 Prompting (LLM Models)
```python
PROMPTING_MODELS = {
    # Hebrew-focused LLMs
    "dicta-il/DictaLM-3.0-1.7B-Instruct": "DictaLM-3.0",
    "dicta-il/DictaLM-3.0-1.7B-Instruct-W4A16": "DictaLM-3.0-Quantized",

    # Multilingual LLMs
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",

    # Optional API model
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B"
}
```

---

## 3. Metrics (Exact Definitions + Computation)

### 3.1 Task 1 (CLS) Metrics

**Primary metric:** `f1` = Macro F1 (average of per-class F1 scores)

**Complete metric set:**
```json
{
  "f1": 0.0,           // REQUIRED: Macro F1 (mean of literal_f1 and figurative_f1)
  "accuracy": 0.0,     // REQUIRED: (TP + TN) / Total
  "precision": 0.0,    // REQUIRED: Macro precision
  "recall": 0.0        // REQUIRED: Macro recall
}
```

**Computation (Python):**
```python
from sklearn.metrics import classification_report

def compute_cls_metrics(true_labels, pred_labels):
    """
    Compute classification metrics following macro averaging.

    Args:
        true_labels: List[int] - Ground truth labels (0 or 1)
        pred_labels: List[int] - Predicted labels (0 or 1)

    Returns:
        dict: Metrics with keys: f1, accuracy, precision, recall
    """
    report = classification_report(
        true_labels,
        pred_labels,
        labels=[0, 1],
        target_names=['literal', 'figurative'],
        output_dict=True,
        zero_division=0
    )

    return {
        "f1": report['macro avg']['f1-score'],
        "accuracy": report['accuracy'],
        "precision": report['macro avg']['precision'],
        "recall": report['macro avg']['recall']
    }
```

**Why Macro F1?**
- Treats literal and figurative classes equally (no bias toward majority class)
- Standard for binary classification in NLP
- Aligns with CoNLL evaluation conventions

### 3.2 Task 2 (SPAN) Metrics

**Primary metric:** `f1` = Exact Span F1 (boundaries must match exactly)

**Complete metric set:**
```json
{
  "f1": 0.0,          // REQUIRED: Exact span F1 (primary metric)
  "precision": 0.0,   // REQUIRED: Span-level precision
  "recall": 0.0,      // REQUIRED: Span-level recall
  "accuracy": 0.0     // OPTIONAL: Token-level accuracy (auxiliary)
}
```

**Exact Span F1 Definition:**
```python
def get_span_indices(tags):
    """
    Extract span indices from IOB2 tags.

    Args:
        tags: List[str] - IOB2 tags (e.g., ['O', 'B-IDIOM', 'I-IDIOM', 'O'])

    Returns:
        List[Tuple[int, int]]: List of (start, end) indices (end exclusive)

    Example:
        ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O'] -> [(1, 4)]
        ['O', 'O', 'O'] -> []
    """
    spans = []
    start = None

    for i, tag in enumerate(tags):
        if tag == 'B-IDIOM':
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == 'O':
            if start is not None:
                spans.append((start, i))
                start = None

    if start is not None:
        spans.append((start, len(tags)))

    return spans


def compute_span_f1(predictions):
    """
    Compute exact span F1 score.

    A span is considered correct ONLY if:
    - Start index matches exactly AND
    - End index matches exactly

    Partial overlaps do NOT count as correct.

    Args:
        predictions: List[dict] with keys 'true_tags' and 'predicted_tags'

    Returns:
        dict: Metrics with keys: f1, precision, recall

    Example:
        True:  ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O']
        Pred:  ['O', 'B-IDIOM', 'I-IDIOM', 'O', 'O']
        Result: Incorrect (end boundary mismatch)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred in predictions:
        true_spans = set(get_span_indices(pred['true_tags']))
        pred_spans = set(get_span_indices(pred['predicted_tags']))

        true_positives += len(true_spans & pred_spans)  # Exact matches
        false_positives += len(pred_spans - true_spans)  # Predicted but not in ground truth
        false_negatives += len(true_spans - pred_spans)  # In ground truth but not predicted

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
```

**Why Exact Span F1?**
- Partial matches are not useful in practice (need precise boundaries)
- Aligns with CoNLL 2003 NER evaluation standard
- Stricter than token-level F1 (more meaningful metric)
- Industry standard for span detection tasks

**CRITICAL FOR PROMPTING:**
- Must use exact same `compute_span_f1()` function
- Available in `src/utils/error_analysis.py`
- Do NOT use token-level F1 or seqeval default behavior

### 3.3 Reporting Standards

#### 3.3.1 Fine-Tuning Reporting
```python
# Required: Multi-seed evaluation
SEEDS = [42, 123, 456]

# Report format:
# Mean ± Std over 3 seeds
# Example: "F1: 94.83 ± 0.42"

def aggregate_multi_seed_results(seed_results):
    """
    Aggregate results across multiple seeds.

    Args:
        seed_results: List[dict] - Results for each seed

    Returns:
        dict: Mean ± Std for all metrics
    """
    import numpy as np

    metrics = {}
    for key in seed_results[0].keys():
        values = [r[key] for r in seed_results]
        metrics[f"{key}_mean"] = np.mean(values)
        metrics[f"{key}_std"] = np.std(values, ddof=1)  # Sample std

    return metrics

# Example output:
# {
#   "f1_mean": 0.9483,
#   "f1_std": 0.0042,
#   "accuracy_mean": 0.9475,
#   "accuracy_std": 0.0038
# }
```

#### 3.3.2 Prompting Reporting

**Deterministic (temperature=0):**
```python
# Single run + bootstrap confidence interval
from scipy import stats

def bootstrap_ci(scores, n_bootstrap=10000, confidence=0.95):
    """
    Compute bootstrap confidence interval.

    Args:
        scores: List[float] - Per-sample scores
        n_bootstrap: int - Number of bootstrap samples
        confidence: float - Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple[float, float]: (lower_bound, upper_bound)
    """
    import numpy as np

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (lower, upper)

# Report format: "F1: 87.50 [85.20, 89.80]"
```

**Stochastic (temperature>0):**
```python
# Run 3 times with different random seeds
# Report: Mean ± Std (same as fine-tuning)

RANDOM_SEEDS = [42, 123, 456]

# Report format: "F1: 88.20 ± 1.35"
```

#### 3.3.3 Required Comparisons

**Always report both splits:**
```python
REQUIRED_SPLITS = ["seen_test", "unseen_test"]

# Example table:
# | Model | Seen F1 | Unseen F1 | Gap (abs) | Gap (%) |
# |-------|---------|-----------|-----------|---------|
# | DictaBERT | 94.83 | 91.08 | -3.75 | -3.96% |
```

**Generalization Gap Calculation:**
```python
def compute_generalization_gap(seen_f1, unseen_f1):
    """
    Compute generalization gap between seen and unseen test sets.

    Args:
        seen_f1: float - F1 score on seen test set
        unseen_f1: float - F1 score on unseen idiom test set

    Returns:
        dict: Absolute and percentage gap
    """
    gap_abs = seen_f1 - unseen_f1
    gap_pct = (gap_abs / seen_f1) * 100 if seen_f1 > 0 else 0.0

    return {
        "gap_absolute": gap_abs,
        "gap_percentage": gap_pct
    }

# Example:
# compute_generalization_gap(94.83, 91.08)
# -> {"gap_absolute": 3.75, "gap_percentage": 3.96}
```

---

## 4. Error Taxonomy (Shared Language)

### 4.1 Task 1 (CLS) Error Categories

**Two error types:**
```python
CLS_ERROR_CATEGORIES = {
    "FP": "False Positive - Predicted Figurative, True Literal",
    "FN": "False Negative - Predicted Literal, True Figurative"
}

def categorize_cls_error(true_label, pred_label):
    """
    Categorize classification error.

    Args:
        true_label: int - Ground truth (0=literal, 1=figurative)
        pred_label: int - Prediction (0=literal, 1=figurative)

    Returns:
        str: 'CORRECT', 'FP', or 'FN'
    """
    if true_label == pred_label:
        return "CORRECT"
    elif true_label == 0 and pred_label == 1:
        return "FP"  # Predicted figurative when actually literal
    else:  # true_label == 1 and pred_label == 0
        return "FN"  # Predicted literal when actually figurative
```

### 4.2 Task 2 (SPAN) Error Categories

**Twelve error types (UPPERCASE with underscores):**

```python
SPAN_ERROR_CATEGORIES = {
    "PERFECT": "Predicted span exactly matches ground truth",
    "MISS": "No span predicted when ground truth has idiom (all O tags)",
    "FALSE_POSITIVE": "Predicted span when no idiom exists in ground truth",
    "PARTIAL_START": "Missing beginning tokens of the idiom",
    "PARTIAL_END": "Missing ending tokens of the idiom",
    "PARTIAL_BOTH": "Span is shorter on both start and end",
    "EXTEND_START": "Span includes extra tokens at the beginning",
    "EXTEND_END": "Span includes extra tokens at the end",
    "EXTEND_BOTH": "Span includes extra tokens at both boundaries",
    "SHIFT": "Span overlaps with true span but is offset",
    "WRONG_SPAN": "Tagged a completely different phrase (no overlap)",
    "MULTI_SPAN": "Predicted multiple spans when only one exists"
}


def categorize_span_error(true_tags, pred_tags):
    """
    Categorize span detection error using exact taxonomy.

    Args:
        true_tags: List[str] - Ground truth IOB2 tags
        pred_tags: List[str] - Predicted IOB2 tags

    Returns:
        str: Error category (one of SPAN_ERROR_CATEGORIES keys)

    Examples:
        True:  ['O', 'B-IDIOM', 'I-IDIOM', 'O']
        Pred:  ['O', 'B-IDIOM', 'I-IDIOM', 'O']
        -> "PERFECT"

        True:  ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O']
        Pred:  ['O', 'B-IDIOM', 'I-IDIOM', 'O', 'O']
        -> "PARTIAL_END"

        True:  ['O', 'B-IDIOM', 'I-IDIOM', 'O']
        Pred:  ['O', 'O', 'O', 'O']
        -> "MISS"
    """
    true_spans = get_span_indices(true_tags)
    pred_spans = get_span_indices(pred_tags)

    # Perfect match
    if true_tags == pred_tags:
        return "PERFECT"

    # No ground truth span
    if len(true_spans) == 0:
        return "FALSE_POSITIVE" if len(pred_spans) > 0 else "PERFECT"

    # No predicted span
    if len(pred_spans) == 0:
        return "MISS"

    # Multiple predicted spans
    if len(pred_spans) > 1:
        return "MULTI_SPAN"

    # Single span comparison
    true_start, true_end = true_spans[0]
    pred_start, pred_end = pred_spans[0]

    # No overlap
    if pred_end <= true_start or pred_start >= true_end:
        return "WRONG_SPAN"

    # Exact match (already handled above, but for clarity)
    if true_start == pred_start and true_end == pred_end:
        return "PERFECT"

    # Boundary analysis
    start_diff = pred_start - true_start  # Positive = pred starts later, Negative = pred starts earlier
    end_diff = pred_end - true_end        # Positive = pred ends later, Negative = pred ends earlier

    # Extension cases
    if start_diff < 0 and end_diff > 0:
        return "EXTEND_BOTH"
    elif start_diff < 0:
        return "EXTEND_START"
    elif end_diff > 0:
        return "EXTEND_END"

    # Partial cases
    if start_diff > 0 and end_diff < 0:
        return "PARTIAL_BOTH"
    elif start_diff > 0:
        return "PARTIAL_START"
    elif end_diff < 0:
        return "PARTIAL_END"

    # Overlapping but shifted
    return "SHIFT"
```

**Visual Examples:**
```
Ground Truth: [O, B-IDIOM, I-IDIOM, I-IDIOM, O]
              "   *******idiom*******     "

PERFECT:      [O, B-IDIOM, I-IDIOM, I-IDIOM, O]  ✓

MISS:         [O, O, O, O, O]                     ❌ (false negative)

PARTIAL_END:  [O, B-IDIOM, I-IDIOM, O, O]         ❌ (missing end)

PARTIAL_START:[O, O, I-IDIOM, I-IDIOM, O]         ❌ (missing start)

EXTEND_END:   [O, B-IDIOM, I-IDIOM, I-IDIOM, I-IDIOM] ❌ (extra at end)

WRONG_SPAN:   [B-IDIOM, I-IDIOM, O, O, O]         ❌ (different phrase)
```

---

## 5. Output Formats (Shared Structure)

### 5.1 Directory Structure

**Recommended path convention:**
```
experiments/results/evaluation/
├── {method}/                    # "fine_tuning" or "prompting"
│   ├── {split}/                # "seen_test" or "unseen_test"
│   │   ├── {model}/            # Model name (e.g., "dictabert", "dictalm-3.0")
│   │   │   ├── {task}/         # "cls" or "span"
│   │   │   │   ├── {run_id}/   # "seed_42", "few_shot_3", "zero_shot", etc.
│   │   │   │   │   ├── eval_results.json
│   │   │   │   │   └── eval_predictions.json
```

**Example paths:**
```
# Fine-tuning example:
experiments/results/evaluation/fine_tuning/seen_test/dictabert/cls/seed_42/eval_results.json

# Prompting example:
experiments/results/evaluation/prompting/unseen_test/dictalm-3.0/span/few_shot_5/eval_results.json
```

### 5.2 eval_results.json (Required Structure)

**Minimum required fields:**
```json
{
  "model": "dicta-il/dictabert",
  "method": "fine_tuning",
  "task": "cls",
  "split": "seen_test",
  "run_id": "seed_42",
  "timestamp": "2025-12-31T10:30:00",
  "metrics": {
    "f1": 0.9483,
    "accuracy": 0.9475,
    "precision": 0.9501,
    "recall": 0.9483
  }
}
```

**Additional fields for fine-tuning:**
```json
{
  "hyperparameters": {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 10,
    "warmup_ratio": 0.1
  },
  "training_info": {
    "total_steps": 1250,
    "best_epoch": 7,
    "best_val_f1": 0.9321
  }
}
```

**Additional fields for prompting:**
```json
{
  "strategy": "few_shot",
  "num_examples": 5,
  "temperature": 0.0,
  "prompt_template": "zero_shot_v1",
  "metrics": {
    "f1": 0.8750,
    "accuracy": 0.8750,
    "precision": 0.8820,
    "recall": 0.8750,
    "parse_success_rate": 0.9875,
    "avg_tokens_per_sample": 256.3,
    "avg_latency_seconds": 1.23,
    "total_cost_usd": 0.15
  }
}
```

### 5.3 eval_predictions.json (Required Structure)

**CLS minimum fields:**
```json
[
  {
    "id": "1_fig_0",
    "sentence": "הוא נתן לה את הגב",
    "true_label": 1,
    "predicted_label": 1,
    "is_correct": true,
    "confidence": 0.9823,
    "error_category": "CORRECT"
  },
  {
    "id": "5_lit_12",
    "sentence": "היא פנתה את הגב למצלמה",
    "true_label": 0,
    "predicted_label": 1,
    "is_correct": false,
    "confidence": 0.6234,
    "error_category": "FP"
  }
]
```

**SPAN minimum fields:**
```json
[
  {
    "id": "1_fig_0",
    "sentence": "הוא נתן לה את הגב",
    "tokens": ["הוא", "נתן", "לה", "את", "הגב"],
    "true_tags": ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", "I-IDIOM"],
    "predicted_tags": ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", "I-IDIOM"],
    "is_correct": true,
    "true_span_text": "נתן לה את הגב",
    "predicted_span_text": "נתן לה את הגב",
    "error_category": "PERFECT"
  },
  {
    "id": "5_lit_12",
    "sentence": "היא פנתה את הגב למצלמה",
    "tokens": ["היא", "פנתה", "את", "הגב", "למצלמה"],
    "true_tags": ["O", "O", "O", "O", "O"],
    "predicted_tags": ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", "O"],
    "is_correct": false,
    "true_span_text": null,
    "predicted_span_text": "פנתה את הגב",
    "error_category": "FALSE_POSITIVE"
  }
]
```

---

## 6. Shared Evaluation Procedure (Both Partners)

### 6.1 Standard Evaluation Flow

**Required steps for ALL methods:**

1. **Load test data**
```python
import pandas as pd

# Always evaluate on BOTH splits
seen_test = pd.read_csv("data/splits/test.csv")
unseen_test = pd.read_csv("data/splits/unseen_idiom_test.csv")
```

2. **Generate predictions**
```python
# Fine-tuning: Use trained model checkpoint
# Prompting: Use LLM with specific strategy

predictions = model.predict(test_data)
```

3. **Compute metrics using standardized functions**
```python
from src.utils.error_analysis import compute_span_f1, compute_cls_metrics

if task == "cls":
    metrics = compute_cls_metrics(true_labels, pred_labels)
elif task == "span":
    metrics = compute_span_f1(predictions)
```

4. **Categorize errors**
```python
from src.utils.error_analysis import categorize_span_error, categorize_cls_error

for pred in predictions:
    if task == "cls":
        pred['error_category'] = categorize_cls_error(
            pred['true_label'],
            pred['predicted_label']
        )
    elif task == "span":
        pred['error_category'] = categorize_span_error(
            pred['true_tags'],
            pred['predicted_tags']
        )
```

5. **Save results and predictions**
```python
import json
from pathlib import Path

# Create directory
output_dir = Path(f"experiments/results/evaluation/{method}/{split}/{model}/{task}/{run_id}")
output_dir.mkdir(parents=True, exist_ok=True)

# Save metrics
with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Save predictions
with open(output_dir / "eval_predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
```

6. **Create per-idiom breakdown**
```python
def compute_per_idiom_f1(predictions_df):
    """
    Compute F1 score for each idiom separately.

    Args:
        predictions_df: DataFrame with columns ['id', 'is_correct', ...]

    Returns:
        DataFrame: Per-idiom F1 scores
    """
    # Extract idiom ID from sample ID (e.g., "1_fig_0" -> idiom=1)
    predictions_df['idiom_id'] = predictions_df['id'].str.split('_').str[0]

    per_idiom_results = []
    for idiom_id in predictions_df['idiom_id'].unique():
        idiom_preds = predictions_df[predictions_df['idiom_id'] == idiom_id]

        if task == "cls":
            metrics = compute_cls_metrics(
                idiom_preds['true_label'].tolist(),
                idiom_preds['predicted_label'].tolist()
            )
        elif task == "span":
            metrics = compute_span_f1(idiom_preds.to_dict('records'))

        per_idiom_results.append({
            'idiom_id': idiom_id,
            'base_pie': idiom_preds['base_pie'].iloc[0],
            'f1': metrics['f1'],
            'num_samples': len(idiom_preds)
        })

    return pd.DataFrame(per_idiom_results)
```

### 6.2 Validation Checklist

**Before reporting results, verify:**

- [ ] Evaluated on BOTH seen_test and unseen_test
- [ ] Used correct metric (Span F1 for SPAN, Macro F1 for CLS)
- [ ] Error categories assigned using shared taxonomy
- [ ] Saved eval_results.json and eval_predictions.json
- [ ] Computed generalization gap
- [ ] Created per-idiom breakdown
- [ ] No data leakage (test samples not in training)
- [ ] Reproducible (saved random seeds, hyperparameters)

---

## 7. Responsibilities By Partner

### 7.1 Fine-Tuning Lead (You)

**Must deliver:**
```python
FINE_TUNING_DELIVERABLES = [
    "Multi-seed evaluation (42/123/456) on Seen + Unseen",
    "Per-idiom F1 (CLS + SPAN)",
    "Error distribution using shared taxonomy",
    "Statistical comparisons among encoder models (paired t-tests)",
    "Generalization gap analysis",
    "eval_results.json + eval_predictions.json for all models/seeds"
]
```

**Method-specific analyses:**
```python
FINE_TUNING_ONLY_ANALYSES = [
    "Frozen backbone vs full fine-tune (Mission 6.2)",
    "Data size impact: 10/25/50/75/100% (Mission 6.4)",
    "Hyperparameter sensitivity (Mission 6.3)",
    "Token importance + attention analysis (Mission 6.1)",
    "Learning curves from TensorBoard logs"
]
```

### 7.2 Prompting Lead (Partner)

**Must deliver:**
```python
PROMPTING_DELIVERABLES = [
    "Zero-shot + few-shot evaluation on Seen + Unseen",
    "Same metric computation (exact span F1)",
    "Same error taxonomy categories",
    "Per-idiom F1 (CLS + SPAN)",
    "Parse success rate + cost/latency metrics",
    "eval_results.json + eval_predictions.json for all strategies"
]
```

**Method-specific analyses:**
```python
PROMPTING_ONLY_ANALYSES = [
    "Prompt strategy comparison (zero-shot, few-shot, CoT)",
    "Few-shot example selection strategy (random vs stratified)",
    "Temperature/sampling sensitivity",
    "Output format adherence rate",
    "Token usage and cost analysis"
]
```

---

## 8. Analysis Matrix (What Applies To Which Method)

| Analysis | Shared | Fine-Tuning Only | Prompting Only |
|---|:---:|:---:|:---:|
| **Evaluation & Metrics** |  |  |  |
| Standard metrics (Seen/Unseen) | ✅ |  |  |
| Generalization gap | ✅ |  |  |
| Per-idiom F1 | ✅ |  |  |
| Error taxonomy + distribution | ✅ |  |  |
| Cross-task analysis (CLS vs SPAN) | ✅ |  |  |
| **Statistical Analysis** |  |  |  |
| Statistical tests (t-test / bootstrap) | ✅ |  |  |
| Effect size (Cohen's d) | ✅ |  |  |
| **Training-Based Ablations** |  |  |  |
| Frozen backbone comparison |  | ✅ |  |
| Data size impact (10-100%) |  | ✅ |  |
| Hyperparameter sensitivity |  | ✅ |  |
| Learning curves |  | ✅ |  |
| **Model Interpretability** |  |  |  |
| Token importance (integrated gradients) |  | ✅ |  |
| Attention analysis |  | ✅ |  |
| **Prompting-Specific** |  |  |  |
| Prompt strategy comparison |  |  | ✅ |
| Few-shot selection impact |  |  | ✅ |
| Temperature sensitivity |  |  | ✅ |
| Cost/latency/token usage |  |  | ✅ |
| Output format adherence |  |  | ✅ |

---

## 9. Mission Coverage (Phase 5-7)

### Phase 5: LLM Evaluation (Prompting Only)
```python
PHASE_5_TASKS = {
    "5.1": "Zero-shot prompting for CLS and SPAN",
    "5.2": "Few-shot prompting with stratified example selection",
    "5.3": "Optional: Chain-of-Thought prompting",
    "5.4": "Parse success rate tracking",
    "5.5": "Cost and latency measurement"
}
```

### Phase 6: Ablations + Interpretability (Fine-Tuning Only)
```python
PHASE_6_TASKS = {
    "6.1": "Token importance analysis (gradients + attention)",
    "6.2": "Frozen backbone vs full fine-tuning",
    "6.3": "Hyperparameter sensitivity analysis",
    "6.4": "Data size impact (10%, 25%, 50%, 75%, 100%)"
}
```

### Phase 7: Comprehensive Analysis (Both Partners)
```python
PHASE_7_TASKS = {
    "7.1": "Error analysis with shared taxonomy",
    "7.2": "Model comparison + statistical significance",
    "7.3": "Cross-task analysis (CLS vs SPAN correlation)",
    "7.4": "Figures for publication",
    "7.5": "Tables for publication"
}
```

---

## 10. Required Outputs (Per Partner)

**Both partners must produce:**
```python
SHARED_OUTPUTS = [
    "eval_results.json for Seen + Unseen (all models/strategies)",
    "eval_predictions.json for Seen + Unseen (all models/strategies)",
    "per_idiom_f1.csv for CLS and SPAN",
    "error_distribution.csv using shared taxonomy",
    "generalization_gap.csv"
]
```

**Joint deliverables (after both complete evaluation):**
```python
JOINT_OUTPUTS = [
    "unified_comparison_table.csv (Fine-Tuning vs Prompting)",
    "cross_method_error_comparison.csv",
    "statistical_significance_report.txt (with Cohen's d)",
    "publication_figures/ (visualizations)",
    "publication_tables/ (LaTeX-formatted)"
]
```

---

## 11. Optional Deep Analyses (Best Practice Additions)

These strengthen the paper and are consistent with the dataset:

```python
OPTIONAL_ANALYSES = {
    "sentence_length_effects": {
        "description": "Performance vs sentence length bins",
        "bins": [5, 10, 15, 20, 25, 30, "35+"],
        "applies_to": "both"
    },
    "idiom_length_effects": {
        "description": "Performance vs idiom token length",
        "bins": [2, 3, 4, 5, "6+"],
        "applies_to": "both"
    },
    "idiom_position_effects": {
        "description": "Performance by idiom position (start/middle/end)",
        "positions": ["start", "middle", "end"],
        "applies_to": "both"
    },
    "morphology_sensitivity": {
        "description": "Group by pie_span variants",
        "applies_to": "both"
    },
    "calibration_analysis": {
        "description": "ECE + Brier score for CLS",
        "metrics": ["ECE", "Brier"],
        "applies_to": "fine_tuning (unless LLM gives calibrated probs)"
    }
}
```

---

## 12. Naming Conventions (Exact Strings)

**Use these exact strings everywhere:**

```python
# Methods
METHODS = ["fine_tuning", "prompting"]

# Tasks
TASKS = ["cls", "span"]

# Splits
SPLITS = ["seen_test", "unseen_test"]

# Seeds (for fine-tuning and stochastic prompting)
SEEDS = ["seed_42", "seed_123", "seed_456"]

# Prompt strategies (for prompting only)
STRATEGIES = ["zero_shot", "few_shot_3", "few_shot_5", "cot"]

# Error categories (Task 1: CLS)
CLS_ERRORS = ["CORRECT", "FP", "FN"]

# Error categories (Task 2: SPAN)
SPAN_ERRORS = [
    "PERFECT", "MISS", "FALSE_POSITIVE",
    "PARTIAL_START", "PARTIAL_END", "PARTIAL_BOTH",
    "EXTEND_START", "EXTEND_END", "EXTEND_BOTH",
    "SHIFT", "WRONG_SPAN", "MULTI_SPAN"
]

# Metric names
METRICS = ["f1", "accuracy", "precision", "recall"]

# Model short names (for file paths)
MODEL_NAMES = {
    "fine_tuning": [
        "alephbert", "alephbertgimmel", "dictabert",
        "neodictabert", "mbert", "xlm-r"
    ],
    "prompting": [
        "dictalm-3.0", "dictalm-3.0-quantized",
        "llama-3.1-8b", "qwen2.5-7b", "llama-3.1-70b"
    ]
}
```

**Formatting rules:**
- Methods, tasks, splits, strategies: `lowercase_with_underscores`
- Error categories: `UPPERCASE_WITH_UNDERSCORES`
- Metric keys: `lowercase_with_underscores`
- File paths: `lowercase_with_hyphens` for model names

---

## 13. Few-Shot Example Selection Protocol (Prompting Only)

### 13.1 Why This Matters
- **Data leakage:** Few-shot examples must NEVER come from test sets
- **Representativeness:** Examples should cover literal and figurative cases
- **Complexity:** Avoid overly complex or ambiguous examples
- **Reproducibility:** Selection must be deterministic (fixed random seed)

### 13.2 Selection Procedure

```python
import pandas as pd
import numpy as np

def select_few_shot_examples(
    train_df,
    validation_df,
    n_examples=5,
    seed=42,
    sentence_length_range=(10, 20),
    idiom_length_range=(2, 4)
):
    """
    Select few-shot examples following best practices.

    Requirements:
    1. Must come from train or validation ONLY (never test)
    2. Stratified by label (balanced literal/figurative)
    3. Moderate complexity (sentence length 10-20 tokens)
    4. Typical idiom length (2-4 tokens)
    5. Reproducible (fixed seed)

    Args:
        train_df: Training set DataFrame
        validation_df: Validation set DataFrame
        n_examples: Total number of examples (will be split 50/50 by label)
        seed: Random seed for reproducibility
        sentence_length_range: (min, max) sentence length
        idiom_length_range: (min, max) idiom span length

    Returns:
        pd.DataFrame: Selected examples
    """
    np.random.seed(seed)

    # Combine train and validation (both are safe to use)
    pool = pd.concat([train_df, validation_df], ignore_index=True)

    # Calculate sentence length
    pool['sentence_length'] = pool['tokens'].apply(len)

    # Calculate idiom length (for figurative samples)
    def get_idiom_length(tags):
        spans = [tag for tag in tags if tag != 'O']
        return len(spans) if spans else 0

    pool['idiom_length'] = pool['iob_tags'].apply(get_idiom_length)

    # Filter for moderate complexity
    min_sent, max_sent = sentence_length_range
    pool_filtered = pool[pool['sentence_length'].between(min_sent, max_sent)]

    # Select examples stratified by label
    n_per_label = n_examples // 2

    # Literal examples (label=0)
    literal_pool = pool_filtered[pool_filtered['label'] == 0]
    literal_examples = literal_pool.sample(n=n_per_label, random_state=seed)

    # Figurative examples (label=1) with idiom length filter
    min_idiom, max_idiom = idiom_length_range
    figurative_pool = pool_filtered[
        (pool_filtered['label'] == 1) &
        (pool_filtered['idiom_length'].between(min_idiom, max_idiom))
    ]
    figurative_examples = figurative_pool.sample(n=n_per_label, random_state=seed)

    # Combine
    few_shot_examples = pd.concat([literal_examples, figurative_examples])
    few_shot_examples = few_shot_examples.sample(frac=1, random_state=seed)  # Shuffle

    return few_shot_examples


def validate_no_leakage(few_shot_ids, test_ids):
    """
    Verify no data leakage between few-shot examples and test set.

    Args:
        few_shot_ids: List of IDs used in few-shot examples
        test_ids: List of IDs in test set

    Raises:
        AssertionError: If leakage detected
    """
    overlap = set(few_shot_ids) & set(test_ids)
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} samples in both few-shot and test: {overlap}"
    print(f"✅ No data leakage: {len(few_shot_ids)} few-shot examples, {len(test_ids)} test samples, 0 overlap")
```

### 13.3 Usage Example

```python
# Load data
train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/validation.csv")
seen_test_df = pd.read_csv("data/splits/test.csv")
unseen_test_df = pd.read_csv("data/splits/unseen_idiom_test.csv")

# Select 5-shot examples
few_shot_5 = select_few_shot_examples(
    train_df,
    val_df,
    n_examples=5,
    seed=42
)

# Validate no leakage
validate_no_leakage(
    few_shot_5['id'].tolist(),
    seen_test_df['id'].tolist() + unseen_test_df['id'].tolist()
)

# Save for reproducibility
few_shot_5.to_csv("experiments/prompting/few_shot_examples_seed42.csv", index=False)
```

### 13.4 Documentation Requirements

**When reporting prompting results, include:**
```json
{
  "few_shot_config": {
    "num_examples": 5,
    "selection_seed": 42,
    "source_split": "train+validation",
    "stratification": "by_label",
    "sentence_length_range": [10, 20],
    "idiom_length_range": [2, 4],
    "example_ids": ["1_lit_3", "5_fig_12", "8_lit_0", "12_fig_8", "20_lit_15"]
  }
}
```

---

## 14. Error Analysis Protocol

### 14.1 Complete Error Analysis Workflow

```python
from src.utils.error_analysis import (
    analyze_span_errors,
    analyze_cls_errors,
    categorize_span_error,
    categorize_cls_error
)
import pandas as pd

def perform_complete_error_analysis(
    predictions_file,
    task,
    output_dir,
    n_examples_per_category=5
):
    """
    Complete error analysis following standardized protocol.

    Steps:
    1. Load predictions
    2. Categorize all errors
    3. Compute error distribution
    4. Extract examples for each error type
    5. Create visualizations
    6. Save reports

    Args:
        predictions_file: Path to eval_predictions.json
        task: "cls" or "span"
        output_dir: Where to save analysis outputs
        n_examples_per_category: Number of examples to extract per error type

    Returns:
        dict: Complete error analysis results
    """
    import json
    from pathlib import Path
    from collections import Counter

    # 1. Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # 2. Categorize errors (if not already done)
    for pred in predictions:
        if 'error_category' not in pred:
            if task == "cls":
                pred['error_category'] = categorize_cls_error(
                    pred['true_label'],
                    pred['predicted_label']
                )
            elif task == "span":
                pred['error_category'] = categorize_span_error(
                    pred['true_tags'],
                    pred['predicted_tags']
                )

    # 3. Compute error distribution
    error_counts = Counter([p['error_category'] for p in predictions])
    total = len(predictions)

    error_distribution = []
    for error_type, count in error_counts.most_common():
        error_distribution.append({
            'error_type': error_type,
            'count': count,
            'percentage': (count / total) * 100
        })

    error_dist_df = pd.DataFrame(error_distribution)

    # 4. Extract examples for each error type
    error_examples = {}
    for error_type in error_counts.keys():
        if error_type in ['CORRECT', 'PERFECT']:
            continue  # Skip correct predictions

        examples = [p for p in predictions if p['error_category'] == error_type]

        # Sample n examples
        import random
        random.seed(42)
        sampled = random.sample(examples, min(n_examples_per_category, len(examples)))

        error_examples[error_type] = sampled

    # 5. Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save error distribution
    error_dist_df.to_csv(output_path / "error_distribution.csv", index=False)

    # Save error examples
    with open(output_path / "error_examples.json", 'w', encoding='utf-8') as f:
        json.dump(error_examples, f, ensure_ascii=False, indent=2)

    # 6. Create markdown report
    create_error_report(
        error_dist_df,
        error_examples,
        task,
        output_path / "error_analysis_report.md"
    )

    return {
        'error_distribution': error_dist_df,
        'error_examples': error_examples,
        'total_samples': total
    }


def create_error_report(error_dist_df, error_examples, task, output_file):
    """
    Create human-readable markdown error report.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Error Analysis Report: Task {task.upper()}\n\n")

        f.write("## Error Distribution\n\n")
        f.write(error_dist_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Error Examples\n\n")
        for error_type, examples in error_examples.items():
            f.write(f"### {error_type}\n\n")
            f.write(f"**Count:** {len(examples)} examples shown\n\n")

            for i, ex in enumerate(examples[:5], 1):
                f.write(f"**Example {i}:**\n")
                f.write(f"- **Sentence:** {ex['sentence']}\n")

                if task == "span":
                    f.write(f"- **True span:** {ex.get('true_span_text', 'N/A')}\n")
                    f.write(f"- **Predicted span:** {ex.get('predicted_span_text', 'N/A')}\n")
                elif task == "cls":
                    f.write(f"- **True label:** {ex['true_label']}\n")
                    f.write(f"- **Predicted label:** {ex['predicted_label']}\n")

                f.write("\n")

            f.write("\n")
```

### 14.2 Per-Idiom Difficulty Analysis

```python
def analyze_idiom_difficulty(predictions, task):
    """
    Identify which idioms are most difficult for the model.

    Args:
        predictions: List of predictions with 'id' field
        task: "cls" or "span"

    Returns:
        pd.DataFrame: Per-idiom F1 scores sorted by difficulty
    """
    import pandas as pd

    # Extract idiom ID from sample ID
    for pred in predictions:
        pred['idiom_id'] = pred['id'].split('_')[0]

    # Group by idiom
    idiom_groups = {}
    for pred in predictions:
        idiom_id = pred['idiom_id']
        if idiom_id not in idiom_groups:
            idiom_groups[idiom_id] = []
        idiom_groups[idiom_id].append(pred)

    # Compute F1 per idiom
    results = []
    for idiom_id, preds in idiom_groups.items():
        if task == "cls":
            from src.utils.error_analysis import compute_cls_metrics
            metrics = compute_cls_metrics(
                [p['true_label'] for p in preds],
                [p['predicted_label'] for p in preds]
            )
        elif task == "span":
            from src.utils.error_analysis import compute_span_f1
            metrics = compute_span_f1(preds)

        results.append({
            'idiom_id': idiom_id,
            'base_pie': preds[0].get('base_pie', 'unknown'),
            'f1': metrics['f1'],
            'num_samples': len(preds),
            'num_errors': sum(1 for p in preds if not p['is_correct'])
        })

    df = pd.DataFrame(results)
    df = df.sort_values('f1')  # Hardest idioms first

    return df
```

### 14.3 Usage Example

```python
# Perform complete error analysis
results = perform_complete_error_analysis(
    predictions_file="experiments/results/evaluation/fine_tuning/seen_test/dictabert/span/seed_42/eval_predictions.json",
    task="span",
    output_dir="experiments/results/analysis/error_analysis/dictabert_span_seed42",
    n_examples_per_category=10
)

# Analyze idiom difficulty
import json
with open("experiments/results/evaluation/fine_tuning/seen_test/dictabert/span/seed_42/eval_predictions.json") as f:
    predictions = json.load(f)

difficulty_df = analyze_idiom_difficulty(predictions, task="span")
difficulty_df.to_csv("experiments/results/analysis/idiom_difficulty.csv", index=False)

print("Top 10 hardest idioms:")
print(difficulty_df.head(10))
```

---

## 15. Statistical Testing Protocol

### 15.1 Within-Method Comparison (Paired t-test)

**Use case:** Compare models within same method (e.g., DictaBERT vs AlephBERT)

```python
from scipy import stats
import numpy as np

def paired_ttest_models(model1_seeds, model2_seeds, alpha=0.05):
    """
    Paired t-test for comparing two models evaluated on same seeds.

    Use when:
    - Comparing models within fine-tuning
    - Both models evaluated on same test set with same seeds

    Args:
        model1_seeds: List[float] - F1 scores for model 1 across seeds
        model2_seeds: List[float] - F1 scores for model 2 across seeds
        alpha: float - Significance level (default: 0.05)

    Returns:
        dict: Test results with interpretation

    Example:
        dictabert_seeds = [0.9483, 0.9501, 0.9467]  # seeds 42, 123, 456
        alephbert_seeds = [0.9421, 0.9438, 0.9405]

        result = paired_ttest_models(dictabert_seeds, alephbert_seeds)
        print(result['interpretation'])
    """
    assert len(model1_seeds) == len(model2_seeds), "Must have same number of seeds"

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(model1_seeds, model2_seeds)

    # Compute effect size (Cohen's d for paired samples)
    differences = np.array(model1_seeds) - np.array(model2_seeds)
    d = np.mean(differences) / np.std(differences, ddof=1)

    # Interpret
    is_significant = p_value < alpha

    mean_diff = np.mean(differences)

    interpretation = f"""
Paired t-test results:
- Mean difference: {mean_diff:.4f}
- t-statistic: {t_stat:.4f}
- p-value: {p_value:.4f}
- Significant at α={alpha}: {'YES' if is_significant else 'NO'}
- Effect size (Cohen's d): {d:.4f} ({'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'})
"""

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'mean_difference': mean_diff,
        'cohens_d': d,
        'interpretation': interpretation
    }
```

### 15.2 Cross-Method Comparison (Independent t-test or Bootstrap)

**Use case:** Compare fine-tuning vs prompting (different evaluation paradigms)

```python
def independent_ttest_or_bootstrap(
    group1_scores,
    group2_scores,
    method='bootstrap',
    alpha=0.05,
    n_bootstrap=10000
):
    """
    Compare two independent groups (e.g., fine-tuning vs prompting).

    Use when:
    - Comparing fine-tuning vs prompting
    - Different number of runs between groups
    - Non-normal distributions (use bootstrap)

    Args:
        group1_scores: List[float] - Scores for group 1
        group2_scores: List[float] - Scores for group 2
        method: 'ttest' or 'bootstrap'
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples

    Returns:
        dict: Test results
    """
    if method == 'ttest':
        # Welch's t-test (does not assume equal variances)
        t_stat, p_value = stats.ttest_ind(group1_scores, group2_scores, equal_var=False)

        # Cohen's d for independent samples
        pooled_std = np.sqrt((np.var(group1_scores, ddof=1) + np.var(group2_scores, ddof=1)) / 2)
        d = (np.mean(group1_scores) - np.mean(group2_scores)) / pooled_std

        return {
            'method': 'Welch t-test',
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'mean_diff': np.mean(group1_scores) - np.mean(group2_scores),
            'cohens_d': d
        }

    elif method == 'bootstrap':
        # Bootstrap test
        observed_diff = np.mean(group1_scores) - np.mean(group2_scores)

        # Combine groups for permutation
        combined = np.concatenate([group1_scores, group2_scores])
        n1 = len(group1_scores)

        # Permutation test
        perm_diffs = []
        np.random.seed(42)
        for _ in range(n_bootstrap):
            perm = np.random.permutation(combined)
            perm_diff = np.mean(perm[:n1]) - np.mean(perm[n1:])
            perm_diffs.append(perm_diff)

        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

        # Bootstrap CI for difference
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1_scores, size=len(group1_scores), replace=True)
            sample2 = np.random.choice(group2_scores, size=len(group2_scores), replace=True)
            bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))

        ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

        return {
            'method': 'Bootstrap permutation test',
            'observed_diff': observed_diff,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'ci_95': (ci_lower, ci_upper)
        }
```

### 15.3 Multiple Comparison Correction (Bonferroni)

**Use case:** Comparing multiple models (e.g., 6 encoders)

```python
def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons.

    Use when:
    - Testing multiple hypotheses simultaneously
    - Example: Comparing best model vs 5 other models (5 tests)

    Args:
        p_values: List[float] - p-values from multiple tests
        alpha: float - Family-wise error rate

    Returns:
        dict: Corrected significance thresholds and results
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    results = []
    for i, p in enumerate(p_values):
        results.append({
            'test_id': i,
            'p_value': p,
            'significant_uncorrected': p < alpha,
            'significant_corrected': p < corrected_alpha
        })

    return {
        'n_tests': n_tests,
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'results': results,
        'summary': f"Bonferroni correction: {corrected_alpha:.6f} (α={alpha}/{n_tests} tests)"
    }
```

### 15.4 Complete Comparison Example

```python
# Example: Compare all fine-tuning models

# 1. Load results for all models across seeds
models_results = {
    'dictabert': [0.9483, 0.9501, 0.9467],
    'alephbert': [0.9421, 0.9438, 0.9405],
    'alephbertgimmel': [0.9402, 0.9415, 0.9389],
    'neodictabert': [0.9456, 0.9471, 0.9443],
    'mbert': [0.8821, 0.8835, 0.8809],
    'xlm-r': [0.8912, 0.8928, 0.8897]
}

# 2. Identify best model
best_model = max(models_results.keys(), key=lambda k: np.mean(models_results[k]))
print(f"Best model: {best_model} (F1: {np.mean(models_results[best_model]):.4f})")

# 3. Compare best model vs all others
p_values = []
comparisons = []

for model_name, scores in models_results.items():
    if model_name == best_model:
        continue

    result = paired_ttest_models(
        models_results[best_model],
        scores,
        alpha=0.05
    )

    p_values.append(result['p_value'])
    comparisons.append({
        'model': model_name,
        'p_value': result['p_value'],
        'cohens_d': result['cohens_d'],
        'mean_diff': result['mean_difference']
    })

# 4. Apply Bonferroni correction
correction = bonferroni_correction(p_values, alpha=0.05)

# 5. Create report
comparison_df = pd.DataFrame(comparisons)
comparison_df['significant_bonferroni'] = [
    r['significant_corrected'] for r in correction['results']
]

print("\nModel Comparisons (vs Best Model):")
print(comparison_df.to_markdown(index=False))

# Save results
comparison_df.to_csv("experiments/results/analysis/statistical_comparison.csv", index=False)
```

### 15.5 Effect Size Interpretation

```python
def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size.

    Standard thresholds (Cohen, 1988):
    - Small: |d| = 0.2
    - Medium: |d| = 0.5
    - Large: |d| = 0.8
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
```

---

## 16. Visualization Standards

### 16.1 Publication-Ready Figure Guidelines

**General requirements:**
- **Resolution:** 300 DPI minimum for print
- **Format:** PDF for vector graphics (preferred), PNG for raster
- **Font size:** 10-12pt for labels, 8-10pt for tick labels
- **Color palette:** Colorblind-friendly (use seaborn "colorblind" palette)
- **Style:** Minimal, clean (use seaborn "whitegrid" or "white")

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
sns.set_palette("colorblind")

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})
```

### 16.2 Required Visualizations

#### 16.2.1 Model Comparison (Bar Chart with Error Bars)

```python
def plot_model_comparison(results_df, task, split, output_file):
    """
    Create bar chart comparing models with error bars.

    Args:
        results_df: DataFrame with columns ['model', 'f1_mean', 'f1_std']
        task: "cls" or "span"
        split: "seen_test" or "unseen_test"
        output_file: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by F1
    results_df = results_df.sort_values('f1_mean', ascending=False)

    # Create bar plot
    x = range(len(results_df))
    bars = ax.bar(
        x,
        results_df['f1_mean'],
        yerr=results_df['f1_std'],
        capsize=5,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    # Customize
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title(f'Model Comparison: Task {task.upper()} ({split.replace("_", " ").title()})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(results_df['f1_mean'], results_df['f1_std'])):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

#### 16.2.2 Generalization Gap (Grouped Bar Chart)

```python
def plot_generalization_gap(results_df, output_file):
    """
    Create grouped bar chart showing seen vs unseen F1.

    Args:
        results_df: DataFrame with columns ['model', 'seen_f1', 'unseen_f1', 'gap']
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(results_df))
    width = 0.35

    # Create grouped bars
    bars1 = ax.bar(x - width/2, results_df['seen_f1'], width, label='Seen Test', alpha=0.8)
    bars2 = ax.bar(x + width/2, results_df['unseen_f1'], width, label='Unseen Test', alpha=0.8)

    # Customize
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Generalization: Seen vs Unseen Test Performance', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add gap annotations
    for i, gap in enumerate(results_df['gap']):
        ax.text(i, results_df['unseen_f1'].iloc[i] - 0.05,
                f'Δ{gap:.1f}%', ha='center', va='top', fontsize=7, style='italic')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

#### 16.2.3 Error Distribution (Stacked Bar Chart)

```python
def plot_error_distribution(error_df, task, output_file):
    """
    Create stacked bar chart showing error category distribution.

    Args:
        error_df: DataFrame with columns ['model', 'error_type', 'percentage']
    """
    # Pivot for stacking
    pivot = error_df.pivot(index='model', columns='error_type', values='percentage').fillna(0)

    # Remove CORRECT/PERFECT (focus on errors)
    if 'CORRECT' in pivot.columns:
        pivot = pivot.drop('CORRECT', axis=1)
    if 'PERFECT' in pivot.columns:
        pivot = pivot.drop('PERFECT', axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.plot(kind='bar', stacked=True, ax=ax, alpha=0.9, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Error Percentage (%)', fontsize=11)
    ax.set_title(f'Error Distribution by Category: Task {task.upper()}', fontsize=12)
    ax.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

#### 16.2.4 Confusion Matrix (Heatmap)

```python
def plot_confusion_matrix(y_true, y_pred, labels, output_file):
    """
    Create confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        output_file: Path to save figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

### 16.3 Figure Naming Convention

```python
FIGURE_NAMES = {
    "model_comparison": "fig1_model_comparison_{task}_{split}.pdf",
    "generalization_gap": "fig2_generalization_gap_{task}.pdf",
    "error_distribution": "fig3_error_distribution_{task}_{split}.pdf",
    "confusion_matrix": "fig4_confusion_matrix_{model}_{task}_{split}.pdf",
    "learning_curves": "fig5_learning_curves_{model}_{task}.pdf",
    "per_idiom_heatmap": "fig6_per_idiom_f1_heatmap_{task}.pdf",
    "attention_weights": "fig7_attention_weights_{model}_example{n}.pdf"
}
```

---

## 17. Per-Idiom F1 Analysis

### 17.1 Computation

```python
def compute_per_idiom_f1(predictions, task):
    """
    Compute F1 score for each idiom.

    Args:
        predictions: List of prediction dicts with 'id' field
        task: "cls" or "span"

    Returns:
        pd.DataFrame: Per-idiom results
    """
    from src.utils.error_analysis import compute_cls_metrics, compute_span_f1
    import pandas as pd

    # Extract idiom ID from sample ID (format: "{idiom_id}_{lit|fig}_{sample_num}")
    for pred in predictions:
        pred['idiom_id'] = int(pred['id'].split('_')[0])

    # Group by idiom
    idiom_groups = pd.DataFrame(predictions).groupby('idiom_id')

    results = []
    for idiom_id, group in idiom_groups:
        group_list = group.to_dict('records')

        if task == "cls":
            metrics = compute_cls_metrics(
                [p['true_label'] for p in group_list],
                [p['predicted_label'] for p in group_list]
            )
        elif task == "span":
            metrics = compute_span_f1(group_list)

        results.append({
            'idiom_id': idiom_id,
            'base_pie': group_list[0].get('base_pie', 'unknown'),
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'num_samples': len(group_list),
            'num_correct': sum(p['is_correct'] for p in group_list)
        })

    return pd.DataFrame(results).sort_values('f1')
```

### 17.2 Visualization (Heatmap)

```python
def plot_per_idiom_heatmap(models_results, task, output_file):
    """
    Create heatmap showing per-idiom F1 across models.

    Args:
        models_results: Dict[str, pd.DataFrame] - Per-idiom results for each model
        task: "cls" or "span"
        output_file: Path to save figure
    """
    # Combine all models
    combined = []
    for model_name, df in models_results.items():
        df_copy = df.copy()
        df_copy['model'] = model_name
        combined.append(df_copy)

    all_results = pd.concat(combined, ignore_index=True)

    # Pivot: rows=models, columns=idioms
    pivot = all_results.pivot(index='model', columns='idiom_id', values='f1')

    # Plot
    fig, ax = plt.subplots(figsize=(20, 8))

    sns.heatmap(
        pivot,
        annot=False,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'F1 Score'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_xlabel('Idiom ID', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.set_title(f'Per-Idiom F1 Scores: Task {task.upper()}', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 18. Learning Curves (Fine-Tuning Only)

### 18.1 Extract from TensorBoard Logs

```python
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_tensorboard_metrics(log_dir):
    """
    Extract training/validation metrics from TensorBoard logs.

    Args:
        log_dir: Path to TensorBoard log directory

    Returns:
        pd.DataFrame: Metrics over training steps/epochs
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get available tags
    tags = ea.Tags()['scalars']

    # Extract metrics
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame([
            {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
            for e in events
        ])

    return data


def combine_train_val_metrics(metrics_dict):
    """
    Combine training and validation metrics into single DataFrame.

    Args:
        metrics_dict: Dict from extract_tensorboard_metrics()

    Returns:
        pd.DataFrame: Combined metrics with columns [epoch, train_loss, val_loss, val_f1, ...]
    """
    # Assuming tags like: "train/loss", "eval/loss", "eval/f1"

    train_loss = metrics_dict.get('train/loss', pd.DataFrame())
    eval_loss = metrics_dict.get('eval/loss', pd.DataFrame())
    eval_f1 = metrics_dict.get('eval/f1', pd.DataFrame())

    # Merge on step
    combined = train_loss.merge(
        eval_loss, on='step', how='outer', suffixes=('_train_loss', '_eval_loss')
    ).merge(
        eval_f1, on='step', how='outer'
    )

    combined = combined.rename(columns={
        'value_train_loss': 'train_loss',
        'value_eval_loss': 'val_loss',
        'value': 'val_f1'
    })

    return combined[['step', 'train_loss', 'val_loss', 'val_f1']]
```

### 18.2 Visualization

```python
def plot_learning_curves(metrics_df, model_name, task, output_file):
    """
    Plot training/validation loss and F1 curves.

    Args:
        metrics_df: DataFrame with columns [step, train_loss, val_loss, val_f1]
        model_name: Name of the model
        task: "cls" or "span"
        output_file: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(metrics_df['step'], metrics_df['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(metrics_df['step'], metrics_df['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Training Step', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title(f'Loss Curves: {model_name} ({task.upper()})', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # F1 curve
    ax2.plot(metrics_df['step'], metrics_df['val_f1'], label='Validation F1', color='green', linewidth=2)
    ax2.set_xlabel('Training Step', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title(f'Validation F1: {model_name} ({task.upper()})', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 19. Frozen Backbone Comparison (Fine-Tuning Only)

### 19.1 Experimental Setup

**Mission 6.2:** Compare full fine-tuning vs frozen backbone

```python
# Training configuration
BACKBONE_EXPERIMENTS = {
    "full_fine_tuning": {
        "freeze_encoder": False,
        "description": "All parameters trainable"
    },
    "frozen_backbone": {
        "freeze_encoder": True,
        "description": "Only task head trainable"
    }
}

# Implementation in training script:
if config.freeze_encoder:
    for param in model.base_model.parameters():
        param.requires_grad = False
```

### 19.2 Analysis

```python
def compare_frozen_vs_full(frozen_results, full_results):
    """
    Compare frozen backbone vs full fine-tuning.

    Args:
        frozen_results: Dict with keys [f1_mean, f1_std]
        full_results: Dict with keys [f1_mean, f1_std]

    Returns:
        dict: Comparison summary
    """
    performance_drop = full_results['f1_mean'] - frozen_results['f1_mean']
    performance_drop_pct = (performance_drop / full_results['f1_mean']) * 100

    # Count trainable parameters
    # (Assume frozen has ~1% of parameters trainable)

    summary = f"""
Frozen Backbone Comparison:
- Full Fine-Tuning F1: {full_results['f1_mean']:.4f} ± {full_results['f1_std']:.4f}
- Frozen Backbone F1: {frozen_results['f1_mean']:.4f} ± {frozen_results['f1_std']:.4f}
- Performance Drop: {performance_drop:.4f} ({performance_drop_pct:.2f}%)
- Conclusion: {'Freezing substantially hurts performance' if performance_drop > 0.05 else 'Freezing has minimal impact'}
"""

    return {
        'full_f1': full_results['f1_mean'],
        'frozen_f1': frozen_results['f1_mean'],
        'performance_drop': performance_drop,
        'performance_drop_pct': performance_drop_pct,
        'summary': summary
    }
```

---

## 20. Data Size Impact Analysis (Fine-Tuning Only)

### 20.1 Experimental Setup

**Mission 6.4:** Train with 10%, 25%, 50%, 75%, 100% of training data

```python
import pandas as pd
import numpy as np

def create_data_size_subsets(train_df, percentages=[10, 25, 50, 75, 100], seed=42):
    """
    Create training data subsets.

    Args:
        train_df: Full training set
        percentages: List of percentages to sample
        seed: Random seed for reproducibility

    Returns:
        Dict[int, pd.DataFrame]: Subsets by percentage
    """
    np.random.seed(seed)

    subsets = {}
    for pct in percentages:
        n_samples = int(len(train_df) * (pct / 100))

        # Stratified sampling to maintain label balance
        literal = train_df[train_df['label'] == 0]
        figurative = train_df[train_df['label'] == 1]

        n_literal = int(n_samples * 0.5)
        n_figurative = n_samples - n_literal

        subset = pd.concat([
            literal.sample(n=n_literal, random_state=seed),
            figurative.sample(n=n_figurative, random_state=seed)
        ])

        subsets[pct] = subset.sample(frac=1, random_state=seed)  # Shuffle

    return subsets
```

### 20.2 Analysis & Visualization

```python
def plot_data_size_impact(data_size_results, output_file):
    """
    Plot F1 vs training data size.

    Args:
        data_size_results: List[dict] with keys [percentage, f1_mean, f1_std]
    """
    df = pd.DataFrame(data_size_results).sort_values('percentage')

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        df['percentage'],
        df['f1_mean'],
        yerr=df['f1_std'],
        marker='o',
        markersize=8,
        capsize=5,
        linewidth=2,
        color='steelblue'
    )

    ax.set_xlabel('Training Data Size (%)', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Impact of Training Data Size on Performance', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Add value labels
    for _, row in df.iterrows():
        ax.text(
            row['percentage'],
            row['f1_mean'] + row['f1_std'] + 0.02,
            f"{row['f1_mean']:.3f}",
            ha='center',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def compute_data_efficiency(data_size_results):
    """
    Compute data efficiency metric.

    Returns percentage of full-data performance achieved with 50% of data.
    """
    df = pd.DataFrame(data_size_results)

    full_f1 = df[df['percentage'] == 100]['f1_mean'].values[0]
    half_f1 = df[df['percentage'] == 50]['f1_mean'].values[0]

    efficiency = (half_f1 / full_f1) * 100

    return {
        'full_data_f1': full_f1,
        'half_data_f1': half_f1,
        'efficiency_pct': efficiency,
        'summary': f"With 50% of data, model achieves {efficiency:.1f}% of full-data performance"
    }
```

---

## 21. Hyperparameter Sensitivity Analysis (Fine-Tuning Only)

### 21.1 Using Optuna Results

**Mission 6.3:** Analyze how F1 varies across hyperparameter space

```python
import json
import pandas as pd

def load_optuna_study(study_file):
    """
    Load Optuna study results.

    Args:
        study_file: Path to Optuna study JSON (e.g., best_params_dictabert_cls.json)

    Returns:
        pd.DataFrame: All trials with hyperparameters and F1
    """
    with open(study_file, 'r') as f:
        study_data = json.load(f)

    # Extract trials
    trials = []
    for trial in study_data.get('trials', []):
        trial_data = {
            'trial_id': trial['number'],
            'f1': trial['value'],
            **trial['params']
        }
        trials.append(trial_data)

    return pd.DataFrame(trials)


def analyze_hyperparameter_sensitivity(trials_df, param_name):
    """
    Analyze sensitivity to a specific hyperparameter.

    Args:
        trials_df: DataFrame from load_optuna_study()
        param_name: Name of hyperparameter (e.g., 'learning_rate')

    Returns:
        dict: Sensitivity analysis results
    """
    import numpy as np

    # Compute correlation
    correlation = trials_df[param_name].corr(trials_df['f1'])

    # Rank by importance (variance in F1 across parameter range)
    param_bins = pd.qcut(trials_df[param_name], q=4, duplicates='drop')
    f1_variance_across_bins = trials_df.groupby(param_bins)['f1'].var().mean()

    return {
        'parameter': param_name,
        'correlation_with_f1': correlation,
        'f1_variance': f1_variance_across_bins,
        'interpretation': 'High sensitivity' if abs(correlation) > 0.5 or f1_variance_across_bins > 0.01 else 'Low sensitivity'
    }


def plot_hyperparameter_sensitivity(trials_df, param_name, output_file):
    """
    Create scatter plot showing F1 vs hyperparameter value.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(trials_df[param_name], trials_df['f1'], alpha=0.6, s=50)

    # Add trend line
    z = np.polyfit(trials_df[param_name], trials_df['f1'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(trials_df[param_name].min(), trials_df[param_name].max(), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.7, label='Trend')

    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title(f'Hyperparameter Sensitivity: {param_name}', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 22. Token Importance & Attention Analysis (Fine-Tuning Only)

### 22.1 Token Importance (Integrated Gradients)

**Mission 6.1:** Identify which tokens are most important for predictions

```python
import torch
from captum.attr import IntegratedGradients

def compute_token_importance(model, tokenizer, sentence, tokens, label):
    """
    Compute token importance scores using Integrated Gradients.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        sentence: Input sentence
        tokens: Pre-tokenized tokens (for alignment)
        label: True label

    Returns:
        List[Tuple[str, float]]: (token, importance_score)
    """
    model.eval()

    # Tokenize
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Forward pass to get embeddings
    def forward_func(inputs_embeds):
        outputs = model(inputs_embeds=inputs_embeds)
        return outputs.logits[:, label]

    # Get embeddings
    embeddings = model.get_input_embeddings()(inputs['input_ids'])

    # Compute integrated gradients
    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(embeddings, target=label)

    # Aggregate attributions (sum over embedding dimension)
    token_importance = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # Map back to original tokens
    word_ids = inputs.word_ids()
    token_scores = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in token_scores:
                token_scores[word_id] = 0
            token_scores[word_id] += token_importance[i]

    # Return (token, score) pairs
    results = [(tokens[word_id], score) for word_id, score in sorted(token_scores.items())]

    return results


def visualize_token_importance(tokens_with_scores, sentence, output_file):
    """
    Create visualization of token importance.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    tokens, scores = zip(*tokens_with_scores)
    scores = np.array(scores)

    # Normalize scores
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 2))

    # Color map: white (low importance) to red (high importance)
    cmap = LinearSegmentedColormap.from_list('importance', ['white', 'yellow', 'red'])

    # Create colored boxes for each token
    for i, (token, score) in enumerate(zip(tokens, scores_norm)):
        color = cmap(score)
        ax.text(i, 0.5, token, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=1))

    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Token Importance: {sentence}', fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

### 22.2 Attention Weights Analysis

```python
def extract_attention_weights(model, tokenizer, sentence, tokens):
    """
    Extract attention weights from the model.

    Args:
        model: Trained model with output_attentions=True
        tokenizer: Tokenizer
        sentence: Input sentence
        tokens: Pre-tokenized tokens

    Returns:
        np.ndarray: Attention weights [num_layers, num_heads, seq_len, seq_len]
    """
    model.eval()

    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack attention weights from all layers
    attentions = torch.stack(outputs.attentions)  # [num_layers, batch, num_heads, seq_len, seq_len]
    attentions = attentions.squeeze(1).cpu().numpy()  # Remove batch dimension

    return attentions


def plot_attention_heatmap(attention_weights, tokens, layer_idx, head_idx, output_file):
    """
    Plot attention heatmap for specific layer and head.
    """
    import seaborn as sns

    att = attention_weights[layer_idx, head_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        att,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )

    ax.set_title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})', fontsize=12)
    ax.set_xlabel('Key', fontsize=10)
    ax.set_ylabel('Query', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 23. Prompting-Specific Metrics & Analysis

### 23.1 Parse Success Rate

**Critical for prompting:** LLM outputs must be parseable

```python
def compute_parse_success_rate(predictions):
    """
    Compute percentage of successfully parsed outputs.

    Args:
        predictions: List of prediction dicts with 'parse_success' field

    Returns:
        dict: Parse success metrics
    """
    total = len(predictions)
    successful = sum(1 for p in predictions if p.get('parse_success', True))
    failed = total - successful

    return {
        'total_samples': total,
        'successful_parses': successful,
        'failed_parses': failed,
        'parse_success_rate': successful / total if total > 0 else 0.0,
        'parse_failure_rate': failed / total if total > 0 else 0.0
    }


def analyze_parse_failures(predictions):
    """
    Analyze why parsing failed.

    Returns:
        pd.DataFrame: Failure reasons and counts
    """
    failures = [p for p in predictions if not p.get('parse_success', True)]

    # Categorize failure reasons
    failure_reasons = []
    for pred in failures:
        reason = pred.get('parse_error_reason', 'unknown')
        failure_reasons.append({
            'id': pred['id'],
            'sentence': pred['sentence'],
            'raw_output': pred.get('raw_output', ''),
            'reason': reason
        })

    return pd.DataFrame(failure_reasons)
```

### 23.2 Cost & Latency Tracking

```python
def track_inference_metrics(model_name, predictions):
    """
    Track cost and latency for LLM inference.

    Args:
        model_name: Name of LLM
        predictions: List of predictions with 'tokens_used' and 'latency' fields

    Returns:
        dict: Cost and latency metrics
    """
    # Pricing (example rates, update with actual)
    PRICING = {
        "dictalm-3.0": {"input": 0.0002, "output": 0.0004},  # per 1K tokens
        "llama-3.1-8b": {"input": 0.0003, "output": 0.0006},
        "qwen2.5-7b": {"input": 0.0002, "output": 0.0005}
    }

    total_input_tokens = sum(p.get('input_tokens', 0) for p in predictions)
    total_output_tokens = sum(p.get('output_tokens', 0) for p in predictions)
    total_latency = sum(p.get('latency_seconds', 0) for p in predictions)

    # Calculate cost
    pricing = PRICING.get(model_name, {"input": 0, "output": 0})
    input_cost = (total_input_tokens / 1000) * pricing['input']
    output_cost = (total_output_tokens / 1000) * pricing['output']
    total_cost = input_cost + output_cost

    return {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'avg_tokens_per_sample': (total_input_tokens + total_output_tokens) / len(predictions),
        'total_cost_usd': total_cost,
        'cost_per_sample_usd': total_cost / len(predictions),
        'total_latency_seconds': total_latency,
        'avg_latency_seconds': total_latency / len(predictions)
    }
```

### 23.3 Temperature Sensitivity (Prompting Only)

```python
def compare_temperature_settings(results_by_temperature):
    """
    Compare performance across temperature settings.

    Args:
        results_by_temperature: Dict[float, dict] - Results for each temperature

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison = []
    for temp, results in results_by_temperature.items():
        comparison.append({
            'temperature': temp,
            'f1_mean': results['f1_mean'],
            'f1_std': results['f1_std'],
            'parse_success_rate': results.get('parse_success_rate', None)
        })

    df = pd.DataFrame(comparison).sort_values('temperature')

    return df


def plot_temperature_sensitivity(comparison_df, output_file):
    """
    Plot F1 vs temperature.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        comparison_df['temperature'],
        comparison_df['f1_mean'],
        yerr=comparison_df['f1_std'],
        marker='o',
        markersize=8,
        capsize=5,
        linewidth=2
    )

    ax.set_xlabel('Temperature', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Temperature Sensitivity', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 24. Cross-Method Comparison (Joint Analysis)

### 24.1 Unified Comparison Table

```python
def create_unified_comparison_table(fine_tuning_results, prompting_results):
    """
    Create unified table comparing fine-tuning and prompting.

    Args:
        fine_tuning_results: List[dict] - Results for all fine-tuned models
        prompting_results: List[dict] - Results for all prompting strategies

    Returns:
        pd.DataFrame: Unified comparison table
    """
    all_results = []

    # Add fine-tuning results
    for result in fine_tuning_results:
        all_results.append({
            'method': 'Fine-Tuning',
            'model': result['model'],
            'strategy': '-',
            'seen_f1_mean': result['seen_f1_mean'],
            'seen_f1_std': result['seen_f1_std'],
            'unseen_f1_mean': result['unseen_f1_mean'],
            'unseen_f1_std': result['unseen_f1_std'],
            'gap_absolute': result['seen_f1_mean'] - result['unseen_f1_mean'],
            'gap_percentage': ((result['seen_f1_mean'] - result['unseen_f1_mean']) / result['seen_f1_mean']) * 100
        })

    # Add prompting results
    for result in prompting_results:
        all_results.append({
            'method': 'Prompting',
            'model': result['model'],
            'strategy': result['strategy'],
            'seen_f1_mean': result['seen_f1_mean'],
            'seen_f1_std': result['seen_f1_std'],
            'unseen_f1_mean': result['unseen_f1_mean'],
            'unseen_f1_std': result['unseen_f1_std'],
            'gap_absolute': result['seen_f1_mean'] - result['unseen_f1_mean'],
            'gap_percentage': ((result['seen_f1_mean'] - result['unseen_f1_mean']) / result['seen_f1_mean']) * 100
        })

    df = pd.DataFrame(all_results)
    df = df.sort_values('seen_f1_mean', ascending=False)

    return df
```

### 24.2 Error Pattern Comparison

```python
def compare_error_patterns(fine_tuning_errors, prompting_errors):
    """
    Compare error distributions between methods.

    Args:
        fine_tuning_errors: Dict[str, float] - Error category percentages for fine-tuning
        prompting_errors: Dict[str, float] - Error category percentages for prompting

    Returns:
        pd.DataFrame: Side-by-side comparison
    """
    comparison = []
    all_categories = set(list(fine_tuning_errors.keys()) + list(prompting_errors.keys()))

    for category in all_categories:
        comparison.append({
            'error_category': category,
            'fine_tuning_pct': fine_tuning_errors.get(category, 0.0),
            'prompting_pct': prompting_errors.get(category, 0.0),
            'difference': prompting_errors.get(category, 0.0) - fine_tuning_errors.get(category, 0.0)
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values('difference', ascending=False)

    return df


def plot_error_pattern_comparison(comparison_df, output_file):
    """
    Visualize error pattern differences.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparison_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, comparison_df['fine_tuning_pct'], width, label='Fine-Tuning', alpha=0.8)
    bars2 = ax.bar(x + width/2, comparison_df['prompting_pct'], width, label='Prompting', alpha=0.8)

    ax.set_xlabel('Error Category', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Error Pattern Comparison: Fine-Tuning vs Prompting', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['error_category'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 25. Reproducibility Checklist

### 25.1 Required Documentation

**Every experiment must document:**

```python
REPRODUCIBILITY_REQUIREMENTS = {
    "code": {
        "required": [
            "Exact commit hash or version tag",
            "Complete environment.yml or requirements.txt",
            "Python version"
        ]
    },
    "data": {
        "required": [
            "Dataset version or commit hash",
            "Exact split files used",
            "Any preprocessing steps"
        ]
    },
    "model": {
        "required": [
            "Exact model identifier (HuggingFace ID or path)",
            "Tokenizer version",
            "Model checkpoint (if applicable)"
        ]
    },
    "training": {
        "required_fine_tuning": [
            "All hyperparameters (learning_rate, batch_size, epochs, etc.)",
            "Random seeds (3 seeds minimum)",
            "Hardware used (GPU type, number)",
            "Training time per seed"
        ],
        "required_prompting": [
            "Exact prompt template (full text)",
            "Few-shot example IDs",
            "Temperature and sampling parameters",
            "Model API version or local model version"
        ]
    },
    "evaluation": {
        "required": [
            "Metric computation code/library version",
            "Test set used (exact file path)",
            "Evaluation timestamp"
        ]
    }
}
```

### 25.2 Reproducibility Report Template

```python
def generate_reproducibility_report(config, results, output_file):
    """
    Generate comprehensive reproducibility report.

    Args:
        config: Experiment configuration dict
        results: Experiment results dict
        output_file: Path to save markdown report
    """
    import platform
    import torch
    import transformers
    from datetime import datetime

    report = f"""
# Reproducibility Report

**Generated:** {datetime.now().isoformat()}

## Environment

- **Python Version:** {platform.python_version()}
- **PyTorch Version:** {torch.__version__}
- **Transformers Version:** {transformers.__version__}
- **CUDA Version:** {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
- **Operating System:** {platform.system()} {platform.release()}

## Code

- **Commit Hash:** {config.get('commit_hash', 'NOT SPECIFIED')}
- **Repository:** {config.get('repository_url', 'NOT SPECIFIED')}

## Data

- **Dataset:** {config['dataset_path']}
- **Split:** {config['split_name']}
- **Number of Samples:** {config['num_samples']}

## Model

- **Model ID:** {config['model_name']}
- **Task:** {config['task']}
- **Method:** {config['method']}

## Hyperparameters

"""

    for key, value in config.get('hyperparameters', {}).items():
        report += f"- **{key}:** {value}\n"

    report += f"""

## Results

- **F1 Score:** {results['f1']:.4f}
- **Accuracy:** {results['accuracy']:.4f}
- **Precision:** {results['precision']:.4f}
- **Recall:** {results['recall']:.4f}

## Files

- **Results:** {results.get('results_file', 'NOT SAVED')}
- **Predictions:** {results.get('predictions_file', 'NOT SAVED')}
- **Model Checkpoint:** {results.get('checkpoint_path', 'NOT SAVED')}

---

**To reproduce this experiment:**

1. Clone repository at commit `{config.get('commit_hash', 'COMMIT_HASH')}`
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py --config {config.get('config_file', 'config.json')}`
4. Run evaluation: `python evaluate.py --checkpoint {results.get('checkpoint_path', 'CHECKPOINT')}`
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
```

---

## 26. Implementation Checklist

### 26.1 Fine-Tuning Partner Checklist

```
Phase 4: Fine-Tuning (COMPLETED ✅)
- [x] Train all 6 encoder models (AlephBERT, AlephBERTGimmel, DictaBERT, NeoDictaBERT, mBERT, XLM-R)
- [x] Both tasks (CLS + SPAN)
- [x] Three seeds (42, 123, 456)
- [x] Seen + Unseen evaluation
- [x] Save eval_results.json and eval_predictions.json

Phase 6: Ablations & Interpretability (IN PROGRESS ⏳)
- [ ] 6.1: Token importance analysis (gradients + attention)
- [ ] 6.2: Frozen backbone comparison
- [ ] 6.3: Hyperparameter sensitivity
- [ ] 6.4: Data size impact (10%, 25%, 50%, 75%, 100%)
- [ ] Extract learning curves from TensorBoard

Phase 7: Comprehensive Analysis (IN PROGRESS ⏳)
- [x] 7.1: Basic error analysis
- [ ] 7.1: Complete error categorization for all models
- [ ] 7.2: Statistical significance testing (paired t-tests + Bonferroni)
- [ ] 7.3: Cross-task analysis (CLS vs SPAN)
- [ ] 7.4: Create all publication figures
- [ ] 7.5: Create all publication tables
- [ ] Per-idiom F1 analysis
- [ ] Confusion matrices for all models
```

### 26.2 Prompting Partner Checklist

```
Phase 5: LLM Evaluation (PENDING ❌)
- [ ] 5.1: Zero-shot prompting (CLS + SPAN)
- [ ] 5.2: Few-shot prompting (CLS + SPAN)
  - [ ] 5.2.1: Select few-shot examples using protocol (Section 13)
  - [ ] 5.2.2: Validate no data leakage
- [ ] 5.3: Optional: Chain-of-Thought prompting
- [ ] Test all 4+ LLMs (DictaLM, Llama-3.1-8B, Qwen2.5, optional Llama-3.1-70B)
- [ ] Seen + Unseen evaluation
- [ ] Save eval_results.json and eval_predictions.json using same format

Phase 6: Prompting-Specific Analysis (PENDING ❌)
- [ ] Prompt strategy comparison (zero-shot vs few-shot vs CoT)
- [ ] Temperature sensitivity analysis
- [ ] Parse success rate tracking
- [ ] Cost and latency measurement
- [ ] Output format adherence rate

Phase 7: Joint Analysis (PENDING ❌)
- [ ] Error categorization using shared taxonomy
- [ ] Per-idiom F1 analysis
- [ ] Create unified comparison table with fine-tuning results
- [ ] Error pattern comparison
```

### 26.3 Joint Deliverables Checklist

```
After Both Partners Complete Evaluation:
- [ ] Merge results into unified comparison table
- [ ] Cross-method error pattern analysis
- [ ] Statistical comparison (fine-tuning vs prompting)
- [ ] Effect size calculation (Cohen's d)
- [ ] Create publication-ready figures
- [ ] Create publication-ready tables
- [ ] Write Results section for paper
- [ ] Write Discussion section
```

---

## 27. Quick Reference: File Locations

### 27.1 Data Files

```
data/
├── expressions_data_tagged.csv          # Main dataset
└── splits/
    ├── train.csv                        # Training set
    ├── validation.csv                   # Validation set
    ├── test.csv                         # Seen test
    └── unseen_idiom_test.csv           # Unseen test
```

### 27.2 Result Files

```
experiments/results/evaluation/
├── fine_tuning/
│   ├── seen_test/{model}/{task}/seed_{seed}/
│   │   ├── eval_results.json
│   │   └── eval_predictions.json
│   └── unseen_test/{model}/{task}/seed_{seed}/
│       ├── eval_results.json
│       └── eval_predictions.json
└── prompting/
    ├── seen_test/{model}/{task}/{strategy}/
    │   ├── eval_results.json
    │   └── eval_predictions.json
    └── unseen_test/{model}/{task}/{strategy}/
        ├── eval_results.json
        └── eval_predictions.json
```

### 27.3 Analysis Files

```
experiments/results/analysis/
├── finetuning_summary.csv
├── finetuning_summary.md
├── statistical_significance.txt
├── generalization/
│   ├── generalization_report.md
│   ├── generalization_gap.csv
│   └── figures/
├── error_analysis/
│   ├── {model}_{task}_{seed}/
│   │   ├── error_distribution.csv
│   │   ├── error_examples.json
│   │   └── error_analysis_report.md
│   └── idiom_difficulty.csv
└── unified_comparison.csv
```

### 27.4 Utility Scripts

```
src/
├── utils/
│   └── error_analysis.py                # Shared error categorization functions
├── analyze_finetuning_results.py        # Aggregate fine-tuning results
├── analyze_generalization.py            # Generalization gap analysis
├── analyze_comprehensive.py             # Comprehensive analysis
└── create_prediction_report.py          # Human-readable prediction reports
```

---

## 28. Contact & Updates

**Version History:**
- v1.0 (Initial): Basic task definitions and metrics
- v2.0: Added error taxonomy and output formats
- v3.0: Unified fine-tuning and prompting standards
- v4.0 (Current): Complete NLP best practices integration

**For Questions:**
- Read this guide first (Sections 0-12 cover 90% of common questions)
- Check MISSION_4.7_SUMMARY_AND_NEXT_STEPS.md for task-specific guidance
- Use `src/utils/error_analysis.py` for all metric computations

**Updates:**
- This guide is version-controlled in the repository
- Any changes must be reviewed by both partners
- Version number increments with each update

---

## 29. Summary Tables

### 29.1 Method Comparison Matrix

| Aspect | Fine-Tuning | Prompting |
|---|---|---|
| **Models** | Encoders (AlephBERT, DictaBERT, etc.) | LLMs (DictaLM, Llama, Qwen) |
| **Evaluation Paradigm** | Multi-seed (42/123/456) | Deterministic (temp=0) or stochastic |
| **Primary Metric** | Span F1 (exact match) | Span F1 (exact match) |
| **Error Taxonomy** | Shared (12 categories for SPAN) | Shared (12 categories for SPAN) |
| **Ablations** | Data size, frozen backbone, hyperparameters | Temperature, prompt strategy |
| **Interpretability** | Gradients, attention weights | Prompt analysis, failure modes |
| **Cost** | Training compute | Inference tokens |
| **Unique Metrics** | Learning curves | Parse success rate, latency |

### 29.2 Metric Summary

| Task | Primary Metric | Formula | Interpretation |
|---|---|---|---|
| CLS | Macro F1 | Mean of literal_f1 and figurative_f1 | Balanced class performance |
| SPAN | Exact Span F1 | 2PR/(P+R) where P,R computed on exact spans | Strict boundary matching |

### 29.3 Error Category Summary

**CLS (2 categories):**
- `FP`: Predicted Figurative, True Literal
- `FN`: Predicted Literal, True Figurative

**SPAN (12 categories):**
- Correct: `PERFECT`
- Boundary errors: `PARTIAL_START`, `PARTIAL_END`, `PARTIAL_BOTH`, `EXTEND_START`, `EXTEND_END`, `EXTEND_BOTH`, `SHIFT`
- Complete errors: `MISS`, `FALSE_POSITIVE`, `WRONG_SPAN`, `MULTI_SPAN`

---

## 30. Conclusion

This guide provides:

✅ **Exact task and data definitions** (no ambiguity)
✅ **Standardized metrics** (same computations for both partners)
✅ **Shared error taxonomy** (common language)
✅ **Complete protocols** (step-by-step with code)
✅ **NLP best practices** (publication-ready)
✅ **Reproducibility** (full documentation requirements)

**Next Steps:**
1. Both partners: Read Sections 0-12 (foundation)
2. Fine-tuning partner: Implement Sections 17-22 (ablations)
3. Prompting partner: Implement Sections 13, 23, 25 (prompting-specific)
4. Both partners: Joint analysis using Sections 24, 28 (comparison)

**Remember:**
- Use `src/utils/error_analysis.py` for all metric computations
- Follow exact naming conventions (Section 12)
- Document everything for reproducibility (Section 25)
- Report both Seen and Unseen results always

---

**End of Guide**
