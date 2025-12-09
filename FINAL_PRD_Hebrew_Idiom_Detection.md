# Product Requirements Document (PRD)
# Hebrew Idiom Detection: Dataset Creation & Model Benchmarking

---

**Document Version:** 3.0 (UPDATED - Fully Aligned with Implementation)
**Last Updated:** December 5, 2025
**Project Type:** Master's Thesis Research
**Primary Researchers:** Igor Nazarenko & Yuval Amit
**Institution:** Reichman University
**Duration:** 12 weeks

---

## Executive Summary

This research project presents **the first comprehensive Hebrew idiom dataset** with dual-task annotations and establishes performance benchmarks across encoder-based transformer models and large language models (LLMs).

**Core Innovation:** A balanced dataset of 4,800 Hebrew sentences enabling:
1. Sentence-level classification (literal vs. figurative)
2. Token-level span identification (IOB2 tagging)

**Research Goals:** Systematically compare 5 fine-tuned transformer models against LLM prompting approaches to determine optimal strategies for Hebrew figurative language understanding.

**Expected Impact:** First systematic benchmark for Hebrew idiom detection with implications for low-resource language NLP and cross-lingual transfer.

---

## 1. Project Overview

### 1.1 Problem Statement

Hebrew idioms present unique NLP challenges:
- **Context-dependent ambiguity**: Same expression can be literal or figurative
- **Multi-word expressions**: Precise boundary detection required
- **Resource scarcity**: No existing Hebrew idiom datasets with dual annotations
- **Model comparison gap**: Unclear whether fine-tuning or prompting is superior

**Example:**
```
Sentence: "הוא שבר את הראש על הבעיה"
          (He broke the head on the problem)

Context 1 (Figurative): Mental effort → "thought hard"
Context 2 (Literal): Physical injury → actual head injury

Challenge: Models must use context to distinguish meaning.
```

### 1.2 Research Questions

1. **Dataset Quality:**
   - Is the dataset balanced and representative?
   - Do IOB2 annotations align correctly with tokens?

2. **Model Performance:**
   - How do Hebrew-specific models (AlephBERT, DictaBERT) compare to multilingual (mBERT, XLM-R)?
   - What is the performance gap between sentence classification and span detection?
   - Do fine-tuned models outperform zero-shot approaches?

3. **LLM Evaluation:**
   - Can open-access LLMs compete with fine-tuned models via prompting?
   - Zero-shot vs. few-shot prompting: What's the improvement?

4. **Training Strategies:**
   - Which hyperparameters matter most for each task?
   - Full fine-tuning vs. frozen backbone: Cost-benefit analysis?

5. **Error Patterns:**
   - What types of idioms are hardest to detect?
   - What contextual patterns cause failures?

### 1.3 Success Criteria

**Minimum Viable Product (MVP):**
- ✅ Dataset validated and splits created
- ✅ 5 encoder models trained and evaluated on both tasks
- ⏳ 1 LLM evaluated via prompting
- ⏳ Comprehensive results comparison
- ⏳ Academic paper draft complete

**Performance Targets:**
- Sentence Classification: F1 > 85%
- Token Classification: F1 > 80%
- LLM competitive within 5% of best fine-tuned model

**Publication Goal:**
- Submit to ACL/EMNLP or similar tier-1 venue
- Dataset released publicly on HuggingFace

---

## 2. Dataset Specification

### 2.1 Dataset Overview

**Name:** Hebrew-Idioms-4800 v2.0
**Size:** 4,800 manually authored sentences (80 per idiom)
**Language:** Hebrew (he)
**Annotation Levels:** 2 (sentence classification + token-level IOB2 spans with character/token indices)
**Balance:** 2,400 literal + 2,400 figurative (perfect 50/50 per idiom)
**Annotators:** 2 native Hebrew speakers (dual annotation)
**IAA:** Cohen's κ = **0.9725** (98.625% agreement, 66 disagreements)
**Quality Score:** 9.2/10 across 14 validation checks
**Status:** ✅ Complete, validated, and packaged with preprocessing + QA notebook (`professor_review/Complete_Dataset_Analysis.ipynb`)

### 2.2 Data Schema

**CRITICAL: Actual Column Names (Updated for v2.0)**

```python
{
    # Identifiers
    "id": str,                       # Informative identifier "{idiom_id}_{lit/fig}_{count}"
                                     # Example: "12_fig_7"
    "split": str,                    # Dataset split: "train", "validation", "test", "unseen_idiom_test"

    # Text Data
    "sentence": str,                 # Full Hebrew sentence (UTF-8 normalized)
    "base_pie": str,                 # Idiom canonical/normalized form
    "pie_span": str,                 # Idiom as it appears in text (with morphology)
    "language": str,                 # "he" (Hebrew)
    "source": str,                   # Data source: "inhouse", "manual"

    # Sentence-Level Label (Task 1)
    "label": int,                    # 0 = literal, 1 = figurative (BINARY)
    "label_str": str,                # "Literal" or "Figurative" (English)

    # Token-Level Annotations (Task 2) - CRITICAL: Pre-tokenized!
    "tokens": list[str],             # PRE-TOKENIZED sentence (punctuation separated)
                                     # Stored as string representation: "['word1', ',', 'word2', '.']"
                                     # Parse with: ast.literal_eval(tokens_str)
    "iob_tags": list[str],          # IOB2 tags aligned to tokens
                                     # Stored as string representation: "['O', 'B-IDIOM', 'I-IDIOM', 'O']"
                                     # Parse with: ast.literal_eval(iob_tags_str)
    "num_tokens": int,               # Total tokens (= len(tokens))
    "start_token": int,              # Token start position (0-indexed)
    "end_token": int,                # Token end position (exclusive, Python-style)

    # Character-Level (Auxiliary)
    "start_char": int,               # Character start position (0-indexed)
    "end_char": int,                 # Character end position (exclusive)
    "char_mask": str,                # Binary character mask (0/1)
}
```

**Example Entry:**
```json
{
    "id": "1_lit_1",
    "split": "train",
    "sentence": "הוא שבר את הראש על הבעיה המורכבת.",
    "base_pie": "שבר את הראש",
    "pie_span": "שבר את הראש",
    "label": 1,
    "label_str": "Figurative",
    "tokens": "['הוא', 'שבר', 'את', 'הראש', 'על', 'הבעיה', 'המורכבת', '.']",
    "iob_tags": "['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O', 'O', 'O', 'O']",
    "num_tokens": 8,
    "start_token": 1,
    "end_token": 4,
    "start_char": 4,
    "end_char": 15,
    "char_mask": "000011111111111000000000000000000"
}
```

**CRITICAL IMPLEMENTATION NOTES:**

1. **Pre-tokenized Data:** The `tokens` column contains the AUTHORITATIVE tokenization
   - Punctuation is SEPARATED (`,` `.` are individual tokens)
   - Do NOT tokenize `sentence` at runtime (causes misalignment)
   - Parse tokens with: `ast.literal_eval(df['tokens'])`

2. **IOB Tags Alignment:** `iob_tags` is aligned to `tokens`, NOT to whitespace-split text
   - Length: `len(iob_tags) == len(tokens) == num_tokens`
   - Parse with: `ast.literal_eval(df['iob_tags'])`

3. **Span Conventions:** Python-style half-open intervals `[start, end)`
   - Character: `sentence[start_char:end_char] == pie_span`
   - Token: `tokens[start_token:end_token]` covers idiom tokens

4. **Data Loading:**
   ```python
   import pandas as pd
   import ast

   df = pd.read_csv('data/expressions_data_tagged_v2.csv')

   # Parse tokens and IOB tags
   df['tokens'] = df['tokens'].apply(ast.literal_eval)
   df['iob_tags'] = df['iob_tags'].apply(ast.literal_eval)
   ```

### 2.3 Dataset Statistics (ACTUAL VALUES from v2.0)

| Metric | Value |
|--------|-------|
| **Total sentences** | 4,800 |
| **Unique idioms** | 60 (100% polysemous) |
| **Samples per idiom** | 80 (40 literal + 40 figurative) |
| **Vocabulary size** | 15,107 unique tokens |
| **Total tokens** | 83,844 |
| **Type-Token Ratio (TTR)** | 0.1802 (word-only: 0.2015) |
| **Hapax legomena** | 8,594 (56.9% of vocabulary) |
| **Dis legomena** | 2,430 tokens |
| **Function word ratio** | 11.43% |

**Sentence Length (Tokens - punctuation separated):**
- Mean: **17.47**
- Median: **13**
- Std: **9.11**
- Range: **5-47**

**Sentence Length (Characters):**
- Mean: **83.03**
- Median: **63**
- Std: **42.55**
- Range: **22-193**

**Idiom Length (Tokens):**
- Mean: **2.48**
- Median: **2**
- Range: **2-5**

**Idiom Length (Characters):**
- Mean: **11.39**
- Median: **11**
- Range: **5-23**

**Sentence Type Distribution:**

| Type | Count | Percentage |
|------|-------|------------|
| Declarative | 4,549 | 94.77% |
| Questions | 239 | 4.98% |
| Exclamatory | 12 | 0.25% |

**Idiom Position (by relative sentence span):**

| Position | Count | Percentage | Literal | Figurative |
|----------|-------|------------|---------|------------|
| Start (0-33%) | 3,123 | 65.06% | 64.71% | 65.42% |
| Middle (33-67%) | 1,449 | 30.19% | 31.50% | 28.88% |
| End (67-100%) | 228 | 4.75% | 3.79% | 5.71% |

**Mean position ratio:** 0.2678 (idioms tend to appear early in sentences)

### 2.4 Split Strategy

**Hybrid Split (Seen + Zero-Shot):**

| Split | Samples | % | Idioms | Literal | Figurative | Notes |
|-------|---------|---|--------|---------|------------|-------|
| **Train** | 3,456 | 72% | 54 (seen) | 1,728 | 1,728 | 32 literal + 32 figurative per idiom |
| **Validation** | 432 | 9% | 54 (seen) | 216 | 216 | 4 literal + 4 figurative per idiom |
| **Test (in-domain)** | 432 | 9% | 54 (seen) | 216 | 216 | 4 literal + 4 figurative per idiom |
| **Unseen idiom test** | 480 | 10% | 6 idioms | 240 | 240 | Zero-shot evaluation only |

**Splitting Methodology:**
- **Seen idioms (54):** Stratified by idiom + label; supports standard training/validation/test loops
- **Unseen idioms (6):** חתך פינה, חצה קו אדום, נשאר מאחור, שבר שתיקה, איבד את הראש, רץ אחרי הזנב של עצמו
- All 80 samples per unseen idiom (balanced literal/figurative) held out for zero-shot transfer evaluation

**Per-Idiom Distribution (Seen Idioms):**
- Train: 64 sentences per idiom (32 literal + 32 figurative)
- Validation: 8 sentences per idiom (4 literal + 4 figurative)
- Test: 8 sentences per idiom (4 literal + 4 figurative)

### 2.5 Linguistic & Structural Highlights

- **Polysemy:** 100% of idioms appear in both literal and figurative contexts
- **Structural Complexity:** Figurative sentences show slightly higher punctuation (mean 1.83 vs 1.73)
- **Collocations:** Top context tokens (±3 around idiom): ',', '.', הוא, היא, לא, הם, על, את
- **Morphological Variance:** Highest variation for שם רגליים (35 forms), שבר את הלב (32), פתח דלתות (29), סגר חשבון (28), הוריד פרופיל (23)

### 2.6 Data Quality & Validation

✅ **14/14 automated validations passed:**
- Missing values: 0/76,800 cells (0%)
- Duplicate rows: 0/4,800 (0%)
- ID sequence: Complete (0-4799)
- Label consistency: 100%
- IOB2 alignment: 100% (tags match token count)
- Character spans: 100% accurate (`sentence[start_char:end_char] == pie_span`)
- Token spans: 100% valid (`0 <= start_token < end_token <= num_tokens`)
- Encoding issues: 0 (BOM removed, Unicode normalized)
- Hebrew text validation: 100%

**Preprocessing Applied:**
- Unicode NFKC normalization
- BOM character removal
- Directional mark removal (LRM/RLM)
- Whitespace normalization
- IOB2 sequence validation (B-IDIOM before I-IDIOM)
- Span verification (character + token)

**Result:** Clean, analysis-ready dataset suitable for publication on HuggingFace

---

## 3. Task Definitions

### 3.1 Task 1: Sentence Classification

**Objective:** Classify sentence as literal (0) or figurative (1)

**Input:**
```python
sentence = "הוא שבר את הראש על הבעיה"
```

**Output:**
```python
label = 1  # Figurative
probability = 0.89
```

**Evaluation Metrics:**
- Accuracy
- Macro F1-score
- Precision & Recall (per class)
- ROC-AUC
- Confusion Matrix (true_positives, false_positives, true_negatives, false_negatives)

**Baseline Performance:**
- Random: 50% (balanced dataset)
- Majority class: 50%
- Expected fine-tuned: 85-92%

**Implementation:**
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2  # Binary: literal (0) vs. figurative (1)
)

# Input: df['sentence'] column
# Output: df['label'] column (0 or 1)
```

### 3.2 Task 2: Token Classification (IOB2)

**Objective:** Identify idiom span using BIO tagging

**CRITICAL: Use Pre-Tokenized Data**

```python
# CORRECT: Use pre-tokenized tokens column
import ast

tokens = ast.literal_eval(df['tokens'][i])
iob_tags = ast.literal_eval(df['iob_tags'][i])

# Example:
# tokens:   ['הוא', 'שבר', 'את', 'הראש', 'על', 'הבעיה', '.']
# iob_tags: ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O', 'O', 'O']
```

**WRONG Approach (DO NOT USE):**
```python
# ❌ WRONG: Runtime tokenization causes misalignment
tokens = sentence.split()  # Attaches punctuation to words!
# Result: ['הוא', 'שבר', 'את', 'הבעיה.']  # Wrong! Only 4 tokens
```

**Label Scheme:**
- `O`: Outside idiom
- `B-IDIOM`: Beginning of idiom
- `I-IDIOM`: Inside idiom (continuation)

**Evaluation Metrics:**
- **Span-level F1** (PRIMARY): Exact idiom span match (start and end must be correct)
- Token-level F1 (macro): Per-token accuracy
- Precision & Recall (per class: O, B-IDIOM, I-IDIOM)
- Boundary accuracy

**Baseline Performance:**
- Random: 33% (3 classes)
- Always-O: varies by dataset
- Expected fine-tuned: 80-90% (span-level F1)

**Implementation:**
```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=3,  # O, B-IDIOM, I-IDIOM
    id2label={0: "O", 1: "B-IDIOM", 2: "I-IDIOM"},
    label2id={"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
)

# Input: Tokenizer processes pre-tokenized tokens with is_split_into_words=True
# Output: IOB2 predictions aligned to word-level tokens
```

#### 3.2.1 Subword Tokenization Alignment (CRITICAL Implementation Detail)

**Challenge:** Transformer tokenizers (especially mBERT, XLM-RoBERTa) split Hebrew words into subwords, but our IOB2 tags are aligned with word-level tokens from the `tokens` column.

**Example:**
```python
# Dataset provides (word-level, punctuation-separated):
tokens:   ['הוא', 'שבר', 'את', 'הראש', 'על', 'הבעיה', '.']
iob_tags: ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O', 'O', 'O']

# After mBERT tokenization (subwords):
# Tokenizer may split Hebrew words into multiple subword pieces
# Need to align 7 word-level tags → N subword tokens
```

**Solution Strategy:**

1. **Tokenize with `is_split_into_words=True`:**
   ```python
   import ast
   from transformers import AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

   # Parse pre-tokenized tokens
   tokens = ast.literal_eval(df['tokens'][i])

   # Tokenize with word boundary tracking
   tokenized = tokenizer(
       tokens,
       truncation=True,
       max_length=128,
       is_split_into_words=True  # CRITICAL: preserves word boundaries
   )
   ```

2. **Align IOB2 labels to subwords:**
   - Use `tokenized.word_ids()` to track which subword belongs to which word
   - First subword of each word gets the word's IOB2 tag
   - Subsequent subwords get label `-100` (ignored in loss)
   - Special tokens ([CLS], [SEP], [PAD]) get label `-100`

3. **Implementation:**
   ```python
   from src.utils.tokenization import align_labels_with_tokens

   # Parse word-level IOB tags
   word_labels = ast.literal_eval(df['iob_tags'][i])

   # Align to subword tokens
   label2id = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
   aligned_labels = align_labels_with_tokens(
       tokenized,
       word_labels,
       label2id,
       label_all_tokens=False  # Only first subword gets label
   )
   ```

4. **During Evaluation:**
   - Model outputs predictions for each subword
   - Convert back to word-level using `word_ids()`
   - Take first subword's prediction as word's prediction
   - Compute metrics on word-level predictions vs ground truth

**Code Location:**
- Alignment function: `src/utils/tokenization.py`
- Test script: `src/test_tokenization_alignment.py`
- Integration: `src/idiom_experiment.py` (Task 2 training)

**Validation Required:**
- Run `python src/test_tokenization_alignment.py` before training
- Verify alignment preserves span boundaries
- Check special tokens have label -100
- Test on all 5 model tokenizers

**Note:** This alignment is MANDATORY for Task 2. Without it, the model will learn incorrect label-token associations and produce meaningless results.

---

## 4. Model Specifications

### 4.1 Encoder Models (Fine-Tuning)

**Models to Evaluate (5 total):**

| Model | HuggingFace ID | Params | Vocab Size | Type | Priority |
|-------|----------------|--------|------------|------|----------|
| **AlephBERT-base** | `onlplab/alephbert-base` | 110M | 52K | Hebrew | ⭐⭐⭐ |
| **AlephBERTGimmel-base** | `dicta-il/alephbertgimmel-base` | 110M | 128K | Hebrew | ⭐⭐⭐ |
| **DictaBERT** | `dicta-il/dictabert` | 110M | 50K | Hebrew | ⭐⭐⭐ |
| **mBERT** | `bert-base-multilingual-cased` | 110M | 119K | Multilingual | ⭐⭐⭐ |
| **XLM-RoBERTa-base** | `xlm-roberta-base` | 125M | 250K | Multilingual | ⭐⭐⭐ |

**Note:** Currently using `imvladikon/alephbertgimmel-base-512` in code. **Recommendation:** Switch to official `dicta-il/alephbertgimmel-base` for consistency.

**Selection Rationale:**
- **Hebrew-specific models (3):** AlephBERT, AlephBERTGimmel, DictaBERT
  - AlephBERT-base: Original Hebrew BERT (52K vocab)
  - AlephBERTGimmel: Enhanced Hebrew BERT with **128K vocabulary** (SOTA on Hebrew benchmarks)
  - DictaBERT: Dicta's Hebrew BERT (50K vocab)
- **Multilingual models (2):** mBERT, XLM-RoBERTa
  - For cross-lingual transfer comparison
- **Similar sizes:** 110-125M params for fair comparison
- **Different vocabulary strategies:** Compare small (50K) vs large (128K) Hebrew vocabularies

**Tokenization Characteristics:**
- **AlephBERT:** Hebrew WordPiece, 52K vocab
- **AlephBERTGimmel:** Hebrew WordPiece, **128K vocab** (fewer splits, better morphology handling)
- **DictaBERT:** Hebrew WordPiece, 50K vocab
- **mBERT:** Multilingual WordPiece, 119K vocab (aggressive Hebrew splitting)
- **XLM-RoBERTa:** SentencePiece, 250K vocab (different segmentation strategy)

**Research Question:** Does larger Hebrew vocabulary (AlephBERTGimmel 128K) outperform smaller vocabularies (AlephBERT 52K, DictaBERT 50K) for idiom detection?

### 4.2 LLM Models (Prompting Evaluation)

**Primary LLMs for Prompting (2 models):**

| Model | HuggingFace ID | Params | Type | API/Local | Priority |
|-------|----------------|--------|------|-----------|----------|
| **DictaLM-3.0-1.7B-Instruct** | `dicta-il/DictaLM-3.0-1.7B-Instruct` | 1.7B | Hebrew-native | Local | ⭐⭐⭐ |
| **DictaLM-3.0-1.7B-Instruct-W4A16** | `dicta-il/DictaLM-3.0-1.7B-Instruct-W4A16` | 1.7B (quantized) | Hebrew-native | Local | ⭐⭐⭐ |
| **Llama-3.1-8B-Instruct** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Multilingual | Local/API | ⭐⭐⭐ |

**Optional Additional Models:**
- Llama-3.1-70B (via API - better accuracy, higher cost)
- DictaLM-3.0-12B-Nemotron-Instruct (larger Hebrew model)

**Selection Rationale:**
- **DictaLM-3.0:** Hebrew-native LLM (SOTA 2025)
  - Matches Hebrew-specific encoder approach (AlephBERT/DictaBERT)
  - Open-weight (free, no API costs)
  - Runs locally (1.7B fits on Mac/24GB GPU)
  - Quantized version (W4A16) for faster inference
- **Llama-3.1-8B:** Multilingual baseline
  - Compare Hebrew-native vs multilingual LLM
  - Strong multilingual capabilities
  - Open-weight, runs locally

**Research Questions:**
1. **Hebrew-native vs Multilingual:** Does DictaLM-3.0 (Hebrew) outperform Llama (multilingual)?
2. **Architecture Comparison:** Do encoders (fine-tuned) or decoders (prompted) perform better?
3. **Cost-Performance:** Fine-tuning vs prompting tradeoff for Hebrew idioms?

**Evaluation Approach:**
- **Zero-shot prompting** (no examples)
- **Few-shot prompting** (3-5 examples)
- **Two-method comparison:**
  - Method 1: Direct classification prompt
  - Method 2: Chain-of-thought reasoning
- **Structured output:** JSON format for consistent parsing

---

## 5. Training Methodology

### 5.1 Training Configuration

**Optimizer:** AdamW
**Scheduler:** Linear decay with warmup
**Early Stopping:** Patience = 3 epochs (configurable)
**Metric for Best Model:** Validation F1 (macro)

**Hyperparameter Search Space:**

| Parameter | Range | Default |
|-----------|-------|---------|
| Learning rate | [1e-5, 2e-5, 3e-5, 5e-5] | 2e-5 |
| Batch size | [8, 16, 32] | 16 |
| Epochs | [3, 5, 8] | 5 |
| Warmup ratio | [0.0, 0.1, 0.2] | 0.1 |
| Weight decay | [0.0, 0.01, 0.05] | 0.01 |
| Gradient accumulation | [1, 2, 4] | 1 |
| Max sequence length | 128 | 128 (fixed) |
| FP16 | false | false (CPU/MPS), true (CUDA optional) |
| Seed | 42 | 42 (or 42, 123, 456 for multi-seed) |

**Hyperparameter Optimization:**
- Method: Optuna (10-15 trials per model-task combination)
- Objective: Maximize validation F1
- Sampler: TPESampler (Bayesian optimization)
- Pruning: MedianPruner (optional, for faster optimization)
- Storage: SQLite database (`experiments/results/optuna_studies/{model}_{task}_hpo.db`)

**Configuration Files:**
- Base training: `experiments/configs/training_config.yaml`
- HPO search space: `experiments/configs/hpo_config.yaml`

### 5.2 Training Modes

**Mode 1: Zero-Shot Evaluation**
- No training
- Direct evaluation on test set
- Baseline for comparison
- Output: `experiments/results/zero_shot/{model}_{split}_{task}.json`

**Mode 2: Full Fine-Tuning** ⭐ **PRIMARY**
- Train all model parameters
- Best expected performance
- ~20-30 minutes per model on GPU
- Output: `experiments/results/full_finetune/{model}/{task}/`

**Mode 3: Frozen Backbone**
- Train only classification head
- Fast baseline (~5-10 minutes)
- Use for comparison only
- Output: `experiments/results/frozen_backbone/{model}/{task}/`

**Mode 4: Hyperparameter Optimization**
- Optuna-based search (15 trials)
- Each trial = full training run
- ~5-8 hours per model-task combination
- Output:
  - Trials: `experiments/hpo_results/{model}/{task}/trial_{n}/`
  - Study: `experiments/results/optuna_studies/{model}_{task}_hpo.db`
  - Best params: `experiments/results/best_hyperparameters/best_params_{model}_{task}.json`

**CLI Usage:**
```bash
# Zero-shot
python src/idiom_experiment.py --mode zero_shot --model_id onlplab/alephbert-base --task cls

# Full fine-tuning
python src/idiom_experiment.py --mode full_finetune --model_id onlplab/alephbert-base --task cls --config experiments/configs/training_config.yaml

# Frozen backbone
python src/idiom_experiment.py --mode frozen_backbone --model_id onlplab/alephbert-base --task cls --config experiments/configs/training_config.yaml

# HPO
python src/idiom_experiment.py --mode hpo --model_id onlplab/alephbert-base --task cls --config experiments/configs/hpo_config.yaml
```

### 5.3 Cross-Seed Validation (Optional)

**Seeds:** 42, 123, 456 (minimum 3)
**Reporting:** Mean ± Standard Deviation
**Statistical Testing:** Paired t-test (α = 0.05)

---

## 6. LLM Prompting Strategy & Evaluation

### 6.1 Model Comparison Design

**Two-Model Comparison:**
1. **DictaLM-3.0-1.7B-Instruct** (Hebrew-native)
2. **Llama-3.1-8B-Instruct** (Multilingual)

**Research Questions:**
- Does Hebrew-native LLM outperform multilingual LLM?
- How much does model size matter (1.7B vs 8B)?
- Which prompting method works best for Hebrew idioms?

### 6.2 Two-Method Comparison

#### **Method 1: Direct Classification Prompt**

**Zero-Shot (No Examples):**
```
Analyze the following Hebrew sentence and determine if the expression
"{base_pie}" is used literally or figuratively.

Sentence: {sentence}
Expression: {base_pie}

Instructions:
- Literal: The expression is used in its physical/actual meaning
- Figurative: The expression is used in its idiomatic/metaphorical meaning

Answer with JSON:
{
  "classification": "literal" or "figurative",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation in Hebrew or English"
}
```

**Few-Shot (3-5 Examples):**
```
Examples of literal vs figurative usage in Hebrew:

Example 1 (Literal):
Sentence: "הוא שבר את הכוס בטעות בזמן הארוחה"
Expression: "שבר את הכוס"
Classification: literal
Reasoning: Actually broke a physical cup

Example 2 (Figurative):
Sentence: "הוא שבר את הראש על הבעיה כל הלילה"
Expression: "שבר את הראש"
Classification: figurative
Reasoning: Thinking hard, not actual head breaking

Example 3 (Figurative):
Sentence: "היא שברה שיא עולמי בריצת 100 מטר"
Expression: "שברה שיא"
Classification: figurative
Reasoning: Breaking a record, not physical breaking

Now analyze:
Sentence: {sentence}
Expression: {base_pie}

[Same JSON output format]
```

#### **Few-Shot Example Selection Methodology**

**Critical for Reproducibility and Publication Quality:**

To ensure rigorous evaluation and avoid data leakage:

1. **Selection Pool:** Training set ONLY (never use test set examples)
2. **Strategy:** Stratified random sampling with fixed seed (42)
3. **N Examples:** 3-5 examples per prompt
4. **Selection Criteria:**
   - Balanced representation (literal and figurative)
   - Different idioms (no duplicates)
   - Medium sentence length (10-20 tokens)
   - Mix of sentence types (question, declarative)
5. **Documentation:**
   - Save selected example IDs to `experiments/configs/few_shot_examples.json`
   - Document exact sentences used in paper methodology
   - Ensure reproducibility with fixed random seed
6. **Validation:** Verify no overlap between few-shot examples and test set

**Rationale:** Many research papers get criticized for poor LLM evaluation methodology. Reviewers will ask:
- Which exact examples were used?
- How were they selected?
- Are they from the test set? (data leakage!)
- Can others reproduce the results?

**Implementation:** See STEP_BY_STEP_MISSIONS.md Mission 5.2.1 for detailed code examples and selection strategies.

#### **Method 2: Chain-of-Thought Reasoning**

**Zero-Shot with CoT:**
```
Analyze the following Hebrew sentence step by step to determine if the
expression "{base_pie}" is used literally or figuratively.

Sentence: {sentence}
Expression: {base_pie}

Think step by step:
1. First, identify the expression in the sentence
2. Look at the surrounding context and other words
3. Consider: Is this describing a physical/actual event or a metaphorical concept?
4. Determine if the expression is used literally (actual meaning) or figuratively (idiomatic meaning)
5. Provide your final classification

Answer with JSON:
{
  "step_by_step_analysis": "your reasoning process",
  "classification": "literal" or "figurative",
  "confidence": 0.0-1.0
}
```

**Few-Shot with CoT:**
```
[Same examples as Method 1, but with step-by-step reasoning shown]

Example 1 Analysis:
1. Expression: "שבר את הכוס"
2. Context: "בטעות בזמן הארוחה" (by mistake during meal)
3. Physical event: Breaking a cup during meal
4. Classification: literal (actual cup breaking)

[Continue with examples 2-3]

Now analyze:
Sentence: {sentence}
Expression: {base_pie}

[Same JSON output format with step-by-step reasoning]
```

### 6.3 Experimental Design

**Comparison Matrix:**

| Model | Prompting Type | Method | Examples | Total Runs |
|-------|----------------|--------|----------|------------|
| DictaLM-3.0 | Zero-shot | Direct | 0 | 1 |
| DictaLM-3.0 | Zero-shot | CoT | 0 | 1 |
| DictaLM-3.0 | Few-shot | Direct | 3 | 1 |
| DictaLM-3.0 | Few-shot | CoT | 3 | 1 |
| Llama-3.1-8B | Zero-shot | Direct | 0 | 1 |
| Llama-3.1-8B | Zero-shot | CoT | 0 | 1 |
| Llama-3.1-8B | Few-shot | Direct | 3 | 1 |
| Llama-3.1-8B | Few-shot | CoT | 3 | 1 |

**Total:** 8 experimental conditions

**Research Questions:**
1. **Model Comparison:** DictaLM-3.0 (Hebrew) vs Llama (multilingual)
2. **Method Comparison:** Direct prompt vs Chain-of-Thought
3. **Example Impact:** Zero-shot vs Few-shot improvement
4. **Interaction:** Which model benefits more from CoT? Which benefits more from examples?

### 6.4 Evaluation Protocol

**Datasets:**
- In-domain test: 432 samples (seen idioms, new contexts)
- Unseen idiom test: 480 samples (zero-shot transfer to new idioms)

**Metrics:**
- **Primary:** F1-score (macro)
- Accuracy
- Precision & Recall (per class: literal, figurative)
- Confusion matrix
- **Cost:** Inference time (seconds per sample), total cost (for API models)
- **Analysis:** Error patterns by idiom type

**Manual Review (Quality Assessment):**
- Randomly sample 50 predictions per model
- Evaluate reasoning quality (1-5 scale):
  - 5: Excellent reasoning, correct prediction
  - 4: Good reasoning, correct prediction
  - 3: Adequate reasoning, correct prediction
  - 2: Poor reasoning, but correct prediction
  - 1: Incorrect prediction
- Document patterns:
  - Does model identify literal/figurative cues correctly?
  - Does model use context effectively?
  - What error types are most common?
- Save analysis to: `experiments/results/llm_evaluation/manual_review_analysis.md`

**Implementation:**
```python
# Example code structure
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_llm_prompting(
    model_id,           # "dicta-il/DictaLM-3.0-1.7B-Instruct" or "meta-llama/Llama-3.1-8B-Instruct"
    method,             # "direct" or "cot"
    num_examples,       # 0 (zero-shot) or 3 (few-shot)
    test_file           # "data/splits/test.csv" or "data/splits/unseen_idiom_test.csv"
):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load test data
    test_df = pd.read_csv(test_file)

    # Prepare prompts
    prompts = [
        create_prompt(row, method=method, num_examples=num_examples)
        for _, row in test_df.iterrows()
    ]

    # Generate predictions
    predictions = []
    for prompt in prompts:
        output = generate_response(model, tokenizer, prompt)
        pred = parse_json_output(output)
        predictions.append(pred)

    # Compute metrics
    results = compute_metrics(test_df['label'], predictions)

    return results
```

**Output:**
- Results saved to: `experiments/results/llm_prompting/{model}/{method}/{shot}/`
- Format: JSON with all metrics + predictions CSV

### 6.5 Comparison with Fine-Tuned Encoders

**Final Analysis:**

| Approach | Model Type | Training | Size | Expected F1 | Cost |
|----------|------------|----------|------|-------------|------|
| **Encoder Fine-Tuning** | BERT-style | Full fine-tune | 110M | 85-92% | ~30 min GPU |
| **LLM Prompting (Hebrew)** | DictaLM-3.0 | None | 1.7B | 75-85% | ~5 min CPU |
| **LLM Prompting (Multi)** | Llama-3.1 | None | 8B | 70-80% | ~10 min GPU |

**Research Contributions:**
1. First evaluation of DictaLM-3.0 on idiom detection
2. Hebrew-native LLM vs encoder comparison
3. Prompting method analysis (Direct vs CoT)
4. Cost-performance tradeoff insights
5. Zero-shot transfer to unseen idioms

**Practical Insights:**
- When to fine-tune vs prompt?
- Is Hebrew-native LLM worth the smaller size?
- Do examples help more than reasoning?

---

## 7. Compute Infrastructure & Development Environment

### 7.1 Development Environment

**IDE:** PyCharm Professional/Community Edition

**Local Development:**
- Code development and testing in PyCharm
- Version control (Git) integration
- Project structure and file management
- All source code maintained in PyCharm project
- Testing on small subsets (CPU/MPS)

**Local Testing Commands:**
```bash
# Test training locally on small subset
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --max_samples 100 \
  --num_epochs 1 \
  --device cpu
```

**Note:** All code must be compatible with remote execution on VAST.ai servers

### 7.2 Compute Infrastructure (VAST.ai)

**Primary Training Platform:** VAST.ai GPU Rental

**Recommended Instance:**
- GPU: RTX 3090 or RTX 4090
- VRAM: ≥ 24GB
- RAM: ≥ 32GB
- Reliability: > 98%
- Cost: ~$0.30-0.50/hour

**Setup Process:**
1. Rent instance via VAST.ai web interface
2. Connect via SSH
3. Run setup script: `bash scripts/setup_vast_instance.sh`
4. Download dataset: `gdown {google_drive_file_id}`
5. Install dependencies: `pip install -r requirements.txt`

**Running Experiments on VAST.ai:**
```bash
# Use screen/tmux to keep running if SSH disconnects
screen -S training

# Run full training
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --device cuda

# Detach: Ctrl+A then D
# Reattach later: screen -r training
```

**Results Management:**
- All results automatically saved to hierarchical folder structure
- Download via SCP: `scp -P {port} root@{ip}:~/project/experiments/results/* ./local/`
- Optional: Automated Google Drive sync with rclone

### 7.3 Docker Support (Optional)

**Docker Image:** `Dockerfile` for reproducible environments

**Usage:**
```bash
docker build -t hebrew-idiom-detection .
docker run --gpus all -v $(pwd)/experiments:/workspace/experiments hebrew-idiom-detection
```

---

## 8. Project Timeline

**Total Duration:** 12 weeks

**Week 1-2: Dataset Preparation** ✅ COMPLETED
- Data validation and preprocessing
- Split creation
- Statistical analysis
- Quality assurance

**Week 3: Zero-Shot Baseline** ✅ COMPLETED
- Model downloads
- Zero-shot evaluation framework
- Baseline results for all models

**Week 4-6: Full Fine-Tuning** 🔄 IN PROGRESS
- Training configuration setup
- Training pipeline implementation
- IOB2 tokenization alignment
- Local testing
- VAST.ai setup
- Full training runs (5 models × 2 tasks = 10 runs)

**Week 7: Hyperparameter Optimization** ⏳ PLANNED
- Optuna integration
- HPO studies (10 model-task combinations)
- Best hyperparameter identification
- Final training with optimal params

**Week 8: LLM Evaluation** ⏳ OPTIONAL
- Prompting strategy design
- Zero-shot evaluation
- Few-shot evaluation
- Cost-performance analysis

**Week 9: Ablation Studies** ⏳ PLANNED
- Training paradigm comparison
- Frozen vs. full fine-tuning
- Hyperparameter sensitivity
- Data size impact

**Week 10: Analysis** ⏳ PLANNED
- Error analysis
- Statistical tests
- Visualization
- Cross-model comparison

**Week 11-12: Writing & Finalization** ⏳ PLANNED
- Draft paper/thesis
- Results tables and figures
- Discussion section
- Revisions and final submission

---

## 9. Evaluation Protocol

### 9.1 Metrics Suite

**Classification Metrics (Task 1):**
- Accuracy
- Macro F1-score (PRIMARY)
- Precision (per class: literal, figurative)
- Recall (per class: literal, figurative)
- Confusion Matrix (TP, FP, TN, FN)
- ROC-AUC

**Token Classification Metrics (Task 2):**
- **Span-level F1** (PRIMARY): Exact idiom span match
- Token-level F1 (macro)
- Precision/Recall (per class: O, B-IDIOM, I-IDIOM)
- Boundary accuracy

**LLM-Specific Metrics (Optional):**
- Cost per prediction (USD)
- Latency (seconds)
- Reasoning quality score (1-5)

**Saved Metrics (All Experiments):**
All metrics saved to `training_results.json`:
```json
{
  "test_metrics": {
    "accuracy": 0.8765,
    "f1": 0.8723,
    "precision": 0.8654,
    "recall": 0.8792,
    "confusion_matrix": {
      "true_positives": 189,
      "false_positives": 27,
      "true_negatives": 194,
      "false_negatives": 22
    }
  },
  "training_history": [
    {
      "epoch": 1.0,
      "loss": 0.5234,
      "eval_f1": 0.7234,
      "eval_accuracy": 0.7345
    }
  ]
}
```

### 9.2 Statistical Testing

**Method:** Paired t-test for cross-seed comparisons
**Significance Level:** α = 0.05
**Multiple Comparisons:** Bonferroni correction

**Reporting Format:**
```
Model A: 89.2 ± 1.3% F1
Model B: 85.7 ± 1.8% F1
Difference: 3.5% (p = 0.018 < 0.05) ✓ Significant
```

### 9.3 Error Analysis

**Categories:**
1. False Positives (Literal → Figurative)
2. False Negatives (Figurative → Literal)
3. High-confidence errors (>0.8 probability)
4. Boundary errors (B/I confusion for Task 2)

**Analysis Dimensions:**
- By idiom type
- By sentence length
- By idiom position
- By context complexity
- By idiom frequency

---

## 10. Deliverables

### 10.1 Code Repository

**Development Environment:** PyCharm IDE (Professional/Community Edition)

**Project Structure:**
```
hebrew-idiom-detection/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── FINAL_PRD_Hebrew_Idiom_Detection.md        # This document
├── MISSIONS_PROGRESS_TRACKER.md               # Progress tracking
├── STEP_BY_STEP_MISSIONS.md                   # Implementation guide
├── TRAINING_OUTPUT_ORGANIZATION.md            # Folder structure guide
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── expressions_data_tagged_v2.csv         # Main dataset (v2.0)
│   ├── expressions_data_tagged_v2.xlsx        # Excel format
│   ├── expressions_data_with_splits.csv       # With split column
│   ├── processed_data.csv                     # Processed version
│   ├── README.md                              # Dataset documentation
│   └── splits/
│       ├── train.csv                          # 3,456 samples
│       ├── validation.csv                     # 432 samples
│       ├── test.csv                           # 432 samples (in-domain)
│       ├── unseen_idiom_test.csv              # 480 samples (zero-shot)
│       └── split_expressions.json             # Split metadata
├── src/
│   ├── __init__.py
│   ├── data_preparation.py                    # Dataset loading and analysis
│   ├── dataset_splitting.py                   # Hybrid split implementation
│   ├── idiom_experiment.py                    # Main training script (all modes)
│   ├── test_tokenization_alignment.py         # IOB2 alignment test
│   ├── llm_evaluation.py                      # LLM prompting (optional)
│   ├── visualization.py                       # Results visualization
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                         # Evaluation metrics
│       ├── tokenization.py                    # IOB2 subword alignment (CRITICAL)
│       ├── storage.py                         # Google Drive sync
│       └── vast_utils.py                      # VAST.ai utilities
├── experiments/
│   ├── configs/
│   │   ├── training_config.yaml               # Base training config
│   │   └── hpo_config.yaml                    # Optuna HPO config
│   ├── results/
│   │   ├── zero_shot/                         # Zero-shot results
│   │   ├── full_finetune/                     # Full fine-tuning results
│   │   │   └── {model}/{task}/
│   │   │       ├── checkpoint-*/
│   │   │       ├── logs/                      # TensorBoard
│   │   │       ├── training_results.json
│   │   │       └── summary.txt
│   │   ├── frozen_backbone/                   # Frozen backbone results
│   │   ├── optuna_studies/                    # Optuna databases
│   │   └── best_hyperparameters/              # Best HPO params
│   ├── hpo_results/                           # HPO trial outputs
│   │   └── {model}/{task}/trial_{n}/
│   └── logs/                                  # Training logs
├── models/                                     # Cached models
├── notebooks/
│   ├── 01_data_validation.ipynb               # Data quality checks
│   ├── 02_model_training.ipynb                # Training experiments
│   ├── 03_llm_evaluation.ipynb                # LLM prompting
│   └── 04_error_analysis.ipynb                # Error categorization
├── scripts/
│   ├── setup_vast_instance.sh                 # VAST.ai automation
│   ├── sync_to_gdrive.sh                      # Google Drive upload
│   ├── download_from_gdrive.sh                # Dataset download
│   ├── run_all_hpo.sh                         # Batch HPO (10 studies)
│   └── run_all_experiments.sh                 # Batch training (30 runs)
├── tests/
│   ├── test_data_preparation.py
│   ├── test_metrics.py
│   └── test_tokenization.py
├── professor_review/                          # QA package
│   ├── README.md                              # Comprehensive analysis
│   ├── Complete_Dataset_Analysis.ipynb        # Full QA notebook
│   └── data/
│       ├── expressions_data_tagged_v2.csv
│       └── splits/
└── paper/
    ├── figures/                               # 17 visualizations
    ├── tables/
    └── main.tex
```

**Key Files:**
- `src/idiom_experiment.py`: Main script supporting all 4 modes
- `src/utils/tokenization.py`: IOB2 subword alignment (CRITICAL for Task 2)
- `src/test_tokenization_alignment.py`: Alignment validation
- `experiments/configs/training_config.yaml`: Base configuration
- `experiments/configs/hpo_config.yaml`: HPO search space

**PyCharm Configuration:**
- Python interpreter: Virtual environment (Python 3.9+)
- Code style: PEP 8
- Git integration enabled
- Requirements: `torch`, `transformers`, `datasets`, `optuna`, `pandas`, `numpy`, `scikit-learn`, `tensorboard`

### 10.2 Dataset Release

**Package Contents:**
- CSV/Excel/JSON files with all annotations
- Data schema documentation (README.md)
- Split files (train/val/test/unseen_idiom_test)
- Statistics report
- Complete QA notebook

**Distribution:**
- GitHub repository: https://github.com/igornazarenko434/hebrew-idiom-detection
- HuggingFace Datasets (planned)
- DOI via Zenodo (planned)

### 10.3 Academic Paper

**Target Venue:** ACL/EMNLP/NAACL
**Format:** 8 pages + references
**Submission Deadline:** [Target conference deadline]

**Structure:**
1. Abstract (250 words)
2. Introduction (1.5 pages)
3. Dataset (2 pages)
4. Methodology (2 pages)
5. Results (2 pages)
6. Analysis (1.5 pages)
7. Conclusion (0.5 pages)

### 10.4 Thesis Document

**Length:** 60-80 pages
**Language:** Hebrew with English abstract
**Chapters:** 8-10
**Defense Date:** [Target date]

---

## 11. Risk Management

### 11.1 Technical Risks

| Risk | Probability | Mitigation |
|------|------------|------------|
| VAST.ai compute unavailable | Low | Book instances in advance, use multiple providers |
| Models don't converge | Low | Use proven architectures, careful HPO |
| LLM API rate limits | Medium | Batch requests, implement retry logic |
| Data quality issues | Very Low | ✅ Already validated (9.2/10 quality score) |
| IOB2 misalignment | Medium | ✅ Mitigated with test_tokenization_alignment.py |

### 11.2 Schedule Risks

| Risk | Probability | Mitigation |
|------|------------|------------|
| HPO takes too long | Medium | Start with 3-5 trials, expand if time permits |
| VAST.ai costs exceed budget | Medium | Monitor costs daily, use Spot instances |
| Paper deadline missed | Low | Start writing Week 10, buffer time built in |

### 11.3 Quality Risks

| Risk | Probability | Mitigation |
|------|------------|------------|
| Low model performance | Low | Multiple baselines, strong pre-trained models |
| Overfitting to training data | Medium | Cross-validation, unseen test set |
| Statistical insignificance | Low | Multi-seed runs, large test sets |

---

## 12. Success Metrics

### 12.1 Technical Success

**Minimum Requirements:**
- ✅ Dataset quality: Cohen's κ > 0.8 (achieved: 0.9725)
- ⏳ Task 1 F1: > 85%
- ⏳ Task 2 F1: > 80%
- ⏳ All 5 models trained successfully
- ⏳ Results reproducible

**Stretch Goals:**
- Task 1 F1: > 90%
- Task 2 F1: > 85%
- LLM competitive with fine-tuned models
- Statistical significance across all comparisons

### 12.2 Research Success

**Publication:**
- Paper accepted to tier-1 venue (ACL/EMNLP)
- Dataset released on HuggingFace
- Code repository public and well-documented

**Impact:**
- First Hebrew idiom detection benchmark
- Contributions to low-resource NLP
- Future research citations

---

## 13. Appendix

### 13.1 Column Name Mapping (v1.0 → v2.0)

**OLD (INCORRECT) → NEW (CORRECT):**

| v1.0 (PRD mistake) | v2.0 (Actual implementation) |
|--------------------|------------------------------|
| `text` | `sentence` |
| `expression` | `base_pie` |
| `matched_expression` | `pie_span` |
| `label_2` | `label` (int, 0/1) |
| `label` (str) | `label_str` ("Literal"/"Figurative") |
| `iob2_tags` | `iob_tags` |
| `token_span_start` | `start_token` |
| `token_span_end` | `end_token` |
| `span_start` | `start_char` |
| `span_end` | `end_char` |

**CRITICAL NEW COLUMN:**
- `tokens`: Pre-tokenized sentence (punctuation separated)
- This is the AUTHORITATIVE tokenization - use this, not runtime tokenization!

### 13.2 Key Implementation Files

**Data Loading:**
- `src/data_preparation.py`: Main data loader
- `src/dataset_splitting.py`: Hybrid split implementation

**Training:**
- `src/idiom_experiment.py`: Lines 900-1500 (main training loop)
- `src/idiom_experiment.py`: Lines 1115-1180 (Task 2 tokenization - CRITICAL)

**Tokenization Alignment:**
- `src/utils/tokenization.py`: Lines 31-97 (`align_labels_with_tokens`)
- `src/test_tokenization_alignment.py`: Full alignment validation

**Configuration:**
- `experiments/configs/training_config.yaml`: Line 58 (TensorBoard enabled)
- `experiments/configs/hpo_config.yaml`: HPO search space

**Validation:**
- `professor_review/Complete_Dataset_Analysis.ipynb`: Full QA notebook
- `data/README.md`: Comprehensive dataset documentation

### 13.3 Critical Implementation Notes

1. **ALWAYS use pre-tokenized `tokens` column** for Task 2
2. **NEVER tokenize `sentence` at runtime** (causes misalignment)
3. **Use `ast.literal_eval()`** to parse tokens and IOB tags
4. **Use `is_split_into_words=True`** when tokenizing for Task 2
5. **Run alignment test** before training: `python src/test_tokenization_alignment.py`
6. **TensorBoard logging enabled** by default in training_config.yaml
7. **All results saved** to hierarchical folder structure

---

## Document Changelog

**Version 3.0 (December 5, 2025):**
- ✅ Updated all column names to match actual v2.0 implementation
- ✅ Updated statistics to actual values from dataset
- ✅ Added critical pre-tokenized data usage notes
- ✅ Clarified IOB2 alignment requirements
- ✅ Added folder structure documentation reference
- ✅ Added column mapping table (v1.0 → v2.0)
- ✅ Added critical implementation file references
- ✅ Verified all code references match actual implementation

**Version 2.1 (November 7, 2025):**
- Added VAST.ai infrastructure section
- Added PyCharm development environment
- Added Google Drive sync workflow

**Version 2.0 (Initial):**
- Initial PRD with dataset specification
- Model and training methodology defined
- Evaluation protocol established

---

**Contact:**
- Igor Nazarenko: igor.nazarenko@post.runi.ac.il
- Yuval Amit: yuval.amit@post.runi.ac.il

**Repository:** https://github.com/igornazarenko434/hebrew-idiom-detection

**Last Reviewed:** December 5, 2025
**Status:** ✅ ALIGNED WITH IMPLEMENTATION
