# Hebrew Idiom Detection Dataset

**Dataset Name:** Hebrew-Idioms-4800
**Version:** 1.0
**Language:** Hebrew (he)
**License:** [Specify License]
**Citation:** [Add citation when published]

---

## Overview

This directory contains the Hebrew Idiom Detection dataset - the first comprehensive Hebrew idiom dataset with dual-task annotations enabling both sentence-level classification and token-level span identification.

**Total Size:** 4,800 manually annotated sentences
**Balance:** 2,400 literal + 2,400 figurative (perfect 50/50)
**Unique Idioms:** 60 expressions
**Annotation Quality:** Manually verified by native Hebrew speakers

---

## Directory Structure

```
data/
├── README.md                           # This file
├── expressions_data_tagged.csv         # Full dataset (4,800 sentences)
├── expressions_data_tagged.xlsx        # Full dataset (Excel format)
├── processed_data.csv                  # Preprocessed version (Mission 2.4)
├── expressions_data_with_splits.csv    # Dataset with split assignments
└── splits/                             # Train/Val/Test splits
    ├── train.csv                       # 3,840 samples (80%, 48 idioms)
    ├── validation.csv                  # 480 samples (10%, 6 idioms)
    ├── test.csv                        # 480 samples (10%, 6 idioms)
    └── split_expressions.json          # Metadata: which idioms in which split
```

---

## Dataset Files

### 1. `expressions_data_tagged.csv` (Main Dataset)

**Size:** 4,800 rows × 16 columns
**Format:** CSV with UTF-8 encoding
**Delimiter:** Comma (,)

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique identifier (0-4799) |
| `split` | str | Original split assignment |
| `text` | str | Full Hebrew sentence |
| `expression` | str | Normalized idiom form |
| `matched_expression` | str | Idiom as it appears in text |
| `language` | str | Language code ("he") |
| `source` | str | Data source |
| `label` | str | Hebrew label ("מילולי" or "פיגורטיבי") |
| `label_2` | int | Binary label (0=literal, 1=figurative) |
| `iob2_tags` | str | IOB2 tags (space-separated) |
| `num_tokens` | int | Total tokens in sentence |
| `token_span_start` | int | Token start position of idiom |
| `token_span_end` | int | Token end position of idiom |
| `span_start` | int | Character start position |
| `span_end` | int | Character end position |
| `char_mask` | str | Binary character-level mask |

**Example Row:**
```csv
42,train,"הוא שבר את הראש על הבעיה המורכבת","שבר את הראש","שבר את הראש",he,manual,פיגורטיבי,1,"O B-IDIOM I-IDIOM I-IDIOM O O O",7,1,4,4,16,"0000111111111110000000000000000"
```

### 2. `splits/` Directory (Expression-Based Splits)

**Splitting Strategy:** Expression-based (NOT random)
**Purpose:** Prevent data leakage - ensure test idioms never seen during training

#### `train.csv`
- **Samples:** 3,840 (80%)
- **Idioms:** 48 unique expressions
- **Label Balance:** 1,920 literal + 1,920 figurative (50/50)
- **Usage:** Model training

#### `validation.csv`
- **Samples:** 480 (10%)
- **Idioms:** 6 unique expressions (different from train/test)
- **Label Balance:** 240 literal + 240 figurative (50/50)
- **Usage:** Hyperparameter tuning, early stopping

#### `test.csv`
- **Samples:** 480 (10%)
- **Idioms:** 6 unique expressions (completely unseen during training)
- **Label Balance:** 240 literal + 240 figurative (50/50)
- **Usage:** Final evaluation, zero-shot generalization

#### `split_expressions.json`
**Metadata file listing which idioms belong to which split.**

Example:
```json
{
  "train_expressions": ["שבר שיא", "זרק כדור", ...],
  "dev_expressions": ["הלך לאיבוד", "נתן יד", ...],
  "test_expressions": ["שבר את הראש", "לב זהב", ...]
}
```

**Critical:** Zero overlap between splits ensures true zero-shot evaluation on test set.

---

## Dataset Statistics

### Size Distribution
- **Total Sentences:** 4,800
- **Train Set:** 3,840 (80%)
- **Validation Set:** 480 (10%)
- **Test Set:** 480 (10%)

### Label Distribution
- **Literal (מילולי):** 2,400 (50.0%)
- **Figurative (פיגורטיבי):** 2,400 (50.0%)

### Idiom Statistics
- **Unique Idioms:** 60 expressions
- **Avg Sentences per Idiom:** 80
- **Avg Idiom Length:** 2.39 tokens (median: 2)
- **Idiom Length Range:** 2-4 tokens

### Sentence Statistics
- **Avg Sentence Length:** 14.93 tokens (median: 10)
- **Sentence Length Range:** 3-89 tokens
- **Declarative Sentences:** 4,427 (92.23%)
- **Questions:** 342 (7.12%)
- **Exclamatory:** 31 (0.65%)

---

## Tasks Supported

### Task 1: Sentence Classification (Binary)
**Objective:** Classify whether the idiom is used literally (0) or figuratively (1)

**Input:** Hebrew sentence with idiom
**Output:** Binary label (0 or 1)

**Evaluation Metrics:**
- Accuracy
- Macro F1-score
- Precision & Recall (per class)
- Confusion Matrix
- ROC-AUC

**Baseline:**
- Random: 50%
- Majority class: 50%
- Expected fine-tuned: 85-92%

### Task 2: Token Classification (IOB2)
**Objective:** Identify the exact token span of the idiom using BIO tagging

**Input:** Tokenized Hebrew sentence
**Output:** IOB2 tag sequence

**Label Scheme:**
- `O`: Outside idiom (not part of idiom)
- `B-IDIOM`: Beginning of idiom span
- `I-IDIOM`: Inside idiom span (continuation)

**Example:**
```
Tokens: ["הוא", "שבר", "את", "הראש", "על", "הבעיה"]
Tags:   ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", "O", "O"]
```

**Evaluation Metrics:**
- Token-level F1 (macro)
- Span-level F1 (exact match)
- Precision & Recall (per class)
- Boundary accuracy

**Baseline:**
- Random: 33% (3 classes)
- Expected fine-tuned: 80-90%

---

## Data Quality

### Manual Validation
- ✅ All sentences verified by native Hebrew speakers
- ✅ IOB2 tags validated for 100% alignment with tokens
- ✅ No duplicate sentences
- ✅ All sentences grammatically correct
- ✅ Context ensures clear literal/figurative distinction

### Preprocessing (Mission 2.1-2.4)
- Unicode normalization (NFKC)
- BOM character removal
- Directional mark removal (LRM/RLM)
- Whitespace normalization
- IOB2 alignment verification

---

## Usage Examples

### Load Full Dataset

```python
import pandas as pd

# Load full dataset
df = pd.read_csv('data/expressions_data_tagged.csv')
print(f"Total samples: {len(df)}")
print(f"Literal: {(df['label_2'] == 0).sum()}")
print(f"Figurative: {(df['label_2'] == 1).sum()}")
```

### Load Splits

```python
import pandas as pd

# Load splits
train = pd.read_csv('data/splits/train.csv')
val = pd.read_csv('data/splits/validation.csv')
test = pd.read_csv('data/splits/test.csv')

print(f"Train: {len(train)} samples")
print(f"Validation: {len(val)} samples")
print(f"Test: {len(test)} samples")
```

### Access IOB2 Tags

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/splits/train.csv')

# Get sentence and tags
for idx, row in df.head(5).iterrows():
    text = row['text']
    tags = row['iob2_tags'].split()

    print(f"Sentence: {text}")
    print(f"IOB2 Tags: {tags}")
    print()
```

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{hebrew_idiom_detection_2025,
  title={Hebrew-Idioms-4800: A Comprehensive Dataset for Idiom Detection with Dual-Task Annotations},
  author={[Your Name]},
  year={2025},
  publisher={[Publisher]},
  url={[Dataset URL]}
}
```

---

## Download Sources

### From Google Drive

```bash
# Install gdown
pip install gdown

# Download CSV
gdown 140zJatqT4LBl7yG-afFSoUrYrisi9276 -O data/expressions_data_tagged.csv

# Download XLSX
gdown 1eKk7w1JDomMQ1zBYcD9iI-qF1pG1LCv_ -O data/expressions_data_tagged.xlsx --fuzzy
```

### From GitHub Repository

```bash
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
cd hebrew-idiom-detection/data/splits/
```

---

## Dataset Limitations

1. **Domain:** Limited to contemporary Hebrew (modern usage)
2. **Size:** 60 idioms (not exhaustive of all Hebrew idioms)
3. **Context:** Sentences may lack broader discourse context
4. **Annotation:** Single annotator per sentence (no inter-annotator agreement scores)
5. **Zero-Shot:** Test set covers only 6 idioms (10% of total)

---

## Future Extensions

- [ ] Add more idioms (target: 100+ expressions)
- [ ] Multi-annotator validation
- [ ] Extended context (discourse-level)
- [ ] Cross-lingual alignment (English translations)
- [ ] Difficulty ratings per idiom
- [ ] Semantic categories for idioms

---

## Contact

For questions, issues, or contributions:
- **GitHub Issues:** https://github.com/igornazarenko434/hebrew-idiom-detection/issues
- **Email:** [Your Email]

---

**Last Updated:** November 9, 2025
**Dataset Version:** 1.0
**Maintained By:** [Your Name]
