# Hebrew Idiom Detection Dataset
## Comprehensive Data Analysis Report

**Dataset:** Hebrew-Idioms-4800 v1.0
**Date:** November 19, 2025
**Purpose:** Master's Thesis - Hebrew Idiom Detection with Dual-Task Annotation

---

## Quick Start

1. **Read this README** for complete statistics, methodology, and findings
2. **Open the notebook** (`Complete_Dataset_Analysis.ipynb`) in any Jupyter-compatible environment to see all visualizations and validations
3. **Explore the data** in the `data/` folder (CSV files can be opened in Excel, Python, R, or any data tool)

---

## Executive Summary

### Dataset at a Glance

| Metric | Value |
|--------|-------|
| **Total Sentences** | 4,800 |
| **Unique Idioms** | 60 |
| **Samples per Idiom** | 80 (perfectly balanced) |
| **Label Balance** | 50% Literal / 50% Figurative |
| **Annotators** | 2 native Hebrew speakers |
| **Inter-Annotator Agreement** | **κ = 0.9725** (near-perfect) |
| **Data Quality Score** | **9.2/10** |

### Key Achievements

- **First Hebrew idiom detection dataset**
- **Dual-task annotation** (sentence classification + token-level span)
- **100% polysemy** (all idioms in both literal & figurative contexts)
- **Near-perfect IAA** (Cohen's κ = 0.9725)
- **High morphological richness** (45.25% prefix attachments)

---

## Data Dictionary

### Column Descriptions

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | int | Unique identifier for each sentence | 0, 1, ..., 4799 |
| `split` | str | Dataset split assignment | train, validation, test, unseen_idiom_test |
| `language` | str | Language code | he |
| `source` | str | Data source identifier | inhouse, manual |
| `text` | str | Full Hebrew sentence (UTF-8) | הוא שבר את הראש על הבעיה |
| `expression` | str | Canonical/normalized idiom form | שבר את הראש |
| `matched_expression` | str | Idiom as it appears in text (with morphology) | שברתי את הראש |
| `span_start` | int | Character start position of idiom (0-indexed) | 4 |
| `span_end` | int | Character end position (exclusive) | 16 |
| `token_span_start` | int | Token start position (0-indexed) | 1 |
| `token_span_end` | int | Token end position (exclusive) | 4 |
| `num_tokens` | int | Total tokens in sentence | 7 |
| `label` | str | Hebrew label | מילולי, פיגורטיבי |
| `label_2` | int | Binary numeric label | 0 (literal), 1 (figurative) |
| `iob2_tags` | str | IOB2 sequence (space-separated) | O B-IDIOM I-IDIOM I-IDIOM O O O |
| `char_mask` | str | Binary character mask | 0000111111111110000000 |

### Data Format Notes

- **Encoding:** UTF-8 with BOM (utf-8-sig)
- **Span Convention:** Python-style half-open intervals `[start, end)`
  - `text[span_start:span_end]` extracts `matched_expression`
- **IOB2 Tags:** Space-separated, one tag per whitespace-tokenized token

### Example Row

```
id: 42
split: train
text: הוא שבר את הראש על הבעיה המורכבת
expression: שבר את הראש
matched_expression: שבר את הראש
span_start: 4
span_end: 16
token_span_start: 1
token_span_end: 4
num_tokens: 7
label: פיגורטיבי
label_2: 1
iob2_tags: O B-IDIOM I-IDIOM I-IDIOM O O O
char_mask: 0000111111111110000000000000000
```

---

## Tokenization & Subword Alignment

### Data Creation Tokenization

The dataset was created using **whitespace-based tokenization** (`text.split()`). This means:
- `num_tokens` = number of whitespace-separated tokens
- `iob2_tags` has exactly `num_tokens` tags
- `token_span_start` and `token_span_end` refer to whitespace token indices

### Model Training Considerations

When using this dataset with **multilingual transformer models** (mBERT, XLM-RoBERTa, AlephBERT, etc.):

1. **Sentence Classification:** No special handling needed - use the full text
2. **Token Classification (IOB2):** Requires subword alignment

#### Subword Alignment Strategy

Multilingual tokenizers produce subwords, not whitespace tokens. For token classification:

```python
# Example alignment strategy
def align_labels_with_tokens(text, iob_tags, tokenizer):
    words = text.split()
    encoding = tokenizer(text, return_offsets_mapping=True)

    aligned_labels = []
    word_idx = 0

    for idx, (start, end) in enumerate(encoding.offset_mapping):
        if start == end:  # Special token
            aligned_labels.append(-100)  # Ignore in loss
        elif start == 0 or text[start-1] == ' ':  # First subword
            aligned_labels.append(tag_to_id[iob_tags[word_idx]])
            if end >= len(text) or text[end] == ' ':
                word_idx += 1
        else:  # Continuation subword
            # Option 1: Copy label (I-IDIOM remains I-IDIOM)
            # Option 2: Use -100 to ignore
            aligned_labels.append(tag_to_id[iob_tags[word_idx-1]])

    return aligned_labels
```

This alignment ensures the model can correctly learn token-level classification despite different tokenization schemes.

---

## 1. Inter-Annotator Agreement (IAA)

### Agreement Metrics

| Metric | Value |
|--------|-------|
| Observed Agreement | 98.625% |
| Expected Agreement (Chance) | 50% |
| **Cohen's Kappa** | **0.9725** |

**Interpretation:** κ > 0.81 indicates "almost perfect agreement"

### Disagreement Analysis

- Total disagreements: **66 items (1.375%)**
- Literal → Figurative: 1 case
- Figurative → Literal: 65 cases

**Finding:** Annotators were more likely to initially label figurative uses as literal, suggesting literal readings are the "default" interpretation.

### Corrections

- Non-label corrections: 223 items (4.65%)
- Type: Text/span corrections (not label changes)

---

## 2. Dataset Statistics

### 2.1 Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Literal (מילולי) | 2,400 | 50.00% |
| Figurative (פיגורטיבי) | 2,400 | 50.00% |

**Perfect 50/50 balance - no class imbalance issues**

### 2.2 Sentence Length Statistics

**Tokens:**
- Mean: 15.71
- Median: 12
- Std: 8.01
- Range: 5-38

**Characters:**
- Mean: 83.04
- Median: 63
- Std: 42.55
- Range: 22-193

### 2.3 Idiom Length Statistics

**Tokens:**
- Mean: 2.48
- Median: 2
- Range: 2-5

**Characters:**
- Mean: 11.39
- Median: 11
- Range: 5-23

### 2.4 Sentence Types

| Type | Count | Percentage |
|------|-------|------------|
| Declarative | 4,549 | 94.77% |
| Questions | 239 | 4.98% |
| Exclamatory | 12 | 0.25% |

### 2.5 Idiom Position in Sentences

| Position | Count | Percentage |
|----------|-------|------------|
| Start (0-33%) | 3,058 | 63.71% |
| Middle (33-67%) | 1,429 | 29.77% |
| End (67-100%) | 313 | 6.52% |

- Mean position ratio: 0.2801

**By Label:**
- Literal: 63.13% start, 31.50% middle, 5.38% end
- Figurative: 64.29% start, 28.04% middle, 7.67% end

---

## 3. Linguistic Analysis

### 3.1 Polysemy Analysis

| Metric | Value |
|--------|-------|
| Total expressions | 60 |
| Polysemous (both contexts) | **60 (100%)** |
| Only literal | 0 |
| Only figurative | 0 |

**All 60 idioms appear in BOTH literal and figurative contexts**

### 3.2 Lexical Diversity

| Metric | Value |
|--------|-------|
| Vocabulary size | 18,784 unique words |
| Total tokens | 75,412 |
| Type-Token Ratio (TTR) | 0.2491 |
| Avg unique words per sentence | 15.38 |
| Function word ratio | 12.57% |

**Top 10 Most Frequent Words:**

1. את - 2,295 (3.04%)
2. לא - 1,289 (1.71%)
3. הוא - 1,105 (1.47%)
4. היא - 1,004 (1.33%)
5. על - 918 (1.22%)
6. של - 756 (1.00%)
7. אם - 623 (0.83%)
8. – - 559 (0.74%)
9. הם - 521 (0.69%)
10. אחרי - 518 (0.69%)

### 3.3 Lexical Richness

| Metric | Value |
|--------|-------|
| Hapax legomena | 11,921 (63.46%) |
| Dis legomena | 2,850 |
| Maas Index | 0.0110 |

**High hapax rate (63.46%) confirms genuine linguistic diversity, not template-based generation**

### 3.4 Structural Complexity

| Metric | Overall | Literal | Figurative |
|--------|---------|---------|------------|
| Mean subclause markers | 0.28 | 0.25 | 0.31 |
| Mean subclause ratio | 0.0146 | 0.0122 | 0.0170 |
| Mean punctuation | 1.81 | 1.75 | 1.87 |
| Sentences with subclauses | 24.52% | - | - |

**Finding:** Figurative sentences are **24% more complex** than literal sentences

### 3.5 Morphological Richness (Hebrew-Specific)

| Metric | Value |
|--------|-------|
| Prefix attachments | 2,172 (45.25%) |
| Mean consistency rate | 39.54% |

**Top 5 Idioms by Morphological Variance:**

1. שם רגליים - 35 variants
2. שבר את הלב - 32 variants
3. פתח דלתות - 29 variants
4. סגר חשבון - 28 variants
5. הוריד פרופיל - 23 variants

### 3.6 Collocational Analysis

**Context Words (±3 tokens around idiom):**
- Total context words: 23,366
- Unique context words: 8,498

**Top 10 Context Words:**

1. הוא - 844 (3.61%)
2. היא - 745 (3.19%)
3. לא - 493 (2.11%)
4. הם - 423 (1.81%)
5. על - 362 (1.55%)
6. את - 328 (1.40%)
7. עם - 256 (1.10%)
8. של - 242 (1.04%)
9. כדי - 241 (1.03%)
10. אחרי - 234 (1.00%)

---

## 4. Data Quality Validation

### 4.1 Critical Validations (14/14 PASSED)

| Check | Result |
|-------|--------|
| Missing values | 0/76,800 cells (0%) |
| Duplicate rows | 0/4,800 (0%) |
| ID sequence | Complete (0-4799) |
| Label consistency | 100% |
| IOB2 alignment | 100% |
| Character spans | 100% accurate |
| Token spans | 100% valid |
| Encoding issues | 0 |
| Hebrew text validation | 100% |
| Length consistency | 100% |
| Expression presence | 100% |
| Data types | All correct |
| Unique values | Valid |

### 4.2 Minor Issues (Acceptable)

- Trailing whitespace: 3.35% (handled by tokenizers)
- Multiple consecutive spaces: 3.42%

**Overall Quality Score: 9.2/10**

---

## 4B. Preprocessing & Data Cleaning

### Preprocessing Steps Applied

1. **Unicode Normalization (NFKC)**
   - Normalizes all Hebrew characters to consistent form
   - Handles combining characters properly

2. **BOM Character Removal**
   - Removed UTF-8 BOM markers from text
   - Ensures clean text processing

3. **Directional Mark Removal**
   - Removed LRM/RLM marks
   - Prevents display issues

4. **Whitespace Normalization**
   - Trimmed leading/trailing whitespace
   - Normalized multiple consecutive spaces

5. **IOB2 Tag Alignment Verification**
   - Verified tag count matches token count for all 4,800 sentences
   - Validated B-IDIOM always precedes I-IDIOM

6. **Span Verification**
   - Character spans: Verified text extraction matches matched_expression
   - Token spans: Verified indices are within bounds

### Validation Checks Summary

| Check | Result | Details |
|-------|--------|---------|
| Missing values | ✅ 0/76,800 | All cells populated |
| Duplicate rows | ✅ 0/4,800 | No duplicates |
| ID sequence | ✅ Complete | 0-4799, no gaps |
| Label consistency | ✅ 100% | מילולי↔0, פיגורטיבי↔1 |
| IOB2 alignment | ✅ 100% | Tags match tokens |
| Character spans | ✅ 100% | Extraction verified |
| Token spans | ✅ 100% | Indices valid |
| Encoding (BOM) | ✅ 0 found | Clean encoding |
| Zero-width chars | ✅ 0 found | No hidden chars |
| Hebrew text | ✅ 100% | All contain Hebrew |
| Data types | ✅ Correct | 8 int64, 8 object |

### Character Span Verification

For each sentence, we verified:
```python
text[span_start:span_end] matches matched_expression
```
Result: **100% match** across all 4,800 sentences

### Token Span Verification

For each sentence, we verified:
```python
0 <= token_span_start < token_span_end <= num_tokens
```
Result: **100% valid** across all 4,800 sentences

### IOB2 Sequence Validation

Valid sequences checked:
- ✅ B-IDIOM always starts idiom span
- ✅ I-IDIOM only follows B-IDIOM or I-IDIOM
- ✅ No invalid tag types found

---

## 5. Dataset Splits

### Split Strategy: Hybrid (Seen + Unseen Idioms)

We use a **hybrid splitting approach** that enables both in-domain evaluation and zero-shot generalization testing:

**1. Unseen Idiom Test (Zero-Shot):**
- 6 idioms are **held out entirely** (all 80 sentences per idiom)
- These idioms never appear in training or validation
- Evaluates true zero-shot generalization to new idioms

**2. Seen Splits (In-Domain):**
- Remaining 54 idioms contribute sentences to **all three splits**
- Split by **sentence** (not expression) using 80/10/10 ratio
- Each idiom and each label maintains the ratio
- Ensures 50/50 literal/figurative balance in all splits

**Why This Approach?**
- **In-domain test:** Measures generalization to new contexts for known idioms
- **Unseen test:** Measures true zero-shot performance on novel idioms
- Best of both worlds for comprehensive evaluation

### Distribution

| Split | Samples | % of Total | Idioms | Literal | Figurative |
|-------|---------|------------|--------|---------|------------|
| **Train** | 3,456 | 72% | 54 (all seen) | 1,728 | 1,728 |
| **Validation** | 432 | 9% | 54 (all seen) | 216 | 216 |
| **Test (in-domain)** | 432 | 9% | 54 (all seen) | 216 | 216 |
| **Unseen Test** | 480 | 10% | 6 (held out) | 240 | 240 |

### Per-Idiom Distribution (Seen Idioms)

Each of the 54 seen idioms contributes:
- **Train:** 32 literal + 32 figurative = 64 sentences
- **Validation:** 4 literal + 4 figurative = 8 sentences
- **Test:** 4 literal + 4 figurative = 8 sentences

### Unseen Test Idioms (Zero-Shot)

These 6 idioms are completely held out from training:

1. חתך פינה (cut corner)
2. חצה קו אדום (crossed red line)
3. נשאר מאחור (stayed behind)
4. שבר שתיקה (broke silence)
5. איבד את הראש (lost head)
6. רץ אחרי הזנב של עצמו (chased own tail)

### Why These 6 Idioms for Zero-Shot Testing?

The selection of these specific idioms was carefully designed to test diverse generalization capabilities:

| Idiom | Tokens | Selection Rationale |
|-------|--------|---------------------|
| **חתך פינה** | 2 | Common idiom with clear literal/figurative distinction; tests basic transfer |
| **חצה קו אדום** | 3 | Boundary/threshold metaphor; different semantic category from training idioms |
| **נשאר מאחור** | 2 | Spatial metaphor; tests whether models learn metaphorical spatial reasoning |
| **שבר שתיקה** | 2 | Very frequent idiom in Hebrew; tests common "break" metaphor pattern |
| **איבד את הראש** | 3 | Body-part idiom; similar to training idioms but different expression |
| **רץ אחרי הזנב של עצמו** | 5 | Longest idiom (5 tokens); tests multi-token span detection ability |

**Key Design Principles:**
- **Length diversity:** 2-5 tokens (tests span detection across lengths)
- **Semantic diversity:** Body parts, spatial, action, boundary metaphors
- **Frequency range:** Common to medium-frequency idioms
- **Structural variety:** Different syntactic patterns (verb-noun, verb-prep-noun, complex clause)

This selection enables evaluation of whether models learn generalizable features or memorize specific idioms

### Unseen Idioms Showcase

Below are examples from each of the 6 unseen idioms, showing both literal and figurative uses:

---

#### 1. חתך פינה (cut a corner)

**Literal:**
> בזמן משחק כדורגל מתוח במיוחד, האוהד הזקן שישב ביציע חתך פינה אחר פינה מהכריך שאשתו הכינה לו מבעוד מועד, נגס ונגס - האם אוכל הוא הדרך היחידה להרגיע אוהד מתוח?

**Figurative:**
> במהלך פרויקט השיפוץ הגדול, הקבלן החליט לחתוך פינה כדי לחסוך בעלויות העבודה, אך אם היה מבין מראש שהמהלך הזה יגרום נזק למערכת החשמל כולה - אולי היה עוצר בזמן ומונע הפסד עצום.

---

#### 2. חצה קו אדום (crossed a red line)

**Literal:**
> אם נהגת חוצה קו אדום באור מלא, גם בלי כוונה רעה, האם זה באמת רק "טעות אנוש" או מעשה שמסכן חיים וצריך להיחשב פלילי?

**Figurative:**
> כשהמנכ"ל חשף נתונים פנימיים לתקשורת כדי להציל את תדמיתו, רבים טענו שהוא חצה קו אדום מוסרי - האם באמת מותר לפגוע באמון החברה כדי לשמור על מוניטין אישי?

---

#### 3. איבד את הראש (lost the head)

**Literal:**
> הפסל הרומי העתיק איבד את הראש שלו במהלך החפירה הארכיאולוגית בירושלים, אך הארכיאולוגים שמחו כי דווקא כך נחשף מבנה נדיר שמעיד על סדנת אומנות עתיקה מהמאה השלישית לפני הספירה.

**Figurative:**
> הוא איבד את הראש מרוב אהבה.

---

#### 4. שבר שתיקה (broke silence)

**Literal:**
> הם עמדו בחושך ליד המדורה, ואף אחד לא העז לדבר, עד שאחד הילדים שבר שתיקה בקול רועד ושאל אם אפשר כבר ללכת הביתה.

**Figurative:**
> אחרי שנים של השתקה במערכת הבריאות, מנהל בית החולים שבר שתיקה והודה כי נעשו טעויות חמורות בטיפול בחולים הקשישים - האם זו תחילתה של רפורמה אמיתית או רק ניסיון להרגיע את הציבור?

---

#### 5. רץ אחרי הזנב של עצמו (chased his own tail)

**Literal:**
> הכלב רץ אחרי הזנב של עצמו בחצר הגדולה

**Figurative:**
> אם מנכ"ל החברה ממשיך לרוץ אחרי הזנב של עצמו במקום להציב חזון ברור, איך אפשר לצפות מהעובדים להבין מה המדיניות של החברה בכלל?

---

#### 6. נשאר מאחור (stayed behind)

**Literal:**
> אם הרצים לא יעזרו אחד לשני בזמן התחרות, מי שנשאר מאחור במקטע האחרון של המסלול עלול לאבד את התקווה ולהפסיד את כל המרוץ.

**Figurative:**
> אם הממשלה לא תתאים את עצמה לשינויים הגלובליים בשוק האנרגיה, היא תישאר מאחור בעוד מדינות אחרות יובילו את המהפכה הירוקה והתעשייתית.

---

### Split Verification

- ✅ Seen idioms appear in train/validation/in-domain test (no empty buckets)
- ✅ Unseen idioms are fully disjoint from seen splits
- ✅ All splits balanced (50/50 literal vs figurative)
- ✅ Total sentences preserved: 4,800
- ✅ No sentence overlap between splits
- ✅ Metadata documented in `split_expressions.json`

---

## 6. Tasks Supported

### Task 1: Sentence Classification (Binary)

**Objective:** Classify whether the idiom is used literally (0) or figuratively (1)

**Evaluation Metrics:**
- Accuracy
- Macro F1-score
- Precision & Recall (per class)
- ROC-AUC

**Baseline:** Random = 50% *(Will be computed as part of this project)*

### Task 2: Token Classification (IOB2)

**Objective:** Identify the exact token span of the idiom using BIO tagging

**Label Scheme:**
- O: Outside idiom
- B-IDIOM: Beginning of idiom span
- I-IDIOM: Inside idiom span

**Example:**
```
Tokens: ["הוא", "שבר", "את", "הראש", "על", "הבעיה"]
Tags:   ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", "O", "O"]
```

**Baseline:** Random = 33% (3 classes) *(Will be computed as part of this project)*

### Task 3: Character Span Detection (Available for Future Research)

**Objective:** Identify the exact character positions of the idiom

**Columns Provided:**
- `span_start`: Character start position
- `span_end`: Character end position
- `char_mask`: Binary character-level mask

**Example:**
```python
text = "הוא שבר את הראש על הבעיה"
span_start = 4
span_end = 16
extracted = text[4:16]  # "שבר את הראש"
char_mask = "0000111111111110000000000"
```

**Note:** This task is available in the dataset for future research but is not the focus of the current thesis.

---

## 7. Key Findings & Research Contributions

### 7.1 Novel Contributions

1. **First Hebrew idiom detection dataset**
2. **Dual-task annotation** (classification + span)
3. **100% polysemy** (unique among similar datasets)
4. **Near-perfect IAA** (κ = 0.9725)
5. **Rich morphological variance** (captures Hebrew complexity)

### 7.2 Linguistic Findings

1. **Figurative sentences are more complex** (24% more subclause markers)
2. **High lexical diversity** (63.46% hapax legomena)
3. **Significant morphological variance** (up to 35 variants per idiom)
4. **Annotators default to literal** (65:1 disagreement ratio)

### 7.3 Comparison to Published Datasets

| Dataset | Language | Size | Idioms | Dual-Task | Polysemy | IAA |
|---------|----------|------|--------|-----------|----------|-----|
| MAGPIE | English | 1,756 | 3 | No | Yes | ~0.80 |
| PIE | Portuguese | 1,248 | 12 | No | Yes | ~0.75 |
| SemEval 2022 | Multi | ~7K | 100+ | No | Partial | 0.70-0.85 |
| **Hebrew-4800** | **Hebrew** | **4,800** | **60** | **Yes** | **100%** | **0.9725** |

---

## 8. Concrete Examples: Literal vs Figurative

### Example 1: שבר את הראש (broke the head)

**Literal Use:**
> הילד שבר את הראש כשהתגלגל מהמיטה החדשה בסלון, ונלקח לבדיקה בבית החולים הקרוב שם גילו רק חבלה קלה.
>
> *"The child broke his head when he rolled off the new bed in the living room, and was taken for examination at the nearby hospital where they found only a minor injury."*

**Figurative Use:**
> היא מעולם לא שברה את הראש כל כך הרבה זמן כדי למצוא רעיון יצירתי למתנת יום הולדת מפתיעה כזאת.
>
> *"She had never racked her brain for so long to find a creative idea for such a surprising birthday gift."*

### Example 2: שיחק באש (played with fire)

**Literal Use:**
> הילד שיחק באש ליד המדורה וכמעט נכווה כשניצוץ קפץ על ידו.
>
> *"The child played with fire near the bonfire and almost got burned when a spark jumped on his hand."*

**Figurative Use:**
> הוא שיחק באש כשהחליט להעתיק במבחן הסופי, בידיעה שאם ייתפס הוא יורחק מהאוניברסיטה.
>
> *"He played with fire when he decided to cheat on the final exam, knowing that if caught he would be expelled from the university."*

### Example 3: איבד את הראש (lost the head)

**Literal Use:**
> הבובה איבדה את הראש אחרי שהכלב שיחק איתה בחצר.
>
> *"The doll lost its head after the dog played with it in the yard."*

**Figurative Use:**
> היא איבדה את הראש כשראתה את המחיר הסופי של החתונה והחלה לצעוק על כולם.
>
> *"She lost her head when she saw the final price of the wedding and started yelling at everyone."*

---

## 9. Data Collection Methodology

### 9.1 Sentence Generation Process

**Phase 1: Initial Generation**
- Used ChatGPT (LLM) to generate initial sentence drafts for each idiom
- Generated multiple sentences per idiom in both literal and figurative contexts
- Ensured variety in sentence structure, length, and context

**Phase 2: Manual Rewriting**
- **Every single sentence** was manually rewritten by native Hebrew speakers
- Corrected grammar, syntax, and natural flow
- Ensured sentences sound authentic and natural in Hebrew
- Maintained proper Hebrew morphological forms

**Phase 3: Unseen Test Set Creation**
- The 6 unseen test idioms (480 sentences) were created **entirely manually**
- No LLM assistance for these sentences
- Ensures clean evaluation of zero-shot generalization

### 9.2 Quality Control

- Multiple rounds of review and revision
- Both annotators reviewed all sentences
- Focus on naturalness and grammatical correctness
- Varied sentence complexity intentionally

---

## 10. Annotation Process & Guidelines

### 10.1 Annotator Qualifications

- **2 native Hebrew speakers**
- Both familiar with Hebrew idioms and their usage
- Trained on the distinction between literal and figurative meanings

### 10.2 Annotation Criteria

**Literal Classification (0):**
- The idiom's words carry their **actual, physical meaning**
- The sentence describes a concrete, real-world event
- Example: "broke the head" = actual physical injury

**Figurative Classification (1):**
- The idiom carries its **idiomatic, metaphorical meaning**
- The sentence describes an abstract concept or state
- Example: "broke the head" = thought hard, struggled mentally

### 10.3 Disagreement Resolution Process

1. Both annotators independently labeled all 4,800 sentences
2. For any disagreement:
   - Discussed the specific sentence and context
   - If ambiguous, **rewrote the sentence** to make the meaning clearer
   - Continued until both annotators agreed
3. Same process applied to both label and span annotations
4. Result: 100% agreement after resolution (κ = 0.9725 before resolution)

### 10.4 Span Annotation

- Annotators marked the exact character and token positions of the idiom
- Included all morphological variants (prefixes, conjugations)
- Verified alignment between text extraction and marked expression

---

## 11. Research Questions & Experiments

### 11.1 Primary Research Questions

This thesis focuses on evaluating model performance for Hebrew idiom detection:

1. **Multilingual vs Hebrew-Specific Models**
   - How do multilingual models (mBERT, XLM-RoBERTa) compare to Hebrew-specific models (AlephBERT, DictaBERT)?
   - Do Hebrew-specific models better capture morphological patterns?

2. **Training Paradigms**
   - Zero-shot: How do models perform without task-specific training?
   - Few-shot backbone training: Does limited fine-tuning help?
   - Full fine-tuning: What is the upper bound performance?

3. **Task Performance**
   - Sentence classification: Which models best distinguish literal vs figurative?
   - Token classification: Which models best identify idiom spans?

4. **Generalization Capability**
   - In-domain test: Generalization to new contexts for known idioms
   - Unseen idiom test: Zero-shot generalization to novel idioms

### 11.2 Planned Experiments

**Models:**
- **Hebrew-Specific:** AlephBERT-base, AlephBERT-Gimmel, DictaBERT
- **Multilingual:** mBERT, XLM-RoBERTa-base

| Experiment | Models | Description |
|------------|--------|-------------|
| **Sentence Classification** | All 5 models | Binary literal/figurative classification |
| **Token Classification** | All 5 models | IOB2 sequence tagging for span detection |
| **LLM Prompting** | GPT-4, Claude | Zero-shot and few-shot prompting on test sets |
| **Training Paradigms** | All 5 models | Zero-shot → Few-shot → Full fine-tuning comparison |
| **Ablation Studies** | Best models | Effect of training size, backbone freezing, etc. |
| **Error Analysis** | All experiments | Analysis of error patterns and failure cases |

### 11.3 Evaluation Strategy

**Two Test Sets for Comprehensive Evaluation:**

1. **In-Domain Test (432 samples)**
   - Same idioms as training (different sentences)
   - Measures: generalization to new contexts

2. **Unseen Idiom Test (480 samples)**
   - Completely held-out idioms (6 idioms, 480 sentences)
   - Measures: zero-shot generalization to new idioms
   - Both test sets used for **all models and all tasks**

### 11.4 Analysis

After experiments, we will perform:
- **Ablation analysis:** Understanding what contributes to performance
- **Error analysis:** Categorizing and explaining model mistakes
- **Comparison analysis:** Why certain models outperform others
- **Generalization analysis:** In-domain vs unseen performance gap

### 11.5 Potential Use Cases Beyond Thesis

- Hebrew NLP resource for the community
- Benchmark for Hebrew figurative language understanding
- Cross-lingual idiom detection research
- Educational tool for Hebrew learners

---

## 12. Limitations & Known Issues

### 12.1 Dataset Limitations

| Limitation | Details | Impact | Mitigation |
|------------|---------|--------|------------|
| **Position Bias** | 63.71% of idioms at sentence start | Models may learn position heuristics | Report position-stratified results |
| **Limited Idiom Coverage** | 60 idioms (not exhaustive) | May not cover all Hebrew idioms | Future expansion planned |
| **Small Unseen Test** | Only 6 held-out idioms | High variance in zero-shot results | Report confidence intervals |
| **Sentence Type Bias** | 95% declarative sentences | Limited to formal written Hebrew | Note as limitation |
| **Domain Limitation** | Contemporary Hebrew only | May not generalize to historical texts | Clearly scope the dataset |

### 12.2 Methodological Limitations

| Limitation | Details | Future Work |
|------------|---------|-------------|
| **No Formal Guidelines Doc** | Annotation criteria not in separate document | Create formal guidelines |
| **No Human Baseline** | Human performance not measured | Collect human accuracy scores |
| **No Error Analysis Yet** | Model mistakes not categorized | Add post-experiment analysis |
| **LLM-Assisted Generation** | Initial sentences from ChatGPT | Mitigated by manual rewriting |

### 12.3 Potential Biases

1. **Position Bias:** Idioms tend to appear at sentence start - may be natural Hebrew pattern or artifact
2. **Complexity Bias:** Figurative sentences are more complex - reflects natural usage
3. **Annotator Bias:** Both annotators from similar background - may affect edge cases

### 12.4 What This Dataset Does NOT Cover

- Historical Hebrew texts
- Spoken/colloquial Hebrew
- Rare or archaic idioms
- Multi-word expressions that aren't idioms
- Idioms with only literal or only figurative uses

---

## 13. Future Work (Post-Project Expansion)

The following extensions are planned for after the thesis is completed:

### 13.1 Dataset Expansion

- [ ] Expand to 100+ idioms
- [ ] Add more sentences per idiom (target: 100+)
- [ ] Include more complex sentence structures
- [ ] Add semantic categories for idioms
- [ ] Create difficulty ratings per idiom

### 13.2 Additional Resources

- [ ] Formal annotation guidelines document
- [ ] Cross-lingual alignment (English translations)
- [ ] Spoken/colloquial Hebrew corpus
- [ ] Historical Hebrew idiom extension

### 13.3 Future Research Directions

- Contextual embeddings analysis
- Attention visualization studies
- Cross-lingual idiom detection
- Multi-word expression detection beyond idioms

---

## 14. Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{hebrew_idioms_4800,
  author = {Nazarenko, Igor and Amit, Yuval},
  title = {Hebrew-Idioms-4800: A Dual-Task Dataset for Hebrew Idiom Detection},
  year = {2025},
  publisher = {Reichman University},
  note = {Master's Project Dataset},
  url = {https://github.com/igornazarenko434/hebrew-idiom-detection}
}
```

---

## 15. Files in This Package

```
professor_review/
├── README.md                           # This file (complete documentation)
├── Complete_Dataset_Analysis.ipynb     # Comprehensive analysis notebook
└── data/
    ├── expressions_data_tagged.csv     # Full dataset (4,800 sentences)
    ├── expressions_data_tagged.xlsx    # Full dataset (Excel format)
    └── splits/
        ├── train.csv                   # Training set (3,456 samples)
        ├── validation.csv              # Validation set (432 samples)
        ├── test.csv                    # In-domain test set (432 samples)
        ├── unseen_idiom_test.csv       # Zero-shot test set (480 samples)
        └── split_expressions.json      # Split metadata
```

---

## 16. How to Use

### View Statistics
Read this README for complete statistics and findings.

### Interactive Analysis
Open `Complete_Dataset_Analysis.ipynb` in any Jupyter-compatible environment to:
- See all validation checks
- View all 16+ visualizations inline
- Explore the data interactively

### Load Data

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/expressions_data_tagged.csv')
print(f"Total samples: {len(df)}")
print(f"Literal: {(df['label_2'] == 0).sum()}")
print(f"Figurative: {(df['label_2'] == 1).sum()}")
```

---

## 17. Summary

**Hebrew-Idioms-4800** is a high-quality, professionally annotated dataset for Hebrew idiom detection featuring:

- **Excellent annotation quality** (κ = 0.9725)
- **Perfect class balance** (50/50)
- **100% polysemous idioms**
- **Dual-task annotations** (classification + span)
- **Rich morphological variance** (Hebrew-specific)
- **Comprehensive validation** (9.2/10 quality score)

The dataset represents a **significant contribution** to Hebrew NLP and figurative language processing research.

---

**Contact:**
- Igor Nazarenko: igor.nazarenko@post.runi.ac.il
- Yuval Amit: yuval.amit@post.runi.ac.il

**Repository:** https://github.com/igornazarenko434/hebrew-idiom-detection
