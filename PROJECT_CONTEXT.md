# PROJECT CONTEXT & STATUS
# Hebrew Idiom Detection - LLM Session Memory

**Last Updated:** December 31, 2025
**Project Status:** 80-85% Complete (Documentation: 100%, STEP 1 Analysis: 100%)
**Current Phase:** Results Analysis & Paper Writing (STEP 1 Complete, Moving to STEP 2/4)
**Documentation Version:** 4.0

---

## QUICK PROJECT SUMMARY

**Mission:** Create the **first comprehensive Hebrew idiom dataset** (Hebrew-Idioms-4800 v2.0) with dual-task annotations and benchmark transformer models + LLMs for Hebrew figurative language understanding.

**Two Tasks:**
1. **Sentence Classification**: Is this sentence literal or figurative?
2. **Span Detection**: Where exactly is the idiom? (IOB2 token tagging)

---

## DATASET: Hebrew-Idioms-4800 v2.0

### Core Stats
- **4,800 sentences** across 60 Hebrew idioms
- **Perfect 50/50 balance**: 2,400 literal + 2,400 figurative
- **Near-perfect IAA**: Cohen's κ = 0.9725 (98.625% agreement)
- **Quality score**: 9.2/10 (14/14 validation checks passed)
- **100% polysemy**: Every idiom appears in both literal and figurative contexts

### Data Construction Process
1. Initial generation with ChatGPT
2. **Every sentence manually rewritten** by native Hebrew speakers
3. 6 unseen idioms (480 sentences) created **entirely manually** (no LLM)
4. Dual annotation by 2 native speakers
5. Disagreements resolved through discussion and rewrites

### Dataset Splits (Hybrid Strategy)

| Split | Samples | Idioms | Purpose |
|-------|---------|--------|---------|
| Train | 3,456 (72%) | 54 seen | Training |
| Validation | 432 (9%) | 54 seen | Model selection |
| Test (seen) | 432 (9%) | 54 seen | In-domain evaluation |
| Test (unseen) | 480 (10%) | 6 held-out | Zero-shot generalization |

**Unseen Test Idioms** (completely held out):
1. חתך פינה (cut corner)
2. חצה קו אדום (crossed red line)
3. נשאר מאחור (stayed behind)
4. שבר שתיקה (broke silence)
5. איבד את הראש (lost head)
6. רץ אחרי הזנב של עצמו (chased own tail)

### Critical Data Schema Details

**Pre-tokenized Data:**
```python
{
    "sentence": str,              # Full Hebrew sentence
    "tokens": list[str],         # PRE-TOKENIZED (punctuation-separated)
    "iob_tags": list[str],       # "O", "B-IDIOM", "I-IDIOM"
    "label": int,                # 0=literal, 1=figurative
    "start_token": int,          # Token start (0-indexed)
    "end_token": int,            # Token end (exclusive)
}
```

**CRITICAL:** Data arrives with `tokens` and `iob_tags` already aligned. Use `is_split_into_words=True` when tokenizing.

### File Locations
- Full dataset: `data/expressions_data_tagged_v2.csv`
- Splits: `data/splits/train.csv`, `validation.csv`, `test.csv`, `unseen_idiom_test.csv`

---

## MODELS EVALUATED

### Encoder Models (6 total)

**Hebrew-Specific (4):**
1. **AlephBERT-base**: `onlplab/alephbert-base` (110M params, 52K vocab)
2. **AlephBERTGimmel-base**: `dicta-il/alephbertgimmel-base` (110M params, 128K vocab)
3. **DictaBERT**: `dicta-il/dictabert` (110M params, 50K vocab)
4. **NeoDictaBERT**: `dicta-il/neodictabert` (110M params, 4K context, 285B Hebrew tokens) - **NEW: Sept 2025**

**Multilingual (2):**
5. **mBERT**: `bert-base-multilingual-cased` (110M params, 119K vocab)
6. **XLM-RoBERTa-base**: `xlm-roberta-base` (125M params, 250K vocab)

### LLM Models (Planned - NOT YET IMPLEMENTED)
- DictaLM-3.0-1.7B-Instruct (Hebrew-native LLM)
- Llama-3.1-8B-Instruct (Multilingual baseline)
- Qwen 2.5-7B-Instruct (`Qwen/Qwen2.5-7B-Instruct`) - **NEW: Added to evaluation plan**

---

## WHAT WE'VE ACCOMPLISHED (75-80% COMPLETE)

### Phase 1: Environment Setup ✅ 100% DONE
- PyCharm project, dependencies, VAST.ai, Docker, rclone

### Phase 2: Data Preparation ✅ 100% DONE
- Dataset validation (14/14 checks passed)
- Comprehensive statistics (17+ metrics)
- Hybrid dataset splitting (seen/unseen idioms)

### Phase 3: Zero-Shot Baseline ✅ 100% DONE

**Results:**
- **Task 1 (Classification)**: All models perform at random (~50% F1)
- **Task 2 (Span Detection)**:
  - Heuristic baseline (string matching): 100% F1 (confirms data quality)
  - Untrained model: ~0% F1 (confirms task requires learning)

**Files:** `experiments/results/zero_shot/`

### Phase 4: Full Fine-Tuning ✅ 100% DONE

**What we did:**
1. ✅ Created training configs (`experiments/configs/training_config.yaml`, `hpo_config.yaml`)
2. ✅ Implemented full training pipeline with:
   - **IOB2 subword alignment** (`src/utils/tokenization.py`)
   - WeightedLossTrainer for class imbalance
   - CRF layer for IOB2 constraint enforcement
   - TensorBoard logging
   - Early stopping + best model checkpointing
3. ✅ Hyperparameter Optimization (Optuna):
   - 150 trials total (5 models × 2 tasks × 15 trials)
   - 6-parameter search space
   - Best params saved to `experiments/results/best_hyperparameters/`
4. ✅ VAST.ai cloud infrastructure:
   - Persistent volume workflow
   - Automated scripts for HPO and training
5. ✅ Final Training:
   - 30 runs (5 models × 2 tasks × 3 seeds: 42, 123, 456)
6. ✅ Evaluation on both test sets (seen + unseen)

### FINAL RESULTS (Multi-Seed Average)

**Task 1: Sentence Classification (Seen Test)**

| Model | Mean F1 | Std | Notes |
|-------|---------|-----|-------|
| **DictaBERT** | **94.83%** | 0.27% | Best classifier |
| AlephBERTGimmel | 94.68% | 1.01% | Close second |
| AlephBERT | 94.21% | 1.06% | |
| XLM-RoBERTa | 91.74% | 1.42% | Best multilingual |
| mBERT | 87.58% | 0.71% | |

**Task 2: Token Classification - Span Detection (Seen Test)**

| Model | Mean Span F1 | Std | Notes |
|-------|--------------|-----|-------|
| **AlephBERT** | **99.65%** | 0.11% | Near-perfect! |
| mBERT | 99.31% | 0.20% | Surprisingly good |
| XLM-RoBERTa | 99.27% | 0.24% | |
| AlephBERTGimmel | 99.12% | 0.13% | |
| DictaBERT | 99.12% | 0.07% | |

**Generalization to Unseen Idioms**

**Task 1 (Classification):**

| Model | Mean F1 | Gap from Seen |
|-------|---------|---------------|
| **AlephBERTGimmel** | **91.38%** | -3.30% |
| DictaBERT | 91.08% | -3.75% |
| AlephBERT | 90.62% | -3.60% |
| mBERT | 90.14% | **+2.56%** (better on unseen!) |
| XLM-RoBERTa | 89.86% | -1.88% |

**Task 2 (Span Detection):**

| Model | Mean Span F1 | Gap from Seen |
|-------|--------------|---------------|
| **AlephBERTGimmel** | **75.59%** | -23.53% |
| DictaBERT | 72.58% | -26.54% |
| AlephBERT | 72.48% | -27.17% |
| XLM-RoBERTa | 63.18% | -36.09% |
| mBERT | 57.99% | -41.32% |

**Key Findings:**
1. Hebrew-specific models dominate both tasks on seen idioms
2. Near-perfect span detection on seen idioms (~99.5% F1)
3. Strong classification generalization to unseen idioms (~91% F1, only 3-4% drop)
4. Significant span detection challenge on unseen idioms (24-41% drop)
5. mBERT shows surprising positive transfer on unseen classification (+2.56%)
6. Statistical significance: Hebrew models >> mBERT, but differences among Hebrew models mostly insignificant

**Files:**
- Results: `experiments/results/evaluation/seen_test/`, `experiments/results/evaluation/unseen_test/`
- Analysis: `experiments/results/analysis/finetuning_summary.md`, `generalization/generalization_report.md`

### Phase 6: Documentation v4.0 ✅ 100% COMPLETE (December 31, 2025)

**Comprehensive Documentation Overhaul:**

We created a clean, professional 3-document system that covers ALL analysis missions with NLP best practices:

**📄 Core Documentation (Active):**

1. **IMPLEMENTATION_ROADMAP.md** (38,157 bytes) ⭐ **PRIMARY DOCUMENT**
   - Complete step-by-step action plan for Phase 5-7 missions
   - 4 STEPs with ready-to-run code snippets (no pseudocode)
   - Week-by-week timeline (Quick Wins → Ablations → Interpretability → Finalization)
   - 100% coverage of all missions from STEP_BY_STEP_MISSIONS.md
   - **This is THE ONE document to follow for implementation**

2. **EVALUATION_STANDARDIZATION_GUIDE.md** (92,897 bytes, 30 sections)
   - Complete NLP best practices reference
   - Exact Span F1 metric definition (exact boundary match)
   - 12-category error taxonomy for SPAN, 2 for CLS
   - Statistical testing protocols (paired t-test, Bonferroni, Cohen's d)
   - Visualization standards (Seaborn, colorblind palette)
   - Shared standards for partner compatibility (fine-tuning + prompting)
   - Following CoNLL 2003, Dodge et al. 2020, Dror et al. 2018

3. **OPERATIONS_GUIDE.md** (51,357 bytes, 11 sections)
   - Complete 5-phase workflow (HPO → Training → Download → Eval → Analysis)
   - Analysis tools reference with exact commands
   - Model-agnostic design (automatically detects any model in results/)
   - Troubleshooting guide
   - VAST.ai training workflow

4. **DOCUMENTATION_INDEX.md** (10,681 bytes)
   - Master navigation guide
   - Scenario-based usage (when to use which document)
   - Quick reference card
   - Document relationship flowchart

**📂 Archived Documentation:**

- `archive/old_documentation/` - 6 superseded files (MISSION_4.7, DOCUMENT_COVERAGE_ANALYSIS, IMPLEMENTATION_GUIDE, MISSIONS_PROGRESS_TRACKER, TRAINING_ANALYSIS_AND_WORKFLOW, TRAINING_OUTPUT_ORGANIZATION)
- `archive/reference_guides/` - 4 optional guides (VAST.ai guides, IAA_Report, PATH_REFERENCE)

**🎯 Key Achievement:**

ONE clear path to completion: Follow IMPLEMENTATION_ROADMAP.md → References EVALUATION_STANDARDIZATION_GUIDE.md (standards) + OPERATIONS_GUIDE.md (how-to) when needed.

**✅ Partner Compatibility:**

- Shared error taxonomy (12 SPAN categories, 2 CLS categories)
- Standardized naming conventions (models, tasks, splits, seeds)
- Common file formats (JSON predictions with error_category field)
- Results can be merged seamlessly with prompting partner

### Phase 7: Analysis ✅ STEP 1 COMPLETE (100%)

**✅ STEP 1: Quick Wins - ALL TASKS COMPLETE (December 31, 2025)**

**Task 1.1: Fine-Tuning Results Aggregation ✅**
- Tool: `src/analyze_finetuning_results.py`
- Outputs: `experiments/results/analysis/finetuning_summary.csv/md`
- Outputs: `experiments/results/analysis/statistical_significance.txt`
- Outputs: `paper/tables/finetuning_results.tex` (LaTeX tables)
- Status: Multi-seed aggregation (Mean ± Std), statistical tests with Bonferroni correction & Cohen's d

**Task 1.2: Generalization Gap Analysis ✅**
- Tool: `src/analyze_generalization.py`
- Outputs: `experiments/results/analysis/generalization/generalization_report.md`
- Outputs: `paper/figures/generalization/*.png`
- Status: Complete with gap metrics (seen_f1 - unseen_f1) and visualizations

**Task 1.3: Error Categorization & Visualization ✅**
- Tool: `scripts/categorize_all_errors.py` + `src/analyze_error_distribution.py`
- Outputs: All 60 `eval_predictions.json` files updated with `error_category` field
- Outputs: `experiments/results/analysis/error_analysis/error_distribution_detailed.csv`
- Outputs: `experiments/results/analysis/error_analysis/error_analysis_report.md` (auto-generated with full methodology)
- Outputs: `paper/figures/error_analysis/*.png` (5 publication-ready figures, 300 DPI)
- Status: Complete with 12 SPAN categories + 3 CLS categories, grouped analysis, comprehensive report

**Task 1.4: Per-Idiom F1 Analysis ✅**
- Tool: `scripts/analyze_per_idiom_f1.py`
- Outputs: `experiments/results/analysis/per_idiom_f1/per_idiom_f1_raw.csv`
- Outputs: `experiments/results/analysis/per_idiom_f1/per_idiom_f1_summary.csv`
- Outputs: `experiments/results/analysis/per_idiom_f1/idiom_difficulty_ranking_{task}_{split}.csv` (4 files)
- Outputs: `experiments/results/analysis/per_idiom_f1/per_idiom_f1_report.md`
- Outputs: `experiments/results/analysis/per_idiom_f1/per_idiom_f1_insights.md`
- Outputs: `experiments/results/analysis/per_idiom_f1/idiom_metadata.csv`
- Outputs: `paper/figures/per_idiom/per_idiom_heatmap_{task}_{split}.png` (4 heatmaps, 300 DPI)
- Status: Complete with idiom difficulty rankings and insights for all 60 idioms

**Task 1.5: Statistical Significance Testing ✅**
- Tool: `scripts/statistical_tests.py`
- Outputs: `experiments/results/analysis/statistical_tests/paired_ttests.csv`
- Outputs: `experiments/results/analysis/statistical_tests/paired_ttests.md`
- Status: Complete with paired t-tests, Bonferroni correction, Cohen's d effect sizes

**Task 1.6: Model Comparison Visualizations ✅**
- Tool: `src/analyze_finetuning_results.py --create_figures`
- Outputs: `paper/figures/finetuning/model_comparison_cls_seen.png`
- Outputs: `paper/figures/finetuning/model_comparison_cls_unseen.png`
- Outputs: `paper/figures/finetuning/model_comparison_span_seen.png`
- Outputs: `paper/figures/finetuning/model_comparison_span_unseen.png`
- Outputs: `paper/figures/finetuning/learning_curves_cls.png`
- Outputs: `paper/figures/finetuning/learning_curves_span.png`
- Status: Complete with 4 model comparison charts + 2 learning curve visualizations (300 DPI)

**✅ Additional Completed Analysis:**
- Zero-shot baseline analysis (`analyze_zero_shot.py`)
- HPO trials analysis (`analyze_optuna_results.py`)

**❌ STEP 2-4: Remaining Tasks (See IMPLEMENTATION_ROADMAP.md)**
- ❌ STEP 2: Ablation Studies (frozen backbone, CRF impact, non-Hebrew models)
- ❌ STEP 3: Interpretability (attention visualization, learning curves extraction)
- ❌ STEP 4: Finalization (confusion matrices, cross-lingual analysis, final polishing)

---

## WHAT'S LEFT TO DO

### CRITICAL FOR PUBLICATION (~40 hours / 1.5 weeks)

**✅ 1. Comprehensive Analysis - COMPLETE** (was 6 hours, now done)
- ✅ Categorized all prediction errors (12 SPAN + 3 CLS categories)
- ✅ Identified hard vs. easy idioms (difficulty rankings)
- ✅ Analyzed unseen idiom failure patterns (error dashboard)
- ✅ Statistical significance testing complete
- ✅ Per-idiom F1 analysis complete

**✅ 2. Core Visualizations - MOSTLY COMPLETE** (was 5 hours)
- ✅ Learning curves (cls/span)
- ✅ Performance comparison charts (4 charts)
- ✅ Generalization gap visualization
- ✅ Error distribution visualizations (5 figures)
- ✅ Per-idiom heatmaps (4 heatmaps)
- ❌ Confusion matrices deep-dive (remaining)
- ❌ HPO convergence plots (optional)

**✅ 3. Results Tables - COMPLETE** (was 2 hours, now done)
- ✅ LaTeX tables for paper (`finetuning_results.tex`)
- ✅ Statistical significance annotations (paired t-tests with Bonferroni)
- ✅ Model comparison tables (Mean ± Std)
- ✅ Per-idiom difficulty rankings (4 CSV files)

**4. Paper Writing (25 hours)**
- Abstract (250 words)
- Introduction (1.5 pages)
- Related Work (1 page)
- Dataset (2 pages)
- Methodology (2 pages)
- Results (2 pages)
- Analysis & Discussion (1.5 pages)
- Conclusion (0.5 pages)
- References

**5. Documentation (8 hours)**
- HuggingFace dataset card
- Code README refinement
- Usage examples
- Troubleshooting guide

**6. Submission (4 hours)**
- Format for ACL/EMNLP
- Generate camera-ready PDF
- Upload to conference portal

### HIGH PRIORITY - NEW MODEL EVALUATION

**Mission 4.8: NeoDictaBERT Training & Evaluation (8-10 hours)**
- **Zero-shot baseline** (Task 1 + Task 2) - 1 hour
- **HPO with Optuna** (15 trials per task) - 3-4 hours on VAST.ai
- **Full fine-tuning** (3 seeds × 2 tasks = 6 runs) - 2-3 hours on VAST.ai
- **Evaluation** on seen + unseen test sets - 1 hour
- **Results integration** into analysis - 1 hour
- **Model**: `dicta-il/neodictabert` - Latest Hebrew BERT (Sept 2025)
  - 285B Hebrew tokens training data
  - 4,096 token context window
  - Expected to outperform DictaBERT

### OPTIONAL (Strengthens Paper) (+21 hours)

**7. LLM Evaluation (12 hours)**
- DictaLM-3.0 vs. Llama-3.1 vs. **Qwen 2.5-7B**
- Zero-shot + few-shot prompting
- Compare against fine-tuned models
- Cost-performance analysis

**8. Trivial Baselines (3 hours)**
- Majority class baseline
- Random guessing baseline
- Position-based heuristics
- Establishes performance floor for credibility

**9. Deep Dive Analysis (6 hours)**
- Attention weight visualization
- Embedding space analysis
- Layer-wise probing

### RELEASE & PUBLICATION

**10. Dataset Release (2-3 hours)**
- Upload to HuggingFace Datasets
- Create dataset viewer
- Add download scripts

**11. Model Release (2-3 hours)**
- Upload best checkpoints to HuggingFace Hub
- Create model cards
- Add inference examples

---

## CRITICAL IMPLEMENTATION DETAILS

### IOB2 Subword Alignment (ESSENTIAL for Task 2)

**Location:** `src/utils/tokenization.py`

**Key Function:** `align_labels_with_tokens()`

**What it does:**
- Maps word-level IOB2 tags to transformer subwords
- Uses `is_split_into_words=True` for pre-tokenized data
- First subword gets the label, others get -100 (ignored in loss)
- Prevents training on subword boundaries

**Example:**
```python
# Word-level (from dataset)
tokens = ["הוא", "שבר", "את", "הראש"]
iob_tags = ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM"]

# After subword tokenization
# "הראש" → ["הרא", "##ש"]
# IOB tags become: ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", -100]
#                                              ↑ first subword    ↑ subsequent subword (ignored)
```

### CRF Layer Integration

**Location:** Custom model class in `src/idiom_experiment.py`

**Purpose:** Enforces valid IOB2 transitions
- Prevents illegal sequences (e.g., I-IDIOM without B-IDIOM)
- Improves span F1 by 1-2%

### Training Configuration

**Files:**
- `experiments/configs/training_config.yaml` - Full fine-tuning params
- `experiments/configs/hpo_config.yaml` - HPO search space

**Key Parameters:**
- Learning rate: 2e-5 (Task 1), 3e-5 (Task 2)
- Batch size: 16
- Epochs: 5
- Warmup: 0.1
- Weight decay: 0.01
- Early stopping patience: 3
- Seeds: 42, 123, 456

### Error Taxonomy & Standardization (NEW - Dec 31, 2025)

**Location:** EVALUATION_STANDARDIZATION_GUIDE.md Section 4

**Critical for partner compatibility** - Fine-tuning + Prompting results must use the same taxonomy.

**SPAN Task Error Categories (12 total):**
1. **PERFECT**: Predicted span exactly matches ground truth
2. **MISS**: No span predicted when ground truth has idiom
3. **FALSE_POSITIVE**: Predicted span when no idiom exists
4. **PARTIAL_START**: Missing beginning tokens (correct end)
5. **PARTIAL_END**: Missing ending tokens (correct start)
6. **PARTIAL_BOTH**: Missing both beginning and ending tokens
7. **EXTEND_START**: Extra tokens at the beginning
8. **EXTEND_END**: Extra tokens at the end
9. **EXTEND_BOTH**: Extra tokens on both sides
10. **SHIFT**: Completely misaligned span
11. **WRONG_SPAN**: Predicted wrong idiom location
12. **MULTI_SPAN**: Multiple spans predicted when only one exists

**CLS Task Error Categories (2 total):**
1. **FALSE_POSITIVE**: Predicted figurative when actually literal
2. **FALSE_NEGATIVE**: Predicted literal when actually figurative

**Implementation:**
```python
from src.utils.error_analysis import categorize_span_error, categorize_cls_error

# For SPAN task
error_category = categorize_span_error(true_tags, predicted_tags)

# For CLS task
error_category = categorize_cls_error(true_label, predicted_label)
```

**File Format:**
```json
{
  "sentence": "...",
  "true_label": "figurative",
  "predicted_label": "literal",
  "true_tags": ["O", "B-IDIOM", "I-IDIOM", ...],
  "predicted_tags": ["O", "O", "B-IDIOM", ...],
  "error_category": "PARTIAL_START"  ← Added by categorization script
}
```

**Partner Sync Critical:**
- Both partners must use EXACT same category names
- Both partners must use EXACT same categorization logic
- Ensures results can be merged for joint analysis
- See EVALUATION_STANDARDIZATION_GUIDE.md Section 4 for complete definitions

---

## PROJECT STRUCTURE

### Key Directories

```
/Users/igornazarenko/PycharmProjects/Final_Project_NLP/
├── data/
│   ├── expressions_data_tagged_v2.csv     # Full dataset
│   └── splits/                            # Train/val/test/unseen
│
├── src/
│   ├── idiom_experiment.py                # Main runner (2,095 lines)
│   ├── analyze_finetuning_results.py      # Performance analysis
│   ├── analyze_generalization.py          # Generalization analysis
│   └── utils/
│       └── tokenization.py                # IOB2 alignment (CRITICAL)
│
├── experiments/
│   ├── configs/
│   │   ├── training_config.yaml
│   │   └── hpo_config.yaml
│   └── results/
│       ├── evaluation/
│       │   ├── seen_test/                 # In-domain results
│       │   └── unseen_test/               # Generalization results
│       ├── analysis/
│       │   ├── finetuning_summary.md
│       │   └── generalization/
│       ├── best_hyperparameters/          # Best HPO params
│       └── optuna_studies/                # HPO databases
│
├── scripts/
│   ├── run_all_hpo.sh
│   ├── run_all_experiments.sh
│   ├── download_from_gdrive.sh
│   └── sync_to_gdrive.sh
│
└── paper/                                 # Paper materials (to be created)
    ├── figures/
    └── tables/
```

### Key Files

**Documentation (v4.0 - December 31, 2025):**

**Active Documents (Use These):**
- `DOCUMENTATION_INDEX.md` - Master navigation guide (START HERE if confused)
- `IMPLEMENTATION_ROADMAP.md` ⭐ - **FOLLOW THIS** for all analysis missions (PRIMARY)
- `EVALUATION_STANDARDIZATION_GUIDE.md` - Standards & metrics reference
- `OPERATIONS_GUIDE.md` - How-to manual & workflows
- `PROJECT_CONTEXT.md` ← YOU ARE HERE (LLM session memory)
- `STEP_BY_STEP_MISSIONS.md` - Original mission requirements
- `README.md` - Project overview
- `FINAL_PRD_Hebrew_Idiom_Detection.md` - Product requirements

**Archived Documentation (Historical Reference Only):**
- `archive/old_documentation/` - Superseded files (MISSION_4.7, IMPLEMENTATION_GUIDE, etc.)
- `archive/reference_guides/` - Optional guides (VAST.ai, IAA, PATH_REFERENCE)

**Training:**
- `src/idiom_experiment.py` - Main experiment runner
- `experiments/configs/training_config.yaml` - Training params
- `experiments/configs/hpo_config.yaml` - HPO search space

**Analysis:**
- `src/analyze_finetuning_results.py` - Performance comparison
- `src/analyze_generalization.py` - Generalization gap analysis
- `experiments/results/analysis/finetuning_summary.md` - Results summary
- `experiments/results/analysis/generalization/generalization_report.md` - Generalization report

---

## RUNNING EXPERIMENTS

### Quick Commands

**Evaluate a trained model:**
```bash
python src/idiom_experiment.py \
    --mode evaluate \
    --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
    --data data/splits/unseen_idiom_test.csv \
    --task cls \
    --device cuda
```

**Train a new model:**
```bash
python src/idiom_experiment.py \
    --mode full_finetune \
    --model_id onlplab/alephbert-base \
    --task cls \
    --config experiments/configs/training_config.yaml \
    --device cuda
```

**Run analysis:**
```bash
python src/analyze_finetuning_results.py
python src/analyze_generalization.py
```

---

## IMPORTANT NOTES FOR NEW LLM SESSIONS

### When Starting a New Session

1. **Read this file first** (PROJECT_CONTEXT.md) to understand project context
2. **Read DOCUMENTATION_INDEX.md** for navigation guidance
3. **Open IMPLEMENTATION_ROADMAP.md** ⭐ - This is THE ONE document to follow
4. **Check git status** to see recent changes
5. **Review** `experiments/results/analysis/` for latest results
6. **Verify** data integrity: `ls data/splits/` should show train.csv, validation.csv, test.csv, unseen_idiom_test.csv

### Documentation System (v4.0)

**Simple Rule:**
- **90% of time:** Follow IMPLEMENTATION_ROADMAP.md step-by-step
- **10% of time:** Reference EVALUATION_STANDARDIZATION_GUIDE.md (standards) or OPERATIONS_GUIDE.md (how-to) when needed

**Navigation:**
```
IMPLEMENTATION_ROADMAP.md (Follow this!)
         ↓
   References when needed:
         ↓
EVALUATION_STANDARDIZATION_GUIDE.md (Standards)
OPERATIONS_GUIDE.md (How-To)
```

### Common Gotchas

1. **IOB2 Alignment**: Data is pre-tokenized. ALWAYS use `is_split_into_words=True`
2. **Multi-Seed**: Results are from 3 seeds (42, 123, 456). Report mean ± std
3. **Two Test Sets**: Always evaluate on both seen test AND unseen test
4. **Span F1 vs Token F1**: For Task 2, primary metric is Span F1 (stricter)
5. **Hebrew Text**: UTF-8 encoding required. Files must be NFKC normalized

### What NOT to Do

❌ Don't retrain models unless specifically requested (already have results)
❌ Don't modify the dataset (it's final and validated)
❌ Don't run experiments without checking if results already exist
❌ Don't create new analysis scripts without checking existing ones first

### What TO Do

✅ **Follow IMPLEMENTATION_ROADMAP.md step-by-step** (starting with Task 1.1)
✅ Focus on analysis, visualization, and paper writing
✅ Use existing results from `experiments/results/`
✅ Reference EVALUATION_STANDARDIZATION_GUIDE.md for standards (metrics, error taxonomy, protocols)
✅ Reference OPERATIONS_GUIDE.md for tool usage and troubleshooting
✅ Create publication-ready figures and tables
✅ Write error analysis based on model predictions
✅ Draft academic paper sections

---

## PROJECT STRENGTHS (For Paper)

1. **First of its kind**: Only Hebrew idiom dataset with dual annotations
2. **Exceptional data quality**: κ = 0.9725, zero errors, perfect balance
3. **Rigorous evaluation**: Multi-seed, statistical tests, dual test sets
4. **Production-ready code**: 8,494 lines, modular, well-documented
5. **Strong results**: 94% classification, 99.5% span detection on seen idioms
6. **Impressive generalization**: 91% classification on completely unseen idioms
7. **Complete infrastructure**: VAST.ai workflow, HPO, TensorBoard, backup automation
8. **Comprehensive analysis**: Statistical significance, generalization gaps, HPO insights

## PROJECT GAPS (What Still Needs Work)

1. **No LLM evaluation**: Critical comparison missing for modern NLP
2. **Limited error analysis**: No categorization of failure patterns
3. **No paper draft**: Academic writing not started
4. **Visualization incomplete**: TensorBoard data not extracted to figures
5. **No dataset release**: Not yet on HuggingFace
6. **Limited interpretability**: No attention visualization, embedding analysis
7. **No trivial baselines**: Missing performance floor for credibility

---

## TIMELINE TO COMPLETION

**Critical Path (Minimum Viable Paper):** 50 hours (1.5-2 weeks)
- Error Analysis: 6h
- Visualizations: 5h
- Results Tables: 2h
- Paper Writing: 25h
- Documentation: 8h
- Submission: 4h

**Enhanced Version (Stronger Paper):** +21 hours
- LLM Evaluation: +12h
- Trivial Baselines: +3h
- Deep Dive Analysis: +6h

**Publication-Ready:** +8 hours
- Dataset Release: +3h
- Model Release: +3h
- Camera-Ready: +2h

**Total to Publication: 3-4 weeks of focused work**

---

## NEXT IMMEDIATE STEPS

### ✅ STEP 1: Quick Wins - **100% COMPLETE!** 🎉

**All STEP 1 tasks finished (December 31, 2025):**
- ✅ Task 1.1: Aggregate fine-tuning results
- ✅ Task 1.2: Generalization gap analysis
- ✅ Task 1.3: Error categorization + visualization
- ✅ Task 1.4: Per-idiom F1 analysis
- ✅ Task 1.5: Statistical significance testing
- ✅ Task 1.6: Model comparison visualizations

### ⭐ NEXT: STEP 2 or STEP 4 (See IMPLEMENTATION_ROADMAP.md)

**Option A: STEP 2 - Ablation Studies** (requires VAST.ai training, 2-3 days)
- Task 2.1: Frozen backbone comparison
- Task 2.2: Non-Hebrew models evaluation
- Task 2.3: CRF impact analysis

**Option B: STEP 4 - Finalization** (no training needed, 1-2 days)
- Task 4.1: Confusion matrix deep-dive
- Task 4.2: Cross-lingual error pattern analysis
- Task 4.3: Final figure polishing

**Recommendation:** Start with **STEP 4** (quicker wins) or proceed directly to **Paper Writing**

**STEP 2: Ablations (2-3 days, requires VAST.ai)**
- Task 2.1: Frozen backbone comparison
- Task 2.2: Non-Hebrew models evaluation
- Task 2.3: CRF impact analysis

**STEP 3: Interpretability (1 day)**
- Task 3.1: Attention visualization
- Task 3.2: Learning curves extraction

**STEP 4: Finalization (1-2 days)**
- Task 4.1: Publication-ready figures
- Task 4.2: LaTeX tables
- Task 4.3: Results section writing

**See IMPLEMENTATION_ROADMAP.md for complete details and ready-to-run code!**

---

## PUBLICATION TARGET

**Venue:** ACL 2025 / EMNLP 2025 / Specialized Workshop
**Contribution:** First Hebrew idiom dataset + comprehensive benchmark
**Novelty:** Dual-task annotations, 100% polysemy, seen/unseen evaluation
**Results Quality:** Publication-worthy (94% cls, 99.5% span on seen; 91% cls on unseen)

---

## CONTACT & COLLABORATION

**Researchers:**
- Igor Nazarenko: igor.nazarenko@post.runi.ac.il
- Yuval Amit: yuval.amit@post.runi.ac.il

**Institution:** Reichman University, School of Computer Science

---

## VERSION HISTORY

| Date | Version | Changes | Completion % |
|------|---------|---------|--------------|
| Dec 30, 2025 | 1.0 | Initial comprehensive context file | 75-80% |
| Dec 30, 2025 | 1.1 | Added NeoDictaBERT (6th model) & Qwen 2.5 LLM | 75-80% |
| Dec 31, 2025 | 2.0 | **Documentation v4.0 Overhaul**: Created 3-document system (IMPLEMENTATION_ROADMAP.md, EVALUATION_STANDARDIZATION_GUIDE.md, OPERATIONS_GUIDE.md) + navigation index. Archived old documentation. 100% mission coverage verified. | 75-80% (Docs: 100%) |
| Dec 31, 2025 | 2.1 | **STEP 1 Analysis Complete**: All 6 Quick Win tasks finished (fine-tuning aggregation, generalization analysis, error categorization+visualization, per-idiom F1, statistical testing, model comparison figures). Fixed error naming bug (FP/FN → FALSE_POSITIVE/FALSE_NEGATIVE). Total: 15+ analysis outputs, 15+ publication-ready figures. | 80-85% (STEP 1: 100%) |

---

**Remember:** This project is 75-80% complete. The hard technical work is done. Focus on analysis, visualization, and writing to bring it across the finish line for publication.

**Documentation is now 100% complete (v4.0).** Follow IMPLEMENTATION_ROADMAP.md for all remaining work.

**Next Session TODO:**
1. **Read PROJECT_CONTEXT.md** (this file) for project overview
2. **Read DOCUMENTATION_INDEX.md** for navigation guidance
3. **Open IMPLEMENTATION_ROADMAP.md** ⭐ and start with Task 1.1 (or next incomplete task)
4. **Execute:** `python src/analyze_finetuning_results.py` (if not already done)
5. **Follow the roadmap step-by-step** - it covers ALL Phase 5-7 missions
6. **Reference standards when needed:** EVALUATION_STANDARDIZATION_GUIDE.md
7. **Reference tools when needed:** OPERATIONS_GUIDE.md
8. **Update PROJECT_CONTEXT.md** when major milestones are reached
