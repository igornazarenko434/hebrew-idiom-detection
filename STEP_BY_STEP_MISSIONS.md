# Step-by-Step Missions Guide
# Hebrew Idiom Detection Project

**Document Version:** 1.0
**Created:** November 7, 2025
**Based on:** PRD v2.1 (VAST.ai + PyCharm Setup)
**Purpose:** Detailed mission breakdown for implementation

---

## How to Use This Document

- Each mission is numbered and must be completed in order
- Each mission has: **Objective**, **Tasks**, **Validation**, and **Success Criteria**
- Do NOT proceed to next mission until current mission validation passes
- This document contains NO code - only instructions and techniques
- Share this with AI assistants or developers for step-by-step execution

---

## PHASE 1: ENVIRONMENT SETUP (Week 1)

### Mission 1.1: PyCharm Project Setup

**Objective:** Create the project structure in PyCharm with proper configuration

**Tasks:**
1. Create new PyCharm project named "hebrew-idiom-detection"
2. Set Python interpreter to virtual environment (Python 3.9 or 3.10 recommended)
3. Create the following folder structure:
   - `data/` - for dataset files
   - `src/` - for source code
   - `src/utils/` - for utility functions
   - `experiments/` with subfolders: `configs/`, `results/`, `logs/`
   - `models/` - for model checkpoints (local cache)
   - `notebooks/` - for Jupyter notebooks
   - `scripts/` - for automation scripts
   - `docker/` - for Docker configurations
   - `tests/` - for unit tests
   - `paper/` with subfolders: `figures/`, `tables/`
4. Initialize Git repository
5. Create `.gitignore` file to exclude:
   - `models/` folder (large files)
   - `__pycache__/`
   - `.env` files
   - `experiments/results/` (will be in Google Drive)
   - Virtual environment folders
6. Create `README.md` with project description
7. Create `requirements.txt` file (initially empty, will populate later)

**Validation:**
- All folders created and visible in PyCharm project explorer
- Git initialized successfully (check with `git status`)
- Python interpreter active and correct version
- Can create and run a simple test file (e.g., print "Hello World")

**Success Criteria:**
✅ Project structure matches PRD Section 10.1
✅ Git repository initialized
✅ Virtual environment activated
✅ PyCharm project opens without errors

---

### Mission 1.2: Dependencies Installation

**Objective:** Install all required Python libraries for local development

**Tasks:**
1. Add the following libraries to `requirements.txt`:
   - `transformers` (HuggingFace - for models)
   - `datasets` (HuggingFace - for data handling)
   - `torch` (PyTorch - deep learning framework)
   - `accelerate` (for efficient training)
   - `optuna` (for hyperparameter optimization)
   - `scikit-learn` (for metrics and evaluation)
   - `pandas` (for data manipulation)
   - `numpy` (for numerical operations)
   - `matplotlib` and `seaborn` (for visualization)
   - `jupyter` (for notebooks)
   - `pytest` (for testing)
   - `gdown` (for Google Drive downloads)
   - `python-dotenv` (for environment variables)
2. Install all dependencies in virtual environment
3. Verify PyTorch installation with GPU support check (if GPU available locally)
4. Verify transformers can load a sample model (e.g., "bert-base-uncased")

**Validation:**
- Run: `pip list` and verify all packages installed
- Run: `python -c "import torch; print(torch.__version__)"`
- Run: `python -c "import transformers; print(transformers.__version__)"`
- Run: `python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased'); print('Success')"`

**Success Criteria:**
✅ All dependencies installed without errors
✅ PyTorch version >= 2.0
✅ Transformers version >= 4.30
✅ Can import all major libraries

---

### Mission 1.3: Google Drive Storage Setup

**Objective:** Configure Google Drive as primary storage for models and results

**Tasks:**
1. Create folder structure in Google Drive:
   - `Hebrew_Idiom_Detection/`
     - `data/` - dataset files
     - `models/` - model checkpoints
     - `results/` - experiment results
     - `logs/` - training logs
     - `backups/` - code backups
2. Upload the dataset file `expressions_data_tagged.csv` to `data/` folder
3. Create shareable links for the dataset (will use for VAST.ai)
4. Note the Google Drive file ID for the dataset (from shareable link)
5. Test download using `gdown` library locally:
   - Use: `gdown <file-id>` to download dataset
6. Create a `.env` file in project root with Google Drive paths and file IDs

**Validation:**
- Dataset visible in Google Drive folder
- Can download dataset using `gdown` command
- `.env` file created with correct file IDs
- Verify downloaded file matches original (check file size)

**Success Criteria:**
✅ Google Drive folder structure created
✅ Dataset uploaded and accessible
✅ Can download dataset programmatically
✅ File IDs documented in `.env`

---

### Mission 1.4: VAST.ai Account Setup

**Objective:** Create VAST.ai account and understand instance rental process

**Tasks:**
1. Create account at https://vast.ai
2. Add payment method (credit card or crypto)
3. Add initial credit ($10-20 recommended for testing)
4. Explore VAST.ai interface:
   - Search for GPU instances
   - Filter by GPU type (RTX 3090, RTX 4090, A5000)
   - Sort by price ($/hour)
   - Check reliability scores
5. Read VAST.ai documentation on:
   - SSH connection
   - File upload/download
   - Instance lifecycle (rent/pause/destroy)
6. **DO NOT rent instance yet** - just familiarize with interface

**Validation:**
- Account created and verified
- Payment method added successfully
- Can see available instances in search
- Understand how to filter instances by specs

**Success Criteria:**
✅ VAST.ai account active
✅ Credit added to account
✅ Familiar with instance search and filtering
✅ Know how to read instance specifications

---

### Mission 1.5: Docker Configuration (Optional but Recommended)

**Objective:** Create Docker configuration for reproducible VAST.ai environment

**Tasks:**
1. Create `Dockerfile` in `docker/` folder with:
   - Base image: PyTorch official image with CUDA support
   - Python 3.9 or 3.10
   - Install all dependencies from `requirements.txt`
   - Set working directory
   - Expose necessary ports (e.g., 8888 for Jupyter)
2. Create `docker-compose.yml` for local testing
3. Create `.dockerignore` file to exclude unnecessary files
4. Test building Docker image locally (if Docker installed)
5. Document Docker commands in `docker/README.md`

**Note:** This mission is optional but highly recommended for reproducibility

**Validation:**
- Dockerfile created with all dependencies
- Can build Docker image locally (if Docker available)
- docker-compose.yml configured correctly
- Documentation clear and complete

**Success Criteria:**
✅ Docker files created
✅ Image builds successfully (or validated syntax)
✅ Documentation complete
✅ Ready for VAST.ai deployment

---

## PHASE 2: DATA PREPARATION & VALIDATION (Week 1-2)

### Mission 2.1: Dataset Loading and Inspection

**Objective:** Load the dataset and understand its structure

**Tasks:**
1. Create `src/data_preparation.py` file
2. Implement function to load CSV dataset using pandas
3. Display basic statistics:
   - Total number of rows
   - Column names and types
   - First 10 rows
   - Check for missing values
   - Check for duplicate rows
4. Verify dataset matches PRD Section 2.2 schema:
   - All required columns present
   - Data types correct
   - No missing critical fields
5. Create Jupyter notebook `notebooks/01_data_validation.ipynb` for interactive exploration

**Validation:**
- Dataset loads without errors
- Row count = 4,800 (as per PRD)
- All columns from PRD schema present
- No missing values in critical columns
- Print summary statistics

**Success Criteria:**
✅ Dataset loads successfully
✅ 4,800 total sentences
✅ All required columns present
✅ No critical missing data
✅ Schema matches PRD Section 2.2

---

### Mission 2.2: Label Distribution Validation

**Objective:** Verify dataset is balanced (50/50 literal vs figurative)

**Tasks:**
1. Count occurrences of each label:
   - "מילולי" (literal) count
   - "פיגורטיבי" (figurative) count
2. Calculate percentage for each class
3. Verify `label_2` column (0/1) matches `label` column
4. Create visualization:
   - Bar chart of label distribution
   - Save to `paper/figures/label_distribution.png`
5. Check if distribution is exactly 50/50 or close enough

**Validation:**
- Total literal samples: ~2,400 (50%)
- Total figurative samples: ~2,400 (50%)
- `label` and `label_2` columns are consistent
- Visualization created and saved

**Success Criteria:**
✅ 2,400 literal samples (50%)
✅ 2,400 figurative samples (50%)
✅ Labels consistent across columns
✅ Visualization saved

---

### Mission 2.3: IOB2 Tags Validation

**Objective:** Verify IOB2 annotations are correctly aligned with tokens

**Tasks:**
1. Implement tokenization function (using whitespace or Hebrew tokenizer)
2. For each row:
   - Tokenize the `text` column
   - Split `iob2_tags` column by spaces
   - Verify number of tokens = number of IOB2 tags
   - Verify number of tags matches `num_tokens` column
3. Check IOB2 tag validity:
   - Only valid tags: "O", "B-IDIOM", "I-IDIOM"
   - No invalid sequences (e.g., "I-IDIOM" without "B-IDIOM" first)
4. Verify token span positions:
   - `token_span_start` and `token_span_end` are valid indices
   - Correspond to idiom boundaries in `iob2_tags`
5. Display examples with misalignments (if any)

**Validation:**
- All rows have matching token count and tag count
- No invalid IOB2 tags
- No IOB2 sequence violations
- Token spans align with B-IDIOM and I-IDIOM tags
- Report any errors found

**Success Criteria:**
✅ 100% alignment between tokens and IOB2 tags
✅ No invalid tags
✅ No sequence violations
✅ Token spans correct
✅ Zero or minimal errors reported

---

### Mission 2.4: Dataset Statistics Analysis (ENHANCED - COMPREHENSIVE)

**Objective:** Generate comprehensive statistics for the dataset (PART 1 + PART 2 analyses)

**Implementation:** All analyses implemented in `src/data_preparation.py`
- Use `loader.run_mission_2_4()` for PART 1 (required analyses)
- Use `loader.run_comprehensive_analysis(include_part2=True)` for PART 1 + PART 2

---

#### **PART 1: REQUIRED ANALYSES** ✅ COMPLETED

**Tasks:**
1. **Basic Dataset Statistics** (`generate_statistics()`):
   - Total sentences: 4,800
   - Literal samples: 2,400 (50%)
   - Figurative samples: 2,400 (50%)
   - Unique idioms: 60 expressions
   - Expression occurrence statistics (min/max/mean/std)
   - Average sentence length (tokens & characters)
   - Average idiom length (tokens & characters)
   - Standard deviations for all metrics

2. **Idiom Distribution Analysis**:
   - Top 10 most frequent idioms
   - Expression occurrence patterns (all idioms: 80 occurrences each)
   - Idioms appearing in both literal and figurative contexts (polysemy)

3. **Sentence Type Analysis** (`analyze_sentence_types()`):
   - Identify sentence types (declarative, question, imperative, exclamatory)
   - Count and percentage of each type
   - Cross-tabulation by label (literal vs figurative)
   - Sentence type distribution per idiom

4. **Idiom Position Analysis** (`analyze_idiom_position()`) - NEW:
   - Compute position_ratio = token_span_start / num_tokens
   - Classify positions: start (0-33%), middle (33-67%), end (67-100%)
   - Position distribution overall and by label
   - Result: 87.96% of idioms at sentence start

5. **Polysemy Analysis** (`analyze_polysemy()`) - NEW:
   - Identify polysemous idioms (both literal & figurative)
   - Calculate figurative usage ratio per idiom
   - Rank idioms by polysemy balance
   - Result: All 60 idioms are polysemous (50/50 split)

6. **Lexical Statistics** (`analyze_lexical_statistics()`) - NEW:
   - Vocabulary size: 17,628 unique words
   - Type-Token Ratio (TTR): 0.2460
   - Top 20 most frequent words overall
   - Top 20 words in idioms
   - Function word frequencies (Hebrew)
   - Lexical statistics by label

7. **Standard Visualizations** (11 total):
   - label_distribution.png
   - sentence_length_distribution.png
   - idiom_length_distribution.png
   - top_10_idioms.png
   - sentence_type_distribution.png
   - sentence_type_by_label.png
   - sentence_length_boxplot_by_label.png (NEW)
   - polysemy_heatmap.png (NEW)
   - idiom_position_histogram.png (NEW)
   - idiom_position_by_label.png (NEW)
   - sentence_length_violin_by_label.png (NEW)

8. **Save Results**:
   - Statistics: `experiments/results/dataset_statistics_comprehensive.txt`
   - Visualizations: `paper/figures/`

---

#### **PART 2: OPTIONAL/RECOMMENDED ANALYSES** ✅ COMPLETED

**Tasks:**
1. **Structural Complexity Analysis** (`analyze_structural_complexity()`) - NEW:
   - Subclause markers detection (ש, כי, אם, כאשר, למרות, etc.)
   - Punctuation counting and analysis
   - Mean subclause count: 0.28 per sentence
   - Sentences with subclauses: 24.48%
   - Complexity comparison by label (figurative more complex)

2. **Lexical Richness Analysis** (`analyze_lexical_richness()`) - NEW:
   - Hapax legomena: 11,182 words (63.43% of vocabulary)
   - Dis legomena: 2,697 words
   - Maas Index: 0.0112
   - Zipf's law validation (frequency vs. rank)
   - Confirms high lexical diversity

3. **Collocational Analysis** (`analyze_collocations()`) - NEW:
   - ±3 token context extraction around idioms
   - Total context words: 23,899
   - Unique context words: 8,053
   - Top context words by label
   - Collocation pattern differences (literal vs figurative)

4. **Annotation Consistency Analysis** (`analyze_annotation_consistency()`) - NEW:
   - Prefix attachment patterns (ו, ה, ל, מ, ב, כ, ש)
   - Prefix attachment rate: 41.40%
   - Variant forms per idiom (up to 28 variants)
   - Mean consistency rate: 0.3971
   - Shows morphological flexibility of Hebrew

5. **Advanced Visualizations** (6 additional):
   - zipf_law_plot.png (NEW)
   - structural_complexity_by_label.png (NEW)
   - collocation_word_clouds.png (NEW)
   - vocabulary_diversity_scatter.png (NEW)
   - hapax_legomena_comparison.png (NEW)
   - context_words_bar_chart.png (NEW)

6. **Save Results**:
   - Comprehensive statistics: `experiments/results/dataset_statistics_full.txt`
   - All visualizations: `paper/figures/` (17 total)

---

**Validation:**
- ✅ Statistics match PRD Section 2.3 expectations
- ✅ Average sentence length: 14.93 tokens (higher than expected)
- ✅ Average idiom length: 2.39 tokens
- ✅ Unique idioms: 60 expressions (as expected)
- ✅ All idioms are polysemous (50/50 literal/figurative)
- ✅ Sentence types analyzed and documented
- ✅ All visualizations saved to paper/figures/
- ✅ PART 1 + PART 2 analyses completed

**Success Criteria:**
✅ All basic statistics calculated correctly
✅ 60 unique idioms identified
✅ Avg sentence length: 14.93 tokens (mean), 10 tokens (median)
✅ Avg idiom length: 2.39 tokens
✅ Sentence types analyzed (7.12% questions, 92.88% declarative)
✅ Idiom positions analyzed (87.96% at start)
✅ Polysemy: All 60 idioms polysemous
✅ Lexical richness: 17,628 vocabulary, 63.43% hapax
✅ Structural complexity: 0.28 mean subclause markers
✅ Collocations: 8,053 unique context words
✅ Annotation consistency: 0.3971 mean consistency rate
✅ 17 visualizations created and saved
✅ 2 comprehensive statistical reports generated

**Key Findings:**
- Dataset is perfectly balanced (50/50 literal/figurative)
- All idioms are polysemous (appear in both contexts)
- Idioms predominantly appear at sentence start (87.96%)
- High lexical diversity (63.43% hapax legomena)
- Figurative sentences are structurally more complex
- Significant morphological variance (Hebrew prefix attachments)

---

### Mission 2.5: Dataset Splitting (Hybrid Strategy)

**Objective:** Support both in-domain and zero-shot evaluation with a unified splitting workflow.

**Strategy Overview:**
- Keep the 6 designated idioms as the **unseen idiom test set** (zero-shot evaluation).
- For every remaining idiom, split its literal and figurative sentences into **train (80%) / validation (10%) / in-domain test (10%)** so each idiom appears in all three splits with disjoint sentences.
- Maintain label balance (≈50/50) within each split.

**Tasks:**
1. **Reserve the Unseen Idiom Test Set**
   - Extract all sentences for the 6 fixed idioms:
     - ״חתך פינה״, ״חצה קו אדום״, ״נשאר מאחור״, ״שבר שתיקה״, ״איבד את הראש״, ״רץ אחרי הזנב של עצמו״
   - Save as `data/splits/unseen_idiom_test.csv`
   - Confirm literal/figurative counts remain balanced
2. **Per-Idiom Stratified Splitting for Remaining Data**
   - For each other idiom:
     - Separate literal vs figurative sentences
     - Shuffle deterministically (seeded)
     - Allocate 80%/10%/10% to train/validation/in-domain test (per label)
     - Ensure each idiom contributes sentences to **all three** splits
   - Aggregate results into:
     - `data/splits/train.csv`
     - `data/splits/validation.csv`
     - `data/splits/test.csv` (now the in-domain test)
3. **Metadata and Documentation**
   - Update `data/splits/split_expressions.json` with per-idiom counts for each split
   - Update `data/expressions_data_with_splits.csv` with a `split` column containing `train`, `validation`, `test_in_domain`, or `unseen_idiom_test`
4. **Verification**
   - Seen vs unseen idioms are disjoint
   - Every seen idiom has sentences in train/validation/test_in_domain
   - Literal/figurative ratios per split stay within ±5% of 50/50
   - Total sentence count equals the original dataset

**Validation:**
- Unseen idiom test set contains exactly the 6 specified idioms
- Train/validation/test_in_domain contain all remaining idioms with disjoint sentences
- Every seen idiom contributes to all three seen splits
- Label balance per split is within ±5% of 50/50
- Saved files match expected counts and can be reloaded successfully

**Success Criteria:**
✅ Four split files (`train`, `validation`, `test_in_domain`, `unseen_idiom_test`) created  
✅ `split_expressions.json` documents per-idiom counts  
✅ No overlap between unseen idioms and seen splits  
✅ Balanced label distribution in every split  
✅ `expressions_data_with_splits.csv` updated

---

### Mission 2.6: Data Preparation Testing

**Objective:** Create unit tests for data preparation functions

**Tasks:**
1. Create `tests/test_data_preparation.py`
2. Write test functions for:
   - Dataset loading (correct shape, columns)
   - Label balance check
   - IOB2 validation
   - Splitting function (correct sizes, stratification, seen vs unseen integrity)
   - Token count alignment
   - Split metadata validation (per-idiom counts, seen vs unseen)
3. Run tests using pytest framework
4. Document any issues found and fix them
5. Ensure all tests pass before proceeding

**Validation:**
- Run: `pytest tests/test_data_preparation.py -v`
- All tests pass
- Test coverage > 80% for data preparation code
- No warnings or errors

**Success Criteria:**
✅ All unit tests created
✅ All tests pass
✅ Split integrity tests pass (seen vs unseen)
✅ Code validated and reliable
✅ Ready for model training

---

## PHASE 3: BASELINE EVALUATION - ZERO-SHOT (Week 2-3)

### Mission 3.1: Model Selection and Download

**Objective:** Download and verify all 5 encoder models for evaluation

**Tasks:**
1. Create list of models from PRD Section 4.1:
   - AlephBERT-base: `onlplab/alephbert-base`
   - AlephBERT-Gimmel: (find correct model ID on HuggingFace)
   - DictaBERT: `dicta-il/dictabert`
   - mBERT: `bert-base-multilingual-cased`
   - XLM-RoBERTa-base: `xlm-roberta-base`
2. For each model:
   - Download tokenizer using `AutoTokenizer`
   - Download model using `AutoModel`
   - Verify model loads without errors
   - Check model size (should be 110-125M parameters)
   - Test tokenization on sample Hebrew sentence
3. Document model information:
   - Model ID
   - Number of parameters
   - Vocabulary size
   - Maximum sequence length
4. Save model info to `experiments/configs/model_info.json`

**Validation:**
- All 5 models download successfully
- Each model loads without errors
- Tokenizers work correctly on Hebrew text
- Model sizes in expected range (110-125M params)
- Can perform forward pass on sample input

**Success Criteria:**
✅ All 5 models downloaded
✅ All tokenizers functional
✅ Hebrew tokenization works
✅ Model info documented
✅ Ready for evaluation

---

### Mission 3.2: Zero-Shot Evaluation Framework

**Objective:** Create evaluation script for zero-shot baseline (no training)

**Tasks:**
1. Create `src/idiom_experiment.py` file with command-line interface

   **Important:** Create the script with a **skeleton structure** that supports all modes (zero_shot, full_finetune, frozen_backbone, hpo), but **only implement zero_shot** in this mission. The skeleton will look like:
   ```python
   def main():
       args = parse_args()

       if args.mode == "zero_shot":
           run_zero_shot(args)  # ← Implement NOW in Mission 3.2
       elif args.mode == "full_finetune":
           run_training(args)  # ← Implement later in Mission 4.2
       elif args.mode == "frozen_backbone":
           run_training(args, freeze_backbone=True)  # ← Implement in Mission 4.2
       elif args.mode == "hpo":
           run_hpo(args)  # ← Implement later in Mission 4.3
       else:
           raise ValueError(f"Unknown mode: {args.mode}")
   ```
   This way, you won't need to restructure the file later - just add implementations
2. Implement functions for:
   - Loading model and tokenizer
   - Preprocessing data (tokenization, padding)
   - Forward pass through model
   - Extracting predictions (for both tasks)
   - Calculating metrics
3. For Task 1 (Sentence Classification):
   - Use [CLS] token representation
   - Apply linear layer if needed
   - Output: probability distribution over 2 classes
4. For Task 2 (Token Classification):
   - Use token-level representations
   - Map to IOB2 labels
   - Output: IOB2 tag sequence
5. Implement evaluation metrics from PRD Section 9.1:
   - Accuracy, Precision, Recall, F1
   - Confusion matrix
   - Token-level and span-level F1
6. Test on small subset first (100 samples)

**Validation:**
- Script runs without errors
- Can evaluate one model on test set
- Metrics calculated correctly
- Results saved to file
- Command-line arguments work

**Success Criteria:**
✅ Evaluation script complete
✅ Both tasks supported
✅ All metrics implemented
✅ Tested on sample data
✅ Results reproducible

---

### Mission 3.3: Zero-Shot Baseline for All Models

**Objective:** Run zero-shot evaluation on all 5 models for both tasks

**Tasks:**
1. For each of the 5 models:
   - Task 1: Sentence Classification (literal vs figurative)
     - Evaluate on test set
     - Calculate all metrics
   - Task 2: Token Classification (IOB2 tagging)
     - Evaluate on test set
     - Calculate token-level and span-level metrics
2. Save results for each model:
   - Metrics summary (JSON format)
   - Predictions (CSV format)
   - Confusion matrix (image)
3. Create comparison table across all models
4. Identify best zero-shot model for each task
5. Save all results to `experiments/results/zero_shot_baseline/`

**Note:** Can run locally if you have GPU, or use VAST.ai for faster execution

**Validation:**
- All 5 models evaluated on both tasks
- Results saved for each model
- Metrics reasonable (not random performance)
- Comparison table created
- Best baseline identified

**Success Criteria:**
✅ 10 total evaluations (5 models × 2 tasks)
✅ All results saved and documented
✅ Performance above random baseline
✅ Best zero-shot model identified
✅ Ready for fine-tuning phase

---

### Mission 3.3.5 (OPTIONAL - HIGH PRIORITY): Trivial Baseline Evaluation

**Objective:** Implement and evaluate trivial baselines to establish performance floor and validate that more complex models are actually learning

**Why this is important (Senior Researcher Recommendation):**
Every ML research paper should include trivial baselines to:
1. Establish a performance floor (sanity check)
2. Verify that sophisticated models actually learn beyond simple heuristics
3. Demonstrate that the task is non-trivial
4. Make results more credible to reviewers and readers
5. Sometimes trivial baselines perform surprisingly well and are more interpretable!

**Tasks:**

**1. Task 1 (Sentence Classification) - Trivial Baselines:**

a) **Majority Class Baseline:**
```python
# Always predict "figurative" (class 1)
predictions = np.ones(len(test_set))
# Calculate accuracy, precision, recall, F1
```

b) **Random Baseline:**
```python
# Random prediction with 50/50 distribution
predictions = np.random.randint(0, 2, size=len(test_set))
# Run 5 times with different seeds and report mean ± std
```

c) **Sentence Length Heuristic:**
```python
# Hypothesis: Longer sentences might be more figurative
# Use median sentence length as threshold
if len(sentence.split()) > median_length:
    predict "figurative"
else:
    predict "literal"
```

d) **Question Mark Heuristic:**
```python
# Based on Mission 2.4 analysis: 7.12% questions
# Test if questions are more literal or figurative
if sentence.endswith("?"):
    predict class_A
else:
    predict class_B
# Try both combinations and report better one
```

**2. Task 2 (Token Classification / Span Detection) - Trivial Baselines:**

a) **Always-O Baseline:**
```python
# Predict no idiom in any sentence
predictions = ["O"] * len(tokens) for all sentences
# This establishes floor for precision/recall
```

b) **Exact String Match Baseline:**
```python
# Already implemented in Mission 3.2 (iob_from_string_match)
# Use the provided 'expression' field
# Search for exact match in sentence
# Tag matched tokens as B-IDIOM/I-IDIOM
```

c) **First-N-Tokens Heuristic:**
```python
# Based on Mission 2.4 analysis: Average idiom length = 2.39 tokens
# Always tag first 2 tokens as B-IDIOM, I-IDIOM
# Or tag middle 2 tokens
# This tests if position matters
```

**3. Implementation:**
1. Create `src/trivial_baselines.py` script:
   ```python
   def majority_class_baseline(test_df):
       """Always predict figurative"""
       pass

   def random_baseline(test_df, n_runs=5):
       """Random predictions with multiple runs"""
       pass

   def sentence_length_baseline(test_df):
       """Use sentence length as heuristic"""
       pass

   def always_o_baseline(test_df):
       """Predict no idioms (all O tags)"""
       pass

   def exact_match_baseline(test_df):
       """Use expression field for exact matching"""
       pass
   ```

2. Run all baselines on test set
3. Save results to `experiments/results/trivial_baselines/`
4. Compare with zero-shot model results from Mission 3.3

**4. Analysis:**
1. Create comparison table:
   ```
   | Method              | Task 1 F1 | Task 2 F1 (span) |
   |---------------------|-----------|------------------|
   | Majority Class      | X.XX      | N/A              |
   | Random              | X.XX±X.XX | N/A              |
   | Sentence Length     | X.XX      | N/A              |
   | Always-O            | N/A       | X.XX             |
   | Exact Match         | N/A       | X.XX             |
   | Best Zero-Shot      | X.XX      | X.XX             |
   | Improvement         | +X.XX     | +X.XX            |
   ```

2. Verify that zero-shot models beat all trivial baselines
3. If any trivial baseline performs surprisingly well, analyze why
4. Document findings in `experiments/results/trivial_baselines_analysis.md`

**Validation:**
- All 5+ trivial baselines implemented
- Results calculated for test set
- Comparison with zero-shot models complete
- Zero-shot models significantly outperform trivial baselines
- Analysis documented

**Success Criteria:**
✅ All trivial baselines implemented and evaluated
✅ Performance floor established
✅ Zero-shot models beat trivial baselines (sanity check passed)
✅ Comparison table ready for paper
✅ Demonstrates task is non-trivial
✅ Increases paper credibility

**Expected Results:**
- Majority class should get ~50% accuracy (balanced dataset)
- Random should get ~50% accuracy
- Heuristics may get 50-65% depending on patterns
- Zero-shot models should get >65% to demonstrate learning
- Exact string match for Task 2 should be decent baseline (reported in Mission 3.2)

**Time Estimate:** 2-3 hours (very quick to implement)

**Recommendation:** This is HIGH PRIORITY and very quick to implement. Adds significant credibility to paper.

---

### Mission 3.4: Zero-Shot Results Analysis

**Objective:** Analyze baseline results and document findings

**Tasks:**
1. Create visualizations:
   - Bar chart comparing all models on Task 1 (F1 score)
   - Bar chart comparing all models on Task 2 (F1 score)
   - Save to `paper/figures/`
2. Analyze performance:
   - Hebrew-specific models vs multilingual models
   - Task 1 vs Task 2 difficulty
   - Common error patterns
3. Generate error analysis:
   - Sample 20 misclassified examples
   - Categorize errors (false positives, false negatives)
   - Note difficult idioms
4. Document findings in `experiments/results/zero_shot_analysis.md`
5. Create presentation-ready table for paper

**Validation:**
- Visualizations created and saved
- Error analysis complete
- Findings documented clearly
- Table ready for paper inclusion
- Insights actionable for next phase

**Success Criteria:**
✅ Comparison visualizations created
✅ Error analysis complete
✅ Findings documented
✅ Baseline established
✅ Ready to proceed to fine-tuning

---

## PHASE 4: FULL FINE-TUNING (Week 4-6)

### Mission 4.1: Training Configuration Setup

**Objective:** Create training configuration system for hyperparameter management

**Tasks:**
1. Create two configuration files in `experiments/configs/`:

   **a) `training_config.yaml` - Base training configuration template:**
   ```yaml
   # Model settings
   model_name: "alephbert-base"          # CLI can override
   model_checkpoint: "onlplab/alephbert-base"
   max_length: 128

   # Task settings
   task: "sequence_classification"       # Options: "sequence_classification", "token_classification"
   num_labels: 2                         # Will be 3 for token_classification

   # Training mode
   training_mode: "full_finetune"        # Options: "zero_shot", "full_finetune", "frozen_backbone"

   # Hyperparameters (can be overridden by Optuna in Mission 4.3-4.5)
   learning_rate: 2e-5
   batch_size: 16
   num_epochs: 5
   warmup_ratio: 0.1
   weight_decay: 0.01
   gradient_accumulation_steps: 1
   fp16: false                           # Mixed precision training
   seed: 42

   # Data paths
   train_file: "data/train.csv"
   dev_file: "data/dev.csv"
   test_file: "data/test.csv"

   # Output settings
   output_dir: "experiments/results/"
   save_steps: 500
   eval_steps: 500
   logging_steps: 100
   save_total_limit: 2

   # Device
   device: "cuda"                        # Options: "cuda", "cpu", "mps"
   ```

   **b) `hpo_config.yaml` - Optuna hyperparameter search space:**
   ```yaml
   # Optuna settings
   optuna:
     n_trials: 15
     direction: "maximize"               # Maximize validation F1
     pruning: true
     study_name: "idiom_hpo"

   # Search space (from PRD Section 5.1)
   search_space:
     learning_rate: [1e-5, 2e-5, 3e-5, 5e-5]
     batch_size: [8, 16, 32]
     num_epochs: [3, 5, 8]
     warmup_ratio: [0.0, 0.1, 0.2]
     weight_decay: [0.0, 0.01, 0.05]
     gradient_accumulation_steps: [1, 2, 4]

   # Fixed settings during HPO
   fixed:
     training_mode: "full_finetune"      # Always full finetune during HPO
     max_length: 128
     fp16: false
   ```

2. Implement configuration loading in `src/idiom_experiment.py`:
   - Add function to load YAML config files:
     ```python
     import yaml

     def load_config(config_path):
         """Load configuration from YAML file"""
         with open(config_path, 'r') as f:
             config = yaml.safe_load(f)
         return config
     ```
   - Add CLI argument `--config` to argparse
   - Test: Load both config files and print their contents to verify structure
   - Support command-line override of config values (e.g., --learning_rate overrides config value)
   - Validate required fields are present
   - Handle different training modes (zero_shot, full_finetune, frozen_backbone)

   **Note:** At this stage, just verify you can LOAD and PRINT config values. Full integration with training happens in Mission 4.2

3. Add command-line interface to support different modes:
   ```bash
   # Zero-shot evaluation
   python src/idiom_experiment.py --mode zero_shot --model alephbert-base --task sequence_classification

   # Full fine-tuning
   python src/idiom_experiment.py --mode full_finetune --config experiments/configs/training_config.yaml

   # Frozen backbone training
   python src/idiom_experiment.py --mode frozen_backbone --config experiments/configs/training_config.yaml

   # Hyperparameter optimization
   python src/idiom_experiment.py --mode hpo --config experiments/configs/hpo_config.yaml
   ```

4. Test configuration system:
   - Load both YAML files without errors
   - Override config values via CLI
   - Validate all hyperparameters accessible

**Validation:**
- Config files created and valid YAML
- Can load configuration from file
- All hyperparameters accessible
- CLI can override config values
- Different modes (zero_shot, full_finetune, frozen_backbone, hpo) supported
- Easy to modify for experiments

**Success Criteria:**
✅ Two configuration files created (training_config.yaml, hpo_config.yaml)
✅ Configuration loading implemented in idiom_experiment.py
✅ CLI supports all training modes
✅ All hyperparameters from PRD Section 5.1 included
✅ Easy to use and modify
✅ Ready for training experiments

---

### Mission 4.2: Training Pipeline Implementation

**Objective:** Implement complete training pipeline with HuggingFace Trainer

**Tasks:**
1. Extend `src/idiom_experiment.py` with training mode
2. Implement training pipeline:
   - Load pre-trained model
   - Add classification head (2 classes for Task 1, 3 for Task 2)
   - Setup AdamW optimizer
   - Setup linear learning rate scheduler with warmup
   - Implement early stopping (patience 2-3 epochs)
3. For Task 1 (Sequence Classification):
   - Use `AutoModelForSequenceClassification`
   - Set num_labels=2
   - Train on training set
   - Validate on validation set

3.5. **CRITICAL: For Task 2 (Token Classification) - Implement Subword Tokenization Alignment:**

   **Problem:** Transformer tokenizers split Hebrew words into subwords, but IOB2 tags are aligned with word-level tokens.

   **Example:**
   ```python
   # Word-level (what we have in dataset)
   Words:     ["הוא", "שבר", "את", "הראש"]
   IOB2 tags: ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM"]

   # After mBERT/XLM-R tokenization (subwords)
   Subwords:  ["הוא", "##ש", "##בר", "את", "##ה", "##ראש"]
   # Need to align IOB2 to these 6 subword tokens!
   ```

   **Solution - Implement alignment utility:**

   a) **Create `src/utils/tokenization.py`** with alignment function:
      ```python
      def align_labels_with_tokens(tokenized_inputs, word_labels, label2id):
          """
          Align word-level IOB2 labels with subword tokens.

          Strategy:
          - Use tokenizer's word_ids() to track which subword belongs to which word
          - First subword of each word gets the word's IOB2 label
          - Subsequent subwords of same word get -100 (ignored in loss)
          - Special tokens ([CLS], [SEP], [PAD]) get -100

          Args:
              tokenized_inputs: Output from tokenizer with return_offsets_mapping=True
              word_labels: List of IOB2 labels aligned with word-level tokens
              label2id: Dictionary mapping label strings to IDs

          Returns:
              aligned_labels: List of label IDs for each subword token
          """
          aligned_labels = []
          word_ids = tokenized_inputs.word_ids()  # Maps each token to its word index

          previous_word_idx = None
          for word_idx in word_ids:
              # Special tokens have word_idx = None
              if word_idx is None:
                  aligned_labels.append(-100)
              # First subword of a new word
              elif word_idx != previous_word_idx:
                  aligned_labels.append(label2id[word_labels[word_idx]])
              # Subsequent subwords of the same word
              else:
                  aligned_labels.append(-100)  # Ignore in loss

              previous_word_idx = word_idx

          return aligned_labels
      ```

   b) **Test alignment thoroughly:**
      - Load a multilingual model tokenizer (mBERT or XLM-R)
      - Test on 10 example sentences from training data
      - For each example:
        * Print original words and IOB2 tags
        * Print subword tokens
        * Print aligned labels
        * Verify span boundaries are preserved
        * Verify [CLS], [SEP] have label -100
      - Save test results to `experiments/results/tokenization_alignment_test.txt`

   c) **Integrate into data collation:**
      - When loading data for Task 2 training:
        * Tokenize text with `return_offsets_mapping=True`
        * Parse word-level IOB2 tags from dataset
        * Call `align_labels_with_tokens()` to get aligned labels
        * Create batch with input_ids, attention_mask, labels
      - Ensure data collator handles variable-length sequences correctly

   d) **Evaluation alignment:**
      - During evaluation, model outputs predictions for each subword
      - Need to convert back to word-level:
        * Use word_ids() to group subword predictions
        * Take first subword's prediction as the word's prediction
        * Ignore subwords with label -100
      - Compute metrics on word-level predictions vs word-level ground truth

   **Validation steps for this subtask:**
   - Print 10 alignment examples and manually verify correctness
   - Verify no span boundary errors (idiom spans preserved)
   - Verify special tokens have -100
   - Test on all 5 model tokenizers (AlephBERT, DictaBERT, mBERT, XLM-R)
   - Integration test: Load data, process batch, run forward pass - no errors

   **WITHOUT THIS ALIGNMENT, TASK 2 WILL NOT WORK!** The model will learn wrong label-token associations.

4. For Task 2 (Token Classification):
   - Use `AutoModelForTokenClassification`
   - Set num_labels=3 (O, B-IDIOM, I-IDIOM)
   - **Use alignment function from Task 3.5 in data collation**
   - Handle IOB2 label mapping
   - Train on training set
   - Validate on validation set
5. Implement checkpointing:
   - Save best model based on validation F1
   - Save training logs
   - Save final metrics

6. **Test training LOCALLY on PyCharm first (important!):**
   - Run on CPU or MPS (Mac GPU)
   - Very small subset: 100 samples from training data
   - 1 epoch only
   - Purpose: Verify code runs without errors, loss decreases, metrics computed correctly
   - **DO NOT skip this step** - finding bugs locally saves VAST.ai costs!
   - Command example:
     ```bash
     python src/idiom_experiment.py \
       --mode full_finetune \
       --config experiments/configs/training_config.yaml \
       --max_samples 100 \
       --num_epochs 1 \
       --device cpu
     ```

7. (Optional) Test on VAST.ai with larger subset:
   - After local test passes, optionally test on VAST.ai GPU
   - Larger subset: 500 samples, 2 epochs
   - Verify GPU acceleration works
   - This is optional - can skip to Mission 4.4 for full VAST.ai setup

**Validation:**
- Training runs without errors
- Loss decreases over epochs
- Validation metrics improve
- Best model saved correctly
- Checkpointing works
- Can resume training from checkpoint
- **FOR TASK 2: IOB2 alignment verified (print 10 examples during first epoch)**
- **FOR TASK 2: Evaluation uses word-level metrics (not subword-level)**

**Success Criteria:**
✅ Training pipeline complete
✅ Both tasks supported
✅ Checkpointing implemented
✅ Early stopping functional
✅ **IOB2 alignment utility created and tested (Task 3.5)**
✅ **Alignment examples printed and verified correct**
✅ Tested successfully

---

### Mission 4.3: Hyperparameter Optimization Setup

**Objective:** Implement Optuna for automated hyperparameter tuning

**Tasks:**
1. Install and import Optuna library:
   ```bash
   pip install optuna optuna-dashboard
   ```

2. Implement `--mode hpo` in `src/idiom_experiment.py`:

   **Add HPO mode to main function:**
   ```python
   def run_hpo(args):
       """
       Run Optuna hyperparameter optimization

       This function:
       1. Loads HPO config (hpo_config.yaml)
       2. Creates Optuna study
       3. For each trial (15 times):
          - Optuna suggests hyperparameters
          - Calls run_training() with those params
          - Returns validation F1 score
       4. Optuna picks next hyperparameters based on results
       5. Saves best hyperparameters after all trials
       """
       import optuna
       import yaml

       # Load HPO configuration
       with open(args.config, 'r') as f:
           hpo_config = yaml.safe_load(f)

       def objective(trial):
           """Optuna objective function - called once per trial"""

           # Optuna suggests hyperparameters from search space
           suggested_params = {
               'learning_rate': trial.suggest_categorical('learning_rate',
                   hpo_config['search_space']['learning_rate']),
               'batch_size': trial.suggest_categorical('batch_size',
                   hpo_config['search_space']['batch_size']),
               'num_epochs': trial.suggest_categorical('num_epochs',
                   hpo_config['search_space']['num_epochs']),
               'warmup_ratio': trial.suggest_categorical('warmup_ratio',
                   hpo_config['search_space']['warmup_ratio']),
               'weight_decay': trial.suggest_categorical('weight_decay',
                   hpo_config['search_space']['weight_decay']),
               'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps',
                   hpo_config['search_space']['gradient_accumulation_steps'])
           }

           # Train model with suggested hyperparameters
           # This calls the same training code from Mission 4.2!
           val_f1 = train_and_evaluate(
               model_name=args.model,
               task=args.task,
               hyperparameters=suggested_params,
               device=args.device
           )

           # Return validation F1 - Optuna will maximize this
           return val_f1

       # Create Optuna study
       study = optuna.create_study(
           study_name=f"{args.model}_{args.task}_hpo",
           direction='maximize',  # Maximize validation F1
           storage=f'sqlite:///experiments/results/optuna_{args.model}_{args.task}.db',
           load_if_exists=True  # Can resume if interrupted
       )

       # Run optimization
       n_trials = hpo_config['optuna']['n_trials']
       study.optimize(objective, n_trials=n_trials)

       # Save best hyperparameters
       best_params = study.best_params
       output_path = f"experiments/results/best_params_{args.model}_{args.task}.json"
       with open(output_path, 'w') as f:
           json.dump(best_params, f, indent=2)

       print(f"\n✅ HPO Complete!")
       print(f"Best F1: {study.best_value:.4f}")
       print(f"Best params saved to: {output_path}")
       print(f"Best params: {best_params}")
   ```

   **Key Points:**
   - The objective function CALLS your training code from Mission 4.2
   - Each trial = one complete training run
   - Optuna automatically picks next hyperparameters based on previous results
   - Can resume if interrupted (uses SQLite database)

3. Configure Optuna study:
   - Objective: Maximize validation F1
   - Number of trials: 10-15 per model
   - Pruning: Enable Successive Halving
   - Storage: SQLite database for persistence
4. Implement pruning callback (optional - for faster optimization):
   - Report validation metric after each epoch
   - Prune unpromising trials early using `optuna.integration.PyTorchLightningPruningCallback`

5. **Test HPO LOCALLY first (important!):**
   - Test on one model (e.g., AlephBERT-base) with just 3 trials
   - Use small subset (500 samples) to verify it works
   - Purpose: Verify Optuna integration works before running on VAST.ai
   - Command:
     ```bash
     # Local test (CPU is fine, just slower)
     python src/idiom_experiment.py \
       --mode hpo \
       --model onlplab/alephbert-base \
       --task sequence_classification \
       --config experiments/configs/hpo_config.yaml \
       --max_samples 500 \
       --device cpu
     ```
   - Should complete 3 trials and save best params

6. Verify best hyperparameters are saved:
   - Check file exists: `experiments/results/best_params_alephbert-base_sequence_classification.json`
   - Contains: learning_rate, batch_size, num_epochs, etc.

**Validation:**
- Optuna study runs successfully
- Trials complete without errors
- Best hyperparameters identified
- Results saved to database and JSON file
- Can visualize optimization history
- **Local test with 3 trials passes before moving to VAST.ai**

**Success Criteria:**
✅ Optuna integrated into idiom_experiment.py
✅ HPO mode implemented with full objective function
✅ Tested locally with 3 trials
✅ Pruning functional (optional)
✅ Best parameters identified and saved
✅ Ready for full HPO experiments on VAST.ai (Mission 4.5)

---

### Mission 4.4: VAST.ai Training Environment Setup

**Objective:** Setup VAST.ai instance for GPU training

**Tasks:**
1. Search for suitable VAST.ai instance:
   - GPU: RTX 3090 or RTX 4090 preferred
   - VRAM: >= 24GB
   - RAM: >= 32GB
   - Reliability: >98%
   - Price: Sort by $/hour ascending
2. Rent instance (hourly rental)
3. Connect via SSH
4. Setup environment on instance:
   - Install Python 3.9/3.10
   - Install all dependencies from `requirements.txt`
   - Or use Docker image (if created in Mission 1.5)
5. Upload code repository:
   - Clone from GitHub, or
   - Upload via SCP/SFTP
6. Download dataset from Google Drive:
   - Use `gdown` with file ID
   - Verify dataset integrity (row count)
7. Test training on small subset to verify GPU works

8. Setup results upload to Google Drive:

   **Option A (Simple - Recommended for beginners):**
   - After training completes, manually download results from VAST.ai:
     ```bash
     # On your local machine
     scp -P <vast-port> root@<vast-ip>:~/hebrew-idiom-detection/experiments/results/* ./local_results/
     ```
   - Then upload to Google Drive via browser interface
   - Simple but manual

   **Option B (Advanced - Automated with rclone):**
   - Install rclone on VAST.ai instance:
     ```bash
     curl https://rclone.org/install.sh | sudo bash
     ```
   - Configure Google Drive (one-time, interactive):
     ```bash
     rclone config
     # Choose: n (new remote)
     # Name: gdrive
     # Storage: drive (Google Drive)
     # Follow prompts for OAuth authentication
     ```
   - Create sync script `scripts/sync_to_gdrive.sh`:
     ```bash
     #!/bin/bash
     # Sync results to Google Drive
     echo "Syncing results to Google Drive..."
     rclone copy experiments/results/ gdrive:Hebrew_Idiom_Detection/results/ -v
     rclone copy experiments/logs/ gdrive:Hebrew_Idiom_Detection/logs/ -v
     echo "Sync complete!"
     ```
   - Make executable: `chmod +x scripts/sync_to_gdrive.sh`
   - Test: Upload one small file to verify it works
   - After each training run: `bash scripts/sync_to_gdrive.sh`

   **Option C (Simplest - gdown for download only):**
   - Use gdown to download from Google Drive to VAST.ai (already working from Mission 1.3)
   - After training, use VAST.ai's file browser to download results
   - Manually upload to Google Drive

   **Choose Option A or B** - Option A is simpler, Option B is more automated for multiple runs

**Validation:**
- Instance rented successfully
- SSH connection works
- Dependencies installed
- GPU detected and available
- Dataset downloaded correctly
- Can run training script
- Results can be synced to Google Drive

**Success Criteria:**
✅ VAST.ai instance configured
✅ Environment ready
✅ GPU functional
✅ Dataset accessible
✅ Training tested
✅ Google Drive sync method chosen and tested (Option A, B, or C)
✅ Ready for full experiments

---

### Mission 4.5: Hyperparameter Optimization for All Models

**Objective:** Run HPO for all 5 models on both tasks **ON VAST.AI**

**IMPORTANT:** This mission MUST run on VAST.ai GPU instance (Mission 4.4 setup required first)

**Tasks:**

1. **Run HPO studies on VAST.ai:**

   For each model-task combination, run:
   ```bash
   # On VAST.ai instance (already setup from Mission 4.4)
   python src/idiom_experiment.py \
     --mode hpo \
     --model <model-name> \
     --task <task-name> \
     --config experiments/configs/hpo_config.yaml \
     --device cuda
   ```

   **Example for one model-task:**
   ```bash
   python src/idiom_experiment.py \
     --mode hpo \
     --model onlplab/alephbert-base \
     --task sequence_classification \
     --config experiments/configs/hpo_config.yaml \
     --device cuda
   ```

2. **Option A: Run manually** (10 commands total)

   5 models: AlephBERT-base, AlephBERT-Gimmel, DictaBERT, mBERT, XLM-R
   2 tasks: sequence_classification, token_classification
   = 10 HPO studies total

   Each study runs 15 trials automatically (defined in hpo_config.yaml)

3. **Option B: Create batch script** (Recommended for convenience)

   Create `scripts/run_all_hpo.sh`:
   ```bash
   #!/bin/bash
   # Batch runner for all 10 HPO studies

   MODELS=("onlplab/alephbert-base" "bert-base-multilingual-cased" "xlm-roberta-base" "dicta-il/dictabert" "alephbert-gimmel")
   TASKS=("sequence_classification" "token_classification")

   for model in "${MODELS[@]}"; do
     for task in "${TASKS[@]}"; do
       echo "======================================"
       echo "Running HPO: Model=$model | Task=$task"
       echo "======================================"

       python src/idiom_experiment.py \
         --mode hpo \
         --model $model \
         --task $task \
         --config experiments/configs/hpo_config.yaml \
         --device cuda

       echo "✅ HPO complete for $model | $task"
       echo ""
     done
   done

   echo "All 10 HPO studies complete!"
   ```

   Then run:
   ```bash
   chmod +x scripts/run_all_hpo.sh

   # Use screen/tmux so it keeps running if SSH disconnects
   screen -S hpo
   bash scripts/run_all_hpo.sh
   # Detach: Ctrl+A then D
   # Reattach later: screen -r hpo
   ```

4. **Total workload:**
   - 10 studies × 15 trials = **150 training runs**
   - Each trial: ~20-30 minutes
   - Total time: **50-75 hours of GPU time**
   - Can run sequentially (one at a time) or parallel (multiple VAST.ai instances)

5. **Save all results:**
   - Best hyperparameters for each model-task: `experiments/results/best_params_<model>_<task>.json`
   - Optuna database: `experiments/results/optuna_<model>_<task>.db`
   - Sync to Google Drive after completion (use sync script from Mission 4.4)

6. **Analyze results:**
   - Visualization of hyperparameter importance (use Optuna dashboard)
   - Which hyperparameters matter most?
   - Different optimal values for different models?
   - Different optimal values for different tasks?

**Time & Cost Estimates:**
- Sequential (one study at a time): 50-75 hours, $20-30 on VAST.ai (~$0.40/hour)
- Parallel (2-3 instances): 20-30 hours, $24-36 total
- **Recommended:** Run sequentially with screen/tmux to save cost

**Validation:**
- All 10 HPO studies complete
- Best hyperparameters identified for each (10 JSON files)
- Results saved and documented
- Hyperparameter importance analyzed
- Results synced to Google Drive

**Success Criteria:**
✅ All 10 HPO studies completed on VAST.ai
✅ Best hyperparameters for all model-task combinations saved
✅ Optuna databases created and accessible
✅ Results documented and analyzed
✅ Insights extracted (which hyperparameters matter)
✅ Ready for Mission 4.6 (final training with best params)

---

### Mission 4.6: Final Training with Best Hyperparameters

**Objective:** Train all models with best hyperparameters from HPO

**Tasks:**
1. For each model and task:
   - Load best hyperparameters from HPO results
   - Train model on full training set
   - Validate on validation set
   - Save final model checkpoint to Google Drive
   - Evaluate on test set
   - Save all metrics and predictions

   **IMPORTANT: Create Batch Execution Script (Makes Running 30 Experiments Easy!)**

   Instead of manually running 30 commands, create `scripts/run_all_experiments.sh`:

   ```bash
   #!/bin/bash
   # Batch runner for all 30 training experiments
   # Usage: bash scripts/run_all_experiments.sh

   # Define arrays
   MODELS=("alephbert-base" "mbert" "xlm-roberta-base" "dicta-il/dictabert" "alephbert-gimmel")
   TASKS=("sequence_classification" "token_classification")
   SEEDS=(42 123 456)

   # Loop through all combinations
   for model in "${MODELS[@]}"; do
     for task in "${TASKS[@]}"; do
       for seed in "${SEEDS[@]}"; do
         echo "======================================"
         echo "Running: Model=$model | Task=$task | Seed=$seed"
         echo "======================================"

         # Create unique output directory
         OUTPUT_DIR="experiments/results/${model}_${task}_seed${seed}"

         # Run training
         python src/idiom_experiment.py \
           --mode full_finetune \
           --model $model \
           --task $task \
           --seed $seed \
           --config experiments/configs/training_config.yaml \
           --output_dir $OUTPUT_DIR \
           --device cuda

         # Check if training succeeded
         if [ $? -eq 0 ]; then
           echo "✅ Success: $model | $task | seed=$seed"

           # Sync results to Google Drive (if using rclone - Option B from Mission 4.4)
           # Uncomment if using automated sync:
           # bash scripts/sync_to_gdrive.sh
         else
           echo "❌ FAILED: $model | $task | seed=$seed"
           # Optional: continue anyway or exit
           # exit 1
         fi

         echo ""
       done
     done
   done

   echo "=========================================="
   echo "All 30 experiments complete!"
   echo "=========================================="
   ```

   **Setup:**
   - Create the file: `touch scripts/run_all_experiments.sh`
   - Make executable: `chmod +x scripts/run_all_experiments.sh`
   - Edit model IDs to match your actual model names

   **Run all experiments:**
   ```bash
   # On VAST.ai (or local with GPU)
   bash scripts/run_all_experiments.sh

   # Runs unattended - can use screen/tmux to keep running if SSH disconnects:
   screen -S training
   bash scripts/run_all_experiments.sh
   # Detach: Ctrl+A then D
   # Reattach: screen -r training
   ```

   **Benefits:**
   - ✅ Run once, walks away
   - ✅ Consistent naming (results organized by model/task/seed)
   - ✅ Automatic error detection
   - ✅ Can sync to Google Drive after each run
   - ✅ Much easier than 30 manual commands!

2. Implement cross-seed validation (PRD Section 5.3):
   - Seeds: 42, 123, 456
   - Train each model 3 times with different seeds
   - Calculate mean ± standard deviation for metrics
3. Total: 30 training runs (5 models × 2 tasks × 3 seeds)
4. Save all results:
   - Model checkpoints (Google Drive)
   - Test metrics for each run
   - Aggregated results (mean ± std)
   - Training logs
5. Create results summary table

**Validation:**
- All 30 training runs complete
- All model checkpoints saved
- Test metrics calculated for all runs
- Mean and std deviation computed
- Results table created

**Success Criteria:**
✅ Batch execution script created (scripts/run_all_experiments.sh)
✅ 30 models trained successfully
✅ All checkpoints saved to Google Drive
✅ Test metrics: F1 > 80% (Task 1), F1 > 75% (Task 2)
✅ Cross-seed validation complete
✅ Results documented with mean ± std

---

### Mission 4.7: Fine-Tuning Results Analysis

**Objective:** Comprehensive analysis of fine-tuning results

**Tasks:**
1. Create comparison visualizations:
   - Bar chart: All models on Task 1 (with error bars)
   - Bar chart: All models on Task 2 (with error bars)
   - Learning curves for best model
   - Confusion matrices for all models
2. Statistical testing (PRD Section 9.2):
   - Paired t-tests between models
   - Bonferroni correction for multiple comparisons
   - Identify significant differences (α = 0.05)
3. Compare fine-tuned vs zero-shot:
   - Improvement for each model
   - Which models benefit most from fine-tuning?
4. Analyze by model type:
   - Hebrew-specific vs multilingual
   - Task 1 vs Task 2 performance
5. Error analysis:
   - Sample 50 errors from best model
   - Categorize error types
   - Identify difficult idioms
6. Document all findings in `experiments/results/finetuning_analysis.md`

**Validation:**
- All visualizations created
- Statistical tests completed
- Significant differences identified
- Error analysis thorough
- Findings documented clearly

**Success Criteria:**
✅ Comprehensive visualizations
✅ Statistical testing complete
✅ Best model identified for each task
✅ Error patterns documented
✅ Ready for LLM comparison

---

## PHASE 5: LLM EVALUATION (Week 7)

### Mission 5.1: LLM Selection and API Setup

**Objective:** Choose LLM and setup API access

**Tasks:**
1. Select one LLM from options (PRD Section 4.2):
   - Option 1: Llama 3.1 70B (via Together AI or Azure)
   - Option 2: Mistral Large (via Azure)
   - Option 3: GPT-3.5-Turbo (via Azure OpenAI)
   - Consider: Cost, Hebrew support, API availability
2. Create account and setup API access:
   - Get API key
   - Test API connection
   - Understand rate limits
   - Check pricing
3. Estimate total cost:
   - Test set samples × 2 tasks = API calls needed
   - Add few-shot examples: ~3-5 per call
   - Calculate total tokens and cost
4. Create `.env` file with API credentials (DO NOT commit to Git)
5. Test API with sample Hebrew sentence

**Validation:**
- API access working
- Can send request and get response
- Hebrew text handled correctly
- Cost estimated and acceptable
- API key secured in `.env`

**Success Criteria:**
✅ LLM selected
✅ API access configured
✅ Cost estimated (<$100)
✅ Test call successful
✅ Ready for evaluation

---

### Mission 5.2: Prompting Strategy Design

**Objective:** Design prompts for zero-shot and few-shot evaluation

**Tasks:**
1. Design Zero-Shot Prompt (PRD Section 6.1):
   - Clear instruction in English or Hebrew
   - Specify task: classify literal vs figurative
   - Include expression and sentence
   - Request JSON output format
   - Test on 5 sample sentences
2. Design Few-Shot Prompt (3-5 examples):
   - Select diverse examples:
     - Mix of literal and figurative
     - Different idioms
     - Different sentence lengths
   - Format examples clearly
   - Add task instruction after examples
   - Test on 5 sample sentences
3. Design Chain-of-Thought Prompt (optional):
   - Step-by-step reasoning instruction
   - Test on 5 sample sentences
4. For Task 2 (Span Detection):
   - Adapt prompts to request idiom span
   - Test on 5 sample sentences
5. Compare prompt variations:
   - Which gets best results on small test?
   - Which is most reliable (consistent output format)?
6. Save final prompts to `experiments/configs/llm_prompts.json`

**Validation:**
- All prompt types created
- Tested on sample sentences
- Output format consistent
- JSON parsing works
- Best prompts selected

**Success Criteria:**
✅ Zero-shot prompt designed
✅ Few-shot prompt designed
✅ Prompts tested and validated
✅ Output parsing reliable
✅ Ready for full evaluation

---

### Mission 5.2.1 (OPTIONAL - HIGH PRIORITY): Enhanced Few-Shot Design and Documentation

**Objective:** Rigorously design, document, and test few-shot prompting strategy to ensure reproducibility and avoid data leakage

**Why this is important (Senior Researcher Recommendation):**
Few-shot prompting is a critical component of LLM evaluation, but it's often poorly documented in research papers. Reviewers will ask:
1. Which exact examples did you use for few-shot prompting?
2. How were these examples selected?
3. Are the few-shot examples from the test set? (data leakage!)
4. Did you try different examples and report the best? (cherry-picking!)
5. Can others reproduce your results?

**Tasks:**

**1. Few-Shot Example Selection Strategy:**

Design and document a principled selection strategy:

a) **Selection Pool:**
```python
# CRITICAL: Never use test set examples!
# Use training or validation set only
selection_pool = train_set  # or validation_set

# Document this decision clearly:
# "Few-shot examples are selected from the training set to avoid data leakage.
#  The test set is never used for prompt design or example selection."
```

b) **Selection Criteria (choose ONE and document):**

**Option A: Stratified Random Sampling (Recommended)**
```python
# Select N examples (e.g., 3-5) using stratified sampling
# Ensures balanced representation of:
# - Both labels (literal and figurative)
# - Different idioms
# - Different sentence lengths
# - Different sentence types (question, declarative)

def select_few_shot_examples(train_df, n_examples=5, seed=42):
    """
    Select few-shot examples using stratified sampling.

    Strategy:
    - n_examples/2 literal, n_examples/2 figurative
    - Different idioms (no duplicates)
    - Medium sentence length (10-20 tokens)
    - Mix of sentence types
    """
    np.random.seed(seed)

    # Select literal examples
    literal_pool = train_df[
        (train_df['label_2'] == 0) &
        (train_df['sentence_length'].between(10, 20))
    ]
    literal_examples = literal_pool.sample(n=n_examples//2)

    # Select figurative examples
    figurative_pool = train_df[
        (train_df['label_2'] == 1) &
        (train_df['sentence_length'].between(10, 20)) &
        (~train_df['expression'].isin(literal_examples['expression']))
    ]
    figurative_examples = figurative_pool.sample(n=n_examples//2)

    return pd.concat([literal_examples, figurative_examples])
```

**Option B: Representative Examples (Manual Selection)**
```python
# Manually select N examples that represent:
# - Clear literal usage (easy example)
# - Clear figurative usage (easy example)
# - Ambiguous literal (hard example)
# - Ambiguous figurative (hard example)
# - One question sentence

# Document exact IDs:
few_shot_ids = [123, 456, 789, 1011, 1213]
few_shot_examples = train_df[train_df['id'].isin(few_shot_ids)]
```

**Option C: Centroids (Embedding-Based)**
```python
# Use sentence embeddings to find representative examples
# Select examples closest to cluster centroids
# More sophisticated but harder to explain

from sklearn.cluster import KMeans
# Embed all training sentences
# Cluster into K clusters
# Select example closest to each centroid
```

**2. Prompt Variation Testing:**

Test multiple prompt variations systematically:

```python
# Define prompt variations
prompt_templates = {
    "version_1_english": """
Task: Classify if the idiom is used literally or figuratively.

Examples:
{few_shot_examples}

Sentence: {test_sentence}
Idiom: {idiom}
Classification: """,

    "version_2_hebrew": """
משימה: סיווג שימוש בביטוי - מילולי או פיגורטיבי

דוגמאות:
{few_shot_examples}

משפט: {test_sentence}
ביטוי: {idiom}
סיווג: """,

    "version_3_cot": """
Task: Classify if the idiom is used literally or figuratively.

Examples:
{few_shot_examples}

Now classify this:
Sentence: {test_sentence}
Idiom: {idiom}

Let's think step by step:
1. What is the literal meaning of the idiom?
2. What is the figurative meaning?
3. Which meaning fits the context?

Classification: """
}

# Test each on validation set (50 samples)
for name, template in prompt_templates.items():
    val_f1 = evaluate_prompt(template, validation_set.head(50))
    print(f"{name}: F1 = {val_f1}")

# Select best performing prompt
best_prompt = select_best(results)
```

**3. Documentation:**

Create `experiments/configs/llm_few_shot_documentation.md`:

```markdown
# Few-Shot Prompting Documentation

## Example Selection Strategy
- **Pool:** Training set only (NO test set leakage)
- **Method:** Stratified random sampling with seed=42
- **N examples:** 5 (3 figurative, 2 literal)
- **Selection criteria:**
  - Different idioms (no duplicates)
  - Medium sentence length (10-20 tokens)
  - Mix of sentence types

## Selected Examples (IDs from train.csv)
1. ID=123: [sentence] - Label: Literal
2. ID=456: [sentence] - Label: Figurative
3. ID=789: [sentence] - Label: Figurative
4. ID=1011: [sentence] - Label: Literal
5. ID=1213: [sentence] - Label: Figurative

## Prompt Template
[Include exact prompt template used]

## Prompt Selection Process
- Tested 3 prompt variations on validation set (50 samples)
- Selected version_2_hebrew based on highest F1: 0.78
- Did NOT iterate on test set (avoided overfitting)

## Reproducibility
- Seed: 42
- Selection pool: data/splits/train.csv
- Example IDs documented above
- Prompt template in experiments/configs/llm_prompts.json
```

**4. Save Artifacts:**
1. Save selected few-shot examples to `experiments/configs/few_shot_examples.json`:
   ```json
   {
     "selection_strategy": "stratified_random",
     "seed": 42,
     "n_examples": 5,
     "examples": [
       {
         "id": 123,
         "text": "...",
         "expression": "...",
         "label": 0,
         "label_name": "literal"
       },
       ...
     ]
   }
   ```

2. Save prompt templates to `experiments/configs/llm_prompts.json`
3. Save prompt selection results to `experiments/results/llm_prompt_selection.json`

**5. Validation Testing:**

Before full evaluation, test on small validation subset:
```python
# Test on 20 validation samples
# Verify:
# 1. Few-shot examples are from training set (not test!)
# 2. Prompt works consistently
# 3. Output format parseable
# 4. Performance reasonable (>60% accuracy)

# Check for data leakage
assert set(few_shot_ids).isdisjoint(set(test_ids)), "Data leakage detected!"
```

**Validation:**
- Few-shot example selection strategy documented
- Examples selected from training set only (NO test set leakage verified)
- Multiple prompt variations tested
- Best prompt selected based on validation performance
- All artifacts saved and versioned
- Selection process reproducible (seed fixed)
- Documentation complete and paper-ready

**Success Criteria:**
✅ Few-shot examples selected using principled strategy
✅ NO data leakage (test set not used)
✅ Example IDs and exact sentences documented
✅ Prompt variations tested systematically
✅ Best prompt selected transparently
✅ Full documentation for paper's methodology section
✅ Reproducible by other researchers
✅ Reviewers can verify no cherry-picking

**What to include in paper:**
- "Few-shot examples were selected from the training set using stratified random sampling (seed=42) to ensure balanced representation while avoiding test set leakage."
- "We tested 3 prompt variations on a validation subset (50 samples) and selected the Hebrew-language prompt based on highest F1 score."
- "The exact few-shot examples and prompt templates are available in our code repository for reproducibility."

**Time Estimate:** 3-4 hours

**Recommendation:** This is HIGH PRIORITY for publication quality and reproducibility. Many papers get rejected due to poor LLM evaluation methodology.

---

### Mission 5.3: LLM Evaluation Script

**Objective:** Create script for automated LLM evaluation

**Tasks:**
1. Create `src/llm_evaluation.py` script
2. Implement functions:
   - Load test set
   - Format prompt for each sample
   - Send API request with retry logic
   - Parse JSON response
   - Extract prediction
   - Handle errors gracefully
3. Implement rate limiting:
   - Respect API rate limits
   - Add delays between requests
   - Batch requests if supported
4. Implement result saving:
   - Save predictions for each sample
   - Save raw LLM responses
   - Calculate metrics
   - Track API costs
5. Add progress tracking:
   - Progress bar
   - Time estimation
   - Cost tracking
6. Test on small subset (20 samples) first

**Validation:**
- Script runs without errors
- API calls successful
- Responses parsed correctly
- Predictions saved
- Metrics calculated
- Cost tracking accurate

**Success Criteria:**
✅ Evaluation script complete
✅ Error handling robust
✅ Rate limiting implemented
✅ Cost tracking working
✅ Tested successfully

---

### Mission 5.4: LLM Evaluation Execution

**Objective:** Run LLM evaluation on full test set

**IMPORTANT:** Run this mission **LOCALLY on PyCharm** (not VAST.ai)
- LLM evaluation only makes API calls - no GPU needed!
- Can run on your local machine (CPU is fine)
- Saves VAST.ai costs

**Tasks:**
1. Task 1: Sentence Classification
   - Zero-shot evaluation:
     - Run on full test set
     - Save predictions and metrics
     - Track cost and latency
   - Few-shot evaluation:
     - Run on full test set
     - Save predictions and metrics
     - Track cost and latency
   - Compare zero-shot vs few-shot
2. Task 2: Token Classification (Span Detection)
   - Zero-shot evaluation:
     - Run on full test set
     - Parse span predictions
     - Convert to IOB2 format
     - Calculate metrics
   - Few-shot evaluation:
     - Run on full test set
     - Calculate metrics
3. Save all results:
   - Predictions (CSV format)
   - Metrics summary (JSON)
   - Cost report
   - Latency statistics
4. Manual review of 50 responses:
   - Evaluate reasoning quality (1-5 scale)
   - Note interesting insights
   - Identify error patterns

**Validation:**
- All evaluations complete
- Predictions saved
- Metrics calculated correctly
- Cost within budget
- Manual review documented

**Success Criteria:**
✅ LLM evaluated on both tasks
✅ Zero-shot and few-shot results
✅ Cost < $100
✅ Metrics competitive with fine-tuned models (within 5-10%)
✅ Results documented

---

### Mission 5.5: LLM vs Fine-Tuned Comparison

**Objective:** Compare LLM performance with fine-tuned models

**Tasks:**
1. Create comprehensive comparison:
   - Task 1 F1: LLM vs best fine-tuned model
   - Task 2 F1: LLM vs best fine-tuned model
   - Zero-shot LLM vs zero-shot encoder models
   - Few-shot LLM vs fine-tuned encoder models
2. Create comparison visualizations:
   - Bar chart: All approaches side-by-side
   - Performance vs cost plot
   - Performance vs latency plot
3. Analyze trade-offs:
   - Accuracy vs cost
   - Accuracy vs development time
   - Accuracy vs inference speed
4. Document findings:
   - When is LLM better?
   - When is fine-tuning better?
   - Cost-benefit analysis
5. Create presentation-ready tables and figures

**Validation:**
- All comparisons complete
- Visualizations created
- Trade-offs analyzed
- Findings documented clearly
- Ready for paper inclusion

**Success Criteria:**
✅ Comprehensive comparison complete
✅ Visualizations created
✅ Trade-offs documented
✅ Clear recommendations
✅ Results ready for paper

---

## PHASE 6: ABLATION STUDIES & INTERPRETABILITY (Week 8)

### Mission 6.1: Word/Token Importance Analysis

**Objective:** Analyze which words/tokens are most important for model predictions using attention and gradient-based methods

**Tasks:**
1. **Select Cases for Analysis:**
   - Identify 10-15 difficult examples:
     - High-confidence errors (model very confident but wrong)
     - Low-confidence correct predictions (model uncertain but right)
     - Specific idioms that are frequently misclassified
     - Examples from test set expressions (the 6 specific expressions)
   - Include mix of literal and figurative examples
   - Include different sentence types (questions, declarative)
2. **Implement Importance Analysis Methods:**
   - Attention-based importance:
     - Extract attention weights from model
     - Aggregate across layers and heads
     - Identify tokens with highest attention
   - Gradient-based importance:
     - Compute input gradients
     - Use integrated gradients or similar technique
     - Identify tokens with highest gradient magnitude
   - Library suggestions: Use `captum` (PyTorch interpretability) or `transformers-interpret`
3. **Analyze Patterns:**
   - For each difficult case:
     - Which tokens does model focus on?
     - Does model attend to idiom tokens?
     - Does model attend to context words?
     - Different patterns for correct vs incorrect predictions?
   - Compare idiom token importance vs context token importance
   - Compare literal vs figurative examples
4. **Visualizations:**
   - Create heatmap visualizations showing token importance
   - Highlight idiom spans in different color
   - Use color intensity for importance scores
   - Save as publication-ready figures
   - Create for at least 10 representative examples
5. **Document Findings:**
   - Which tokens matter most for predictions?
   - Does model learn to focus on idioms?
   - What context clues does model use?
   - Different patterns for errors vs correct predictions?
   - Save analysis to `experiments/results/interpretability_analysis.md`

**Validation:**
- Importance scores computed for all selected examples
- Attention and gradient methods both implemented
- Heatmap visualizations created and clear
- Patterns identified and documented
- At least 10 visualizations saved to `paper/figures/interpretability/`

**Success Criteria:**
✅ Token importance analysis complete for 10-15 cases
✅ Multiple importance methods implemented
✅ Visualizations created and publication-ready
✅ Clear patterns identified
✅ Insights documented for paper inclusion

---

### Mission 6.2: Frozen Backbone Comparison

**Objective:** Compare full fine-tuning vs frozen backbone (optional)

**Note:** This requires training - can run on VAST.ai or locally (faster on GPU)

**Tasks:**
1. Select best-performing model from fine-tuning phase
2. Train with frozen backbone:
   - Freeze all transformer layers
   - Train only classification head
   - Use best hyperparameters from Mission 4.5 (not running HPO again)
   - Train on both tasks
3. Compare with full fine-tuning:
   - Performance difference
   - Training time difference
   - Which layers matter most?
4. Document findings:
   - Is full fine-tuning necessary?
   - Cost-benefit analysis

**Note:** This is optional - only if time permits

**Validation:**
- Frozen backbone training complete
- Comparison with full fine-tuning done
- Training time measured
- Performance gap documented

**Success Criteria:**
✅ Frozen backbone evaluated
✅ Comparison complete
✅ Insights documented
✅ Recommendations clear

---

### Mission 6.3: Hyperparameter Sensitivity Analysis

**Objective:** Analyze which hyperparameters matter most

**Tasks:**
1. Use Optuna results from Mission 4.5
2. For each hyperparameter:
   - Visualize relationship with validation F1
   - Calculate importance score
   - Identify optimal range
3. Create visualizations:
   - Hyperparameter importance plot
   - Learning rate vs F1
   - Batch size vs F1
   - Epochs vs F1
4. Analyze patterns:
   - Different for different models?
   - Different for different tasks?
   - Generalizable insights?
5. Document recommendations for future work

**Validation:**
- All hyperparameters analyzed
- Importance scores calculated
- Visualizations created
- Insights documented
- Recommendations clear

**Success Criteria:**
✅ Sensitivity analysis complete
✅ Important hyperparameters identified
✅ Visualizations created
✅ Recommendations documented
✅ Useful for future research

---

### Mission 6.4: Data Size Impact Analysis

**Objective:** Analyze performance vs training data size

**Note:** This requires training multiple models - **recommend running on VAST.ai** for speed

**Tasks:**
1. Create reduced training sets:
   - 10% of training data (by expressions)
   - 25% of training data (by expressions)
   - 50% of training data (by expressions)
   - 100% of training data (full)
   - Maintain expression-based split (no data leakage)
   - Maintain stratification
2. Select best model from fine-tuning
3. Train on each data size:
   - Use best hyperparameters from Mission 4.5 (not running HPO again)
   - Train on both tasks
   - Evaluate on same test set
4. Create learning curve:
   - Plot: Training size vs Test F1
   - For both tasks
   - With error bars (cross-seed validation)
5. Analyze:
   - Diminishing returns point
   - How much data is needed?
   - Compare Task 1 vs Task 2 data efficiency
6. Document findings and implications

**Note:** This is optional - only if time permits

**Validation:**
- All data sizes evaluated
- Learning curves created
- Analysis complete
- Findings documented
- Implications clear

**Success Criteria:**
✅ Data size impact analyzed
✅ Learning curves created
✅ Optimal data size identified
✅ Insights documented
✅ Recommendations for future datasets

---

## PHASE 7: COMPREHENSIVE ANALYSIS (Week 9)

### Mission 7.1: Error Analysis and Categorization

**Objective:** Deep dive into errors across all models with interpretability insights

**Tasks:**
1. Collect errors from all models:
   - False Positives (Literal → Figurative)
   - False Negatives (Figurative → Literal)
   - High-confidence errors (probability > 0.8)
   - Low-confidence errors (probability 0.4-0.6)
2. Sample 100 errors for manual analysis
3. Categorize errors by:
   - Error type (FP vs FN)
   - Idiom type
   - Sentence length
   - Sentence type (question, declarative, etc.)
   - Context complexity
   - Idiom frequency in training data
4. Identify patterns:
   - Which idioms are hardest?
   - What contexts cause confusion?
   - Systematic model weaknesses?
   - Do questions have higher error rate?
5. For Task 2 errors:
   - Boundary errors (B/I confusion)
   - Missing idioms (predicted all O)
   - Extra spans (false idiom detection)
6. **Connect to interpretability analysis:**
   - For difficult cases, reference token importance analysis
   - Explain WHY model made errors based on attention patterns
   - Include token importance visualizations for key error examples
7. Create error analysis report with examples and visualizations

**OPTIONAL (HIGH PRIORITY): Deeper Error Analysis**

If you want to make your error analysis more comprehensive and publication-ready, add these optional deeper analyses:

**7a. Idiom Difficulty Ranking:**
```python
# Rank idioms by error rate across all models
# For each idiom:
#   - Error rate (% misclassified)
#   - Average model confidence on errors
#   - Frequency in training data
#   - Average sentence length for this idiom
#   - Ambiguity score (if both literal/figurative common)

# Create difficulty taxonomy:
difficult_idioms = {
    "very_hard": [],      # >50% error rate
    "hard": [],           # 30-50% error rate
    "medium": [],         # 15-30% error rate
    "easy": []            # <15% error rate
}

# Analyze what makes idioms hard:
# - Rare idioms (low training frequency)?
# - Ambiguous idioms (both usages common)?
# - Multi-word idioms (longer spans)?
# - Context-dependent idioms?

# Visualize: Heatmap of idiom difficulty across models
```

**7b. Sentence Length and Complexity Impact:**
```python
# Analyze error rate vs sentence characteristics:

# 1. Sentence length impact
sentence_length_bins = [0, 5, 10, 15, 20, 30, 100]
for bin in bins:
    error_rate = calculate_error_rate(sentences_in_bin)
    plot(bin_midpoint, error_rate)

# 2. Sentence type impact (from Mission 2.4 analysis)
# - Declarative: 92.23% → error rate?
# - Question: 7.12% → error rate?
# - Exclamatory: 0.65% → error rate?
# Are questions harder?

# 3. Context complexity
# - Number of clauses
# - Presence of negation
# - Passive vs active voice
# - Multiple idioms in same sentence?

# Create scatter plot: sentence_length vs error_confidence
# Color by sentence type
```

**7c. Cross-Lingual Error Pattern Analysis:**
```python
# Compare Hebrew-specific vs multilingual models:

# For each error type:
#   - AlephBERT errors vs mBERT errors
#   - Which idioms confuse multilingual models more?
#   - Do Hebrew-specific models fail on different patterns?

# Hypothesis testing:
# - Do multilingual models struggle with Hebrew morphology?
# - Do Hebrew models overfit to training idioms?
# - Are there systematic differences in error types?

# Create Venn diagram:
# - Errors only AlephBERT makes
# - Errors only mBERT makes
# - Errors both make (systematic difficulty)

# Insight: Where does Hebrew pretraining help most?
```

**7d. Error Progression Analysis:**
```python
# Track how errors change across training:

# If you saved checkpoints during training:
# - Which errors are corrected first? (easy patterns)
# - Which errors persist? (hard patterns)
# - Which new errors appear? (overfitting?)

# Compare zero-shot → fine-tuned errors:
# - Which zero-shot errors are fixed by fine-tuning?
# - Which errors are introduced by fine-tuning?
# - Are models learning generalizable patterns or memorizing?

# Visualization: Error type distribution evolution
```

**7e. Contextual Ambiguity Analysis:**
```python
# Deep dive into ambiguous cases:

# Select 20 examples where:
# - Model confidence is low (0.4-0.6)
# - Multiple models disagree
# - Annotator confidence might be low (if available)

# For each case:
# - Manual linguistic analysis
# - Could it be both literal AND figurative?
# - Is the gold label debatable?
# - What contextual clues determine the label?

# Create "ambiguity spectrum":
# - Clear literal → Ambiguous → Clear figurative
# - Where do errors concentrate?

# Insight: Is the task inherently ambiguous?
#          Should we use soft labels instead?
```

**Implementation:**
- Create `src/deep_error_analysis.py` for advanced analysis
- Save results to `experiments/results/deep_error_analysis/`
- Create visualizations for paper
- Document findings in `experiments/results/deep_error_analysis.md`

**Expected Insights:**
- Which factors predict errors (length, type, idiom difficulty)
- Where Hebrew pretraining helps most
- Task ambiguity and annotation quality
- Model-specific failure modes
- Recommendations for future dataset/model improvements

**Time Estimate:** 6-8 hours (but provides rich paper content)

**Recommendation:** Choose 2-3 of these deeper analyses based on:
- What story you want to tell in the paper
- What gaps you found in preliminary error analysis
- What would be most novel/interesting to readers

---

**Standard Validation:**
- 100 errors analyzed
- Categories clearly defined
- Patterns identified
- Interpretability insights integrated
- Examples documented with visualizations
- Report comprehensive

**Success Criteria (Standard):**
✅ Error analysis complete
✅ Error categories identified
✅ Patterns documented
✅ Interpretability analysis integrated
✅ Examples with visualizations included
✅ Insights actionable

**Success Criteria (If Optional Deeper Analysis Done):**
✅ All standard criteria met
✅ 2-3 deeper analyses completed
✅ Idiom difficulty ranking created
✅ Sentence complexity impact analyzed
✅ Cross-lingual patterns identified
✅ Rich insights for paper discussion section
✅ Publication-ready visualizations created

---

### Mission 7.2: Model Comparison and Statistical Testing

**Objective:** Rigorous statistical comparison of all models

**Tasks:**
1. Compile all results:
   - Zero-shot baseline (5 models × 2 tasks)
   - Fine-tuned models (5 models × 2 tasks × 3 seeds)
   - LLM results (1 model × 2 tasks × 2 approaches)
2. Statistical testing (PRD Section 9.2):
   - Paired t-tests between models
   - Significance level: α = 0.05
   - Bonferroni correction for multiple comparisons
   - Report p-values
3. Create results tables:
   - Mean ± Std for all models
   - Statistical significance markers
   - Rank models by performance
4. Analyze by dimensions:
   - Hebrew-specific vs multilingual
   - Zero-shot vs fine-tuned improvement
   - Task 1 vs Task 2 difficulty
   - Best overall model
5. Create final comparison visualization:
   - All models, all approaches, both tasks
   - Publication-ready figure

**Validation:**
- All statistical tests complete
- Results tables created
- Significance properly marked
- Visualizations professional
- Ready for paper inclusion

**Success Criteria:**
✅ Statistical testing complete
✅ Significant differences identified
✅ Results tables publication-ready
✅ Best model identified for each task
✅ Comprehensive comparison done

---

### Mission 7.3: Cross-Task Analysis

**Objective:** Analyze relationship between Task 1 and Task 2 performance

**Tasks:**
1. For each model:
   - Compare Task 1 F1 vs Task 2 F1
   - Which task is harder?
   - Correlation between tasks?
2. Analyze predictions:
   - When Task 1 is wrong, is Task 2 also wrong?
   - Can correct Task 2 span help Task 1?
   - Joint modeling potential?
3. Case study analysis:
   - 20 examples where both tasks correct
   - 20 examples where both tasks wrong
   - 20 examples where only one task correct
   - What patterns emerge?
4. Document findings:
   - Task interdependence
   - Which task is fundamental?
   - Implications for multi-task learning

**Validation:**
- Cross-task analysis complete
- Correlations calculated
- Case studies documented
- Patterns identified
- Findings clear

**Success Criteria:**
✅ Task relationship analyzed
✅ Correlations computed
✅ Case studies complete
✅ Insights documented
✅ Implications for future work

---

### Mission 7.4: Visualization and Figure Creation

**Objective:** Create all figures for paper and thesis

**Tasks:**
1. Dataset figures:
   - Label distribution
   - Sentence length distribution
   - Sentence type distribution
   - Idiom frequency distribution
   - Example annotations
2. Results figures:
   - Zero-shot baseline comparison (both tasks)
   - Fine-tuned models comparison (both tasks)
   - LLM vs fine-tuned comparison
   - Learning curves (if ablation done)
   - Hyperparameter importance (if ablation done)
3. Analysis figures:
   - Confusion matrices (best models)
   - Error category distribution
   - Task 1 vs Task 2 correlation
   - Statistical significance heatmap
4. **Interpretability figures:**
   - Token importance heatmaps (10-15 examples)
   - Attention visualization for key examples
   - Error case visualizations with importance scores
5. All figures:
   - Publication quality (300 DPI minimum)
   - Consistent style and colors
   - Clear labels and legends
   - Saved in multiple formats (PNG, PDF)
6. Save to `paper/figures/` with descriptive names

**Validation:**
- All figures created
- High resolution (300+ DPI)
- Consistent styling
- Labels clear and readable
- Interpretability visualizations included
- Saved in required formats

**Success Criteria:**
✅ 20-25 figures created (including interpretability)
✅ Publication quality
✅ Consistent style
✅ Well-organized
✅ Ready for paper inclusion

---

### Mission 7.5: Results Tables Creation

**Objective:** Create all tables for paper and thesis

**Tasks:**
1. Dataset statistics table (PRD Section 2.3)
   - Include sentence type distribution
2. Model specifications table (PRD Section 4.1)
3. Hyperparameter ranges table (PRD Section 5.1)
4. Zero-shot baseline results table
5. Fine-tuned models results table:
   - All models, both tasks
   - Mean ± Std
   - Statistical significance markers
6. LLM evaluation results table:
   - Zero-shot and few-shot
   - Both tasks
   - Cost and latency
7. Model comparison summary table:
   - Best model per task
   - Performance vs cost
8. Error analysis summary table
9. Format all tables:
   - LaTeX format for paper
   - Markdown format for README
   - CSV format for records
10. Save to `paper/tables/`

**Validation:**
- All tables created
- Correct data and formatting
- LaTeX compiles correctly
- Consistent styling
- Well-organized

**Success Criteria:**
✅ 8-10 tables created
✅ Multiple formats (LaTeX, Markdown, CSV)
✅ Accurate data
✅ Professional formatting
✅ Ready for paper

---

## PHASE 8: PAPER & DOCUMENTATION (Week 10-11)

### Mission 8.1: Dataset Documentation

**Objective:** Create comprehensive dataset documentation for release

**Tasks:**
1. Create `data/README.md` with:
   - Dataset description
   - Schema documentation
   - Statistics summary (including sentence types)
   - Split information (expression-based strategy explained)
   - Test set expressions documented
   - Usage examples
   - Citation information
2. Create annotation guidelines document:
   - How annotations were created
   - IOB2 tagging rules
   - Quality control process
   - Examples
3. Create dataset card for HuggingFace:
   - Dataset description
   - Languages
   - Data fields
   - Split methodology (expression-based to avoid leakage)
   - Licensing
   - Citation
4. Prepare release files:
   - Train/val/test splits
   - Full dataset with split column
   - Expression-to-split mapping file
   - Statistics file
   - README
5. Test dataset loading with HuggingFace datasets library

**Validation:**
- Documentation complete
- Split strategy clearly explained
- Dataset card ready
- All files prepared
- HuggingFace format validated
- Ready for upload

**Success Criteria:**
✅ Comprehensive documentation
✅ Dataset card complete
✅ HuggingFace format ready
✅ Split methodology documented
✅ All files prepared
✅ Ready for public release

---

### Mission 8.2: Code Documentation

**Objective:** Document all code for reproducibility

**Tasks:**
1. Update main `README.md`:
   - Project description
   - Installation instructions
   - Usage examples
   - VAST.ai setup guide
   - Google Drive integration
   - Citation
2. Document all Python scripts:
   - Docstrings for all functions
   - Type hints
   - Usage examples
   - Command-line arguments
3. Create usage guides:
   - `docs/training_guide.md`
   - `docs/evaluation_guide.md`
   - `docs/vast_ai_guide.md`
   - `docs/interpretability_guide.md`
4. Create reproducibility guide:
   - Environment setup
   - Exact commands to reproduce results
   - Expected outputs
5. Add license file (MIT or Apache 2.0)
6. Clean up code:
   - Remove debug prints
   - Remove unused imports
   - Format with Black or autopep8
   - Run linter (flake8 or pylint)

**Validation:**
- All code documented
- README complete
- Guides clear and tested
- Code cleaned and formatted
- License added

**Success Criteria:**
✅ Comprehensive code documentation
✅ Usage guides complete
✅ Reproducibility ensured
✅ Code clean and formatted
✅ Ready for open-source release

---

### Mission 8.3: Academic Paper Writing - Structure

**Objective:** Write complete academic paper draft

**Tasks:**
1. Create paper structure (PRD Section 10.3):
   - Abstract (250 words)
   - Introduction (1.5 pages)
   - Related Work (1 page)
   - Dataset (2 pages)
   - Methodology (2 pages)
   - Results (2 pages)
   - Analysis (1.5 pages)
   - Conclusion (0.5 pages)
   - References
2. Write Abstract:
   - Research problem
   - Contribution (dataset + benchmarks)
   - Main results
   - Implications
3. Write Introduction:
   - Motivation
   - Research questions
   - Contributions
   - Paper organization
4. Write Related Work:
   - Idiom detection in NLP
   - Hebrew NLP resources
   - Model comparison studies
   - Interpretability in NLP
   - Position this work
5. Save draft to `paper/main.tex`

**Validation:**
- Structure complete
- Abstract and introduction written
- Related work comprehensive
- LaTeX compiles
- Word count appropriate

**Success Criteria:**
✅ Paper structure complete
✅ Abstract written (≤250 words)
✅ Introduction written (~1.5 pages)
✅ Related work written (~1 page)
✅ LaTeX compiles without errors

---

### Mission 8.4: Academic Paper Writing - Content

**Objective:** Complete methodology, results, and analysis sections

**Tasks:**
1. Write Dataset section:
   - Collection process
   - Annotation methodology
   - Hybrid seen/unseen splitting strategy
   - Statistics (table and description, including sentence types)
   - Examples (figure)
   - Quality control
2. Write Methodology section:
   - Task definitions
   - Models evaluated
   - Training procedure
   - Hyperparameter optimization
   - LLM evaluation approach
   - Interpretability methods
   - Evaluation metrics
3. Write Results section:
   - Zero-shot baseline results (table)
   - Fine-tuned models results (table)
   - LLM results (table)
   - Best model for each task
   - Statistical significance
4. Write Analysis section:
   - Model comparison
   - Hebrew-specific vs multilingual
   - Fine-tuning vs prompting trade-offs
   - Error analysis
   - Interpretability insights (with visualizations)
   - Limitations
5. Write Conclusion:
   - Summary of contributions
   - Main findings
   - Future work
   - Dataset release announcement

**Validation:**
- All sections complete
- Results accurately reported
- Figures and tables referenced
- Interpretability analysis included
- Clear and concise writing
- LaTeX compiles

**Success Criteria:**
✅ All sections written
✅ 8 pages + references
✅ All figures and tables included
✅ Interpretability analysis integrated
✅ Results accurate
✅ Paper complete

---

### Mission 8.5: Paper Refinement and Proofreading

**Objective:** Polish paper to submission quality

**Tasks:**
1. Review and edit:
   - Check for clarity
   - Remove redundancy
   - Improve flow
   - Strengthen arguments
2. Format checking:
   - Follows ACL/EMNLP guidelines
   - Correct citation format
   - Figure quality and placement
   - Table formatting
3. Proofreading:
   - Grammar and spelling
   - Consistency in terminology
   - Reference completeness
4. Get feedback:
   - Share with advisor
   - Share with peers
   - Incorporate feedback
5. Final checks:
   - Abstract represents paper accurately
   - Introduction and conclusion aligned
   - All claims supported by results
   - All figures and tables referenced
6. Create supplementary materials if needed

**Validation:**
- Paper reads smoothly
- No grammatical errors
- Formatting correct
- All references complete
- Feedback incorporated

**Success Criteria:**
✅ Paper polished and professional
✅ Formatting correct
✅ No errors
✅ Feedback incorporated
✅ Ready for submission

---

### Mission 8.6: Thesis Document (If Required)

**Objective:** Expand paper into full thesis document

**Tasks:**
1. Create thesis structure (PRD Section 10.4):
   - 60-80 pages
   - Hebrew with English abstract
   - 8-10 chapters
2. Expand paper sections into chapters:
   - Chapter 1: Introduction (expanded)
   - Chapter 2: Background and Related Work (expanded)
   - Chapter 3: Dataset Creation
   - Chapter 4: Methodology
   - Chapter 5: Experimental Setup
   - Chapter 6: Results
   - Chapter 7: Analysis and Discussion (including interpretability)
   - Chapter 8: Conclusions and Future Work
   - Appendices: Detailed results, code samples, etc.
3. Add thesis-specific content:
   - Detailed literature review
   - More background on Hebrew NLP
   - Detailed experimental setup
   - Extended ablation studies
   - More error analysis examples
   - Extended interpretability analysis
4. Translate to Hebrew if required
5. Format according to university guidelines

**Note:** Only if thesis is separate from paper

**Validation:**
- All chapters complete
- 60-80 pages
- Follows university format
- Comprehensive and detailed
- Ready for defense

**Success Criteria:**
✅ Thesis complete (60-80 pages)
✅ All chapters written
✅ Translated if needed
✅ Formatted correctly
✅ Ready for submission and defense

---

## PHASE 9: RELEASE & SUBMISSION (Week 12)

### Mission 9.1: Dataset Release on HuggingFace

**Objective:** Publicly release dataset on HuggingFace Hub

**Tasks:**
1. Create HuggingFace account (if not already)
2. Prepare dataset files:
   - Train/val/test CSV files
   - Expression-to-split mapping file
   - Dataset loading script
   - Dataset card (from Mission 8.1)
   - README
3. Upload to HuggingFace Datasets:
   - Create new dataset repository
   - Upload all files
   - Test dataset loading
4. Add metadata:
   - License
   - Tags (Hebrew, idioms, NLP, token-classification, text-classification)
   - Task categories
   - Languages
5. Test dataset:
   - Load using datasets library
   - Verify all fields correct
   - Check splits
6. Announce release:
   - Tweet/social media
   - Relevant forums/communities
   - Email to Hebrew NLP researchers

**Validation:**
- Dataset uploaded successfully
- Can load with `datasets.load_dataset()`
- All data correct
- Documentation complete
- Public and accessible

**Success Criteria:**
✅ Dataset on HuggingFace Hub
✅ Loadable and usable
✅ Well-documented
✅ License clear
✅ Announced publicly

---

### Mission 9.2: Code Release on GitHub

**Objective:** Release code repository publicly

**Tasks:**
1. Prepare repository:
   - Clean up code
   - Remove sensitive information (API keys, personal paths)
   - Add comprehensive README
   - Add LICENSE file
   - Add CONTRIBUTING.md
2. Add model checkpoints:
   - Upload best models to HuggingFace Hub
   - Link to models in README
   - Or provide download script
3. Add documentation:
   - Installation guide
   - Usage examples
   - Reproduction guide
   - VAST.ai setup guide
   - Interpretability analysis guide
4. Create releases:
   - Tag version 1.0
   - Create release notes
   - Include paper link when available
5. Make repository public
6. Add badges to README:
   - License
   - Python version
   - HuggingFace dataset link

**Validation:**
- Repository clean and organized
- Documentation complete
- Can clone and run
- Models accessible
- Public and usable

**Success Criteria:**
✅ GitHub repository public
✅ Code well-documented
✅ Models accessible
✅ Reproduction possible
✅ Ready for community use

---

### Mission 9.3: Model Release on HuggingFace

**Objective:** Release trained models on HuggingFace Model Hub

**Tasks:**
1. For each best model (Task 1 and Task 2):
   - Upload to HuggingFace Model Hub
   - Create model card
   - Include usage example
   - Add evaluation metrics
2. Model card contents:
   - Model description
   - Training data
   - Hyperparameters
   - Performance metrics
   - Usage example
   - Citation
3. Test models:
   - Load using `AutoModel`
   - Run inference on example
   - Verify predictions
4. Link models in:
   - GitHub README
   - Paper
   - Dataset card

**Validation:**
- Models uploaded successfully
- Can load and use easily
- Model cards complete
- Inference works
- Well-documented

**Success Criteria:**
✅ Best models on HuggingFace Hub
✅ Model cards complete
✅ Easy to use
✅ Performance documented
✅ Linked from other resources

---

### Mission 9.4: Paper Submission

**Objective:** Submit paper to target venue

**Tasks:**
1. Select target venue (PRD Section 10.3):
   - ACL 2026
   - EMNLP 2026
   - NAACL 2026
   - Or relevant workshop
   - Check deadlines
2. Prepare submission:
   - Format according to guidelines
   - Anonymize (if required)
   - Check page limits
   - Prepare supplementary materials
3. Create submission materials:
   - Main paper PDF
   - Supplementary materials (if any)
   - Code link (anonymized if needed)
   - Dataset link (anonymized if needed)
4. Review checklist:
   - All sections complete
   - All figures and tables included
   - References complete
   - No identifying information (if anonymous)
   - Follows formatting guidelines
5. Submit through conference system
6. Track submission:
   - Note submission ID
   - Save submission confirmation
   - Mark review deadline on calendar

**Validation:**
- Paper formatted correctly
- All materials prepared
- Submission successful
- Confirmation received
- Ready for review process

**Success Criteria:**
✅ Paper submitted to target venue
✅ All guidelines followed
✅ Submission confirmed
✅ Awaiting reviews
✅ On track for publication

---

### Mission 9.5: Results Archive and DOI

**Objective:** Archive complete project for long-term preservation

**Tasks:**
1. Compile all materials:
   - Dataset
   - Code
   - Models
   - Paper
   - Results
   - Figures
   - Documentation
2. Create Zenodo archive:
   - Create account at zenodo.org
   - Upload complete archive
   - Add metadata
   - Add keywords
   - Link to GitHub
3. Get DOI:
   - Zenodo provides DOI automatically
   - Add DOI to paper
   - Add DOI to README
   - Add DOI to dataset card
4. Update all links:
   - Paper: add DOI
   - GitHub: add DOI badge
   - HuggingFace: add DOI
5. Create project website (optional):
   - GitHub Pages
   - Showcase dataset
   - Link to paper
   - Link to models

**Validation:**
- Archive complete
- DOI obtained
- All materials accessible
- Links updated
- Preservation ensured

**Success Criteria:**
✅ Complete archive on Zenodo
✅ DOI obtained
✅ All materials preserved
✅ Links updated everywhere
✅ Long-term accessibility ensured

---

## FINAL CHECKLIST

Before considering the project complete, verify:

### Dataset
- [ ] Dataset validated (4,800 samples, balanced)
- [ ] Sentence types analyzed
- [ ] Splits created (expression-based, no data leakage)
- [ ] Test set: 6 specific expressions
- [ ] Released on HuggingFace
- [ ] Well-documented
- [ ] DOI assigned

### Models
- [ ] 5 encoder models evaluated (zero-shot)
- [ ] 5 encoder models fine-tuned (both tasks)
- [ ] Cross-seed validation (3 seeds)
- [ ] 1 LLM evaluated (zero-shot + few-shot)
- [ ] Best models released on HuggingFace

### Results
- [ ] All metrics calculated
- [ ] Statistical testing complete
- [ ] Error analysis done
- [ ] Token importance analysis complete
- [ ] All figures created (publication quality)
- [ ] All tables created (LaTeX + Markdown)

### Analysis
- [ ] Model comparison complete
- [ ] Hebrew-specific vs multilingual analyzed
- [ ] Fine-tuning vs prompting compared
- [ ] Error patterns identified
- [ ] Interpretability insights documented
- [ ] Limitations documented

### Code
- [ ] All code documented
- [ ] IOB2 subword alignment utility implemented and tested (src/utils/tokenization.py)
- [ ] Training config files created (training_config.yaml, hpo_config.yaml)
- [ ] Batch execution scripts created (scripts/run_all_hpo.sh, scripts/run_all_experiments.sh)
- [ ] Optuna HPO mode implemented in idiom_experiment.py
- [ ] Tests passing
- [ ] GitHub repository public
- [ ] Reproduction guide complete (see WORKFLOW_SUMMARY.md)
- [ ] License added

### Paper
- [ ] 8 pages + references
- [ ] All sections complete
- [ ] Figures and tables included
- [ ] Interpretability analysis included
- [ ] Proofread and polished
- [ ] Submitted to conference

### Thesis (if required)
- [ ] 60-80 pages complete
- [ ] All chapters written
- [ ] Translated if needed
- [ ] Formatted per university guidelines
- [ ] Ready for defense

### Release
- [ ] Dataset on HuggingFace
- [ ] Models on HuggingFace
- [ ] Code on GitHub
- [ ] Paper submitted
- [ ] DOI obtained
- [ ] Announced publicly

---

## SUCCESS CRITERIA SUMMARY

**Minimum Viable Product (MVP):**
✅ Dataset validated and splits created (expression-based)
✅ 5 encoder models trained and evaluated on both tasks
✅ 1 LLM evaluated via prompting
✅ Comprehensive results comparison
✅ Interpretability analysis complete
✅ Academic paper draft complete

**Performance Targets:**
✅ Sentence Classification: F1 > 85%
✅ Token Classification: F1 > 80%
✅ LLM competitive within 5% of best fine-tuned model

**Publication Goal:**
✅ Submit to ACL/EMNLP or similar tier-1 venue
✅ Dataset released publicly on HuggingFace

---

## NOTES FOR EXECUTION

1. **Work Sequentially**: Complete validation before moving to next mission
2. **Document Everything**: Keep detailed logs of experiments and results
3. **Save Regularly**: Sync to Google Drive frequently
4. **Track Costs**: Monitor VAST.ai and API spending
5. **Ask for Help**: Consult advisor when stuck
6. **Stay Organized**: Keep files and experiments well-organized
7. **Test Incrementally**: Always test on small subsets first
8. **Version Control**: Commit to Git regularly
9. **Backup**: Multiple backups (Git + Google Drive)
10. **Reproducibility**: Document exact steps for reproducibility
11. **Interpretability**: Save token importance visualizations for key cases
12. **Data Leakage**: Always verify expression-based splits to avoid leakage

---

**END OF MISSIONS DOCUMENT**

*This guide provides the complete roadmap from setup to publication. Follow each mission carefully, validate at each step, and you'll successfully complete the Hebrew Idiom Detection project.*
