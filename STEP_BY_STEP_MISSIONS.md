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

### Mission 2.4: Dataset Statistics Analysis

**Objective:** Generate comprehensive statistics for the dataset

**Tasks:**
1. Calculate and verify:
   - Total sentences: 4,800
   - Literal samples: 2,400 (50%)
   - Figurative samples: 2,400 (50%)
   - Unique idioms: count unique values in `expression` column
   - Average sentence length: mean of `num_tokens`
   - Average idiom length: calculate from token spans
2. Analyze idiom distribution:
   - Most frequent idioms (top 10)
   - Least frequent idioms
   - Idioms that appear in both literal and figurative contexts
3. **Sentence Type Analysis:**
   - Analyze sentence types (declarative, question, imperative, exclamatory)
   - Identify questions (sentences ending with "?")
   - Identify imperatives (command sentences)
   - Identify exclamatory (sentences with "!")
   - Count and percentage of each type
   - Check if sentence types are balanced across literal/figurative labels
   - Check if certain idioms appear more in specific sentence types
4. Create visualizations:
   - Sentence length distribution histogram
   - Idiom length distribution histogram
   - Top 10 idioms bar chart
   - Sentence type distribution pie chart
   - Sentence type by label (literal vs figurative) stacked bar chart
5. Save statistics to `experiments/results/dataset_statistics.txt`
6. Save visualizations to `paper/figures/`

**Validation:**
- Statistics match PRD Section 2.3 expectations
- Average sentence length: ~12-13 tokens
- Average idiom length: ~3-4 tokens
- ~60-80 unique idioms
- Sentence types analyzed and documented
- All visualizations saved

**Success Criteria:**
✅ Statistics calculated correctly
✅ Unique idioms: 60-80 expressions
✅ Avg sentence length: ~12.5 tokens
✅ Avg idiom length: ~3.2 tokens
✅ Sentence types analyzed (questions, declarative, etc.)
✅ Visualizations created and saved

---

### Mission 2.5: Dataset Splitting (Expression-Based Strategy)

**Objective:** Create train/validation/test splits based on specific expressions to avoid data leakage

**Tasks:**
1. **Test Set Creation (Expression-Based):**
   - Define test set expressions:
     - ״חתך פינה״
     - ״חצה קו אדום״
     - ״נשאר מאחור״
     - ״שבר שתיקה״
     - ״איבד את הראש״
     - ״רץ אחרי הזנב של עצמו״
   - Filter all samples with these expressions to test set
   - Include both literal and figurative examples for each expression
   - Verify test set has balanced labels (50/50 literal/figurative or close)
   - Count total samples in test set
2. **Train/Validation Split (From Remaining Expressions):**
   - Take all remaining expressions (not in test set)
   - Group samples by unique expression
   - Split expressions (not individual samples) into train/validation
   - Use stratification to maintain balance
   - Common percentage: 80/20 or 85/15 for train/val from remaining data
   - Ensure each expression appears in ONLY ONE split (no data leakage)
   - Verify both splits have balanced labels
3. **Data Leakage Verification:**
   - Verify no expression appears in multiple splits
   - Check: set(test_expressions) ∩ set(train_expressions) = ∅
   - Check: set(test_expressions) ∩ set(val_expressions) = ∅
   - Check: set(train_expressions) ∩ set(val_expressions) = ∅
   - Document which expressions are in which split
4. **Save Splits:**
   - `data/splits/train.csv`
   - `data/splits/validation.csv`
   - `data/splits/test.csv`
   - `data/splits/split_expressions.json` (mapping of expressions to splits)
5. Add `split` column to original dataset indicating train/val/test
6. Save updated dataset with split column

**Validation:**
- Test set contains only the 6 specified expressions
- No expression overlap between splits (zero data leakage)
- Each split has balanced labels (50% ± 5%)
- All idioms from test expressions present in test set
- Train and validation splits from remaining expressions only
- Files saved successfully
- Can reload splits and verify expression separation

**Success Criteria:**
✅ Test set: 6 specific expressions only
✅ Train/Val: remaining expressions (80/20 or 85/15 split)
✅ Zero data leakage verified (no shared expressions)
✅ Each split balanced (50/50 ± 5%)
✅ Expression-to-split mapping documented
✅ Files saved and validated

---

### Mission 2.6: Data Preparation Testing

**Objective:** Create unit tests for data preparation functions

**Tasks:**
1. Create `tests/test_data_preparation.py`
2. Write test functions for:
   - Dataset loading (correct shape, columns)
   - Label balance check
   - IOB2 validation
   - Splitting function (correct sizes, stratification, no data leakage)
   - Token count alignment
   - Expression-based split validation
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
✅ Data leakage test passes
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
1. Create configuration file structure in `experiments/configs/`
2. Define hyperparameter ranges from PRD Section 5.1:
   - Learning rate: [1e-5, 2e-5, 3e-5, 5e-5]
   - Batch size: [8, 16, 32]
   - Epochs: [3, 5, 8]
   - Warmup ratio: [0.0, 0.1, 0.2]
   - Weight decay: [0.0, 0.01, 0.05]
   - Gradient accumulation steps: [1, 2, 4]
3. Create default configuration file (YAML or JSON format):
   - Default values for all hyperparameters
   - Model selection
   - Task selection
   - Paths to data
   - Output directory
   - Random seed
4. Implement configuration loading in training script
5. Test loading and validating configuration

**Validation:**
- Config files created and valid
- Can load configuration from file
- All hyperparameters accessible
- Ranges documented clearly
- Easy to modify for experiments

**Success Criteria:**
✅ Configuration system implemented
✅ All hyperparameters defined
✅ Ranges from PRD included
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
4. For Task 2 (Token Classification):
   - Use `AutoModelForTokenClassification`
   - Set num_labels=3 (O, B-IDIOM, I-IDIOM)
   - Handle IOB2 label mapping
   - Train on training set
   - Validate on validation set
5. Implement checkpointing:
   - Save best model based on validation F1
   - Save training logs
   - Save final metrics
6. Test training on small subset (500 samples, 2 epochs)

**Validation:**
- Training runs without errors
- Loss decreases over epochs
- Validation metrics improve
- Best model saved correctly
- Checkpointing works
- Can resume training from checkpoint

**Success Criteria:**
✅ Training pipeline complete
✅ Both tasks supported
✅ Checkpointing implemented
✅ Early stopping functional
✅ Tested successfully

---

### Mission 4.3: Hyperparameter Optimization Setup

**Objective:** Implement Optuna for automated hyperparameter tuning

**Tasks:**
1. Install and import Optuna library
2. Define Optuna objective function:
   - Suggest hyperparameters from ranges
   - Train model with suggested parameters
   - Return validation F1 score
3. Configure Optuna study:
   - Objective: Maximize validation F1
   - Number of trials: 10-15 per model
   - Pruning: Enable Successive Halving
   - Storage: SQLite database for persistence
4. Implement pruning callback:
   - Report validation metric after each epoch
   - Prune unpromising trials early
5. Test HPO on one model (e.g., AlephBERT) with 3 trials
6. Verify best hyperparameters are saved

**Validation:**
- Optuna study runs successfully
- Trials complete without errors
- Pruning works (some trials stopped early)
- Best hyperparameters identified
- Results saved to database
- Can visualize optimization history

**Success Criteria:**
✅ Optuna integrated
✅ HPO runs successfully
✅ Pruning functional
✅ Best parameters identified
✅ Ready for full HPO experiments

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
8. Setup automatic result sync to Google Drive

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
✅ Ready for full experiments

---

### Mission 4.5: Hyperparameter Optimization for All Models

**Objective:** Run HPO for all 5 models on both tasks

**Tasks:**
1. For each model (AlephBERT-base, AlephBERT-Gimmel, DictaBERT, mBERT, XLM-R):
   - Task 1: Sentence Classification
     - Run Optuna study (10-15 trials)
     - Search hyperparameters: LR, batch size, epochs, warmup, weight decay
     - Track validation F1 score
     - Save best hyperparameters
   - Task 2: Token Classification
     - Run Optuna study (10-15 trials)
     - Same hyperparameter search
     - Track validation F1 score
     - Save best hyperparameters
2. Total: 10 HPO studies (5 models × 2 tasks)
3. Save all results:
   - Best hyperparameters for each model-task combination
   - Optimization history
   - Visualization of hyperparameter importance
4. Document findings:
   - Which hyperparameters matter most?
   - Different optimal values for different models?
   - Different optimal values for different tasks?

**Note:** This will take significant time - use VAST.ai and can run multiple studies in parallel if budget allows

**Validation:**
- All 10 HPO studies complete
- Best hyperparameters identified for each
- Results saved and documented
- Hyperparameter importance analyzed
- Ready for final training with best configs

**Success Criteria:**
✅ 10 HPO studies completed
✅ Best hyperparameters for all model-task combinations
✅ Results documented
✅ Insights extracted
✅ Ready for final training

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

**Tasks:**
1. Select best-performing model from fine-tuning phase
2. Train with frozen backbone:
   - Freeze all transformer layers
   - Train only classification head
   - Use same hyperparameters
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
   - Use best hyperparameters
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

**Validation:**
- 100 errors analyzed
- Categories clearly defined
- Patterns identified
- Interpretability insights integrated
- Examples documented with visualizations
- Report comprehensive

**Success Criteria:**
✅ Error analysis complete
✅ Error categories identified
✅ Patterns documented
✅ Interpretability analysis integrated
✅ Examples with visualizations included
✅ Insights actionable

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
   - Expression-based splitting strategy
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
- [ ] Tests passing
- [ ] GitHub repository public
- [ ] Reproduction guide complete
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
