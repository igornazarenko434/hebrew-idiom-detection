# Operations & Workflow Guide
# Hebrew Idiom Detection Project

**Version:** 1.0
**Last Updated:** December 31, 2025
**Purpose:** Step-by-step operational guide for training, evaluation, and analysis workflows

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Project Structure](#2-project-structure)
3. [Complete Workflow Overview](#3-complete-workflow-overview)
4. [Phase 1: Hyperparameter Optimization (VAST.ai)](#4-phase-1-hyperparameter-optimization-vastai)
5. [Phase 2: Full Fine-Tuning (VAST.ai)](#5-phase-2-full-fine-tuning-vastai)
6. [Phase 3: Download Results](#6-phase-3-download-results)
7. [Phase 4: Run Evaluations](#7-phase-4-run-evaluations)
8. [Phase 5: Analysis Workflow](#8-phase-5-analysis-workflow)
9. [Analysis Tools Reference](#9-analysis-tools-reference)
10. [Adding New Models](#10-adding-new-models)
11. [Troubleshooting](#11-troubleshooting)
12. [Appendix: Script Templates](#12-appendix-script-templates)

---

## 1. Quick Start

### 1.1 For Complete Beginners

**If you just want to analyze existing results:**
```bash
# 1. Navigate to project directory
cd /path/to/Final_Project_NLP

# 2. Activate virtual environment
source activate_env.sh

# 3. Run analysis
python src/analyze_finetuning_results.py

# 4. Check outputs
ls experiments/results/analysis/
```

**If you want to train a new model:**
See [Section 4: Hyperparameter Optimization](#4-phase-1-hyperparameter-optimization-vastai)

**If you want a full end-to-end re-run:**
Use `FULL_RERUN_CHECKLIST.md` (one-page, step-by-step).

### 1.2 Common Operations Quick Reference

| Task | Command | Output Location |
|------|---------|----------------|
| Activate virtual environment | `source activate_env.sh` | N/A |
| Analyze fine-tuning results | `python src/analyze_finetuning_results.py` | `experiments/results/analysis/` |
| Analyze generalization gap | `python src/analyze_generalization.py` | `experiments/results/analysis/generalization/` |
| Analyze error distribution | `python src/analyze_error_distribution.py` | `paper/figures/error_analysis/` |
| Categorize errors | `python scripts/categorize_all_errors.py` | Updates `eval_predictions.json` |

---

## 2. Project Structure

### 2.1 Directory Layout

```
Final_Project_NLP/
├── data/
│   ├── expressions_data_tagged.csv          # Main dataset
│   └── splits/
│       ├── train.csv                        # Training set (3,360 samples)
│       ├── validation.csv                   # Validation set (480 samples)
│       ├── test.csv                         # Seen test (480 samples)
│       └── unseen_idiom_test.csv           # Unseen test (480 samples)
│
├── src/
│   ├── train.py                             # Main training script
│   ├── evaluate.py                          # Evaluation script
│   ├── analyze_finetuning_results.py        # Aggregate all model results
│   ├── analyze_generalization.py            # Generalization gap analysis
│   ├── analyze_error_distribution.py        # Error visualization dashboard
│   └── utils/
│       ├── error_analysis.py                # Error categorization functions
│       └── data_utils.py                    # Data loading utilities
│
├── experiments/
│   ├── checkpoints/                         # Trained model checkpoints
│   │   └── {model}/{task}/seed_{seed}/
│   │       └── best_model/
│   ├── logs/                                # TensorBoard logs
│   │   └── {model}/{task}/seed_{seed}/
│   └── results/
│       ├── evaluation/                      # Evaluation outputs
│       │   ├── seen_test/{model}/{task}/seed_{seed}/
│       │   │   ├── eval_results*.json
│       │   │   └── eval_predictions.json
│       │   └── unseen_test/{model}/{task}/seed_{seed}/
│       │       ├── eval_results*.json
│       │       └── eval_predictions.json
│       └── analysis/                        # Analysis outputs
│           ├── finetuning_summary.csv
│           ├── finetuning_summary.md
│           ├── statistical_significance.txt
│           ├── generalization/
│           └── error_analysis/
│               ├── error_distribution_detailed.csv
│               └── error_analysis_report.md
│
├── configs/
│   ├── hpo/                                 # HPO configs
│   │   ├── alephbert_cls_hpo.yaml
│   │   ├── dictabert_span_hpo.yaml
│   │   └── ...
│   ├── training/                            # Full fine-tuning configs
│   │   ├── alephbert_cls_train.yaml
│   │   ├── dictabert_span_train.yaml
│   │   └── ...
│   └── best_hyperparameters/                # Best params from HPO
│       ├── best_params_alephbert_cls.json
│       ├── best_params_dictabert_span.json
│       └── ...
│
├── scripts/
│   ├── run_all_hpo.sh                       # Batch HPO on VAST.ai
│   ├── run_all_experiments.sh               # Batch training on VAST.ai
│   ├── run_evaluation_batch.sh              # Batch evaluation on seen + unseen
│   ├── download_checkpoints.sh              # Download from VAST.ai
│   ├── download_evaluation_results.sh       # Download eval results
│   └── categorize_all_errors.py             # Categorize all predictions (Task 1.3)
│
├── paper/                                    # Publication materials
│   ├── figures/
│   │   ├── generalization/                   # Generalization gap figures
│   │   └── error_analysis/                   # Error distribution figures
│   └── tables/
│       └── finetuning_results.tex            # LaTeX tables
│
├── activate_env.sh                           # Activate virtual environment
├── venv/                                     # Virtual environment (git-ignored)
│
├── EVALUATION_STANDARDIZATION_GUIDE.md      # Metrics & standards reference
├── OPERATIONS_GUIDE.md                      # This file
└── README.md                                # Project overview
```

### 2.2 Key Files and Their Purposes

| File | Purpose | When to Use |
|------|---------|-------------|
| `src/idiom_experiment.py` | Unified entrypoint for HPO, training, and evaluation | All experiment modes |
| `src/analyze_finetuning_results.py` | Aggregate multi-seed results, compute Mean ± Std | After evaluating all seeds |
| `src/analyze_generalization.py` | Compute generalization gap | After evaluating seen + unseen |
| `src/analyze_error_distribution.py` | Visualize error distributions across models | After error categorization |
| `scripts/categorize_all_errors.py` | Add error_category to all predictions | After evaluation complete |
| `src/utils/error_analysis.py` | Error categorization functions | Imported by categorize script |

---

## 3. Complete Workflow Overview

### 3.1 Visual Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: HPO (VAST.ai)                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Create HPO config for each model/task                    │
│ 2. Upload data + code to VAST.ai                            │
│ 3. Run HPO (Optuna) with 30-50 trials                       │
│ 4. Download best hyperparameters                            │
│ 5. Save to experiments/results/best_hyperparameters/        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: Full Fine-Tuning (VAST.ai)            │
├─────────────────────────────────────────────────────────────┤
│ 1. Create training config using best hyperparameters        │
│ 2. Train with 3 seeds (42, 123, 456)                        │
│ 3. Monitor with TensorBoard                                 │
│ 4. Wait for training completion (~2-4 hours per seed)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: Download Results (Local)              │
├─────────────────────────────────────────────────────────────┤
│ 1. Download best checkpoints from VAST.ai                   │
│ 2. Download TensorBoard logs                                │
│ 3. Verify checkpoint integrity                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               PHASE 4: Evaluation (Local)                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Evaluate on seen test (data/splits/test.csv)             │
│ 2. Evaluate on unseen test (data/splits/unseen_idiom_test.csv)│
│ 3. Save eval_results.json and eval_predictions.json         │
│ 4. Repeat for all models × tasks × seeds                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 5: Analysis (Local)                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Run analyze_finetuning_results.py → Summary tables       │
│ 2. Run analyze_generalization.py → Gap analysis             │
│ 3. Run analyze_comprehensive.py → Full analysis             │
│ 4. Create visualizations → Publication figures              │
│ 5. Generate error reports → Manual inspection               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Timeline Estimates

| Phase | Time | Hardware |
|-------|------|----------|
| HPO (per model/task) | 4-8 hours | VAST.ai GPU (RTX 3090/A6000) |
| Full Training (per seed) | 2-4 hours | VAST.ai GPU (RTX 3090/A6000) |
| Full Training (all 3 seeds) | 6-12 hours | Can parallelize on multiple GPUs |
| Evaluation (all models) | 1-2 hours | Local CPU/GPU |
| Analysis | 30 minutes | Local CPU |
| **Total (6 models × 2 tasks)** | **3-5 days** | With parallelization |

---

## 4. Phase 1: Hyperparameter Optimization (VAST.ai)

### 4.1 Prerequisites

**Required:**
- VAST.ai account with credits
- SSH key configured
- Data files prepared (`data/splits/`)

**Optional but recommended:**
- tmux/screen for persistent sessions
- rsync for efficient file transfer

### 4.2 Step-by-Step HPO Workflow

#### Step 1: HPO Configuration

**File:** `experiments/configs/hpo_config.yaml`

This is a single, model-agnostic HPO config used by all models/tasks. It defines:
- Search space (learning rate, batch size, epochs, warmup, weight decay, grad accumulation)
- Fixed settings (max_length, data paths, CRF usage, logging)
- Output locations (Optuna DB + trial outputs + best hyperparameters)

#### Step 2: Start VAST.ai Instance

```bash
# 1. Go to VAST.ai and create instance
# Recommended specs:
# - GPU: RTX 3090 or A6000
# - RAM: 32GB+
# - Disk: 50GB+
# - Image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 2. Note the SSH command (e.g., ssh -p 12345 root@123.456.789.0)
```

#### Step 3: Upload Code and Data

```bash
# On your local machine

# Set VAST.ai connection details
VAST_HOST="root@123.456.789.0"
VAST_PORT="12345"

# Upload project files (excluding checkpoints/logs)
rsync -avz -e "ssh -p $VAST_PORT" \
  --exclude 'experiments/checkpoints/' \
  --exclude 'experiments/logs/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  ./ $VAST_HOST:/workspace/project/

# Verify upload
ssh -p $VAST_PORT $VAST_HOST "ls -lh /workspace/project/"
```

#### Step 4: Run HPO on VAST.ai

```bash
# SSH into VAST.ai instance
ssh -p $VAST_PORT $VAST_HOST

# Navigate to project directory
cd /workspace/project

# Install dependencies (first time only)
pip install -r requirements.txt

# Start tmux session (recommended for long-running jobs)
tmux new -s hpo_dictabert_span

# Run HPO for a single model/task
python src/idiom_experiment.py \
  --mode hpo \
  --model_id dicta-il/dictabert \
  --task span \
  --config experiments/configs/hpo_config.yaml \
  --device cuda \
  2>&1 | tee experiments/results/hpo/dictabert/span/hpo.log

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t hpo_dictabert_span
```

#### Step 5: Monitor HPO Progress

**Option A: Check log file**
```bash
# From local machine
ssh -p $VAST_PORT $VAST_HOST "tail -f /workspace/project/experiments/results/hpo/dictabert/span/hpo.log"
```

**Option B: Use Optuna Dashboard (optional)**
```bash
# On VAST.ai instance
pip install optuna-dashboard
optuna-dashboard sqlite:///experiments/results/optuna_studies/dictabert_span_hpo.db

# Access via browser: http://VAST_IP:8080
```

#### Step 6: Download Best Hyperparameters

HPO writes best params to:
`experiments/results/best_hyperparameters/best_params_{model}_{task}.json`

After syncing to Google Drive (`bash scripts/sync_to_gdrive.sh`), download them locally:
```bash
rclone copy gdrive:Hebrew_Idiom_Detection/results/best_hyperparameters \
  experiments/results/best_hyperparameters \
  --include "*.json" --verbose
```

**Expected output:**
```json
{
  "learning_rate": 2.3e-05,
  "batch_size": 16,
  "num_epochs": 10,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "best_val_f1": 0.9321
}
```

### 4.3 Batch HPO for All Models

**File:** `scripts/run_all_hpo.sh`

```bash
cd /workspace/project
bash scripts/run_all_hpo.sh
```

---

## 5. Phase 2: Full Fine-Tuning (VAST.ai)

### 5.1 Training Configuration

**File:** `experiments/configs/training_config.yaml`

This is a single, model-agnostic training config used for all models/tasks.
Best hyperparameters are injected via CLI by `scripts/run_all_experiments.sh`.

### 5.2 Run Full Fine-Tuning

#### Single Seed Training (Manual)

```bash
# SSH into VAST.ai
ssh -p $VAST_PORT $VAST_HOST
cd /workspace/project

# Example: DictaBERT span, seed 42
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id dicta-il/dictabert \
  --task span \
  --config experiments/configs/training_config.yaml \
  --seed 42 \
  --output_dir experiments/results/full_fine-tuning/dictabert/span/seed_42
```

#### Multi-Seed Training (Batch)

**If you have multiple GPUs on VAST.ai:**

```bash
# Run full batch (all models × tasks × seeds)
cd /workspace/project
bash scripts/run_all_experiments.sh
```

### 5.3 Monitor Training with TensorBoard

**On VAST.ai instance:**

```bash
# Start TensorBoard
tensorboard --logdir experiments/logs/ --port 6006 --bind_all

# Access via browser:
# http://VAST_IP:6006
```

**What to monitor:**
- Training loss (should decrease smoothly)
- Validation loss (should decrease without overfitting)
- Validation F1 (should increase and plateau)
- Best checkpoint saved (usually around epoch 7-10)

### 5.4 Batch Training Script

**File:** `scripts/run_all_experiments.sh`

```bash
cd /workspace/project
bash scripts/run_all_experiments.sh
```

**Outputs:**
- `experiments/results/full_fine-tuning/<model>/<task>/seed_<seed>/`
- `experiments/logs/`

---

## 6. Phase 3: Download Results

### 6.1 Download Results (Google Drive)

**Best model weights only:**
```bash
bash scripts/download_best_checkpoints.sh
```

**Training metrics only (lightweight):**
```bash
bash scripts/download_results_for_analysis.sh
```

**Evaluation JSONs:**
```bash
bash scripts/download_evaluation_results.sh
```
      rsync -avz -e "ssh -p $VAST_PORT" \
        "$VAST_HOST:$REMOTE_DIR/$model/$task/seed_$seed/best_model/" \
        "$LOCAL_DIR/$model/$task/seed_$seed/best_model/"

      # Verify download
      if [ -f "$LOCAL_DIR/$model/$task/seed_$seed/best_model/pytorch_model.bin" ]; then
        echo "✓ Successfully downloaded $model $task seed_$seed"
      else
        echo "✗ Failed to download $model $task seed_$seed"
      fi
    done
  done
done

echo "All checkpoints downloaded!"
```

### 6.2 Download TensorBoard Logs (Optional)

```bash
#!/bin/bash

# Download TensorBoard logs for learning curve analysis

GDRIVE_LOGS="gdrive:Hebrew_Idiom_Detection/logs"
LOCAL_DIR="experiments/logs"

rclone copy "${GDRIVE_LOGS}" "${LOCAL_DIR}" \
  --include "events.out.tfevents.*" \
  --verbose \
  --transfers 8
```

### 6.3 Verify Download Integrity

```bash
# Check trained outputs are present
python src/audit_results.py
```

---

## 7. Phase 4: Run Evaluations

### 7.1 Single Model Evaluation

```bash
# Evaluate DictaBERT SPAN task seed 42 on SEEN test
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_fine-tuning/dictabert/span/seed_42 \
  --data data/splits/test.csv \
  --task span \
  --output_dir experiments/results/evaluation/seen_test/dictabert/span/seed_42

# Evaluate same model on UNSEEN test
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_fine-tuning/dictabert/span/seed_42 \
  --data data/splits/unseen_idiom_test.csv \
  --task span \
  --output_dir experiments/results/evaluation/unseen_test/dictabert/span/seed_42
```

**Output files created:**
```
experiments/results/evaluation/seen_test/dictabert/span/seed_42/
├── eval_results_20251231_103045.json     # Metrics with timestamp
└── eval_predictions.json                 # All predictions with error categories
```

### 7.2 Batch Evaluation Script

**File:** `scripts/run_evaluation_batch.sh`

```bash
#!/bin/bash

# Batch evaluation script for all models, tasks, seeds, and splits
# Usage: ./scripts/run_evaluation_batch.sh

MODELS=("dictabert" "alephbert" "alephbertgimmel" "neodictabert" "mbert" "xlm-r")
TASKS=("cls" "span")
SEEDS=(42 123 456)
SPLITS=("seen_test:data/splits/test.csv" "unseen_test:data/splits/unseen_idiom_test.csv")

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      checkpoint="experiments/checkpoints/$model/$task/seed_$seed/best_model"

      # Check if checkpoint exists
      if [ ! -d "$checkpoint" ]; then
        echo "⚠️  Checkpoint not found: $checkpoint"
        continue
      fi

      for split_info in "${SPLITS[@]}"; do
        split=$(echo $split_info | cut -d':' -f1)
        test_file=$(echo $split_info | cut -d':' -f2)

        output_dir="experiments/results/evaluation/$split/$model/$task/seed_$seed"

        echo "Evaluating: $model / $task / seed_$seed / $split"

        python src/idiom_experiment.py \
          --mode evaluate \
          --model_checkpoint "$checkpoint" \
          --data "$test_file" \
          --task "$task" \
          --output_dir "$output_dir"

        echo "✓ Completed: $model $task seed_$seed $split"
      done
    done
  done
done

echo "All evaluations completed!"
```

### 7.3 Verify Evaluation Completeness

```bash
# Quick counts (see Section 12.2 for snippets)
find experiments/results/evaluation -name "eval_results*.json" | wc -l
```

---

## 8. Phase 5: Analysis Workflow

### 8.1 Analysis Pipeline Overview

```
┌──────────────────────────────────────────────────────┐
│ Step 1: Activate Virtual Environment                 │
│ Tool: activate_env.sh                                │
│ Output: Activated venv with all dependencies         │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ Step 2: Aggregate Results                            │
│ Tool: analyze_finetuning_results.py                  │
│ Output: Summary tables (Mean ± Std across seeds)     │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ Step 3: Generalization Gap Analysis                  │
│ Tool: analyze_generalization.py                      │
│ Output: Gap metrics, visualizations                  │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ Step 4: Error Categorization                         │
│ Tool: categorize_all_errors.py                       │
│ Output: error_category added to eval_predictions.json│
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ Step 5: Error Visualization Dashboard                │
│ Tool: analyze_error_distribution.py                  │
│ Output: 5 figures + CSV + report                     │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ Step 6: Per-Idiom F1 Analysis                         │
│ Tool: analyze_per_idiom_f1.py                         │
│ Output: per-idiom CSVs + heatmaps + report            │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ Step 7: Statistical Significance Testing             │
│ Tool: statistical_tests.py                            │
│ Output: paired_ttests.csv + paired_ttests.md          │
└──────────────────────────────────────────────────────┘
```

### 8.2 Step-by-Step Analysis

#### Step 0: Activate Virtual Environment

**Purpose:** Ensure all analysis dependencies are available

```bash
# Navigate to project directory
cd /path/to/Final_Project_NLP

# Activate virtual environment (creates if needed)
source activate_env.sh

# Verify activation
which python
# Should show: ./venv/bin/python
```

**What this does:**
- Creates `venv/` if it doesn't exist
- Installs analysis packages: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, tabulate
- Activates the environment

**See also:** `VENV_USAGE.md` for detailed venv guide

#### Step 1: Aggregate Fine-Tuning Results

**Purpose:** Compute Mean ± Std across 3 seeds for all models/tasks

```bash
python src/analyze_finetuning_results.py \
  --results_dir experiments/results/evaluation \
  --output_dir experiments/results/analysis

# Optional: generate model comparison figures
python src/analyze_finetuning_results.py --create_figures

# Check outputs
ls experiments/results/analysis/
# finetuning_summary.csv
# finetuning_summary.md
# statistical_significance.txt
# paper/tables/finetuning_results.tex  # LaTeX tables

ls paper/figures/finetuning/
# model_comparison_cls_seen.png
# model_comparison_cls_unseen.png
# model_comparison_span_seen.png
# model_comparison_span_unseen.png
```

**What it does:**
1. Scans `experiments/results/evaluation/` for all eval_results*.json files
2. Groups by model, task, split
3. Computes Mean ± Std across seeds (42, 123, 456)
4. Performs paired t-tests (best model vs others)
5. Saves summary tables
6. (Optional) Generates model comparison figures per task/split

**Output Example:**

`finetuning_summary.md`:
```markdown
# Fine-Tuning Results Summary

## Task: CLS (Classification) - Seen Test

| Model | F1 (Mean ± Std) | Accuracy | Precision | Recall |
|-------|-----------------|----------|-----------|--------|
| DictaBERT | 94.83 ± 0.42 | 94.75 ± 0.38 | 95.01 ± 0.51 | 94.83 ± 0.42 |
| AlephBERT | 94.21 ± 0.35 | 94.12 ± 0.33 | 94.38 ± 0.41 | 94.21 ± 0.35 |
| ... | ... | ... | ... | ... |

## Task: SPAN - Unseen Test

| Model | F1 (Mean ± Std) | Precision | Recall |
|-------|-----------------|-----------|--------|
| DictaBERT | 91.08 ± 0.58 | 91.23 ± 0.62 | 90.95 ± 0.55 |
| ... | ... | ... | ... |
```

#### Step 2: Generalization Gap Analysis

**Purpose:** Analyze performance drop from seen to unseen idioms

```bash
python src/analyze_generalization.py \
  --results_dir experiments/results/evaluation \
  --output_dir experiments/results/analysis/generalization

# Check outputs
ls experiments/results/analysis/generalization/
# generalization_report.md
# generalization_gap.csv
# figures/
#   └── generalization_gap_cls.png
#   └── generalization_gap_span.png
```

**What it does:**
1. Loads seen and unseen results for each model/task
2. Computes gap: `seen_f1 - unseen_f1`
3. Computes percentage gap: `(gap / seen_f1) * 100`
4. Creates visualizations (grouped bar charts)

**Output Example:**

`generalization_gap.csv`:
```csv
model,task,seen_f1_mean,seen_f1_std,unseen_f1_mean,unseen_f1_std,gap_absolute,gap_percentage
dictabert,cls,94.83,0.42,91.08,0.58,3.75,3.96
dictabert,span,93.21,0.51,89.45,0.62,3.76,4.03
alephbert,cls,94.21,0.35,90.62,0.45,3.59,3.81
...
```

#### Step 3: Error Categorization

**Purpose:** Add error_category field to all predictions

```bash
# Categorize all 27,360 predictions (60 eval files)
python scripts/categorize_all_errors.py

# Verify categorization
python -c "
import json
with open('experiments/results/evaluation/seen_test/dictabert/span/seed_42/eval_predictions.json') as f:
    preds = json.load(f)
print('Sample error categories:', [p['error_category'] for p in preds[:5]])
"
```

**What it does:**
1. Scans all 60 evaluation files (5 models × 2 tasks × 3 seeds × 2 splits)
2. Applies standardized error taxonomy using `src/utils/error_analysis.py`
3. Adds `error_category` field to each prediction
4. Saves updated `eval_predictions.json` files

**Error Taxonomies:**
- **CLS:** CORRECT, FALSE_POSITIVE, FALSE_NEGATIVE
- **SPAN:** PERFECT, MISS, FALSE_POSITIVE, PARTIAL_START, PARTIAL_END, PARTIAL_BOTH, EXTEND_START, EXTEND_END, EXTEND_BOTH, SHIFT, WRONG_SPAN, MULTI_SPAN

#### Step 4: Error Visualization Dashboard

**Purpose:** Generate comprehensive error analysis with 5 publication-ready figures

```bash
# Run error visualization dashboard
python src/analyze_error_distribution.py

# Check outputs
ls paper/figures/error_analysis/
# error_distribution_cls.png (203 KB)
# error_distribution_span_aggregated.png (203 KB)
# error_heatmap_span.png (435 KB)
# seen_unseen_comparison.png (131 KB)
# model_error_profiles.png (385 KB)

ls experiments/results/analysis/error_analysis/
# error_distribution_detailed.csv
# error_analysis_report.md
```

**What it generates:**
1. **error_distribution_cls.png**: Stacked bar chart (CORRECT/FP/FN)
2. **error_distribution_span_aggregated.png**: Grouped bar chart with 4 error groups
3. **error_heatmap_span.png**: Heatmap showing all 12 SPAN categories
4. **seen_unseen_comparison.png**: Error shift visualization
5. **model_error_profiles.png**: Radar chart per model
6. **error_distribution_detailed.csv**: Detailed statistics
7. **error_analysis_report.md**: Summary report with methodology

**Error Grouping (SPAN Task):**
- **PERFECT**: Exact matches
- **BOUNDARY_ERRORS**: PARTIAL_*, EXTEND_* (6 categories)
- **DETECTION_ERRORS**: MISS, FALSE_POSITIVE (2 categories)
- **POSITION_ERRORS**: SHIFT, WRONG_SPAN, MULTI_SPAN (3 categories)

#### Step 5: Per-Idiom F1 Analysis

**Purpose:** Compute idiom-level F1 for each model/task/split and create difficulty rankings

```bash
python scripts/analyze_per_idiom_f1.py

# Check outputs
ls experiments/results/analysis/per_idiom_f1/
# per_idiom_f1_raw.csv
# per_idiom_f1_summary.csv
# idiom_metadata.csv
# idiom_difficulty_ranking_cls_seen_test.csv
# idiom_difficulty_ranking_cls_unseen_test.csv
# idiom_difficulty_ranking_span_seen_test.csv
# idiom_difficulty_ranking_span_unseen_test.csv
# per_idiom_f1_report.md
# per_idiom_f1_insights.md

ls paper/figures/per_idiom/
# per_idiom_heatmap_cls_seen_test.png
# per_idiom_heatmap_cls_unseen_test.png
# per_idiom_heatmap_span_seen_test.png
# per_idiom_heatmap_span_unseen_test.png
```

**What it does:**
1. Loads eval_predictions.json for all models/seeds/splits
2. Joins with split CSVs to attach `base_pie`
3. Computes per-idiom F1 (CLS macro F1, SPAN exact span F1)
4. Aggregates mean ± std across seeds
5. Ranks idioms by difficulty (avg F1 across models)
6. Generates heatmaps and a summary report

#### Step 6: Statistical Significance Testing

**Purpose:** Run paired t-tests with Bonferroni correction and Cohen’s d

```bash
python scripts/statistical_tests.py

# Check outputs
ls experiments/results/analysis/statistical_tests/
# paired_ttests.csv
# paired_ttests.md
```

**What it does:**
1. Loads eval_results.json for all models/seeds
2. Identifies best model per task/split
3. Runs paired t-tests vs other models
4. Computes Cohen’s d (paired)
5. Applies Bonferroni correction per task/split
6. Writes CSV + Markdown summary

### 8.3 Analysis Output Summary

| Analysis Type | Output Location | Key Files |
|---------------|-----------------|-----------|
| Aggregated Results | `experiments/results/analysis/` | `finetuning_summary.csv`, `finetuning_summary.md`, `statistical_significance.txt` |
| LaTeX Tables | `paper/tables/` | `finetuning_results.tex` |
| Generalization Gap | `experiments/results/analysis/generalization/` | `generalization_gap.csv`, `generalization_report.md` |
| Generalization Figures | `paper/figures/generalization/` | `generalization_gap_*.png` |
| Error Categorization | `experiments/results/evaluation/` | `eval_predictions.json` (updated with error_category) |
| Error Analysis | `experiments/results/analysis/error_analysis/` | `error_distribution_detailed.csv`, `error_analysis_report.md` |
| Error Figures | `paper/figures/error_analysis/` | 5 PNG files (300 DPI) |
| Per-Idiom Analysis | `experiments/results/analysis/per_idiom_f1/` | `per_idiom_f1_raw.csv`, `per_idiom_f1_summary.csv`, `per_idiom_f1_report.md`, `per_idiom_f1_insights.md` |
| Per-Idiom Figures | `paper/figures/per_idiom/` | 4 PNG heatmaps (300 DPI) |
| Statistical Tests | `experiments/results/analysis/statistical_tests/` | `paired_ttests.csv`, `paired_ttests.md` |

---

## 9. Analysis Tools Reference

### 9.1 analyze_finetuning_results.py

**Purpose:** Aggregate multi-seed results and compute summary statistics

**Usage:**
```bash
python src/analyze_finetuning_results.py \
  [--results_dir RESULTS_DIR] \
  [--output_dir OUTPUT_DIR] \
  [--models MODEL1 MODEL2 ...] \
  [--tasks TASK1 TASK2 ...] \
  [--seeds SEED1 SEED2 ...]
```

**Arguments:**
- `--results_dir`: Path to evaluation results (default: `experiments/results/evaluation`)
- `--output_dir`: Where to save analysis (default: `experiments/results/analysis`)
- `--models`: Specific models to analyze (default: all)
- `--tasks`: Specific tasks to analyze (default: all)
- `--seeds`: Seeds to include (default: [42, 123, 456])

**Outputs:**
1. `finetuning_summary.csv`: CSV with Mean ± Std for all models/tasks
2. `finetuning_summary.md`: Markdown tables for easy reading
3. `statistical_significance.txt`: Paired t-test results

**When to use:**
- After completing all evaluations
- To get overview of model performance
- To identify best model for each task

**Example output:**
```
Best model for CLS (Seen Test): DictaBERT (F1: 94.83 ± 0.42)
Best model for SPAN (Seen Test): DictaBERT (F1: 93.21 ± 0.51)

Statistical Significance (Paired t-tests, α=0.05):
DictaBERT vs AlephBERT (CLS, Seen): p=0.0234, Significant ✓
DictaBERT vs mBERT (CLS, Seen): p<0.001, Significant ✓
```

### 9.2 analyze_generalization.py

**Purpose:** Analyze generalization gap between seen and unseen test sets

**Usage:**
```bash
python src/analyze_generalization.py \
  [--results_dir RESULTS_DIR] \
  [--output_dir OUTPUT_DIR] \
  [--create_figures]
```

**Arguments:**
- `--results_dir`: Path to evaluation results
- `--output_dir`: Where to save analysis (default: `experiments/results/analysis/generalization`)
- `--create_figures`: Generate visualization figures

**Outputs:**
1. `generalization_gap.csv`: Gap metrics for all models/tasks
2. `generalization_report.md`: Detailed report
3. `figures/generalization_gap_{task}.png`: Visualizations

**When to use:**
- To understand model generalization capability
- To identify which models generalize best to unseen idioms
- For paper results section

**Key metrics:**
- `gap_absolute`: Seen F1 - Unseen F1
- `gap_percentage`: (gap_absolute / seen_f1) × 100

**Interpretation:**
- Smaller gap = better generalization
- Typical gap: 3-5% for good models
- Gap > 10% indicates overfitting to seen idioms

### 9.3 analyze_per_idiom_f1.py

**Purpose:** Compute per-idiom F1 across models/seeds/splits and generate difficulty rankings + heatmaps

**Usage:**
```bash
python scripts/analyze_per_idiom_f1.py
```

**Outputs:**
1. `per_idiom_f1_raw.csv`: Per-idiom metrics per seed
2. `per_idiom_f1_summary.csv`: Mean ± std per idiom across seeds
3. `idiom_difficulty_ranking_{task}_{split}.csv`: Idiom difficulty (avg across models)
4. `per_idiom_f1_report.md`: Summary report with rankings
5. `per_idiom_f1_insights.md`: Deep-dive insights for unseen idioms
6. `paper/figures/per_idiom/per_idiom_heatmap_{task}_{split}.png`: Heatmaps

**When to use:**
- After all evaluations are complete
- For Mission 7.1 per-idiom analysis and difficulty ranking
- For paper figures showing idiom-level performance patterns

### 9.4 statistical_tests.py

**Purpose:** Statistical significance testing with paired t-tests, Bonferroni correction, and Cohen’s d

**Usage:**
```bash
python scripts/statistical_tests.py
```

**Outputs:**
1. `paired_ttests.csv`: Full comparison table
2. `paired_ttests.md`: Human-readable summary

**When to use:**
- After all evaluations are complete
- For Mission 7.2 statistical reporting

### 9.3 categorize_all_errors.py

**Purpose:** Add error_category field to all evaluation predictions

**Location:** `scripts/categorize_all_errors.py`

**Usage:**
```bash
# Categorize all 60 evaluation files
python scripts/categorize_all_errors.py

# No arguments needed - scans all evaluation results automatically
```

**What it does:**
1. Scans `experiments/results/evaluation/` for all eval_predictions.json files
2. For each prediction, applies standardized error taxonomy
3. Adds `error_category` field to each prediction dict
4. Saves updated eval_predictions.json files

**Error Taxonomy Used:**
- **CLS Task:** Uses `categorize_cls_error()` from `src/utils/error_analysis.py`
  - Categories: CORRECT, FP, FN
- **SPAN Task:** Uses `categorize_span_error()` from `src/utils/error_analysis.py`
  - Categories: PERFECT, MISS, FALSE_POSITIVE, PARTIAL_START, PARTIAL_END, PARTIAL_BOTH, EXTEND_START, EXTEND_END, EXTEND_BOTH, SHIFT, WRONG_SPAN, MULTI_SPAN

**When to use:**
- After completing all model evaluations (all seeds, all splits)
- Before running error visualization dashboard
- As part of Task 1.3 from IMPLEMENTATION_ROADMAP.md

**Output:**
- Modifies eval_predictions.json files in place
- Adds `error_category` field to each prediction
- Total files updated: 60 (5 models × 2 tasks × 3 seeds × 2 splits)
- Total predictions categorized: ~27,360

**Verification:**
```bash
# Check a sample file
python -c "
import json
with open('experiments/results/evaluation/seen_test/dictabert/span/seed_42/eval_predictions.json') as f:
    preds = json.load(f)
print('Has error_category:', 'error_category' in preds[0])
print('Sample categories:', [p['error_category'] for p in preds[:3]])
"
```

**Runtime:** ~30 seconds for all 60 files

### 9.4 analyze_error_distribution.py

**Purpose:** Generate comprehensive error visualization dashboard

**Location:** `src/analyze_error_distribution.py`

**Usage:**
```bash
# Run after categorize_all_errors.py
python src/analyze_error_distribution.py

# No arguments needed - automatically processes all categorized predictions
```

**Prerequisites:**
- Must run `categorize_all_errors.py` first
- All eval_predictions.json files must have `error_category` field

**What it generates:**

**Figures (in `paper/figures/error_analysis/`):**
1. `error_distribution_cls.png` - Stacked bar chart (CLS: CORRECT/FALSE_POSITIVE/FALSE_NEGATIVE)
2. `error_distribution_span_aggregated.png` - Grouped bar chart (4 error groups)
3. `error_heatmap_span.png` - Heatmap of all 12 SPAN categories
4. `seen_unseen_comparison.png` - Error shift visualization
5. `model_error_profiles.png` - Radar chart per model

**Data (in `experiments/results/analysis/error_analysis/`):**
1. `error_distribution_detailed.csv` - Detailed statistics per model/task/split
2. `error_analysis_report.md` - Summary report with methodology and key findings

**Error Grouping for SPAN Task:**
```python
PERFECT = ['PERFECT']
BOUNDARY_ERRORS = ['PARTIAL_START', 'PARTIAL_END', 'PARTIAL_BOTH',
                   'EXTEND_START', 'EXTEND_END', 'EXTEND_BOTH']
DETECTION_ERRORS = ['MISS', 'FALSE_POSITIVE']
POSITION_ERRORS = ['SHIFT', 'WRONG_SPAN', 'MULTI_SPAN']
```

**When to use:**
- After error categorization complete
- For publication-ready error analysis figures
- As part of Task 1.3 from IMPLEMENTATION_ROADMAP.md

**Output Format:**
- All figures: 300 DPI PNG (publication-ready)
- CSV: Detailed percentages per model/task/split/category
- Report: Markdown with methodology, taxonomy, and key findings

**Runtime:** ~1 minute for all visualizations

---

## 10. Adding New Models

### 10.1 Quick Start: Add New Model

**Scenario:** You want to add a new Hebrew BERT model to your analysis

**Steps:**

#### 1. Add model to project

**Update:** `EVALUATION_STANDARDIZATION_GUIDE.md` Section 2.1

```python
FINE_TUNING_MODELS = {
    # Existing models...
    "dicta-il/dictabert": "DictaBERT",

    # New model
    "your-org/your-hebrew-bert": "YourBERT"
}
```

#### 2. Add model to batch scripts

Update both:
- `scripts/run_all_hpo.sh`
- `scripts/run_all_experiments.sh`

Add the new model ID to their `MODELS=(...)` arrays.

#### 3. Run HPO

```bash
cd /workspace/project
bash scripts/run_all_hpo.sh
```

Best params are saved to:
`experiments/results/best_hyperparameters/best_params_{model}_{task}.json`

#### 4. Run training (3 seeds)

```bash
cd /workspace/project
bash scripts/run_all_experiments.sh
```

Outputs:
`experiments/results/full_fine-tuning/<model>/<task>/seed_<seed>/`

#### 5. Run evaluations

```bash
cd /workspace/project
bash scripts/run_evaluation_batch.sh
```

#### 8. Run analysis (automatically includes new model)

```bash
# All analysis scripts automatically detect new model results
python src/analyze_finetuning_results.py
python src/analyze_generalization.py
python src/analyze_comprehensive.py --create_figures

# Your new model will appear in all tables and figures!
```

### 10.2 That's It!

The analysis scripts are **model-agnostic**. They automatically:
- Scan `experiments/results/evaluation/` for all models
- Aggregate results for any model with 3 seeds
- Include new models in statistical comparisons
- Generate figures with new models

**No code changes needed!**

---

## 11. Troubleshooting

### 11.1 Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```bash
# Option 1: Reduce batch size
# Edit config file: batch_size: 16 → batch_size: 8

# Option 2: Use gradient accumulation
# Edit config file:
gradient_accumulation_steps: 2  # Effective batch size = 8 × 2 = 16

# Option 3: Use smaller model
# Try alephbert-base instead of dictabert (slightly smaller)

# Option 4: Disable fp16 (uses more memory but might help with stability)
fp16: false
```

#### Issue 2: Evaluation Results Not Found

**Symptom:**
```
FileNotFoundError: No evaluation results found for model X
```

**Solutions:**
```bash
# Check if evaluation was run
ls experiments/results/evaluation/seen_test/MODEL/TASK/seed_42/

# If empty, run evaluation
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_fine-tuning/MODEL/TASK/seed_42 \
  --data data/splits/test.csv \
  --task TASK \
  --output_dir experiments/results/evaluation/seen_test/MODEL/TASK/seed_42

# Verify file creation
ls experiments/results/evaluation/seen_test/MODEL/TASK/seed_42/
# Should see: eval_results_*.json, eval_predictions.json
```

#### Issue 3: Missing Seeds

**Symptom:**
```
Warning: Model X task Y only has 2/3 seeds. Skipping aggregation.
```

**Solutions:**
```bash
# Find which seed is missing
ls experiments/results/full_fine-tuning/MODEL/TASK/
# Expected: seed_42, seed_123, seed_456

# Re-run batch training (will skip completed runs)
bash scripts/run_all_experiments.sh

# Then evaluate (batch)
bash scripts/run_evaluation_batch.sh
```

#### Issue 4: VAST.ai Connection Lost

**Symptom:**
```
ssh: connect to host X.X.X.X port XXXXX: Connection refused
```

**Solutions:**
```bash
# Check if instance is still running on VAST.ai dashboard

# If instance stopped, restart it
# Note: Your files in /workspace/ should persist

# Re-bootstrap and resume with batch scripts
bash /workspace/project/scripts/instance_bootstrap.sh

# Re-run training (skips completed runs)
bash /workspace/project/scripts/run_all_experiments.sh
```

#### Issue 5: Incorrect Metric Values

**Symptom:**
```
Span F1 seems too high (>99%) or too low (<50%)
```

**Solutions:**
```bash
# Verify using src/utils/error_analysis.py
python -c "
from src.utils.error_analysis import compute_span_f1
import json

with open('experiments/results/evaluation/seen_test/MODEL/TASK/seed_42/eval_predictions.json') as f:
    preds = json.load(f)

metrics = compute_span_f1(preds)
print(f'Span F1: {metrics[\"f1\"]:.4f}')
print(f'Precision: {metrics[\"precision\"]:.4f}')
print(f'Recall: {metrics[\"recall\"]:.4f}')
"

# If values still seem wrong, check:
# 1. Is eval_predictions.json using correct format? (see EVALUATION_STANDARDIZATION_GUIDE.md Section 5.3)
# 2. Are IOB tags aligned with tokens?
# 3. Did tokenization use is_split_into_words=True?
```

### 11.2 Debug Mode

**Enable verbose logging:**

```bash
# For training
python src/idiom_experiment.py --mode full_finetune --config CONFIG --logging_level DEBUG

# For evaluation
python src/idiom_experiment.py --mode evaluate --model_checkpoint CHECKPOINT --data TEST_FILE --task TASK --output_dir OUTPUT --logging_level DEBUG

# For analysis
python src/analyze_finetuning_results.py --verbose
```

### 11.3 Validation Scripts

**Check tokenization alignment:**
```bash
python src/test_tokenization_alignment.py

# Expected output:
# ✓ Training set: 3360 samples
# ✓ Validation set: 480 samples
# ✓ Seen test: 480 samples
# ✓ Unseen test: 480 samples
# ✓ All columns present
# ✓ No missing values
# ✓ IOB tags aligned with tokens
```

**Check training outputs:**
```bash
python src/audit_results.py
```

---

## 12. Appendix: Script Templates

### 12.1 Custom Evaluation Script

**File:** `scripts/custom_evaluation.py`

```python
#!/usr/bin/env python3
"""
Custom evaluation script for specific analysis needs.

Usage:
    python scripts/custom_evaluation.py --model dictabert --task span
"""

import argparse
import json
from pathlib import Path
from src.utils.error_analysis import compute_span_f1, categorize_span_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--split', default='seen_test')
    args = parser.parse_args()

    # Load predictions for all seeds
    predictions_all_seeds = []
    for seed in [42, 123, 456]:
        pred_file = f"experiments/results/evaluation/{args.split}/{args.model}/{args.task}/seed_{seed}/eval_predictions.json"
        with open(pred_file) as f:
            predictions_all_seeds.extend(json.load(f))

    # Compute overall metrics
    if args.task == "span":
        metrics = compute_span_f1(predictions_all_seeds)
        print(f"\nOverall Span F1 (pooled across seeds): {metrics['f1']:.4f}")

    # Categorize errors
    error_counts = {}
    for pred in predictions_all_seeds:
        error = categorize_span_error(pred['true_tags'], pred['predicted_tags'])
        error_counts[error] = error_counts.get(error, 0) + 1

    print("\nError Distribution:")
    for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(predictions_all_seeds)) * 100
        print(f"  {error}: {count} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
```

### 12.2 Quick Evaluation Sanity Checks

```bash
# Count total evaluations
find experiments/results/evaluation -name "eval_results*.json" | wc -l

# Count by split
find experiments/results/evaluation/seen_test -name "eval_results*.json" | wc -l
find experiments/results/evaluation/unseen_test -name "eval_results*.json" | wc -l

# Count by task
find experiments/results/evaluation -path "*/cls/*" -name "eval_results*.json" | wc -l
find experiments/results/evaluation -path "*/span/*" -name "eval_results*.json" | wc -l
```

### 12.3 Batch Download Script

**File:** `scripts/download_evaluation_results.sh`

```bash
#!/bin/bash

# Download evaluation results from VAST.ai
# Usage: ./scripts/download_evaluation_results.sh

VAST_HOST="root@123.456.789.0"
VAST_PORT="12345"

echo "Downloading evaluation results from Google Drive..."

bash scripts/download_evaluation_results.sh

echo "Download complete!"
```

---

## End of Operations Guide

**Quick Reference Card:**

| Need to... | Use this command |
|------------|------------------|
| Activate environment | `source activate_env.sh` |
| Train a new model | `python src/idiom_experiment.py --mode full_finetune --model_id MODEL_ID --task TASK --config experiments/configs/training_config.yaml --seed 42` |
| Evaluate a model | `python src/idiom_experiment.py --mode evaluate --model_checkpoint PATH --data data/splits/test.csv --task TASK` |
| Aggregate results | `python src/analyze_finetuning_results.py` |
| Analyze generalization | `python src/analyze_generalization.py` |
| Categorize errors | `python scripts/categorize_all_errors.py` |
| Error visualization | `python src/analyze_error_distribution.py` |
| Check completeness | `find experiments/results/evaluation -name "eval_results*.json" | wc -l` |

**For detailed metric definitions and standards, see:** `EVALUATION_STANDARDIZATION_GUIDE.md`

**For mission-specific tasks, see:** `STEP_BY_STEP_MISSIONS.md`
