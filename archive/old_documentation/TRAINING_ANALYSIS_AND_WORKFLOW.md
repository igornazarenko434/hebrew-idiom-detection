# Training Analysis & Workflow Guide
## Hebrew Idiom Detection Project

**Created:** December 8, 2025
**Purpose:** Complete guide to understanding training outputs, model evaluation, and workflow

---

## Table of Contents

1. [What Gets Saved During Training](#what-gets-saved-during-training)
2. [Where Are The Model Weights](#where-are-the-model-weights)
3. [What's Synced to Google Drive](#whats-synced-to-google-drive)
4. [How to Analyze Results in PyCharm](#how-to-analyze-results-in-pycharm)
5. [How to Evaluate Trained Models](#how-to-evaluate-trained-models)
6. [Current Limitations & Missing Features](#current-limitations--missing-features)
7. [Workflow Summary](#workflow-summary)

---

## What Gets Saved During Training

### During Training (`full_finetune` mode)

**Location:** `/workspace/project/experiments/results/full_finetune/{model_name}/{task}/`

**Files Created:**

```
full_finetune/alephbert-base/cls/
├── checkpoint-864/              # Intermediate checkpoint (epoch 4)
│   ├── model.safetensors        # Model weights (503 MB)
│   ├── optimizer.pt             # Optimizer state (1 GB)
│   ├── scheduler.pt             # LR scheduler state
│   ├── trainer_state.json       # Training state
│   ├── rng_state.pth            # Random state (reproducibility)
│   ├── config.json              # Model config
│   ├── tokenizer files          # Tokenizer for this checkpoint
│   └── training_args.bin        # Training arguments
│
├── checkpoint-1080/             # Final checkpoint (epoch 5) - BEST MODEL
│   └── (same files as above)
│
├── model.safetensors            # Final model weights (503 MB)
├── config.json                  # Model configuration
├── tokenizer files              # Tokenizer (vocab.txt, etc.)
├── training_args.bin            # Training arguments used
├── training_results.json        # **MOST IMPORTANT** - All metrics
├── summary.txt                  # Quick text summary
└── logs/                        # TensorBoard logs
    └── events.out.tfevents.*    # Training/validation curves
```

---

### Key Files Explained

#### 1. **`training_results.json`** ⭐ MOST IMPORTANT

Contains ALL metrics from training:

```json
{
  "test_metrics": {
    "loss": 0.3397,
    "f1": 0.9404,               // ← Main metric!
    "accuracy": 0.9398,
    "precision": 0.9318,
    "recall": 0.9491,
    "confusion_matrix_tn": 201,
    "confusion_matrix_fp": 15,
    "confusion_matrix_fn": 11,
    "confusion_matrix_tp": 205,
    "runtime": 0.2421,
    "samples_per_second": 1784.31,
    "epoch": 5.0
  },
  "training_history": [
    {
      "epoch": 1.0,
      "loss": 0.5234,
      "eval_loss": 0.4123,
      "eval_f1": 0.7234,
      "eval_accuracy": 0.7345
    },
    // ... more epochs
  ],
  "config": {
    "learning_rate": 2e-05,
    "batch_size": 16,
    "num_epochs": 5,
    // ... all training parameters
  }
}
```

**This file has EVERYTHING you need for analysis!**

---

#### 2. **`checkpoint-{step}/`** - Model Checkpoints

**What they are:**
- Saved after each epoch (or based on `save_strategy`)
- Contains full model state + optimizer + scheduler

**Which one is the "best model"?**
- Training uses `load_best_model_at_end=True`
- Best model is determined by `metric_for_best_model="f1"` (highest validation F1)
- The **final checkpoint** (`checkpoint-1080` in your case) is loaded as the best model at the end
- That's why the root directory also has `model.safetensors` (it's the best model)

**File sizes:**
- `model.safetensors`: 503 MB (just model weights)
- `optimizer.pt`: ~1 GB (optimizer state - needed to resume training)
- Total per checkpoint: ~1.5 GB

**Do you need all checkpoints?**
- For inference/evaluation: NO - just need `model.safetensors` + `config.json` + tokenizer
- For resuming training: YES - need optimizer + scheduler states
- For analysis: Just the final checkpoint is enough

---

#### 3. **`logs/`** - TensorBoard Logs

Contains training curves:
- Loss per step
- Validation metrics per epoch
- Learning rate schedule
- Gradient norms

**View them:**
```bash
tensorboard --logdir experiments/results/full_finetune/alephbert-base/cls/logs/
```

---

## Where Are The Model Weights?

### On Vast.ai Volume

**Best model weights:**
```
/workspace/project/experiments/results/full_finetune/alephbert-base/cls/model.safetensors
```

This is the **final best model** (503 MB). It has:
- ✅ Best validation F1 score from all epochs
- ✅ Evaluated on test set
- ✅ Ready to use for inference

**All checkpoints:**
```
/workspace/project/experiments/results/full_finetune/alephbert-base/cls/checkpoint-{step}/
```

---

### On Google Drive (After Sync)

**Location:** `gdrive:Hebrew_Idiom_Detection/results/`

**What gets synced:**
- ✅ `model.safetensors` (503 MB) - Best model weights
- ✅ `training_results.json` - All metrics
- ✅ `summary.txt` - Quick summary
- ✅ All checkpoints (checkpoint-864/, checkpoint-1080/)
- ✅ Tokenizer files
- ✅ TensorBoard logs

**Total size per model:** ~3-4 GB (with all checkpoints)

---

### On Your Local Mac (After Download)

**You DON'T have the weights yet on your Mac!**

To get them:

```bash
# Option 1: Download from Google Drive via rclone
rclone copy gdrive:Hebrew_Idiom_Detection/results/full_finetune/alephbert-base/cls/ \
  ~/Desktop/training_results/alephbert-base-cls/

# Option 2: Just download the essential files (not full checkpoints)
rclone copy gdrive:Hebrew_Idiom_Detection/results/full_finetune/alephbert-base/cls/model.safetensors \
  ~/Desktop/models/alephbert-base-cls/
rclone copy gdrive:Hebrew_Idiom_Detection/results/full_finetune/alephbert-base/cls/config.json \
  ~/Desktop/models/alephbert-base-cls/
rclone copy gdrive:Hebrew_Idiom_Detection/results/full_finetune/alephbert-base/cls/tokenizer* \
  ~/Desktop/models/alephbert-base-cls/
rclone copy gdrive:Hebrew_Idiom_Detection/results/full_finetune/alephbert-base/cls/vocab.txt \
  ~/Desktop/models/alephbert-base-cls/
```

**Recommended:** Download just `training_results.json` for analysis, download model weights only when you need them for evaluation.

---

## What's Synced to Google Drive

### By `sync_to_gdrive.sh`

**What it syncs:**

```bash
# From /workspace/project/experiments/results/
#   → gdrive:Hebrew_Idiom_Detection/results/

# Everything in experiments/results/ including:
- Full model checkpoints (all epochs)
- Best model weights
- Training metrics JSON
- TensorBoard logs
- Summary files
```

**What it DOESN'T sync by default:**
- Cached models from HuggingFace (`/workspace/cache/`) - too large
- Python environment (`/workspace/env/`) - not needed
- Raw data (`/workspace/data/`) - already in Drive

**Size considerations:**
- Per training run: ~3-4 GB (with checkpoints)
- If you train 10 models: ~30-40 GB
- Google Drive free tier: 15 GB

**Recommendation:** After HPO or bulk training, consider:
1. Keep best models only (delete intermediate checkpoints)
2. Or use Google Drive paid plan

---

## How to Analyze Results in PyCharm

### Step 1: Download Results from Google Drive

**On your Mac:**

```bash
cd ~/Desktop
mkdir training_results

# Download just the JSON results (lightweight)
rclone copy gdrive:Hebrew_Idiom_Detection/results/ \
  ~/Desktop/training_results/ \
  --include "*.json" \
  --include "*.txt"

# Or download everything (including model weights)
rclone copy gdrive:Hebrew_Idiom_Detection/results/ \
  ~/Desktop/training_results/
```

---

### Step 2: Analyze in PyCharm

**Open PyCharm and create a new notebook:**

```python
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load training results
results_path = Path("~/Desktop/training_results/full_finetune/alephbert-base/cls/training_results.json").expanduser()

with open(results_path, 'r') as f:
    results = json.load(f)

# Extract test metrics
test_metrics = results['test_metrics']
print("Test Metrics:")
print(f"  F1 Score: {test_metrics['f1']:.4f}")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall: {test_metrics['recall']:.4f}")

# Confusion matrix
cm = [
    [test_metrics['confusion_matrix_tn'], test_metrics['confusion_matrix_fp']],
    [test_metrics['confusion_matrix_fn'], test_metrics['confusion_matrix_tp']]
]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Literal', 'Figurative'],
            yticklabels=['Literal', 'Figurative'])
plt.title('Confusion Matrix - AlephBERT')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Training history
history_df = pd.DataFrame(results['training_history'])

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history_df['epoch'], history_df['loss'], label='Train Loss')
axes[0, 0].plot(history_df['epoch'], history_df['eval_loss'], label='Val Loss')
axes[0, 0].set_title('Loss over Epochs')
axes[0, 0].legend()

# F1 Score
axes[0, 1].plot(history_df['epoch'], history_df['eval_f1'])
axes[0, 1].set_title('Validation F1 Score')
axes[0, 1].axhline(y=test_metrics['f1'], color='r', linestyle='--', label='Test F1')
axes[0, 1].legend()

# Accuracy
axes[1, 0].plot(history_df['epoch'], history_df['eval_accuracy'])
axes[1, 0].set_title('Validation Accuracy')

# Learning Rate (if saved)
if 'learning_rate' in history_df.columns:
    axes[1, 1].plot(history_df['epoch'], history_df['learning_rate'])
    axes[1, 1].set_title('Learning Rate Schedule')

plt.tight_layout()
plt.show()
```

---

### Step 3: Compare Multiple Models

```python
import glob

# Load all results
all_results = {}
results_dir = Path("~/Desktop/training_results/full_finetune/").expanduser()

for model_dir in results_dir.glob("*/cls/"):
    model_name = model_dir.parent.name
    results_file = model_dir / "training_results.json"

    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results[model_name] = json.load(f)

# Create comparison DataFrame
comparison = []
for model_name, results in all_results.items():
    test_metrics = results['test_metrics']
    comparison.append({
        'Model': model_name,
        'F1': test_metrics['f1'],
        'Accuracy': test_metrics['accuracy'],
        'Precision': test_metrics['precision'],
        'Recall': test_metrics['recall'],
        'Training Time (s)': results.get('training_metrics', {}).get('runtime', 'N/A')
    })

df_comparison = pd.DataFrame(comparison).sort_values('F1', ascending=False)
print(df_comparison)

# Plot comparison
plt.figure(figsize=(12, 6))
x = range(len(df_comparison))
width = 0.2

plt.bar([i - width*1.5 for i in x], df_comparison['F1'], width, label='F1', alpha=0.8)
plt.bar([i - width*0.5 for i in x], df_comparison['Accuracy'], width, label='Accuracy', alpha=0.8)
plt.bar([i + width*0.5 for i in x], df_comparison['Precision'], width, label='Precision', alpha=0.8)
plt.bar([i + width*1.5 for i in x], df_comparison['Recall'], width, label='Recall', alpha=0.8)

plt.xticks(x, df_comparison['Model'], rotation=45, ha='right')
plt.ylabel('Score')
plt.title('Model Comparison - Task 1 (Sentence Classification)')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## How to Evaluate Trained Models

### Built-in Evaluation Mode

**✅ AVAILABLE:** Your `idiom_experiment.py` has a comprehensive standalone evaluation mode!

**Available modes:**
- `zero_shot` - Evaluate pre-trained models (no fine-tuning)
- `full_finetune` - Train and auto-evaluate on test set
- `frozen_backbone` - Train (frozen) and auto-evaluate
- `hpo` - Hyperparameter optimization
- **`evaluate`** - Load trained model and evaluate on any dataset ⭐

---

### Option 1: Automatic Evaluation (During Training)

**When you run `full_finetune`**, it **automatically evaluates** on:
- ✅ Validation set (after each epoch)
- ✅ Test set (at the end)

**The results are saved in `training_results.json`:**

```json
{
  "test_metrics": {
    "f1": 0.9404,
    "accuracy": 0.9398,
    "precision": 0.9318,
    "recall": 0.9491,
    "confusion_matrix_tn": 201,
    "confusion_matrix_fp": 15,
    "confusion_matrix_fn": 11,
    "confusion_matrix_tp": 205
  }
}
```

**This gives you:**
- ✅ Performance on `test.csv` (in-domain, seen idioms)
- ✅ All metrics automatically saved
- ✅ Confusion matrix included

**What you DON'T have automatically:**
- ❌ Results on `unseen_idiom_test.csv` (6 unseen idioms)
- ❌ Evaluation on custom datasets

**→ Use Option 2 for these cases!**

---

### Option 2: Standalone Evaluation Mode ⭐ RECOMMENDED

Use the built-in `evaluate` mode to test trained models on any dataset.

#### Usage: Evaluate on Unseen Idioms

```bash
# Activate environment
source /workspace/env/bin/activate
cd /workspace/project

# Evaluate trained model on unseen idioms
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
  --data data/splits/unseen_idiom_test.csv \
  --task cls \
  --device cuda
```

#### What This Does:

1. **Loads your trained model** from the checkpoint directory
2. **Loads the specified dataset** (any CSV with `sentence` and `label` columns)
3. **Runs evaluation** with proper metrics
4. **Saves results** to `experiments/results/evaluation/` with structure:
   ```
   experiments/results/evaluation/
   └── full_finetune/
       └── alephbert-base/
           └── cls/
               └── eval_results_unseen_idiom_test_20251208_143052.json
   ```

#### Output Structure:

**Saved to:** `experiments/results/evaluation/{mode}/{model}/{task}/eval_results_{dataset}_{timestamp}.json`

```json
{
  "model_checkpoint": "experiments/results/full_finetune/alephbert-base/cls/",
  "dataset": "data/splits/unseen_idiom_test.csv",
  "task": "cls",
  "num_samples": 480,
  "metrics": {
    "loss": 0.4523,
    "f1": 0.8234,
    "accuracy": 0.8271,
    "precision": 0.8156,
    "recall": 0.8314,
    "confusion_matrix_tn": 198,
    "confusion_matrix_fp": 42,
    "confusion_matrix_fn": 41,
    "confusion_matrix_tp": 199
  },
  "config": {
    "batch_size": 16,
    "max_length": 128,
    "device": "cuda"
  }
}
```

---

### Complete Evaluation Workflow

#### Step 1: Train Model

```bash
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --device cuda
```

**Result:** Model saved to `experiments/results/full_finetune/alephbert-base/cls/`

---

#### Step 2: Evaluate on In-Domain Test Set

**Already done automatically!** Results in `experiments/results/full_finetune/alephbert-base/cls/training_results.json`

---

#### Step 3: Evaluate on Unseen Idioms

```bash
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
  --data data/splits/unseen_idiom_test.csv \
  --task cls \
  --device cuda
```

**Result:** Saved to `experiments/results/evaluation/full_finetune/alephbert-base/cls/eval_results_unseen_idiom_test_*.json`

---

#### Step 4: Evaluate on Custom Dataset

```bash
# Evaluate on any CSV with 'sentence' and 'label' columns
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
  --data path/to/your/custom_dataset.csv \
  --task cls \
  --device cuda \
  --output path/to/custom_results.json  # Optional: specify output path
```

---

### Advanced Evaluation Options

#### Evaluate on Validation Split Only

```bash
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
  --data data/expressions_data_tagged_v2.csv \
  --split validation \
  --task cls \
  --device cuda
```

#### Evaluate with Limited Samples (Quick Test)

```bash
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
  --data data/splits/test.csv \
  --max_samples 100 \
  --task cls \
  --device cuda
```

#### Evaluate Task 2 (Token Classification)

```bash
python src/idiom_experiment.py \
  --mode evaluate \
  --model_checkpoint experiments/results/full_finetune/alephbert-base/span/ \
  --data data/splits/test.csv \
  --task span \
  --device cuda
```

---

### What Gets Evaluated

#### For Task 1 (cls - Sequence Classification):

**Metrics computed:**
- ✅ F1 Score (macro-averaged)
- ✅ Accuracy
- ✅ Precision (macro-averaged)
- ✅ Recall (macro-averaged)
- ✅ Confusion Matrix (TN, FP, FN, TP)

#### For Task 2 (span - Token Classification):

**Metrics computed:**
- ✅ F1 Score (seqeval - entity level)
- ✅ Precision (seqeval - entity level)
- ✅ Recall (seqeval - entity level)
- ✅ Accuracy (token level)

---

### Evaluation Results Organization

**All evaluation results are saved to:**

```
experiments/results/evaluation/
├── full_finetune/          # Evaluating full fine-tuned models
│   ├── alephbert-base/
│   │   ├── cls/
│   │   │   ├── eval_results_unseen_idiom_test_20251208_143052.json
│   │   │   ├── eval_results_test_20251208_150234.json
│   │   │   └── eval_results_custom_dataset_20251208_152145.json
│   │   └── span/
│   │       └── eval_results_unseen_idiom_test_20251208_160312.json
│   ├── dictabert/
│   └── xlm-roberta-base/
└── frozen_backbone/        # Evaluating frozen backbone models
    └── ...
```

**Structure mirrors training outputs!**

**This gets synced to Google Drive** automatically by `sync_to_gdrive.sh`:
- From: `experiments/results/evaluation/`
- To: `gdrive:Hebrew_Idiom_Detection/results/evaluation/`

---

## Current Limitations & Missing Features

### ❌ What's Missing in `idiom_experiment.py`

1. **No cross-validation mode**
   - Can't easily run k-fold cross-validation
   - Would need to implement manually

2. **No ensemble evaluation**
   - Can't evaluate ensemble of multiple models
   - Would need custom script to aggregate predictions

3. **No automated unseen idiom testing**
   - Training auto-evaluates on `test.csv` (seen idioms)
   - Need to manually run `--mode evaluate` with `unseen_idiom_test.csv`
   - Could add this to training pipeline as additional automatic step

---

### ✅ What Works Perfectly

1. **Training & Auto-Evaluation**
   - ✅ Trains on `train.csv`
   - ✅ Validates on `validation.csv` (each epoch)
   - ✅ Tests on `test.csv` (at end)
   - ✅ Saves best model based on validation F1
   - ✅ Saves all metrics to JSON

2. **Standalone Evaluation Mode** ⭐ NEW!
   - ✅ Load any trained model checkpoint
   - ✅ Evaluate on any dataset (CSV with sentence/label)
   - ✅ Full metrics (F1, accuracy, precision, recall, confusion matrix)
   - ✅ Results saved to organized structure
   - ✅ Supports both Task 1 (cls) and Task 2 (span)

3. **Model Checkpointing**
   - ✅ Saves checkpoints per epoch
   - ✅ Loads best model at end
   - ✅ Saves final model with best weights

4. **Hyperparameter Optimization**
   - ✅ HPO mode with Optuna
   - ✅ Saves best hyperparameters
   - ✅ Can resume interrupted HPO

5. **Results Organization**
   - ✅ Hierarchical folder structure
   - ✅ Separate folders per model/task/mode
   - ✅ Evaluation results mirror training structure
   - ✅ Clear naming conventions with timestamps

6. **Google Drive Sync**
   - ✅ Syncs all results automatically
   - ✅ Includes evaluation results
   - ✅ Preserves folder structure
   - ✅ Doesn't re-upload unchanged files

---

### ✅ What's Correctly Configured

**Your configs are perfect:**

#### `training_config.yaml`
```yaml
learning_rate: 2e-5        # ✅ Good for Task 1
batch_size: 16             # ✅ Fits 24GB GPU
num_epochs: 5              # ✅ Reasonable
warmup_ratio: 0.1          # ✅ Standard
weight_decay: 0.01         # ✅ Prevents overfitting
early_stopping_patience: 3 # ✅ Stops if no improvement
save_total_limit: 2        # ✅ Keeps last 2 checkpoints
load_best_model_at_end: true  # ✅ Uses best validation model
metric_for_best_model: "f1"   # ✅ Optimizes for F1
```

#### `hpo_config.yaml`
```yaml
n_trials: 15               # ✅ Good balance
learning_rate:             # ✅ Reasonable search space
  - 1e-5
  - 2e-5
  - 3e-5
  - 5e-5
batch_size: [8, 16, 32]    # ✅ Covers GPU memory range
# ... all parameters well-chosen
```

#### `run_all_hpo.sh`
- ✅ Runs 10 studies (5 models × 2 tasks)
- ✅ Uses correct paths
- ✅ Saves to organized folders
- ✅ Logs everything

#### `run_all_experiments.sh`
- ✅ Runs 30 experiments (5 models × 2 tasks × 3 seeds)
- ✅ Uses best hyperparameters from HPO
- ✅ Organized output structure

#### `sync_to_gdrive.sh`
- ✅ Syncs from correct paths (`experiments/results/`)
- ✅ Uploads to organized Drive structure
- ✅ Excludes unnecessary files (`.DS_Store`, `__pycache__`)
- ✅ Shows progress and summary

**Everything aligns perfectly!** ✅

---

## Workflow Summary

### Current Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│  LOCAL MAC (Development)                                │
│  ├─ Edit code in PyCharm                                │
│  ├─ Commit & push to GitHub                             │
│  └─ Analyze results (download from Google Drive)        │
└─────────────────────────────────────────────────────────┘
                    │
                    │ git push
                    ▼
┌─────────────────────────────────────────────────────────┐
│  GITHUB (Code Source)                                   │
│  └─ All code, configs, scripts                          │
└─────────────────────────────────────────────────────────┘
                    │
                    │ git pull (in bootstrap.sh)
                    ▼
┌─────────────────────────────────────────────────────────┐
│  VAST.AI VOLUME (/workspace/)                           │
│  ├─ env/          (Python + packages - permanent)       │
│  ├─ data/         (Dataset - permanent)                 │
│  ├─ project/      (Git clone - updates on bootstrap)    │
│  ├─ outputs/      (Training results - persistent)       │
│  ├─ cache/        (HF models - persistent)              │
│  └─ config/       (rclone, .env - permanent)            │
└─────────────────────────────────────────────────────────┘
                    ▲
                    │ attached
┌───────────────────┴─────────────────────────────────────┐
│  VAST.AI GPU INSTANCE (Ephemeral)                       │
│  ├─ Rent when needed                                    │
│  ├─ Attach volume at /workspace                         │
│  ├─ Run: bash /workspace/project/scripts/instance_bootstrap.sh  │
│  ├─ Train models                                        │
│  ├─ Sync to Google Drive                                │
│  └─ Destroy instance                                    │
└─────────────────────────────────────────────────────────┘
                    │
                    │ rclone sync
                    ▼
┌─────────────────────────────────────────────────────────┐
│  GOOGLE DRIVE (Long-term Storage)                       │
│  └─ Hebrew_Idiom_Detection/                             │
│      ├─ results/  (All training outputs)                │
│      │   ├─ training_results.json  ⭐ Download this     │
│      │   ├─ model.safetensors       (for evaluation)    │
│      │   └─ checkpoints/            (optional)          │
│      └─ logs/     (TensorBoard)                         │
└─────────────────────────────────────────────────────────┘
                    │
                    │ rclone copy (manual, when needed)
                    ▼
┌─────────────────────────────────────────────────────────┐
│  LOCAL MAC (Analysis)                                   │
│  └─ ~/Desktop/training_results/                         │
│      └─ Analyze in PyCharm/Jupyter                      │
└─────────────────────────────────────────────────────────┘
```

---

### Step-by-Step: Full Training Cycle

#### On Vast.ai:

```bash
# 1. Rent instance + attach volume
# 2. SSH to instance
ssh -p <PORT> root@<IP>

# 3. Bootstrap (30 seconds)
bash /workspace/project/scripts/instance_bootstrap.sh

# 4. Activate environment
source /workspace/env/bin/activate
cd /workspace/project

# 5. Train model
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --device cuda

# 6. Sync to Google Drive
bash scripts/sync_to_gdrive.sh

# 7. Exit and destroy instance
exit
# (Destroy in Vast.ai console)
```

#### On Your Mac (Analysis):

```bash
# 1. Download results from Google Drive
rclone copy gdrive:Hebrew_Idiom_Detection/results/ \
  ~/Desktop/training_results/ \
  --include "*.json" --include "*.txt"

# 2. Open PyCharm
# 3. Create analysis notebook (see examples above)
# 4. Load training_results.json
# 5. Visualize metrics, compare models
```

---

## Recommendations

### Immediate Next Steps

1. ✅ **Your current workflow is perfect** - keep using it!

2. 📊 **For analysis:** Download `training_results.json` files to your Mac and analyze in PyCharm

3. 💾 **For model weights:** Only download when you need them for evaluation (500 MB each)

4. 🎯 **For unseen idiom evaluation:** Use Workaround 2 or 3 (create simple eval script)

---

### Optional Improvements

If you want to enhance the workflow:

1. **Add evaluation mode to `idiom_experiment.py`**
   - Add `--mode evaluate`
   - Load trained model from checkpoint
   - Evaluate on any CSV file
   - Save results to JSON

2. **Create analysis scripts**
   - `scripts/analyze_results.py` - Compare all models
   - `scripts/plot_training_curves.py` - Visualize training
   - `scripts/generate_report.py` - Auto-generate report

3. **Optimize storage**
   - After HPO, delete intermediate checkpoints
   - Keep only best model + training_results.json
   - Reduces Google Drive usage

---

## Summary

### ✅ What You Have

- ✅ Complete training pipeline
- ✅ Automatic evaluation on test set
- ✅ **Standalone evaluation mode** for any dataset ⭐
- ✅ All metrics saved to JSON
- ✅ Model weights saved (best model)
- ✅ TensorBoard logs
- ✅ Google Drive sync working
- ✅ Persistent volume setup
- ✅ All configs aligned perfectly

### ❌ What's Missing (Very Minor)

- ❌ Automated unseen idiom evaluation (manual command works)
- ❌ Cross-validation mode (not needed for this project)
- ❌ Ensemble evaluation (nice-to-have)

### 🎯 Bottom Line

**Your setup is production-ready!** You have everything needed:
- ✅ Train all 5 models on both tasks
- ✅ Run HPO with Optuna
- ✅ Evaluate on in-domain test set (automatic)
- ✅ Evaluate on unseen idioms (standalone mode)
- ✅ Evaluate on custom datasets (standalone mode)
- ✅ Get all metrics with confusion matrices
- ✅ Analyze results in PyCharm
- ✅ Use trained models for inference

**Everything aligns and works correctly!** 🚀

---

**Created:** December 8, 2025
**Last Updated:** After first successful training run
**Status:** ✅ Production Ready
