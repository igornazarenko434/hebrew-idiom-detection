# Training Output Organization Guide

This document explains how all training runs, evaluations, and experiments are organized in the `experiments/` directory.

**Last Updated:** 2025-12-05

---

## Overview

All experimental outputs are saved to the `experiments/` directory with a hierarchical structure that makes it easy to:

1. **Compare different models** (AlephBERT, DictaBERT, mBERT, XLM-RoBERTa, etc.)
2. **Compare different tasks** (sequence classification vs. token classification)
3. **Compare different training modes** (zero-shot, full fine-tuning, frozen backbone, HPO)
4. **Track training progress** (TensorBoard logs, loss curves, validation metrics)
5. **Reproduce results** (complete configuration and metrics saved for each run)

---

## Folder Structure

### 1. Zero-Shot Evaluation (Mission 3.2-3.4)

**Location:** `experiments/results/zero_shot/`

**Structure:**
```
experiments/results/zero_shot/
└── {model_name}_{split_name}_{task_name}.json
```

**Example Files:**
```
experiments/results/zero_shot/
├── alephbert-base_all_cls.json
├── alephbert-base_all_span.json
├── dictabert_all_cls.json
├── mbert_all_cls.json
└── xlm-roberta-base_all_span.json
```

**Contents:**
- Complete evaluation metrics (accuracy, F1, precision, recall)
- Confusion matrices
- Per-class performance
- Dataset validation results

**How to Run:**
```bash
# Zero-shot evaluation for Task 1 (sequence classification)
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --task cls \
  --data data/splits/test.csv

# Zero-shot evaluation for Task 2 (token classification)
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --task span \
  --data data/splits/test.csv
```

---

### 2. Full Fine-Tuning (Mission 4.2, 4.6)

**Location:** `experiments/results/full_finetune/`

**Structure:**
```
experiments/results/full_finetune/
└── {model_name}/              # e.g., alephbert-base, dictabert
    └── {task}/                # cls or span
        ├── checkpoint-216/    # Best model checkpoint
        ├── checkpoint-432/    # Latest checkpoint
        ├── logs/              # TensorBoard logs
        │   └── events.out.tfevents.*
        ├── training_results.json  # Complete results
        └── summary.txt        # Quick reference
```

**Example:**
```
experiments/results/full_finetune/
├── alephbert-base/
│   ├── cls/
│   │   ├── checkpoint-216/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── trainer_state.json
│   │   │   └── training_args.bin
│   │   ├── logs/
│   │   │   └── events.out.tfevents.1733432789.Igors-MacBook-Pro-2.local.12345.0
│   │   ├── training_results.json
│   │   └── summary.txt
│   └── span/
│       ├── checkpoint-324/
│       ├── logs/
│       ├── training_results.json
│       └── summary.txt
├── dictabert/
│   ├── cls/
│   └── span/
└── xlm-roberta-base/
    ├── cls/
    └── span/
```

**File Contents:**

**`training_results.json`:**
```json
{
  "model": "onlplab/alephbert-base",
  "task": "cls",
  "mode": "full_finetune",
  "freeze_backbone": false,
  "config": {
    "learning_rate": 2e-05,
    "batch_size": 16,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01
  },
  "dataset": {
    "train_samples": 3456,
    "dev_samples": 432,
    "test_samples": 432
  },
  "train_metrics": {
    "runtime": 432.56,
    "samples_per_second": 8.0,
    "final_loss": 0.2345,
    "epochs_completed": 5
  },
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
      "step": 216,
      "loss": 0.5234,
      "learning_rate": 1.8e-05,
      "eval_loss": 0.4567,
      "eval_f1": 0.7234,
      "eval_accuracy": 0.7345
    },
    {
      "epoch": 2.0,
      "step": 432,
      "loss": 0.3456,
      "learning_rate": 1.6e-05,
      "eval_loss": 0.3123,
      "eval_f1": 0.8012,
      "eval_accuracy": 0.8123
    }
  ]
}
```

**`summary.txt`:**
```
Hebrew Idiom Detection - Training Summary
==========================================

Model: onlplab/alephbert-base
Task: cls (Sequence Classification)
Mode: full_finetune

Configuration:
--------------
Learning Rate: 2e-05
Batch Size: 16
Epochs: 5
Warmup Ratio: 0.1
Weight Decay: 0.01

Dataset:
--------
Train: 3,456 samples
Validation: 432 samples
Test: 432 samples

Final Test Results:
-------------------
F1 Score: 0.8723
Accuracy: 0.8765
Precision: 0.8654
Recall: 0.8792

Training Time: 7.2 minutes (432.56 seconds)
```

**`logs/` Directory:**
Contains TensorBoard event files. View with:
```bash
tensorboard --logdir experiments/results/full_finetune/alephbert-base/cls/logs/
```

**How to Run:**
```bash
# Full fine-tuning for Task 1
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml

# Full fine-tuning for Task 2
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task span \
  --config experiments/configs/training_config.yaml
```

---

### 3. Frozen Backbone Training (Mission 4.2)

**Location:** `experiments/results/frozen_backbone/`

**Structure:** Same as full fine-tuning, but under `frozen_backbone/` instead of `full_finetune/`

```
experiments/results/frozen_backbone/
└── {model_name}/
    └── {task}/
        ├── checkpoint-*/
        ├── logs/
        ├── training_results.json
        └── summary.txt
```

**Difference from Full Fine-Tuning:**
- Only trains the classification head (final layer)
- Backbone transformer layers are frozen
- Faster training (fewer parameters to update)
- Usually lower performance than full fine-tuning

**How to Run:**
```bash
python src/idiom_experiment.py \
  --mode frozen_backbone \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml
```

---

### 4. Hyperparameter Optimization (Mission 4.3-4.5)

**Location:** `experiments/hpo_results/` and `experiments/results/`

**Structure:**
```
experiments/hpo_results/
└── {model_name}/              # e.g., alephbert-base
    └── {task}/                # cls or span
        ├── trial_0/
        │   ├── checkpoint-*/
        │   ├── logs/
        │   ├── training_results.json
        │   └── summary.txt
        ├── trial_1/
        │   ├── checkpoint-*/
        │   ├── logs/
        │   ├── training_results.json
        │   └── summary.txt
        └── trial_14/           # 15 trials total (0-14)
            ├── checkpoint-*/
            ├── logs/
            ├── training_results.json
            └── summary.txt

experiments/results/optuna_studies/
└── {model_name}_{task}_hpo.db  # Optuna SQLite database

experiments/results/best_hyperparameters/
└── best_params_{model_name}_{task}.json  # Best hyperparameters
```

**Example:**
```
experiments/hpo_results/
├── alephbert-base/
│   ├── cls/
│   │   ├── trial_0/
│   │   │   ├── checkpoint-216/
│   │   │   ├── logs/
│   │   │   ├── training_results.json
│   │   │   └── summary.txt
│   │   ├── trial_1/
│   │   └── trial_14/
│   └── span/
│       ├── trial_0/
│       └── trial_14/
└── dictabert/
    ├── cls/
    └── span/

experiments/results/optuna_studies/
├── alephbert-base_cls_hpo.db
├── alephbert-base_span_hpo.db
├── dictabert_cls_hpo.db
└── dictabert_span_hpo.db

experiments/results/best_hyperparameters/
├── best_params_alephbert-base_cls.json
├── best_params_alephbert-base_span.json
├── best_params_dictabert_cls.json
└── best_params_dictabert_span.json
```

**`best_params_{model}_{task}.json` Contents:**
```json
{
  "model": "onlplab/alephbert-base",
  "task": "cls",
  "best_trial_number": 7,
  "best_validation_f1": 0.8923,
  "best_hyperparameters": {
    "learning_rate": 3e-05,
    "batch_size": 16,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01
  },
  "study_name": "alephbert-base_cls_hpo",
  "n_trials": 15,
  "fixed_parameters": {
    "max_length": 128,
    "fp16": false,
    "seed": 42
  }
}
```

**How to Run:**
```bash
# Run HPO for one model-task combination
python src/idiom_experiment.py \
  --mode hpo \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/hpo_config.yaml \
  --device cuda

# View Optuna study results
optuna-dashboard sqlite:///experiments/results/optuna_studies/alephbert-base_cls_hpo.db
```

**Batch Run All HPO Studies:**
```bash
# Create and run batch script (5 models × 2 tasks = 10 studies)
bash scripts/run_all_hpo.sh
```

---

## Viewing Training Progress

### TensorBoard (Recommended)

**View a Single Run:**
```bash
# View specific training run
tensorboard --logdir experiments/results/full_finetune/alephbert-base/cls/logs/

# Open browser to http://localhost:6006
```

**Compare Multiple Runs:**
```bash
# Compare all models for Task 1 (sequence classification)
tensorboard --logdir experiments/results/full_finetune/ \
  --path_prefix alephbert-base/cls,dictabert/cls,xlm-roberta-base/cls

# Compare all tasks for AlephBERT
tensorboard --logdir experiments/results/full_finetune/alephbert-base/

# Compare all HPO trials for one model
tensorboard --logdir experiments/hpo_results/alephbert-base/cls/
```

**Metrics Available in TensorBoard:**
- Training loss (per step)
- Validation loss (per epoch)
- Validation F1 score (per epoch)
- Validation accuracy (per epoch)
- Learning rate schedule
- Gradient norms (if enabled)

### Programmatic Analysis

**Load Results from JSON:**
```python
import json
from pathlib import Path

# Load training results
results_path = Path("experiments/results/full_finetune/alephbert-base/cls/training_results.json")
with open(results_path) as f:
    results = json.load(f)

# Access metrics
print(f"Test F1: {results['test_metrics']['f1']:.4f}")
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")

# Plot learning curves
import matplotlib.pyplot as plt

history = results['training_history']
epochs = [h['epoch'] for h in history]
train_loss = [h['loss'] for h in history]
val_f1 = [h['eval_f1'] for h in history]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, val_f1)
plt.title('Validation F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')

plt.tight_layout()
plt.savefig('learning_curves.png')
```

---

## Running Experiments on Different Platforms

### Local (Mac/Linux)

**CPU Training (Small Experiments):**
```bash
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --device cpu \
  --max_samples 500 \
  --num_epochs 1
```

**MPS (Mac GPU):**
```bash
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --device mps
```

### VAST.ai (Cloud GPU)

**Setup:**
1. Rent VAST.ai instance (RTX 3090/4090 recommended)
2. Connect via SSH: `ssh -p {port} root@{ip}`
3. Clone repository or upload code
4. Download dataset: `gdown {google_drive_file_id}`
5. Install dependencies: `pip install -r requirements.txt`

**Run Training:**
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

**Download Results:**
```bash
# On your local machine
scp -P {vast_port} root@{vast_ip}:~/hebrew-idiom-detection/experiments/results/* ./local_results/
```

### Output Location Consistency

**All platforms save to the same folder structure:**
- Local (CPU/MPS): `experiments/results/full_finetune/{model}/{task}/`
- VAST.ai (CUDA): `experiments/results/full_finetune/{model}/{task}/`
- Optuna HPO: `experiments/hpo_results/{model}/{task}/trial_{n}/`

**No special handling needed** - the folder structure is platform-agnostic and determined only by:
1. Training mode (zero_shot, full_finetune, frozen_backbone, hpo)
2. Model name (alephbert-base, dictabert, etc.)
3. Task (cls, span)

---

## Debugging Failed Runs

### Check Training Progress

**View TensorBoard logs:**
```bash
tensorboard --logdir experiments/results/full_finetune/alephbert-base/cls/logs/
```

**Check latest checkpoint:**
```bash
ls -lh experiments/results/full_finetune/alephbert-base/cls/checkpoint-*/
```

**Read summary file:**
```bash
cat experiments/results/full_finetune/alephbert-base/cls/summary.txt
```

### Common Issues

**1. Out of Memory (OOM)**
- **Symptom:** CUDA out of memory error
- **Solution:** Reduce batch size or use gradient accumulation
```bash
python src/idiom_experiment.py \
  --mode full_finetune \
  --batch_size 8 \
  --gradient_accumulation_steps 2  # Effective batch size = 16
```

**2. No Improvement / Overfitting**
- **Symptom:** Validation F1 doesn't improve or decreases
- **Solution:** Check TensorBoard for training vs. validation curves
- **Action:** Adjust learning rate, add regularization, or use early stopping

**3. Alignment Errors (Task 2 only)**
- **Symptom:** KeyError or IndexError during training
- **Solution:** Run tokenization alignment test first
```bash
python src/test_tokenization_alignment.py
```

---

## Comparing Results Across Models

### Create Comparison Table

**Python Script:**
```python
import json
from pathlib import Path
import pandas as pd

# Collect results
results = []
base_dir = Path("experiments/results/full_finetune")

for model_dir in base_dir.iterdir():
    if not model_dir.is_dir():
        continue

    for task_dir in model_dir.iterdir():
        if not task_dir.is_dir():
            continue

        results_file = task_dir / "training_results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        results.append({
            "Model": data["model"],
            "Task": data["task"],
            "F1": data["test_metrics"]["f1"],
            "Accuracy": data["test_metrics"]["accuracy"],
            "Precision": data["test_metrics"]["precision"],
            "Recall": data["test_metrics"]["recall"],
            "Training Time (s)": data["train_metrics"]["runtime"]
        })

# Create comparison table
df = pd.DataFrame(results)
df = df.sort_values(["Task", "F1"], ascending=[True, False])

print(df.to_string(index=False))

# Save to CSV
df.to_csv("experiments/results/model_comparison.csv", index=False)
```

---

## Mission Progress Tracking

### Missions Completed
- ✅ Mission 3.2: Zero-shot evaluation framework
- ✅ Mission 4.1: Training configuration setup
- ✅ Mission 4.2: Training pipeline implementation

### Current Output Organization Status

**✅ All modes properly organized:**
1. Zero-shot: `experiments/results/zero_shot/`
2. Full fine-tuning: `experiments/results/full_finetune/{model}/{task}/`
3. Frozen backbone: `experiments/results/frozen_backbone/{model}/{task}/`
4. HPO trials: `experiments/hpo_results/{model}/{task}/trial_{n}/`
5. HPO studies: `experiments/results/optuna_studies/{model}_{task}_hpo.db`
6. Best params: `experiments/results/best_hyperparameters/best_params_{model}_{task}.json`

**✅ Comprehensive logging enabled:**
- TensorBoard logs in `{output_dir}/logs/`
- Complete training history in `training_results.json`
- Quick reference in `summary.txt`
- Per-epoch metrics tracked

**✅ Platform-agnostic:**
- Works on local Mac (CPU/MPS)
- Works on VAST.ai (CUDA)
- Same folder structure regardless of platform

**✅ Easy debugging:**
- TensorBoard for visualization
- JSON files for programmatic analysis
- Summary files for quick reference
- Checkpoints for resuming training

---

## Summary

**All training runs are organized hierarchically by:**
1. **Mode** (zero_shot, full_finetune, frozen_backbone, hpo)
2. **Model** (alephbert-base, dictabert, mbert, xlm-roberta-base)
3. **Task** (cls, span)
4. **Trial** (for HPO only)

**Every run saves:**
- Complete configuration
- Training history (per-epoch metrics)
- Final test results
- TensorBoard logs
- Model checkpoints
- Summary text file

**Easy comparison:**
- TensorBoard for visual comparison
- JSON files for programmatic comparison
- Consistent structure across all runs

**Platform-agnostic:**
- Works identically on local Mac, Linux, and VAST.ai
- No special handling needed for different platforms
