# Hebrew Idiom Detection - Complete Implementation Guide
**Easy-to-Follow Instructions for Running All Experiments**

Version: 2.0
Last Updated: December 6, 2025
Based on: FINAL_PRD v3.0, STEP_BY_STEP_MISSIONS.md, Actual Implementation

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Zero-Shot Evaluation](#zero-shot-evaluation)
5. [Model Training](#model-training)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [LLM Evaluation](#llm-evaluation)
8. [VAST.ai GPU Training](#vastai-gpu-training)
9. [Results Analysis](#results-analysis)
10. [Complete Experiment Matrix](#complete-experiment-matrix)

---

## Quick Start

### Minimal Commands to Get Started

```bash
# 1. Setup environment
conda create -n hebrew-idiom python=3.10
conda activate hebrew-idiom
pip install -r requirements.txt

# 2. Verify data is ready
python -c "import pandas as pd; df = pd.read_csv('data/splits/train.csv'); print(f'Train: {len(df)} samples')"

# 3. Run zero-shot evaluation (baseline)
python src/idiom_experiment.py --mode zero_shot --model_id onlplab/alephbert-base --task cls --device cpu

# 4. Train a model (requires GPU)
python src/idiom_experiment.py --mode full_finetune --config experiments/configs/training_config.yaml --task cls --device cuda
```

---

## Environment Setup

### 1. Initial Setup (One-Time)

```bash
# Clone repository (if not already done)
git clone <your-repo-url>
cd hebrew-idiom-detection

# Create conda environment
conda create -n hebrew-idiom python=3.10
conda activate hebrew-idiom

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 2. Environment Activation (Every Session)

```bash
conda activate hebrew-idiom
cd /path/to/hebrew-idiom-detection
```

### 3. Verify Data Files Exist

```bash
ls data/splits/
# Should show: train.csv, validation.csv, test.csv, split_expressions.json

# Check data splits
python -c "
import pandas as pd
train = pd.read_csv('data/splits/train.csv')
val = pd.read_csv('data/splits/validation.csv')
test = pd.read_csv('data/splits/test.csv')
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
"
```

**Expected output:** `Train: 3456, Val: 432, Test: 432, Unseen: 480, Total: 4800`

---

## Data Preparation

### Check Dataset Statistics

```bash
# View dataset statistics
cat experiments/results/dataset_statistics.txt

# Or generate fresh statistics
python src/data_preperation.py --action stats
```

### Validate IOB2 Alignment

```bash
# Test tokenization alignment for Task 2
python src/test_tokenization_alignment.py

# View results
cat experiments/results/tokenization_alignment_test.txt
```

### Understanding the Data Splits

**Hybrid Strategy (Seen + Unseen Idioms):**
- **Train:** 3,456 samples (54 seen idioms) – 64 sentences per idiom (32 literal + 32 figurative)
- **Validation:** 432 samples (54 seen idioms) – 8 sentences per idiom (4 literal + 4 figurative)
- **Test (in-domain):** 432 samples (54 seen idioms) – 8 sentences per idiom (4 literal + 4 figurative)
- **Unseen Idiom Test:** 480 samples (6 held-out idioms) – 80 sentences per idiom (40 literal + 40 figurative)

**Perfect 50/50 literal/figurative balance maintained across all splits**

**Unseen Idioms (Zero-Shot Evaluation):**
1. חתך פינה (cut corner)
2. חצה קו אדום (crossed red line)
3. נשאר מאחור (stayed behind)
4. שבר שתיקה (broke silence)
5. איבד את הראש (lost head)
6. רץ אחרי הזנב של עצמו (chased own tail)

This allows you to report both **in-domain** and **zero-shot** results without regenerating splits.

---

## Zero-Shot Evaluation

### What is Zero-Shot?
Evaluate pre-trained models **without any fine-tuning** - just use them as-is.

### Available Models

| Model ID | Name | Type | Priority |
|----------|------|------|----------|
| `onlplab/alephbert-base` | AlephBERT | Hebrew-specific | ⭐⭐⭐ |
| `dicta-il/dictabert` | DictaBERT | Hebrew-specific | ⭐⭐⭐ |
| `bert-base-multilingual-cased` | mBERT | Multilingual | ⭐⭐⭐ |
| `xlm-roberta-base` | XLM-RoBERTa | Multilingual | ⭐⭐⭐ |
| `dicta-il/alephbertgimmel-base` | AlephBERT-Gimmel | Hebrew-specific | ⭐⭐⭐ |

### Task Options

- `cls` = Task 1: Sequence Classification (literal vs. figurative)
- `span` = Task 2: Token Classification (IOB2 tagging)
- `both` = Run both tasks sequentially

### Zero-Shot Commands

#### Task 1: Sentence Classification

```bash
# AlephBERT (recommended starting point)
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/test.csv \
  --task cls \
  --device cpu

# DictaBERT
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id dicta-il/dictabert \
  --data data/splits/test.csv \
  --task cls \
  --device cpu

# mBERT (multilingual baseline)
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id bert-base-multilingual-cased \
  --data data/splits/test.csv \
  --task cls \
  --device cpu

# XLM-RoBERTa
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id xlm-roberta-base \
  --data data/splits/test.csv \
  --task cls \
  --device cpu
```

#### Task 2: Token Classification (IOB2)

```bash
# AlephBERT - Token Classification
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/test.csv \
  --task span \
  --device cpu

# DictaBERT - Token Classification
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id dicta-il/dictabert \
  --data data/splits/test.csv \
  --task span \
  --device cpu
```

#### Both Tasks Together

```bash
# Run both tasks in one command
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/test.csv \
  --task both \
  --device cpu
```

### Results Location

Results saved to: `experiments/results/zero_shot/<model_name>_<task>.json`

Example:
```bash
cat experiments/results/zero_shot/alephbert-base_cls.json
```

---

## Model Training

### Training Modes

1. **Zero-Shot** - No training (covered above)
2. **Full Fine-Tuning** - Train all model parameters (⭐ PRIMARY)
3. **Frozen Backbone** - Train only classification head (fast baseline)
4. **HPO** - Hyperparameter optimization with Optuna

### Device Options

- `cpu` - Use CPU (slow, for testing only)
- `cuda` - Use NVIDIA GPU (required for real training)
- `mps` - Use Apple Silicon GPU (Mac M1/M2)

### Configuration Files

- **Training Config:** `experiments/configs/training_config.yaml`
- **HPO Config:** `experiments/configs/hpo_config.yaml`

### Full Fine-Tuning

#### Task 1: Sequence Classification

```bash
# Using default config
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task cls \
  --device cuda

# Override specific parameters
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task cls \
  --model_id onlplab/alephbert-base \
  --learning_rate 3e-5 \
  --batch_size 32 \
  --num_epochs 10 \
  --device cuda

# Custom output directory
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task cls \
  --output_dir experiments/results/my_custom_run \
  --device cuda
```

#### Task 2: Token Classification (IOB2)

```bash
# ⚠️ IMPORTANT: Task 2 uses IOB2 alignment automatically!
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task span \
  --device cuda

# With custom parameters
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task span \
  --model_id dicta-il/dictabert \
  --learning_rate 2e-5 \
  --batch_size 16 \
  --num_epochs 8 \
  --device cuda
```

### Frozen Backbone Training

Train only the classification head (much faster, lower performance):

```bash
# Task 1 with frozen backbone
python src/idiom_experiment.py \
  --mode frozen_backbone \
  --config experiments/configs/training_config.yaml \
  --task cls \
  --device cuda

# Task 2 with frozen backbone
python src/idiom_experiment.py \
  --mode frozen_backbone \
  --config experiments/configs/training_config.yaml \
  --task span \
  --device cuda
```

### Training Configuration Parameters

Edit `experiments/configs/training_config.yaml`:

```yaml
# Model configuration
model_id: "onlplab/alephbert-base"
task: "cls"  # or "span"

# Training hyperparameters
learning_rate: 2e-5
batch_size: 16
num_epochs: 5
warmup_ratio: 0.1
weight_decay: 0.01

# Data paths
train_data: "data/splits/train.csv"
validation_data: "data/splits/validation.csv"
test_data: "data/splits/test.csv"

# Output
output_dir: "experiments/results/"  # Code appends mode/model/task structure
save_strategy: "epoch"
evaluation_strategy: "epoch"

# Device
device: "cuda"  # or "cpu" or "mps"
```

### Results Location

Training results saved to:
- **Checkpoints:** `experiments/results/full_finetune/<model_name>/<task>/`
- **Training metrics:** `experiments/results/full_finetune/<model_name>/<task>/training_results.json`

---

## Hyperparameter Optimization

### What is HPO?
Automatically find the best hyperparameters using Optuna (Bayesian optimization).

### HPO Configuration

Edit `experiments/configs/hpo_config.yaml`:

```yaml
# Model to optimize
model_id: "onlplab/alephbert-base"
task: "cls"  # or "span"

# Optuna settings
n_trials: 15  # Number of different configurations to try
study_name: "alephbert-base_cls_hpo"

# Search space
search_space:
  learning_rate:
    type: "loguniform"
    low: 1e-5
    high: 5e-5

  batch_size:
    type: "categorical"
    choices: [8, 16, 32]

  num_epochs:
    type: "int"
    low: 3
    high: 10

  warmup_ratio:
    type: "uniform"
    low: 0.0
    high: 0.2

  weight_decay:
    type: "uniform"
    low: 0.0
    high: 0.05

# Data paths
train_data: "data/splits/train.csv"
validation_data: "data/splits/validation.csv"

# Output
output_dir: "experiments/results/hpo"
database_path: "experiments/results/optuna_studies"
```

### Running HPO

```bash
# Start HPO for Task 1 (Sequence Classification)
python src/idiom_experiment.py \
  --mode hpo \
  --config experiments/configs/hpo_config.yaml \
  --device cuda

# HPO for Task 2 (Token Classification)
# Edit hpo_config.yaml first: change task: "span"
python src/idiom_experiment.py \
  --mode hpo \
  --config experiments/configs/hpo_config.yaml \
  --device cuda
```

### HPO for All Models

```bash
# AlephBERT - Task 1
python src/idiom_experiment.py \
  --mode hpo \
  --config experiments/configs/hpo_config.yaml \
  --model_id onlplab/alephbert-base \
  --task cls \
  --device cuda

# DictaBERT - Task 1
python src/idiom_experiment.py \
  --mode hpo \
  --config experiments/configs/hpo_config.yaml \
  --model_id dicta-il/dictabert \
  --task cls \
  --device cuda

# mBERT - Task 1
python src/idiom_experiment.py \
  --mode hpo \
  --config experiments/configs/hpo_config.yaml \
  --model_id bert-base-multilingual-cased \
  --task cls \
  --device cuda

# XLM-RoBERTa - Task 1
python src/idiom_experiment.py \
  --mode hpo \
  --config experiments/configs/hpo_config.yaml \
  --model_id xlm-roberta-base \
  --task cls \
  --device cuda
```

### Monitoring HPO Progress

```bash
# Check Optuna study database
ls experiments/results/optuna_studies/

# View best parameters so far
cat experiments/results/best_hyperparameters/best_params_<model>_<task>.json
```

### Using HPO Results for Final Training

After HPO completes, use the best parameters:

```bash
# Best parameters saved to:
cat experiments/results/best_hyperparameters/best_params_alephbert-base_cls.json

# Train with best parameters
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task cls \
  --learning_rate 2.5e-5 \  # From HPO results
  --batch_size 16 \           # From HPO results
  --num_epochs 7 \            # From HPO results
  --warmup_ratio 0.15 \       # From HPO results
  --weight_decay 0.02 \       # From HPO results
  --device cuda
```

---

## LLM Evaluation

### Coming in Mission 5
LLM evaluation (prompting approaches) will be implemented separately.

### Available Strategies (Future)

1. **Zero-Shot Prompting** - Direct question without examples
2. **Few-Shot Prompting** - Include 3-5 examples in prompt
3. **Chain-of-Thought** - Step-by-step reasoning

### Target LLMs

- Llama 3.1 70B (via Azure/Together AI)
- Mistral Large
- GPT-3.5-Turbo

---

## VAST.ai GPU Training

### Automation Scripts

We've created 5 automation scripts to simplify VAST.ai workflow:

| Script | Purpose | Mission |
|--------|---------|---------|
| **`setup_vast_instance.sh`** | One-command instance setup (installs everything) | 4.4 |
| **`download_from_gdrive.sh`** | Download dataset from Google Drive | 4.4 |
| **`sync_to_gdrive.sh`** | Upload results to Google Drive (rclone automation) | 4.4-4.6 |
| **`run_all_hpo.sh`** | Batch run all 10 HPO studies | 4.5 |
| **`run_all_experiments.sh`** | Batch run all 30 training runs | 4.6 |

📚 **Complete script documentation:** [scripts/README.md](scripts/README.md)

### Quick Start with Scripts

For a fully automated workflow, use our scripts:

```bash
# 1. Rent VAST.ai instance and SSH in
ssh -p <port> root@<host>

# 2. Clone repository
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
cd hebrew-idiom-detection

# 3. One-command setup (installs deps, downloads data, verifies GPU)
bash scripts/setup_vast_instance.sh

# 4. Configure rclone for Google Drive sync (one-time, 5 min)
curl https://rclone.org/install.sh | sudo bash
rclone config  # Follow prompts to add 'gdrive' remote

# 5. Run HPO for all models (Mission 4.5)
bash scripts/run_all_hpo.sh  # Runs 10 HPO studies automatically

# 6. Run final training (Mission 4.6)
bash scripts/run_all_experiments.sh  # Runs 30 training runs automatically

# 7. Sync results to Google Drive
bash scripts/sync_to_gdrive.sh
```

✅ **That's it!** All automation handled by scripts.

### Manual Workflow (Alternative)

If you prefer manual control or need to understand each step, follow the detailed instructions below.

### When to Use VAST.ai

Use VAST.ai for:
- ✅ Hyperparameter optimization (10-15 trials, ~2-4 hours)
- ✅ Final training with best hyperparameters (all models)
- ✅ Cross-seed validation (3 seeds per model)
- ✅ Any training that takes > 30 minutes

Use local machine for:
- ✅ Zero-shot evaluation (no training needed)
- ✅ Quick tests (100 samples, 1 epoch)
- ✅ Code development and debugging

### Step 1: Prepare Your Code

```bash
# Commit all changes to Git
git add .
git commit -m "Ready for VAST.ai training"
git push
```

### Step 2: Rent VAST.ai Instance

1. Go to https://vast.ai
2. Search for instances:
   - **GPU:** RTX 3090, RTX 4090, or A5000
   - **VRAM:** ≥ 24GB
   - **Reliability:** > 98%
   - **DLPerf:** > 50
3. Sort by: `$/hour` (ascending)
4. **Rent instance** (recommended: ~$0.20-0.50/hour)

### Step 3: Connect to Instance

```bash
# Copy SSH command from VAST.ai dashboard
ssh root@<instance-ip> -p <port> -L 8080:localhost:8080
```

### Step 4: Setup Environment on VAST.ai

```bash
# Update system
apt-get update

# Clone your repository
git clone https://github.com/<username>/hebrew-idiom-detection.git
cd hebrew-idiom-detection

# Install Python dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 5: Upload Dataset to VAST.ai

**Option A: Git LFS (if dataset in repo)**
```bash
# Already cloned with repository
ls data/splits/
```

**Option B: Download from Google Drive**
```bash
# Install gdown
pip install gdown

# Download dataset (get file ID from Google Drive share link)
cd data
gdown <file-id>  # For expressions_data_tagged.csv

# Or download splits directly
cd splits
gdown <train-file-id>
gdown <val-file-id>
gdown <test-file-id>
```

**Option C: SCP from local machine**
```bash
# From your local machine (in another terminal)
scp -P <port> data/splits/*.csv root@<instance-ip>:/root/hebrew-idiom-detection/data/splits/
```

### Step 6: Run Training on VAST.ai

```bash
# Activate screen session (prevents disconnection)
screen -S training

# Run HPO (example)
python src/idiom_experiment.py \
  --mode hpo \
  --config experiments/configs/hpo_config.yaml \
  --device cuda

# Detach from screen: Press Ctrl+A, then D
# Reattach later: screen -r training
```

### Step 7: Monitor Progress

```bash
# View logs in real-time
tail -f experiments/logs/training.log

# Or reattach to screen
screen -r training
```

### Step 8: Download Results

**Option A: SCP to local machine**
```bash
# From your local machine
scp -P <port> -r root@<instance-ip>:/root/hebrew-idiom-detection/experiments/results/ ./experiments/
```

**Option B: Upload to Google Drive (from VAST.ai)**
```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure rclone for Google Drive
rclone config

# Upload results
rclone copy experiments/results/ gdrive:Hebrew_Idiom_Detection/results/
```

**Option C: Download via browser**
```bash
# Start a simple file server on VAST.ai
cd experiments/results
python -m http.server 8080

# Then access in browser: http://localhost:8080
# Download files manually
```

### Step 9: Destroy Instance

```bash
# After downloading all results
exit  # Exit SSH

# Go to VAST.ai dashboard → Destroy instance
# (Important: You're charged per minute!)
```

### VAST.ai Cost Estimation

| Task | Time | Cost @ $0.40/hr |
|------|------|-----------------|
| Single HPO (15 trials) | 2-3 hours | $0.80-$1.20 |
| Single model training | 20-40 min | $0.15-$0.25 |
| All 5 models × 2 tasks | 3-4 hours | $1.20-$1.60 |
| Full project (HPO + final training) | 10-15 hours | $4-$6 |

**Total estimated cost: $5-10** (vs. $80-100 on Azure!)

---

## Results Analysis

### View Training Results

```bash
# Task 1 results (Sequence Classification)
cat experiments/results/full_finetune/alephbert-base/cls/training_results.json

# Task 2 results (Token Classification)
cat experiments/results/full_finetune/alephbert-base/span/training_results.json
```

### Python Script for Analysis

```python
import json
import pandas as pd

# Load results
with open('experiments/results/full_finetune/alephbert-base/cls/training_results.json') as f:
    results = json.load(f)

# Print metrics
print("Test Set Performance:")
print(f"Accuracy: {results['test_accuracy']:.2%}")
print(f"F1 Score: {results['test_f1']:.2%}")
print(f"Precision: {results['test_precision']:.2%}")
print(f"Recall: {results['test_recall']:.2%}")
```

### Compare Multiple Models

```python
import json
import pandas as pd

models = ['alephbert-base', 'dictabert', 'bert-base-multilingual-cased', 'xlm-roberta-base']
results_list = []

for model in models:
    path = f'experiments/results/full_finetune/{model}/cls/training_results.json'
    with open(path) as f:
        data = json.load(f)
        results_list.append({
            'Model': model,
            'Accuracy': data['test_accuracy'],
            'F1': data['test_f1'],
            'Precision': data['test_precision'],
            'Recall': data['test_recall']
        })

df = pd.DataFrame(results_list)
print(df.sort_values('F1', ascending=False))
```

---

## Complete Experiment Matrix

### Phase 1: Zero-Shot Baseline (Mission 3)

**Total: 10 runs (5 models × 2 tasks)**

```bash
# Task 1: Sequence Classification
for model in onlplab/alephbert-base dicta-il/dictabert bert-base-multilingual-cased xlm-roberta-base onlplab/alephbert-gimmel
do
  python src/idiom_experiment.py --mode zero_shot --model_id $model --task cls --device cpu
done

# Task 2: Token Classification
for model in onlplab/alephbert-base dicta-il/dictabert bert-base-multilingual-cased xlm-roberta-base onlplab/alephbert-gimmel
do
  python src/idiom_experiment.py --mode zero_shot --model_id $model --task span --device cpu
done
```

### Phase 2: Hyperparameter Optimization (Mission 4.5)

**Total: 10 HPO studies (5 models × 2 tasks)**
**Time: ~20-30 hours on VAST.ai**
**Cost: ~$8-15**

```bash
# Task 1 HPO for all models (run on VAST.ai)
for model in onlplab/alephbert-base dicta-il/dictabert bert-base-multilingual-cased xlm-roberta-base onlplab/alephbert-gimmel
do
  python src/idiom_experiment.py --mode hpo --model_id $model --task cls --config experiments/configs/hpo_config.yaml --device cuda
done

# Task 2 HPO for all models (run on VAST.ai)
for model in onlplab/alephbert-base dicta-il/dictabert bert-base-multilingual-cased xlm-roberta-base onlplab/alephbert-gimmel
do
  python src/idiom_experiment.py --mode hpo --model_id $model --task span --config experiments/configs/hpo_config.yaml --device cuda
done
```

### Phase 3: Final Training with Best Hyperparameters (Mission 4.6)

**Total: 30 runs (5 models × 2 tasks × 3 seeds)**
**Time: ~10-15 hours on VAST.ai**
**Cost: ~$4-8**

```bash
# After HPO completes, train with best parameters for each seed

# Example for AlephBERT Task 1 with 3 seeds
for seed in 42 123 456
do
  python src/idiom_experiment.py \
    --mode full_finetune \
    --config experiments/configs/training_config.yaml \
    --model_id onlplab/alephbert-base \
    --task cls \
    --seed $seed \
    --learning_rate 2.5e-5 \  # From HPO
    --batch_size 16 \          # From HPO
    --num_epochs 7 \           # From HPO
    --device cuda
done
```

### Phase 4: LLM Evaluation (Mission 5)

**To be implemented**

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Solution:**
```bash
# Reduce batch size
python src/idiom_experiment.py ... --batch_size 8

# Or use gradient accumulation
python src/idiom_experiment.py ... --batch_size 8 --gradient_accumulation_steps 2
```

### Issue 2: Tokenization Alignment Errors

**Solution:**
```bash
# Verify IOB2 alignment first
python src/test_tokenization_alignment.py

# Check results
cat experiments/results/tokenization_alignment_test.txt
```

### Issue 3: Model Download Fails

**Solution:**
```bash
# Download model manually first
python src/model_download.py --model_id onlplab/alephbert-base

# Then run training
python src/idiom_experiment.py ...
```

### Issue 4: VAST.ai Connection Lost

**Solution:**
```bash
# Always use screen for long-running tasks
screen -S training
python src/idiom_experiment.py ...
# Press Ctrl+A, then D to detach

# Reconnect later
ssh root@<instance-ip> -p <port>
screen -r training
```

---

## Key Configuration Files Reference

### 1. Training Config (`experiments/configs/training_config.yaml`)

Controls: Learning rate, batch size, epochs, warmup, weight decay

### 2. HPO Config (`experiments/configs/hpo_config.yaml`)

Controls: Number of trials, search space, Optuna settings

### 3. Model Info (`experiments/configs/model_info.json`)

Contains: Model IDs, names, types, download status

---

## Expected Performance Targets

### Task 1: Sequence Classification

| Model | Expected F1 | Status |
|-------|-------------|--------|
| AlephBERT | 85-92% | Primary |
| DictaBERT | 85-90% | Primary |
| mBERT | 80-88% | Baseline |
| XLM-RoBERTa | 82-90% | Strong |

### Task 2: Token Classification

| Model | Expected F1 | Status |
|-------|-------------|--------|
| AlephBERT | 80-88% | Primary |
| DictaBERT | 80-86% | Primary |
| mBERT | 75-83% | Baseline |
| XLM-RoBERTa | 78-85% | Strong |

---

## Summary: Complete Workflow

1. **Setup** (1 hour)
   - Install dependencies
   - Verify data
   - Test zero-shot locally

2. **Zero-Shot** (2-3 hours, local CPU)
   - Run all 5 models on both tasks
   - Establish baseline

3. **HPO** (20-30 hours, VAST.ai)
   - Find best hyperparameters
   - 10 HPO studies total

4. **Final Training** (10-15 hours, VAST.ai)
   - Train with best params
   - 3 seeds per model/task

5. **LLM Eval** (4-6 hours, local)
   - API calls only
   - No GPU needed

6. **Analysis** (8-10 hours, local)
   - Error analysis
   - Visualizations
   - Statistical tests

**Total Time:** ~6-7 days
**Total Cost:** ~$15-25 (VAST.ai) + $50-100 (LLM APIs)

---

## Questions?

If you encounter issues:
1. Check the error message carefully
2. Review this guide's troubleshooting section
3. Check PRD Section 11 (Risk Management)
4. Review MISSIONS_PROGRESS_TRACKER.md for detailed status

---

**END OF IMPLEMENTATION GUIDE**

*Keep this file handy during development. Update it as you discover new patterns or solutions.*
