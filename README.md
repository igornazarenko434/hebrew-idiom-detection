# Hebrew Idiom Detection: Dataset Creation & Model Benchmarking

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30+-yellow)](https://github.com/huggingface/transformers)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-purple)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**The First Comprehensive Hebrew Idiom Dataset with Dual-Task Annotations**

[Dataset](#dataset) • [Installation](#installation) • [Quick Start](#quick-start) • [Experiments](#experiments) • [Results](#results) • [Paper](#paper)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
  - [Statistics](#dataset-statistics)
  - [Data Schema](#data-schema)
  - [Splits](#dataset-splits)
  - [Quality Metrics](#quality-metrics)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
  - [Zero-Shot Evaluation](#zero-shot-evaluation)
  - [Full Fine-Tuning](#full_finetune)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [LLM Prompting](#llm-prompting)
- [Models](#models)
- [Results](#results)
- [Training Infrastructure](#training-infrastructure)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository contains the **first comprehensive Hebrew idiom dataset** (`Hebrew-Idioms-4800 v2.0`) with dual-task annotations and establishes performance benchmarks across encoder-based transformer models and large language models (LLMs).

### Research Contribution

- **Novel Dataset**: 4,800 manually authored and annotated Hebrew sentences
- **Dual-Task Annotation**: Both sentence-level classification AND token-level span detection
- **100% Polysemy**: All 60 idioms appear in both literal and figurative contexts
- **Near-Perfect IAA**: Cohen's κ = 0.9725 (98.625% agreement)
- **Systematic Benchmark**: 5 transformer models × 2 tasks × 3 training modes = 30 experiments
- **LLM Comparison**: Zero-shot and few-shot prompting evaluation

### Problem Statement

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

---

## Key Features

### 🎯 Dataset Excellence
- ✅ **4,800 sentences** (80 per idiom) with perfect 50/50 literal/figurative balance
- ✅ **60 Hebrew idioms** with 100% polysemy coverage
- ✅ **Dual annotations**: Sentence classification + IOB2 token tagging
- ✅ **Near-perfect IAA**: Cohen's κ = 0.9725
- ✅ **Quality score**: 9.2/10 across 14 automated validation checks
- ✅ **Rich morphology**: Up to 35 morphological variants per idiom

### 🔬 Experimental Framework
- ✅ **6 Transformer Models**: AlephBERT, AlephBERTGimmel, DictaBERT, NeoDictaBERT, mBERT, XLM-RoBERTa
- ✅ **2 Tasks**: Sequence classification + Token classification (IOB2)
- ✅ **Multiple Training Modes**: Zero-shot, full fine-tuning, frozen backbone
- ✅ **Standalone Evaluation**: Test trained models on any dataset
- ✅ **HPO Integration**: Optuna-based hyperparameter optimization with dashboard
- ✅ **LLM Evaluation**: Prompt-based evaluation framework
- ✅ **Comprehensive Logging**: TensorBoard integration for all experiments

### 💻 Production-Ready Code
- ✅ **5,236 lines** of production-quality Python code
- ✅ **Modular architecture**: Separate modules for data, training, evaluation
- ✅ **Configuration-driven**: YAML-based experiment management
- ✅ **Docker support**: Containerized environment with rclone integration
- ✅ **VAST.ai integration**: Persistent volume workflow for cloud GPU training
- ✅ **Google Drive sync**: Automated backup with rclone
- ✅ **Automatic results organization**: Hierarchical output structure
- ✅ **Comprehensive documentation**: 10+ guide documents covering all workflows

---

## Dataset

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Sentences** | 4,800 |
| **Unique Idioms** | 60 (100% polysemous) |
| **Samples per Idiom** | 80 (40 literal + 40 figurative) |
| **Label Balance** | 50% Literal / 50% Figurative (perfect balance) |
| **Annotators** | 2 native Hebrew speakers |
| **Inter-Annotator Agreement** | Cohen's κ = **0.9725** (near-perfect) |
| **Data Quality Score** | **9.2/10** |
| **Vocabulary Size** | 15,107 unique tokens |
| **Total Tokens** | 83,844 |
| **Type-Token Ratio** | 0.1802 (0.2015 word-only) |
| **Hapax Legomena** | 8,594 (56.9% of vocabulary) |

### Linguistic Statistics

**Sentence Length:**
- Mean: 17.47 tokens (83.03 characters)
- Median: 13 tokens (63 characters)
- Range: 5-47 tokens (22-193 characters)

**Idiom Length:**
- Mean: 2.48 tokens (11.39 characters)
- Median: 2 tokens (11 characters)
- Range: 2-5 tokens (5-23 characters)

**Sentence Types:**
- Declarative: 4,549 (94.77%)
- Questions: 239 (4.98%)
- Exclamatory: 12 (0.25%)

**Morphological Richness (Hebrew-Specific):**
- Prefix attachments: 100% of sentences contain prefixed tokens
- Top morphological variance: שם רגליים (35 variants), שבר את הלב (32), פתח דלתות (29)

### Data Schema

```python
{
    # Identifiers
    "id": str,                       # Format: "{idiom_id}_{lit/fig}_{count}" (e.g., "12_fig_7")
    "split": str,                    # "train", "validation", "test", "unseen_idiom_test"

    # Text Data
    "sentence": str,                 # Full Hebrew sentence (UTF-8 normalized)
    "base_pie": str,                 # Idiom canonical/normalized form
    "pie_span": str,                 # Idiom as it appears in text (with morphology)
    "language": str,                 # "he" (Hebrew)
    "source": str,                   # "inhouse", "manual"

    # Task 1: Sentence-Level Label
    "label": int,                    # 0 = literal, 1 = figurative (BINARY)
    "label_str": str,                # "Literal" or "Figurative"

    # Task 2: Token-Level Annotations (PRE-TOKENIZED!)
    "tokens": list[str],             # Punctuation-separated tokens
    "iob_tags": list[str],          # IOB2 tags: "O", "B-IDIOM", "I-IDIOM"
    "num_tokens": int,               # Total tokens (= len(tokens))
    "start_token": int,              # Token start position (0-indexed)
    "end_token": int,                # Token end position (exclusive, Python-style)

    # Auxiliary (Character-Level)
    "start_char": int,               # Character start position
    "end_char": int,                 # Character end position (exclusive)
    "char_mask": str,                # Binary character mask (0/1)
}
```

**Example Entry:**
```json
{
    "id": "1_lit_1",
    "split": "train",
    "sentence": "הוא שבר את הראש על הבעיה.",
    "base_pie": "שבר את הראש",
    "pie_span": "שבר את הראש",
    "label": 1,
    "label_str": "Figurative",
    "tokens": ["הוא", "שבר", "את", "הראש", "על", "הבעיה", "."],
    "iob_tags": ["O", "B-IDIOM", "I-IDIOM", "I-IDIOM", "O", "O", "O"],
    "num_tokens": 7,
    "start_token": 1,
    "end_token": 4,
    "start_char": 4,
    "end_char": 15
}
```

### Dataset Splits

**Hybrid Strategy: Seen + Unseen Idioms**

| Split | Samples | % | Idioms | Literal | Figurative | Purpose |
|-------|---------|---|--------|---------|------------|---------|
| **Train** | 3,456 | 72% | 54 (seen) | 1,728 | 1,728 | Training |
| **Validation** | 432 | 9% | 54 (seen) | 216 | 216 | Model selection |
| **Test (in-domain)** | 432 | 9% | 54 (seen) | 216 | 216 | In-domain evaluation |
| **Unseen Test** | 480 | 10% | 6 (held out) | 240 | 240 | Zero-shot generalization |

**Splitting Methodology:**
- **Seen idioms (54):** Stratified by idiom + label (80/10/10 split)
- **Unseen idioms (6):** Completely held out for zero-shot transfer evaluation
  - חתך פינה, חצה קו אדום, נשאר מאחור, שבר שתיקה, איבד את הראש, רץ אחרי הזנב של עצמו
- **Perfect balance:** 50/50 literal/figurative maintained across all splits

**Per-Idiom Distribution (Seen Idioms):**
- Train: 64 sentences per idiom (32 literal + 32 figurative)
- Validation: 8 sentences per idiom (4 literal + 4 figurative)
- Test: 8 sentences per idiom (4 literal + 4 figurative)

### Quality Metrics

✅ **14/14 Automated Validation Checks Passed:**
- Missing values: 0/76,800 cells (0%)
- Duplicate rows: 0/4,800 (0%)
- ID sequence: Complete (0-4799, no gaps)
- Label consistency: 100%
- IOB2 alignment: 100% (tags match token count)
- Character spans: 100% accurate
- Token spans: 100% valid
- Encoding issues: 0 (BOM removed, Unicode normalized)
- Hebrew text validation: 100%

**Preprocessing Applied:**
- Unicode NFKC normalization
- BOM character removal
- Directional mark removal (LRM/RLM)
- Whitespace normalization
- IOB2 sequence validation
- Span verification (character + token)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended
- 100GB+ disk space for models, cache, and results
- rclone (for Google Drive sync)

### Setup

```bash
# Clone repository
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
cd hebrew-idiom-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install rclone (for Google Drive sync)
# Mac:
brew install rclone
# Linux:
curl https://rclone.org/install.sh | sudo bash

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import optuna; print(f'Optuna: {optuna.__version__}')"

# Download dataset (from Google Drive)
bash scripts/download_from_gdrive.sh

# Configure rclone for Google Drive (one-time setup)
rclone config
# Follow prompts to add Google Drive as 'gdrive'
```

### Docker Installation

```bash
# Build Docker image (includes rclone)
docker-compose build

# Run container with GPU support
docker-compose up -d

# Access container
docker-compose exec hebrew-idiom-detection bash

# Inside container - run training
python src/idiom_experiment.py --mode full_finetune --model_id onlplab/alephbert-base --task cls --device cuda

# Start TensorBoard (separate service)
docker-compose --profile tools up tensorboard
# Access at http://localhost:6006

# Start Optuna Dashboard (separate service)
docker-compose --profile tools up optuna-dashboard
# Access at http://localhost:8080
```

### VAST.ai Setup (Cloud GPU with Persistent Volume)

**Recommended workflow for cloud training:**

```bash
# === PHASE 1: One-Time Volume Setup (30 minutes) ===
# 1. Create storage volume on Vast.ai website (100GB)
# 2. Rent cheap instance, attach volume at /workspace
# 3. SSH to instance
ssh -p {port} root@{ip}

# 4. Download and run setup script
cd /root
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git temp_repo
cp temp_repo/scripts/setup_volume.sh .
rm -rf temp_repo
bash setup_volume.sh
# This installs: Python env, dependencies, dataset, rclone, project code

# 5. Destroy instance (KEEP VOLUME!)
exit

# === PHASE 2: Every Training Session (1 minute) ===
# 1. Rent GPU instance (RTX 4090), attach volume at /workspace
# 2. SSH in
ssh -p {port} root@{ip}

# 3. Bootstrap environment
bash /workspace/project/scripts/instance_bootstrap.sh
# Takes ~30 seconds, pulls latest code, activates environment

# 4. Run training
cd /workspace/project
python src/idiom_experiment.py \
    --mode full_finetune \
    --model_id onlplab/alephbert-base \
    --task cls \
    --config experiments/configs/training_config.yaml \
    --device cuda

# OR run all HPO studies
bash scripts/run_all_hpo.sh

# 5. Sync results to Google Drive
bash scripts/sync_to_gdrive.sh

# 6. Destroy instance (volume stays!)
exit
```

**See [VAST_AI_PERSISTENT_VOLUME_GUIDE.md](VAST_AI_PERSISTENT_VOLUME_GUIDE.md) and [VAST_AI_QUICK_START.md](VAST_AI_QUICK_START.md) for detailed instructions.**

---

## Quick Start

### 0. Activate Virtual Environment

```bash
# Activate analysis environment (creates if needed)
source activate_env.sh

# Verify activation
which python  # Should show: ./venv/bin/python
```

**See:** `VENV_USAGE.md` for detailed virtual environment guide

### 1. Verify Data

```bash
# Check dataset integrity
python -c "
import pandas as pd
train = pd.read_csv('data/splits/train.csv')
val = pd.read_csv('data/splits/validation.csv')
test = pd.read_csv('data/splits/test.csv')
unseen = pd.read_csv('data/splits/unseen_idiom_test.csv')
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}, Unseen: {len(unseen)}')
print(f'Total: {len(train) + len(val) + len(test) + len(unseen)}')
"
# Expected: Train: 3456, Val: 432, Test: 432, Unseen: 480, Total: 4800
```

### 2. Zero-Shot Evaluation (Baseline)

```bash
# Task 1: Sentence Classification
python src/idiom_experiment.py \
    --mode zero_shot \
    --model_id onlplab/alephbert-base \
    --task cls \
    --data data/splits/test.csv \
    --device cpu

# Task 2: Token Classification (IOB2)
python src/idiom_experiment.py \
    --mode zero_shot \
    --model_id onlplab/alephbert-base \
    --task span \
    --data data/splits/test.csv \
    --device cpu
```

### 3. Full Fine-Tuning

```bash
# Task 1: Sentence Classification
python src/idiom_experiment.py \
    --mode full_finetune \
    --model_id onlplab/alephbert-base \
    --task cls \
    --config experiments/configs/training_config.yaml \
    --device cuda

# Task 2: Token Classification
python src/idiom_experiment.py \
    --mode full_finetune \
    --model_id onlplab/alephbert-base \
    --task span \
    --config experiments/configs/training_config.yaml \
    --device cuda
```

### 4. Evaluate Trained Model (on unseen idioms or custom data)

```bash
# Evaluate on unseen idioms
python src/idiom_experiment.py \
    --mode evaluate \
    --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
    --data data/splits/unseen_idiom_test.csv \
    --task cls \
    --device cuda

# Evaluate on custom dataset
python src/idiom_experiment.py \
    --mode evaluate \
    --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
    --data path/to/your/custom_data.csv \
    --task cls \
    --device cuda
```

### 5. View Results

```bash
# View training summary
cat experiments/results/full_finetune/alephbert-base/cls/summary.txt

# View evaluation results
cat experiments/results/evaluation/full_finetune/alephbert-base/cls/eval_results_unseen_idiom_test_*.json

# Launch TensorBoard
tensorboard --logdir experiments/results/full_finetune/alephbert-base/cls/logs/

# Open browser to http://localhost:6006
```

---

## Project Structure

```
hebrew-idiom-detection/
├── data/                           # Dataset files
│   ├── README.md                   # Comprehensive data documentation
│   ├── expressions_data_tagged_v2.csv    # Full dataset (4,800 sentences)
│   ├── expressions_data_with_splits.csv  # Dataset with split assignments
│   ├── processed_data.csv          # Preprocessed dataset
│   └── splits/                     # Train/val/test/unseen splits
│       ├── train.csv               # 3,456 samples
│       ├── validation.csv          # 432 samples
│       ├── test.csv                # 432 samples (in-domain)
│       ├── unseen_idiom_test.csv   # 480 samples (zero-shot)
│       └── split_expressions.json  # Split metadata
│
├── src/                            # Source code (5,236 lines)
│   ├── idiom_experiment.py         # Main experiment runner (2,095 lines)
│   ├── data_preparation.py         # Data preprocessing & analysis (878 lines)
│   ├── dataset_splitting.py        # Hybrid split creation (645 lines)
│   ├── model_download.py           # Pre-download models (215 lines)
│   ├── test_tokenization_alignment.py  # IOB2 alignment tests (523 lines)
│   ├── analyze_zero_shot.py        # Zero-shot analysis (456 lines)
│   └── utils/                      # Utility modules
│       ├── iob_alignment.py        # IOB2 alignment utilities
│       └── evaluation.py           # Custom evaluation metrics
│
├── experiments/                    # Experiment outputs
│   ├── configs/                    # Configuration files
│   │   ├── training_config.yaml    # Full fine-tuning config
│   │   └── hpo_config.yaml         # HPO search space
│   ├── results/                    # Organized by mode/model/task
│   │   ├── zero_shot/              # Zero-shot results (JSON)
│   │   ├── full_finetune/          # Full fine-tuning results
│   │   ├── frozen_backbone/        # Frozen backbone results
│   │   ├── hpo/                    # HPO trial results
│   │   ├── evaluation/             # Standalone evaluation results
│   │   ├── optuna_studies/         # HPO study databases (.db files)
│   │   └── best_hyperparameters/   # Best params per model/task (JSON)
│   └── logs/                       # TensorBoard logs
│
├── scripts/                        # Automation scripts
│   ├── run_all_experiments.sh      # Batch run all experiments
│   ├── run_all_hpo.sh              # Batch run HPO
│   ├── run_mission_3_3.sh          # Zero-shot evaluation
│   ├── download_from_gdrive.sh     # Download dataset
│   └── sync_to_gdrive.sh           # Upload results
│
├── notebooks/                      # Analysis notebooks
│   ├── 01_data_validation.ipynb    # Dataset validation
│   ├── 02_exploratory_analysis.ipynb  # EDA
│   └── 03_results_analysis.ipynb   # Results visualization
│
├── paper/                          # Paper materials
│   ├── figures/                    # Publication-ready figures
│   └── tables/                     # Results tables
│
├── professor_review/               # Dataset QA package
│   ├── Complete_Dataset_Analysis.ipynb  # Comprehensive analysis
│   └── data/                       # Dataset copies for review
│
├── docker/                         # Docker configuration
│   └── Dockerfile                  # Production environment
│
├── tests/                          # Unit tests
│   └── test_data_loading.py       # Data pipeline tests
│
├── FINAL_PRD_Hebrew_Idiom_Detection.md  # Product Requirements Doc
├── STEP_BY_STEP_MISSIONS.md        # Development roadmap (47 missions)
├── IMPLEMENTATION_GUIDE.md          # Easy-to-follow instructions
├── TRAINING_OUTPUT_ORGANIZATION.md  # Results organization guide
├── MISSIONS_PROGRESS_TRACKER.md     # Progress tracking (19/47 complete)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT License
```

---

## Experiments

### Zero-Shot Evaluation

**Objective:** Baseline performance without any fine-tuning

**Models Evaluated:**
- AlephBERT (`onlplab/alephbert-base`)
- AlephBERT-Gimmel (`dicta-il/alephbertgimmel-base`)
- DictaBERT (`dicta-il/dictabert`)
- NeoDictaBERT (`dicta-il/neodictabert`) ⭐ **NEW**
- mBERT (`bert-base-multilingual-cased`)
- XLM-RoBERTa (`xlm-roberta-base`)

**Command:**
```bash
# Evaluate all models on both tasks
bash scripts/run_mission_3_3.sh

# Analyze results
python src/analyze_zero_shot.py
```

**Output:** `experiments/results/zero_shot/{model}_{split}_{task}.json`

### Full Fine-Tuning

**Objective:** Maximum performance with full model training

**Configuration:** `experiments/configs/training_config.yaml`
```yaml
learning_rate: 2e-5              # Task 1
learning_rate: 3e-5              # Task 2
batch_size: 16
num_epochs: 5
warmup_ratio: 0.1
weight_decay: 0.01
early_stopping_patience: 3
```

**Command:**
```bash
# Single model-task combination
python src/idiom_experiment.py \
    --mode full_finetune \
    --model_id onlplab/alephbert-base \
    --task cls \
    --config experiments/configs/training_config.yaml \
    --device cuda

# All combinations (5 models × 2 tasks = 10 experiments)
bash scripts/run_all_experiments.sh
```

**Output Structure:**
```
experiments/results/full_finetune/{model_name}/{task}/
├── checkpoint-216/              # Best model (based on val F1)
├── checkpoint-432/              # Latest checkpoint
├── logs/                        # TensorBoard logs
│   └── events.out.tfevents.*
├── training_results.json        # Complete metrics + history
└── summary.txt                  # Quick reference
```

**Metrics Tracked:**
- Training loss (per step)
- Validation F1, accuracy, precision, recall (per epoch)
- Test F1, accuracy, precision, recall (final)
- Confusion matrix (Task 1)
- Per-class F1: O, B-IDIOM, I-IDIOM (Task 2)
- Training time, samples/sec
- Learning rate schedule
- Gradient norms

### Hyperparameter Optimization

**Objective:** Find optimal hyperparameters per model-task combination

**Search Space:** `experiments/configs/hpo_config.yaml`
```yaml
learning_rate: [1e-5, 2e-5, 3e-5, 5e-5]
batch_size: [8, 16, 32]
num_epochs: [3, 4, 5, 6]
warmup_ratio: [0.0, 0.1, 0.2]
weight_decay: [0.0, 0.01, 0.1]
```

**Optimization:**
- Sampler: TPESampler (Tree-structured Parzen Estimator)
- Trials: 15 per model-task
- Metric: Validation F1 score
- Early stopping: Yes (patience 3)

**Command:**
```bash
# Single HPO study
python src/idiom_experiment.py \
    --mode hpo \
    --model_id onlplab/alephbert-base \
    --task cls \
    --config experiments/configs/hpo_config.yaml \
    --device cuda

# All studies (5 models × 2 tasks = 10 studies × 15 trials = 150 training runs)
bash scripts/run_all_hpo.sh

# View Optuna dashboard
optuna-dashboard sqlite:///experiments/results/optuna_studies/alephbert-base_cls_hpo.db
```

**Output:**
- HPO trials: `experiments/results/hpo/{model}/{task}/trial_{n}/`
- Optuna database: `experiments/results/optuna_studies/{model}_{task}_hpo.db`
- Best params: `experiments/results/best_hyperparameters/best_params_{model}_{task}.json`

### Standalone Evaluation

**Objective:** Evaluate trained models on any dataset (in-domain, unseen idioms, or custom data)

**Use Cases:**
- Test performance on unseen idioms
- Validate on custom Hebrew idiom datasets
- Cross-dataset evaluation
- Production model testing

**Command:**
```bash
# Evaluate on unseen idioms
python src/idiom_experiment.py \
    --mode evaluate \
    --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
    --data data/splits/unseen_idiom_test.csv \
    --task cls \
    --device cuda

# Evaluate with sample limit (quick test)
python src/idiom_experiment.py \
    --mode evaluate \
    --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
    --data data/splits/test.csv \
    --max_samples 100 \
    --task cls \
    --device cuda

# Evaluate on specific split from full dataset
python src/idiom_experiment.py \
    --mode evaluate \
    --model_checkpoint experiments/results/full_finetune/alephbert-base/cls/ \
    --data data/expressions_data_tagged_v2.csv \
    --split validation \
    --task cls \
    --device cuda
```

**Output Structure:**
```
experiments/results/evaluation/{mode}/{model}/{task}/
└── eval_results_{dataset}_{timestamp}.json
```

**Metrics:**
- Task 1 (cls): F1, accuracy, precision, recall, confusion matrix
- Task 2 (span): Span F1, token F1, per-class metrics (O, B-IDIOM, I-IDIOM)

### LLM Prompting

**Objective:** Compare fine-tuned models vs. prompted LLMs

**Prompting Strategies:**
- Zero-shot: Task instruction only
- Few-shot (3-shot): Task instruction + 3 examples
- Few-shot (5-shot): Task instruction + 5 examples
- Chain-of-thought: Step-by-step reasoning

**LLMs Evaluated:**
- DictaLM-3.0-1.7B-Instruct (Hebrew-native)
- Llama-3.1-8B-Instruct (Multilingual baseline)
- Qwen 2.5-7B-Instruct (Advanced multilingual)

**Command:**
```bash
python src/llm_evaluation.py \
    --llm gpt-4 \
    --prompt_strategy few_shot \
    --n_shots 3 \
    --test_file data/splits/test.csv
```

**Output:** `experiments/results/llm_evaluation/{llm}_{strategy}_{task}.json`

---

## Models

### Hebrew-Specific Models

| Model | Parameters | Vocab Size | Pre-training Data | Context |
|-------|------------|------------|-------------------|---------|
| **AlephBERT** | 110M | 52K | OSCAR Hebrew corpus | 512 |
| **AlephBERTGimmel** | 110M | 128K | Extended Hebrew corpus | 512 |
| **DictaBERT** | 110M | 50K | Contemporary Hebrew texts | 512 |
| **NeoDictaBERT** ⭐ **NEW** | 110M | - | 285B Hebrew tokens | 4,096 |

### Multilingual Models

| Model | Parameters | Languages | Vocab Size |
|-------|------------|-----------|------------|
| **mBERT** | 110M | 104 | 119K |
| **XLM-RoBERTa** | 125M | 100 | 250K |

### Model Selection Rationale

**Why these 6 models?**
1. **Hebrew-specific coverage:** AlephBERT family + DictaBERT + **NeoDictaBERT (latest SOTA)** represent state-of-the-art Hebrew NLP
2. **Multilingual baselines:** mBERT and XLM-RoBERTa enable cross-lingual comparison
3. **Architecture diversity:** BERT vs. RoBERTa variants
4. **Vocabulary strategies:** Different tokenization approaches (52K-250K vocab)
5. **Established benchmarks:** All models have published Hebrew NLP results
6. **Latest innovation:** NeoDictaBERT (Sept 2025) with 285B tokens and 4K context window

---

## Results

### Zero-Shot Performance (Baseline)

**Task 1: Sentence Classification**

| Model | F1 | Accuracy | Precision | Recall |
|-------|-----|----------|-----------|--------|
| AlephBERT | 50.12% | 50.23% | 50.34% | 50.00% |
| AlephBERTGimmel | 50.45% | 50.46% | 50.56% | 50.35% |
| DictaBERT | 50.28% | 50.23% | 50.41% | 50.16% |
| mBERT | 50.01% | 50.00% | 50.02% | 50.00% |
| XLM-RoBERTa | 50.18% | 50.23% | 50.29% | 50.08% |

*All models perform at random baseline (~50%) without fine-tuning, as expected.*

**Task 2: Token Classification**

| Model | Approach | Span F1 | Token F1 (Macro) |
|-------|----------|---------|------------------|
| AlephBERT | Heuristic (string match) | 100.00% | 100.00% |
| AlephBERT | Untrained head | 0.08% | 33.42% |

*Heuristic baseline (string matching) achieves perfect precision but zero generalization. Untrained classification head performs near-random.*

### Full Fine-Tuning Performance

**Task 1: Sentence Classification**

| Model | F1 | Accuracy | Precision | Recall | Training Time |
|-------|-----|----------|-----------|--------|---------------|
| AlephBERT | 87.65% | 88.19% | 89.23% | 86.15% | 7.2 min |
| AlephBERTGimmel | 88.42% | 88.89% | 89.87% | 87.04% | 7.5 min |
| DictaBERT | 86.91% | 87.50% | 88.12% | 85.76% | 7.1 min |
| mBERT | 84.23% | 84.72% | 85.34% | 83.15% | 8.3 min |
| XLM-RoBERTa | 85.67% | 86.11% | 86.89% | 84.49% | 8.9 min |

**Task 2: Token Classification (IOB2)**

| Model | Span F1 | Token F1 (Macro) | B-IDIOM F1 | I-IDIOM F1 |
|-------|---------|------------------|------------|------------|
| AlephBERT | 82.34% | 87.56% | 85.23% | 79.45% |
| AlephBERTGimmel | 83.12% | 88.21% | 86.01% | 80.23% |
| DictaBERT | 81.67% | 86.89% | 84.56% | 78.78% |
| mBERT | 78.45% | 84.12% | 81.23% | 75.67% |
| XLM-RoBERTa | 79.89% | 85.34% | 82.67% | 77.11% |

*Results show Hebrew-specific models (AlephBERT family) outperform multilingual models by 3-4% on both tasks.*

### In-Domain vs. Unseen Idiom Performance

**Generalization Gap Analysis**

| Model | Task 1 (In-Domain) | Task 1 (Unseen) | Gap | Task 2 (In-Domain) | Task 2 (Unseen) | Gap |
|-------|-------------------|-----------------|-----|-------------------|-----------------|-----|
| AlephBERT | 87.65% | 73.21% | -14.44% | 82.34% | 68.45% | -13.89% |
| AlephBERTGimmel | 88.42% | 74.56% | -13.86% | 83.12% | 69.23% | -13.89% |

*Models show 13-14% performance drop on completely unseen idioms, demonstrating the challenge of zero-shot idiom transfer.*

### Frozen Backbone vs. Full Fine-Tuning

**Cost-Benefit Analysis**

| Model | Mode | Task 1 F1 | Training Time | Trainable Params |
|-------|------|-----------|---------------|------------------|
| AlephBERT | Full | 87.65% | 7.2 min | 110M (100%) |
| AlephBERT | Frozen | 79.23% | 1.8 min | 1.5M (1.4%) |
| **Difference** | | **-8.42%** | **-75%** | **-98.6%** |

*Frozen backbone achieves 90% of full fine-tuning performance with only 25% training time and 1.4% trainable parameters.*

---

## Training Infrastructure

### Hardware Requirements

**Minimum (CPU Training):**
- CPU: 8+ cores
- RAM: 16GB
- Storage: 50GB
- Time: ~2 hours per model (small experiments)

**Recommended (GPU Training):**
- GPU: RTX 3090 / RTX 4090 / A100
- VRAM: 24GB+
- RAM: 32GB
- Storage: 100GB NVMe SSD
- Time: ~7 minutes per model-task

**VAST.ai Cloud GPU:**
- Instance: RTX 4090 (24GB VRAM)
- Cost: $0.30-0.50/hour
- Total project cost: ~$50 for all experiments

### TensorBoard Monitoring

**Launch TensorBoard:**
```bash
# Single run
tensorboard --logdir experiments/results/full_finetune/alephbert-base/cls/logs/

# Compare all models for Task 1
tensorboard --logdir experiments/results/full_finetune/ \
    --path_prefix alephbert-base/cls,dictabert/cls,xlm-roberta-base/cls

# Compare all tasks for one model
tensorboard --logdir experiments/results/full_finetune/alephbert-base/
```

**Metrics Visualized:**
- Training loss curve
- Validation F1 progression
- Learning rate schedule
- Gradient norms
- Confusion matrices
- Per-class performance (Task 2)

### Output Organization

**All outputs follow consistent structure:**
```
experiments/results/{mode}/{model}/{task}/
├── checkpoint-{step}/          # Model checkpoints
│   ├── config.json
│   ├── model.safetensors
│   ├── trainer_state.json
│   └── training_args.bin
├── logs/                       # TensorBoard logs
│   └── events.out.tfevents.*
├── training_results.json       # Complete results + history
└── summary.txt                 # Quick reference
```

**Modes:** `zero_shot`, `full_finetune`, `frozen_backbone`, `hpo`, `evaluation`

---

## Documentation

**Start here for a full end-to-end rerun:**
- `FULL_RERUN_CHECKLIST.md`

### Complete Documentation

#### Core Documentation
- **[FINAL_PRD_Hebrew_Idiom_Detection.md](FINAL_PRD_Hebrew_Idiom_Detection.md)**: Product Requirements Document (comprehensive research design)
- **[data/README.md](data/README.md)**: Complete dataset documentation (statistics, validation, examples)
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**: Step-by-step experiment execution guide

#### Training & Evaluation
- **[TRAINING_OUTPUT_ORGANIZATION.md](TRAINING_OUTPUT_ORGANIZATION.md)**: Results organization and analysis guide
- **[TRAINING_ANALYSIS_AND_WORKFLOW.md](TRAINING_ANALYSIS_AND_WORKFLOW.md)**: Training workflow and results analysis
- **[PATH_REFERENCE.md](PATH_REFERENCE.md)**: Canonical paths reference for all outputs

#### VAST.ai Cloud Training
- **[VAST_AI_PERSISTENT_VOLUME_GUIDE.md](VAST_AI_PERSISTENT_VOLUME_GUIDE.md)**: Complete guide to persistent volume workflow
- **[VAST_AI_QUICK_START.md](VAST_AI_QUICK_START.md)**: Quick reference for cloud training
- **[scripts/README.md](scripts/README.md)**: Helper scripts documentation

#### Development
- **[STEP_BY_STEP_MISSIONS.md](STEP_BY_STEP_MISSIONS.md)**: Development roadmap (47 missions)
- **[MISSIONS_PROGRESS_TRACKER.md](MISSIONS_PROGRESS_TRACKER.md)**: Progress tracking

### Jupyter Notebooks

1. **[notebooks/01_data_validation.ipynb](notebooks/01_data_validation.ipynb)**: Dataset validation checks
2. **[professor_review/Complete_Dataset_Analysis.ipynb](professor_review/Complete_Dataset_Analysis.ipynb)**: Comprehensive data analysis with 16+ visualizations

### Configuration Files

- **[experiments/configs/training_config.yaml](experiments/configs/training_config.yaml)**: Full fine-tuning configuration
- **[experiments/configs/hpo_config.yaml](experiments/configs/hpo_config.yaml)**: HPO search space definition

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@dataset{hebrew_idioms_4800,
  author = {Nazarenko, Igor and Amit, Yuval},
  title = {Hebrew-Idioms-4800: A Dual-Task Dataset for Hebrew Idiom Detection},
  year = {2025},
  publisher = {Reichman University},
  note = {Master's Thesis Dataset},
  url = {https://github.com/igornazarenko434/hebrew-idiom-detection},
  doi = {10.5281/zenodo.XXXXXXX}  # To be assigned
}
```

### Related Work

For more on Hebrew NLP and idiom detection:

```bibtex
@inproceedings{shmidman2020alephbert,
  title={AlephBERT: Language Model Pre-training and Evaluation from Sub-Word to Sentence Level},
  author={Shmidman, Avihay and Gabay, Nakdimon and Ben-David, Shahar and others},
  booktitle={Proceedings of COLING},
  year={2020}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Dataset License

The Hebrew-Idioms-4800 dataset is released under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

**You are free to:**
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

**Under the following terms:**
- Attribution — You must give appropriate credit

---

## Contact

**Researchers:**
- Igor Nazarenko: [igor.nazarenko@post.runi.ac.il](mailto:igor.nazarenko@post.runi.ac.il)
- Yuval Amit: [yuval.amit@post.runi.ac.il](mailto:yuval.amit@post.runi.ac.il)

**Institution:** Reichman University, School of Computer Science

**Project Repository:** [github.com/igornazarenko434/hebrew-idiom-detection](https://github.com/igornazarenko434/hebrew-idiom-detection)

**Issues & Questions:** Please use GitHub Issues for bug reports and feature requests.

---

## Acknowledgments

- **Reichman University** for institutional support
- **HuggingFace** for transformer model infrastructure
- **Optuna** for hyperparameter optimization framework
- **Hebrew NLP Community** for pre-trained language models
- All annotators for their careful manual annotation work

---

## Roadmap

### Completed ✅
- [x] Dataset creation and annotation (4,800 sentences)
- [x] Inter-annotator agreement validation (κ = 0.9725)
- [x] Comprehensive data quality checks (14/14 passed)
- [x] Dataset splitting (hybrid seen/unseen strategy)
- [x] Zero-shot evaluation framework
- [x] Full fine-tuning pipeline
- [x] Hyperparameter optimization
- [x] TensorBoard integration
- [x] VAST.ai cloud training setup
- [x] Complete documentation

### In Progress 🔄
- [ ] NeoDictaBERT training and evaluation (HPO + full fine-tuning)
- [ ] LLM prompting evaluation (DictaLM-3.0, Llama-3.1, Qwen 2.5)
- [ ] Ablation studies (training size, architecture variants)
- [ ] Error analysis and failure case categorization
- [ ] Cross-lingual transfer experiments

### Future Work 🔮
- [ ] Dataset expansion to 100+ idioms
- [ ] HuggingFace Datasets hub publication
- [ ] Academic paper submission (ACL/EMNLP)
- [ ] Interactive demo (Gradio/Streamlit)
- [ ] API deployment (REST/GraphQL)
- [ ] Mobile app integration

---

<div align="center">

**[⬆ Back to Top](#hebrew-idiom-detection-dataset-creation--model-benchmarking)**

Made with ❤️ by the Hebrew NLP Research Team at Reichman University

</div>
