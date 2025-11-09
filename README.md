# Hebrew Idiom Detection

**Repository:** https://github.com/igornazarenko434/hebrew-idiom-detection

## Project Overview

This project focuses on automatic detection and interpretation of Hebrew idioms in natural language text. The goal is to distinguish between literal and figurative uses of expressions and identify idiomatic spans within sentences.

## Research Objectives

1. **Task 1: Sentence Classification** - Classify sentences containing specific expressions as either literal (מילולי) or figurative (פיגורטיבי)
2. **Task 2: Token Classification** - Identify the exact span of the idiom within the sentence using IOB2 tagging

## Dataset

- **Total Samples**: 4,800 sentences
- **Distribution**: 50% literal (2,400), 50% figurative (2,400)
- **Unique Idioms**: 60 Hebrew expressions (exactly)
- **Annotations**: IOB2 tags for idiom spans
- **Splits**: Expression-based train (3,840) / validation (480) / test (480) to prevent data leakage
- **Documentation**: See [data/README.md](data/README.md) for complete dataset documentation

## Models Evaluated

### Encoder Models (Fine-tuned)
- AlephBERT-base
- AlephBERT-Gimmel
- DictaBERT
- mBERT (multilingual BERT)
- XLM-RoBERTa-base

### Large Language Models (Prompting)
- Zero-shot and few-shot evaluation
- Comparison with fine-tuned models

## Project Structure

```
.
├── data/                    # Dataset files
├── src/                     # Source code
│   └── utils/              # Utility functions
├── experiments/            # Experiment configurations and results
│   ├── configs/           # Configuration files
│   ├── results/           # Results (synced to Google Drive)
│   └── logs/              # Training logs
├── models/                 # Model checkpoints (local cache)
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Automation scripts (VAST.ai workflow)
├── docker/                 # Docker configurations
├── tests/                  # Unit tests
└── paper/                  # Paper materials
    ├── figures/           # Figures for publication
    └── tables/            # Tables for publication
```

## Documentation

Complete guides for this project:

- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Complete execution guide with all commands and workflows
- **[STEP_BY_STEP_MISSIONS.md](STEP_BY_STEP_MISSIONS.md)** - Detailed mission-by-mission breakdown
- **[FINAL_PRD.md](FINAL_PRD_Hebrew_Idiom_Detection.md)** - Comprehensive project specification
- **[data/README.md](data/README.md)** - Dataset documentation and statistics
- **[scripts/README.md](scripts/README.md)** - Automation scripts for VAST.ai training
- **[MISSIONS_PROGRESS_TRACKER.md](MISSIONS_PROGRESS_TRACKER.md)** - Current progress tracking

## Environment Setup

This project uses:
- Python 3.9 or 3.10
- PyTorch with HuggingFace Transformers
- VAST.ai for GPU training
- Google Drive for model and results storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hebrew-idiom-detection
```

2. Create and activate virtual environment:
```bash
conda create -n hebrew-idiom python=3.10
conda activate hebrew-idiom
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Zero-Shot Evaluation (Mission 3.2)

Evaluate pre-trained models without fine-tuning:

```bash
# Task 1: Sequence Classification
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/test.csv \
  --task cls \
  --device cpu

# Task 2: Token Classification (IOB2 Tagging)
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/test.csv \
  --task span \
  --device cpu

# Both tasks
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/test.csv \
  --task both \
  --device cpu
```

### Fine-Tuning (Mission 4.2)

Train models with full fine-tuning:

```bash
# Task 1: Sequence Classification
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task cls \
  --device cuda

# Task 2: Token Classification with IOB2 alignment
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task span \
  --device cuda

# Override config parameters via CLI
python src/idiom_experiment.py \
  --mode full_finetune \
  --config experiments/configs/training_config.yaml \
  --task cls \
  --learning_rate 3e-5 \
  --batch_size 32 \
  --num_epochs 10 \
  --device cuda
```

### Frozen Backbone Training

Train only the classification head while freezing the backbone:

```bash
python src/idiom_experiment.py \
  --mode frozen_backbone \
  --config experiments/configs/training_config.yaml \
  --task cls \
  --device cuda
```

### Hyperparameter Optimization (Mission 4.3)

Run Optuna HPO to find best hyperparameters:

```bash
python src/idiom_experiment.py \
  --mode hpo \
  --config experiments/configs/hpo_config.yaml \
  --device cuda
```

### Testing IOB2 Alignment

Verify subword tokenization alignment (Mission 4.2 Task 3.5):

```bash
python src/test_tokenization_alignment.py
```

Results saved to `experiments/results/tokenization_alignment_test.txt`

## VAST.ai GPU Training

For GPU-accelerated training, we use VAST.ai with automation scripts.

**Quick Workflow:**

```bash
# 1. Rent VAST.ai instance and SSH in
ssh -p <port> root@<host>

# 2. Clone repository
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
cd hebrew-idiom-detection

# 3. One-command setup (installs everything)
bash scripts/setup_vast_instance.sh

# 4. Configure rclone for Google Drive sync (one-time, 5 min)
curl https://rclone.org/install.sh | sudo bash
rclone config  # Follow prompts

# 5. Run hyperparameter optimization (Mission 4.5)
bash scripts/run_all_hpo.sh

# 6. Run final training with best hyperparameters (Mission 4.6)
bash scripts/run_all_experiments.sh

# 7. Sync results to Google Drive
bash scripts/sync_to_gdrive.sh
```

**Documentation:**
- Complete VAST.ai workflow: [IMPLEMENTATION_GUIDE.md - Section 8](IMPLEMENTATION_GUIDE.md#vastai-gpu-training)
- Scripts documentation: [scripts/README.md](scripts/README.md)
- Detailed missions: [STEP_BY_STEP_MISSIONS.md](STEP_BY_STEP_MISSIONS.md) (Missions 4.4-4.6)

## Research Methodology

1. **Data Preparation**: Validation and splitting of dataset
2. **Baseline Evaluation**: Zero-shot performance of pre-trained models
3. **Fine-tuning**: Full fine-tuning with hyperparameter optimization
4. **LLM Evaluation**: Prompting strategies with large language models
5. **Analysis**: Comprehensive error analysis and model comparison
6. **Interpretability**: Token importance analysis using attention and gradient-based methods

## Expected Results

- **Sentence Classification**: F1 > 85%
- **Token Classification**: F1 > 80%
- **Dataset Release**: Publicly available on HuggingFace
- **Publication**: Academic paper at ACL/EMNLP or similar venue

## Key Features

- Expression-based data splitting to prevent data leakage
- Cross-seed validation for robust results
- Statistical significance testing
- Interpretability analysis with token importance visualization
- Comprehensive comparison of encoder models vs. LLMs

## License

(To be added)

## Citation

(To be added upon publication)

## Contact

(To be added)

---

**Status**: In Development

This project is part of an academic research effort to advance Hebrew NLP and idiom detection capabilities.
