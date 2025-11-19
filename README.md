# Hebrew Idiom Detection

**Repository:** https://github.com/igornazarenko434/hebrew-idiom-detection

## Project Overview

This project focuses on automatic detection and interpretation of Hebrew idioms in natural language text. The goal is to distinguish between literal and figurative uses of expressions and identify idiomatic spans within sentences.

## Research Objectives

1. **Task 1: Sentence Classification** - Classify sentences containing specific expressions as either literal (מילולי) or figurative (פיגורטיבי)
2. **Task 2: Token Classification** - Identify the exact span of the idiom within the sentence using IOB2 tagging

## Dataset

### Overview (Hebrew-Idioms-4800 v1.0)

| Metric | Value |
|--------|-------|
| Total sentences | 4,800 (manually authored) |
| Unique idioms | 60 (80 samples per idiom, 100% polysemous) |
| Label balance | 2,400 literal / 2,400 figurative (perfect 50/50) |
| Annotators | 2 native Hebrew speakers |
| IAA | Cohen's κ = **0.9725** (98.625% observed agreement) |
| Data quality | 14/14 validation checks passed, score **9.2/10** |
| Annotations | Sentence label + token spans (IOB2 + char spans + token spans) |

Additional details plus visualizations live in [professor_review/](professor_review/), including the full `Complete_Dataset_Analysis.ipynb`.

### Split Strategy

We follow a **hybrid split** that supports both in-domain and zero-shot evaluation:

| Split | Samples | % | Idioms | Literal | Figurative |
|-------|---------|---|--------|---------|------------|
| Train | 3,456 | 72% | 54 seen idioms | 1,728 | 1,728 |
| Validation | 432 | 9% | Same 54 | 216 | 216 |
| Test (seen) | 432 | 9% | Same 54 | 216 | 216 |
| Unseen idiom test | 480 | 10% | 6 held-out idioms | 240 | 240 |

- **Seen idioms:** Stratified 80/10/10 split per idiom and label (32/4/4 literal + figurative).  
- **Unseen idioms:** חתך פינה, חצה קו אדום, נשאר מאחור, שבר שתיקה, איבד את הראש, רץ אחרי הזנב של עצמו (all 80 samples per idiom held out).

### Key Statistics (from professor review package)

- **Sentence length:** 15.71 tokens on average (median 12, range 5-38); 83.04 characters on average (median 63, range 22-193).  
- **Idiom span length:** 2.48 tokens (median 2, range 2-5); 11.39 characters (median 11, range 5-23).  
- **Sentence types:** 94.77% declarative, 4.98% interrogative, 0.25% exclamatory.  
- **Idiom position:** 63.71% start, 29.77% middle, 6.52% end (figurative spans skew slightly later than literal).  
- **Lexical diversity:** 18,784 unique tokens across 75,412 total words (TTR 0.2491, hapax 63.46%, Maas 0.0110, function word ratio 12.57%).  
- **Morphology:** 45.25% of tokens have prefix attachments; top idioms by morphological variance—שם רגליים (35 variants), שבר את הלב (32), פתח דלתות (29), סגר חשבון (28), הוריד פרופיל (23).  
- **Complexity:** Figurative sentences contain 24% more subclause markers (mean subclause ratio 0.017 vs. 0.012) and slightly more punctuation.  
- **Collocations:** 23,366 context words collected (8,498 unique); הו/היא/לא/הם/על/את/עם/של/כדי/אחרי dominate.  
- **Quality validation:** 0 missing values, 0 duplicates, character/token spans verified 100%, IOB2 sequences valid, encoding normalized (NFKC, BOM removed, no zero-width chars).

### Documentation

- [data/README.md](data/README.md) — column definitions and usage notes  
- [professor_review/README.md](professor_review/README.md) — full statistical report and notebook links

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

Evaluate pre-trained models on the **unseen idiom test set** (`data/splits/unseen_idiom_test.csv`):

```bash
# Task 1: Sequence Classification
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/unseen_idiom_test.csv \
  --task cls \
  --device cpu

# Task 2: Token Classification (IOB2 Tagging)
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/unseen_idiom_test.csv \
  --task span \
  --device cpu

# Both tasks
python src/idiom_experiment.py \
  --mode zero_shot \
  --model_id onlplab/alephbert-base \
  --data data/splits/unseen_idiom_test.csv \
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

> **Note:** The default configs use `data/splits/test.csv` (in-domain test). Evaluate the final model on `data/splits/unseen_idiom_test.csv` as well to report zero-shot performance.

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

- Hybrid splitting (seen vs unseen idioms) to report both in-domain and zero-shot performance
- Cross-seed validation for robust results
- Statistical significance testing
- Interpretability analysis with token importance visualization
- Comprehensive comparison of encoder models vs. LLMs

## License

(To be added)

## Citation

If you use this dataset or code in your research, please cite:

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

## Contact

- **Igor Nazarenko**: igor.nazarenko@post.runi.ac.il
- **Yuval Amit**: yuval.amit@post.runi.ac.il

**Repository:** https://github.com/igornazarenko434/hebrew-idiom-detection

---

**Status**: In Development

This project is part of a Master's research effort at Reichman University to advance Hebrew NLP and idiom detection capabilities.
