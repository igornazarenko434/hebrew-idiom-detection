# Hebrew Idiom Detection

## Project Overview

This project focuses on automatic detection and interpretation of Hebrew idioms in natural language text. The goal is to distinguish between literal and figurative uses of expressions and identify idiomatic spans within sentences.

## Research Objectives

1. **Task 1: Sentence Classification** - Classify sentences containing specific expressions as either literal (מילולי) or figurative (פיגורטיבי)
2. **Task 2: Token Classification** - Identify the exact span of the idiom within the sentence using IOB2 tagging

## Dataset

- **Total Samples**: 4,800 sentences
- **Distribution**: 50% literal, 50% figurative
- **Unique Idioms**: 60-80 Hebrew expressions
- **Annotations**: IOB2 tags for idiom spans
- **Splits**: Expression-based train/validation/test splits to prevent data leakage

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
├── scripts/                # Automation scripts
├── docker/                 # Docker configurations
├── tests/                  # Unit tests
└── paper/                  # Paper materials
    ├── figures/           # Figures for publication
    └── tables/            # Tables for publication
```

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

(To be added as development progresses)

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
