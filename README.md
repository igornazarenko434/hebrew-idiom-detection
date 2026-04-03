# Detection Without Localization: Compositional Generalization Failures in Transformer Models for Hebrew Idiom Boundaries

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30+-yellow)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Igor Nazarenko & Yuval Amit**
M.Sc. Machine Learning & Data Science | Efi Arazi School of Computer Science, Reichman University
Supervised by Dr. Kfir Bar

[Paper (PDF)](paper/report/Hebrew_Idiom_Detection_Report.pdf) | [Presentation](presentation/) | [Dataset](data/) | [Results](#key-results) | [Reproduce](docs/REPRODUCING.md)

</div>

---

## Abstract

Do transformer models learn compositional representations of multi-word expressions, or do they rely on memorization? We investigate through idiom boundary prediction, where models must generalize structural knowledge to novel expressions. Using **Hebrew-Idioms-4800**, a new dual-task dataset of 4,800 sentences covering 60 idioms with near-perfect annotation agreement (Cohen's kappa = 0.97), we evaluate six transformer encoders and five LLMs on both idiom usage classification (CLS) and token-level idiom identification (SPAN; BIO tagging) across seen and held-out idioms.

**Key finding**: Classification generalizes robustly to unseen idioms (F1 0.90-0.92; 0.5-3.5 point drop), while span identification degrades substantially (F1 0.59-0.76; 23-41 point drop). LLM prompting exhibits *reversed* error patterns compared to fine-tuning, suggesting that fine-tuning induces boundary memorization while prompting preserves compositional reasoning.

---

## Key Results

### The Generalization Gap: CLS vs SPAN

<p align="center">
  <img src="paper/figures/generalization/generalization_bar_cls.png" width="48%" alt="CLS Generalization">
  <img src="paper/figures/generalization/generalization_bar_span.png" width="48%" alt="SPAN Generalization">
</p>

**Classification generalizes. Span identification collapses.** Models detect *that* an idiom is present (CLS) even for unseen idioms, but fail to locate *where* it is (SPAN) -- revealing memorization of surface boundaries rather than compositional understanding.

### Fine-Tuned Encoder Results

| Model | CLS Seen F1 | CLS Unseen F1 | SPAN Seen F1 | SPAN Unseen F1 |
|-------|:-----------:|:-------------:|:------------:|:--------------:|
| **NeoDictaBERT** | **0.958** +/- 0.012 | **0.924** +/- 0.003 | **0.997** +/- 0.001 | 0.661 +/- 0.070 |
| DictaBERT | 0.941 +/- 0.003 | 0.922 +/- 0.004 | 0.994 +/- 0.002 | **0.761** +/- 0.048 |
| AlephBERTGimmel | 0.940 +/- 0.012 | 0.908 +/- 0.012 | 0.990 +/- 0.001 | 0.747 +/- 0.015 |
| AlephBERT | 0.930 +/- 0.003 | 0.910 +/- 0.007 | 0.994 +/- 0.003 | 0.668 +/- 0.021 |
| XLM-RoBERTa | 0.912 +/- 0.014 | 0.907 +/- 0.005 | 0.994 +/- 0.002 | 0.615 +/- 0.036 |
| mBERT | 0.888 +/- 0.005 | 0.904 +/- 0.007 | 0.995 +/- 0.002 | 0.588 +/- 0.080 |

*Mean F1 +/- standard deviation across 3 random seeds (42, 123, 456). 95% bootstrap confidence intervals available in [full analysis](experiments/results/analysis/finetuning_summary.md).*

### Error Analysis Highlights

- **Boundary truncation dominates**: 1,761 partial-end errors vs. only 1 partial-start across all fine-tuned models
- **Length effect**: Unseen idioms with 5+ tokens exhibit 70-100% error rates despite near-perfect seen performance
- **LLM reversal**: Prompted LLMs show 5-13% partial-start errors (opposite of fine-tuned models), and Gemini 3 Flash achieves 90% SPAN F1 on unseen idioms, exceeding the best fine-tuned model

---

## Dataset: Hebrew-Idioms-4800

The **first comprehensive Hebrew idiom dataset** with dual-task annotations for both sentence-level classification and token-level span identification.

| Metric | Value |
|--------|-------|
| Total sentences | 4,800 |
| Unique idioms | 60 (100% polysemous) |
| Label balance | 50/50 literal/figurative |
| Inter-annotator agreement | Cohen's kappa = 0.97 |
| Annotators | 2 native Hebrew speakers |
| Mean sentence length | 17.5 tokens |
| Idiom length range | 2-5 tokens |

### Data Splits

| Split | Sentences | Idioms | Purpose |
|-------|-----------|--------|---------|
| Train | 3,456 (72%) | 54 seen | Model training |
| Validation | 432 (9%) | 54 seen | Model selection |
| Seen Test | 432 (9%) | 54 seen | In-domain evaluation |
| Unseen Test | 480 (10%) | 6 held-out | Zero-shot generalization |

### Example

```
Sentence:  "אבי שבר את הקרח במועדון הספורט בשיחה על נבחרת ישראל."
Translation: Avi broke the ice at the sports club in a conversation about the Israeli national team.
Idiom:      שבר את הקרח (broke the ice)
Label:      Figurative
BIO tags:   O  B-IDIOM  I-IDIOM  I-IDIOM  O  O  O  O  O  O  O
```

See [`data/README.md`](data/README.md) for full schema, quality metrics, and construction methodology.

---

## Models

### Fine-Tuned Encoders

| Model | HuggingFace ID | Type |
|-------|---------------|------|
| AlephBERT | `onlplab/alephbert-base` | Hebrew |
| AlephBERTGimmel | `dicta-il/alephbertgimmel-base` | Hebrew |
| DictaBERT | `dicta-il/dictabert` | Hebrew |
| NeoDictaBERT | `dicta-il/neodictabert` | Hebrew |
| mBERT | `bert-base-multilingual-cased` | Multilingual |
| XLM-RoBERTa | `xlm-roberta-base` | Multilingual |

### LLM Prompting Baselines

| Model | Prompting |
|-------|-----------|
| Gemini 3 Flash Preview | Zero-shot & 3-shot |
| Llama 4 Maverick Instruct | Zero-shot & 3-shot |
| Llama 4 Scout Instruct | Zero-shot & 3-shot |
| Dicta 3 Nemotron 12B | Zero-shot & 3-shot |
| Qwen 3 235B A22B | Zero-shot & 3-shot |

---

## Repository Structure

```
hebrew-idiom-detection/
├── README.md
├── LICENSE                      # MIT (code) + CC BY 4.0 (dataset)
├── CONTRIBUTING.md
├── CITATION.cff                 # Machine-readable citation metadata
├── requirements.txt
│
├── .github/                     # Issue & PR templates
│
├── data/                        # Hebrew-Idioms-4800 dataset
│   ├── README.md                # Dataset documentation
│   └── splits/                  # Train/val/test splits (CSV + JSON)
│
├── src/                         # Core training & analysis code
│   ├── idiom_experiment.py      # Main experiment runner (train/eval/HPO)
│   ├── data_preparation.py      # Data loading & preprocessing
│   └── utils/                   # Tokenization alignment, error analysis
│
├── scripts/                     # Analysis & automation scripts
│   ├── statistical_tests.py     # Paired t-tests, significance analysis
│   ├── analyze_*.py             # Interpretability & error analysis
│   └── *.sh                     # Training & evaluation batch scripts
│
├── experiments/
│   ├── configs/                 # Training & HPO configurations (YAML)
│   └── results/                 # All experiment outputs
│       ├── analysis/            # Aggregated statistics, CSV tables
│       ├── evaluation/          # Per-model evaluation JSONs
│       ├── full_fine-tuning/    # Trained model checkpoints
│       └── zero_shot/           # Zero-shot baseline results
│
├── paper/
│   ├── report/                  # Final thesis
│   │   └── Hebrew_Idiom_Detection_Report.pdf
│   ├── conll2026_paper.tex      # Conference paper source
│   ├── figures/                 # All publication-ready figures
│   └── tables/                  # LaTeX tables
│
├── presentation/                # HTML5 slide deck (open index.html)
│
├── notebooks/                   # Jupyter notebooks for exploration
├── docker/                      # Dockerfile + docker-compose
├── tests/                       # Unit tests
└── docs/                        # Supplementary documentation
    └── REPRODUCING.md           # Full reproduction guide
```

---

## Getting Started

```bash
# Clone
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
cd hebrew-idiom-detection

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify
python -c "import torch, transformers; print('Ready')"
```

To reproduce all experiments, see [docs/REPRODUCING.md](docs/REPRODUCING.md).

To analyze pre-computed results (no GPU required):

```bash
python src/analyze_finetuning_results.py    # Summary tables
python src/analyze_generalization.py        # Generalization gap
python scripts/statistical_tests.py         # Statistical significance
```

---

## Paper & Presentation

- **Full Report (PDF)**: [`paper/report/Hebrew_Idiom_Detection_Report.pdf`](paper/report/Hebrew_Idiom_Detection_Report.pdf)
- **Presentation**: Open [`presentation/index.html`](presentation/index.html) in a browser (self-contained, no dependencies)
- **Conference Paper Source**: [`paper/conll2026_paper.tex`](paper/conll2026_paper.tex)

---

## Citation

```bibtex
@mastersthesis{nazarenko2026hebrew,
    title     = {Detection Without Localization: Compositional Generalization
                 Failures in Transformer Models for Hebrew Idiom Boundaries},
    author    = {Nazarenko, Igor and Amit, Yuval},
    school    = {Reichman University, Efi Arazi School of Computer Science},
    year      = {2026},
    type      = {M.Sc. Thesis},
    note      = {Machine Learning and Data Science Track}
}
```

---

## Authors & Acknowledgments

**Igor Nazarenko** & **Yuval Amit**
M.Sc. Machine Learning & Data Science, Reichman University

**Supervisor**: Dr. Kfir Bar, Efi Arazi School of Computer Science, Reichman University

We thank Kai Golan Hashiloni for his invaluable guidance throughout the project, including initial review and continuous support.

---

## License

- **Code**: [MIT License](LICENSE)
- **Dataset (Hebrew-Idioms-4800)**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
