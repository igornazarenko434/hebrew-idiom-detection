# Notebooks

Interactive Jupyter notebooks for data exploration and results analysis.

## Contents

| Notebook | Purpose |
|----------|---------|
| `01_data_validation.ipynb` | Dataset quality checks, statistics, and visualization of Hebrew-Idioms-4800 |
| `Complete_Dataset_Analysis.ipynb` | Comprehensive dataset exploration including per-idiom distributions, length analysis, and annotation consistency |
| `training_results_analysis.ipynb` | Interactive analysis of fine-tuning results across all models and seeds |

## Usage

```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

All notebooks can be run without GPU access -- they analyze pre-computed results and the dataset itself.
