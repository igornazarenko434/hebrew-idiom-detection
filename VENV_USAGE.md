# Virtual Environment Usage Guide

## Quick Start

### Activate the environment:
```bash
source activate_env.sh
```

This will:
- Activate the virtual environment
- Show which Python is being used
- Display installed package versions

### Run analysis scripts:
```bash
# After activating the environment above:

# Task 1.1: Fine-tuning results analysis
python src/analyze_finetuning_results.py

# Task 1.2: Generalization gap analysis
python src/analyze_generalization.py

# Task 1.3: Error categorization
python scripts/categorize_all_errors.py
```

### Deactivate when done:
```bash
deactivate
```

---

## Manual Setup (if needed)

### Create virtual environment:
```bash
python3 -m venv venv
```

### Activate:
```bash
source venv/bin/activate
```

### Install analysis packages:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn tabulate
```

### Or install full requirements (for training):
```bash
pip install -r requirements.txt
```

---

## Installed Packages (Analysis Only)

- **pandas** - Data manipulation and CSV handling
- **numpy** - Numerical computing
- **scipy** - Statistical tests (t-test, Cohen's d)
- **scikit-learn** - Machine learning metrics
- **matplotlib** - Plotting
- **seaborn** - Statistical visualization
- **tabulate** - Markdown table generation

---

## Requirements.txt

The `requirements.txt` file contains ALL project dependencies including:
- PyTorch and transformers (for training)
- Analysis packages (listed above)
- Jupyter notebooks
- Testing frameworks

For analysis-only work, you don't need the full requirements.txt.

---

## Troubleshooting

**Problem:** `ModuleNotFoundError`
**Solution:** Make sure you've activated the environment:
```bash
source venv/bin/activate
python --version  # Should show venv python
which python      # Should show ./venv/bin/python
```

**Problem:** Package not found
**Solution:** Install missing package:
```bash
pip install <package_name>
```

**Problem:** Want to add a new dependency
**Solution:**
1. Install it: `pip install <package>`
2. Update requirements: `pip freeze > requirements.txt`

---

## Notes

- Virtual environment is in `venv/` directory (git-ignored)
- Always activate before running Python scripts
- Use `deactivate` to exit the environment
- The venv is isolated - won't affect system Python

---

Last Updated: December 31, 2025
