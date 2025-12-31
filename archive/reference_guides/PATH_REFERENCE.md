# Path Reference Guide
## Hebrew Idiom Detection Project - Canonical Paths

**Last Updated:** 2025-12-08
**Purpose:** Single source of truth for all directory paths and file locations

---

## Volume Structure (Vast.ai)

### Volume Root
**Path:** `/workspace/`
**Mount Point:** Always attach volume at `/workspace` when renting instances

### Directory Structure

```
/workspace/
├── env/                              # Python virtual environment (persistent)
│   ├── bin/python                    # Python 3.10+
│   └── lib/python3.10/site-packages/ # All installed packages
│
├── data/                             # Dataset files (persistent)
│   ├── expressions_data_tagged_v2.csv
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── unseen_idiom_test.csv
│
├── project/                          # Git repository (updated each session)
│   ├── .git/
│   ├── src/                          # Python source code
│   ├── scripts/                      # Shell scripts
│   ├── experiments/
│   │   ├── configs/                  # YAML configuration files
│   │   │   ├── training_config.yaml
│   │   │   └── hpo_config.yaml
│   │   └── results/                  # All training outputs (GROWS OVER TIME)
│   │       ├── zero_shot/            # Zero-shot evaluation results
│   │       │   └── {model}_{split}_{task}.json
│   │       ├── full_finetune/        # Full fine-tuning results
│   │       │   └── {model}/
│   │       │       └── {task}/
│   │       │           ├── checkpoint-*/
│   │       │           ├── logs/
│   │       │           ├── training_results.json
│   │       │           └── summary.txt
│   │       ├── frozen_backbone/      # Frozen backbone training results
│   │       │   └── {model}/
│   │       │       └── {task}/
│   │       │           └── (same structure as full_finetune)
│   │       ├── hpo/                  # HPO trial results
│   │       │   └── {model}/
│   │       │       └── {task}/
│   │       │           ├── trial_0/
│   │       │           ├── trial_1/
│   │       │           └── trial_N/
│   │       ├── optuna_studies/       # Optuna SQLite databases
│   │       │   └── {model}_{task}_hpo.db
│   │       ├── best_hyperparameters/ # Best params from HPO
│   │       │   └── best_params_{model}_{task}.json
│   │       └── evaluation/           # Evaluation results
│   │           └── {mode}/
│   │               └── {model}/
│   │                   └── {task}/
│   │                       └── eval_results_{dataset}_{timestamp}.json
│   ├── requirements.txt
│   └── README.md
│
├── cache/                            # HuggingFace model cache (persistent)
│   └── huggingface/
│       ├── transformers/
│       └── datasets/
│
└── config/                           # Persistent configuration
    ├── .env                          # Environment variables
    ├── .rclone.conf                  # Google Drive auth (DO NOT COMMIT!)
    └── secrets/                      # Any API keys
```

---

## Path Constants by Component

### 1. Volume Paths (Vast.ai)

| Description | Path | Created By | Persistent? |
|-------------|------|------------|-------------|
| Volume root | `/workspace/` | Vast.ai | ✅ Yes |
| Python environment | `/workspace/env/` | setup_volume.sh | ✅ Yes |
| Dataset | `/workspace/data/` | setup_volume.sh | ✅ Yes |
| Data splits | `/workspace/data/splits/` | Git repo | ✅ Yes |
| Project code | `/workspace/project/` | Git clone | ✅ Yes (updated) |
| Training results | `/workspace/project/experiments/results/` | Code execution | ✅ Yes |
| HF cache | `/workspace/cache/huggingface/` | setup_volume.sh | ✅ Yes |
| Config files | `/workspace/config/` | setup_volume.sh | ✅ Yes |

### 2. Training Output Paths (Project-Relative)

| Mode | Base Path | Full Structure |
|------|-----------|----------------|
| Zero-shot | `experiments/results/zero_shot/` | `experiments/results/zero_shot/{model}_{split}_{task}.json` |
| Full finetune | `experiments/results/full_finetune/` | `experiments/results/full_finetune/{model}/{task}/` |
| Frozen backbone | `experiments/results/frozen_backbone/` | `experiments/results/frozen_backbone/{model}/{task}/` |
| HPO trials | `experiments/results/hpo/` | `experiments/results/hpo/{model}/{task}/trial_{n}/` |
| HPO databases | `experiments/results/optuna_studies/` | `experiments/results/optuna_studies/{model}_{task}_hpo.db` |
| Best params | `experiments/results/best_hyperparameters/` | `experiments/results/best_hyperparameters/best_params_{model}_{task}.json` |
| Evaluation | `experiments/results/evaluation/` | `experiments/results/evaluation/{mode}/{model}/{task}/eval_results_{dataset}_{timestamp}.json` |

### 3. Configuration File Paths

| File | Local (Mac) | Volume (Vast.ai) |
|------|-------------|------------------|
| Training config | `experiments/configs/training_config.yaml` | `/workspace/project/experiments/configs/training_config.yaml` |
| HPO config | `experiments/configs/hpo_config.yaml` | `/workspace/project/experiments/configs/hpo_config.yaml` |
| Environment vars | `.env` | `/workspace/config/.env` |
| rclone config | `~/.config/rclone/rclone.conf` | `/workspace/config/.rclone.conf` |

### 4. Script Paths

| Script | Location |
|--------|----------|
| Volume setup | `scripts/setup_volume.sh` |
| Instance bootstrap | `scripts/instance_bootstrap.sh` |
| Sync to Google Drive | `scripts/sync_to_gdrive.sh` |
| Download from Google Drive | `scripts/download_from_gdrive.sh` |
| Run all HPO | `scripts/run_all_hpo.sh` |
| Run all experiments | `scripts/run_all_experiments.sh` |

---

## Path Construction Logic (Code Behavior)

### Training Output Path Construction

**Code Location:** `src/idiom_experiment.py` lines 1106-1108

```python
output_dir = Path(config.get('output_dir', 'experiments/results/'))
output_dir = output_dir / mode_name.lower().replace(' ', '_') / Path(model_checkpoint).name / task
output_dir.mkdir(parents=True, exist_ok=True)
```

**Example:**
- Config: `output_dir: "experiments/results/"`
- Mode: `"Full Finetune"` → becomes `"full_finetune"`
- Model: `"onlplab/alephbert-base"` → becomes `"alephbert-base"`
- Task: `"cls"`
- **Result:** `experiments/results/full_finetune/alephbert-base/cls/`

### HPO Output Path Construction

**Code Location:** `src/idiom_experiment.py` lines 2125-2127

```python
trial_output_dir = Path(fixed_params.get('output_dir', 'experiments/hpo_results/'))
trial_output_dir = trial_output_dir / model_name / task / f"trial_{trial.number}"
trial_config['output_dir'] = str(trial_output_dir)
```

**Example:**
- Config: `output_dir: "experiments/results/hpo/"`
- Model: `"alephbert-base"`
- Task: `"cls"`
- Trial: `0`
- **Result:** `experiments/results/hpo/alephbert-base/cls/trial_0/`

---

## Important Notes

### ⚠️ Common Mistakes

1. **WRONG:** Using `/mnt/volume` instead of `/workspace`
   - **CORRECT:** Always use `/workspace` as mount point

2. **WRONG:** Creating `/workspace/outputs/` directory
   - **CORRECT:** Results go to `/workspace/project/experiments/results/`

3. **WRONG:** Using `full_fine-tuning` or `full-fine-tuning`
   - **CORRECT:** Code creates `full_finetune` (underscore, no hyphen)

4. **WRONG:** Hardcoding full path in config: `output_dir: "experiments/results/full_finetune/alephbert-base/cls"`
   - **CORRECT:** Use base path only: `output_dir: "experiments/results/"` (code appends structure)

5. **WRONG:** Assuming HPO outputs go to `experiments/hpo_results/`
   - **CORRECT:** Config says `experiments/results/hpo/` (code uses config value)

### ✅ Best Practices

1. **Always use project-relative paths** in code and configs
2. **Let the code append structure** (mode/model/task) automatically
3. **Volume paths are absolute** (`/workspace/...`)
4. **Project paths are relative** (`experiments/results/...`)
5. **Bootstrap script handles symlinks** (don't manually link configs)

---

## Sync Behavior (Google Drive)

**Script:** `scripts/sync_to_gdrive.sh`

| Local Source | Google Drive Destination |
|--------------|--------------------------|
| `experiments/results/` | `gdrive:Hebrew_Idiom_Detection/results/` |
| `experiments/logs/` | `gdrive:Hebrew_Idiom_Detection/logs/` |
| `models/` | `gdrive:Hebrew_Idiom_Detection/models/` (optional) |

**Note:** Sync script uses project-relative paths and works both locally and on Vast.ai

---

## Environment Variables

**File:** `/workspace/config/.env` (on volume)

```bash
# HuggingFace cache
HF_HOME=/workspace/cache/huggingface
TRANSFORMERS_CACHE=/workspace/cache/huggingface

# Dataset metadata
DATASET_NAME=Hebrew-Idioms-4800
DATASET_FILE_ID=140zJatqT4LBl7yG-afFSoUrYrisi9276

# Add any API keys below (e.g., OPENAI_API_KEY if needed)
```

**Note:** No need to set `LOCAL_RESULTS_DIR` - code uses project-relative paths

---

## Quick Reference Commands

### Check Paths on Vast.ai Instance

```bash
# Check volume is mounted
df -h | grep /workspace

# Check volume structure
ls -la /workspace/

# Check project results
ls -la /workspace/project/experiments/results/

# Check HF cache
du -sh /workspace/cache/huggingface/

# Check Python environment
source /workspace/env/bin/activate
which python
```

### Verify Code Output Paths

```bash
# Check what code creates
cd /workspace/project
python -c "
from pathlib import Path
output_dir = Path('experiments/results/') / 'full_finetune' / 'alephbert-base' / 'cls'
print(f'Output path: {output_dir}')
"
```

---

## Path Migration History

### Before (Inconsistent)

- Volume: `/mnt/volume/` ❌
- Results: `/mnt/volume/outputs/` ❌
- Mode: `full_fine-tuning` or `full-fine-tuning` ❌
- HPO: Mixed `experiments/hpo_results/` and `experiments/results/hpo/` ❌

### After (Consistent)

- Volume: `/workspace/` ✅
- Results: `/workspace/project/experiments/results/` ✅
- Mode: `full_finetune` ✅
- HPO: `experiments/results/hpo/` ✅

---

**This document is the single source of truth for all paths in the project. When in doubt, refer here.**
