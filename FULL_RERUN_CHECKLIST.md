# Full Re-Run Checklist (VAST.ai → Google Drive → Local Analysis)
# Hebrew Idiom Detection Project

This checklist runs **all models** (AlephBERT, AlephBERTGimmel, DictaBERT, NeoDictaBERT, mBERT, XLM-R) for **both tasks** (cls, span) end-to-end.

---

## A) One-Time Setup (Persistent Volume)

- [ ] Create a VAST.ai storage volume
  - Name: `hebrew-idiom-volume`
  - Size: 100–150 GB (150 GB recommended for full re-run)

- [ ] Rent a cheap temporary instance
  - Attach the volume at `/workspace`

- [ ] Run volume setup
  - `bash /root/temp_repo/scripts/setup_volume.sh`

- [ ] Destroy the instance (keep volume)

**What it creates:**
- `/workspace/env` (venv + packages)
- `/workspace/data` (dataset + splits)
- `/workspace/project` (repo clone)
- `/workspace/cache/huggingface`
- `/workspace/config/.rclone.conf`

---

## B) Every Training Session (Bootstrap)

- [ ] Rent GPU instance (RTX 3090 / A6000 / 4090)
  - Attach volume at `/workspace`

- [ ] SSH in, then bootstrap
  - `bash /workspace/project/scripts/instance_bootstrap.sh`

---

## C) HPO (All Models, All Tasks)

- [ ] Run batch HPO
  - `cd /workspace/project`
  - `bash scripts/run_all_hpo.sh`

**What it saves:**
- Optuna DBs: `experiments/results/optuna_studies/{model}_{task}_hpo.db`
- Trial outputs: `experiments/results/hpo/{model}/{task}/trial_{N}/`
- Best params: `experiments/results/best_hyperparameters/best_params_{model}_{task}.json`

**Notes:**
- HPO uses **dev F1** (no test leakage).
- Only final weights per trial are saved (no checkpoints).

---

## D) Full Training (All Models, All Tasks, 3 Seeds)

- [ ] Run batch training
  - `cd /workspace/project`
  - `bash scripts/run_all_experiments.sh`

**What it saves per run:**
`experiments/results/full_fine-tuning/<model>/<task>/seed_<seed>/`
- `model.safetensors` (best weights)
- `config.json` (ensured for span + CRF)
- tokenizer files
- `training_results.json`
- `logs/`
- no `checkpoint-*` folders (auto-deleted)

---

## E) Batch Evaluation (Seen + Unseen)

- [ ] Run batch evaluation
  - `cd /workspace/project`
  - `bash scripts/run_evaluation_batch.sh`

**What it saves:**
- Seen: `experiments/results/evaluation/seen_test/<model>/<task>/seed_<seed>/`
- Unseen: `experiments/results/evaluation/unseen_test/<model>/<task>/seed_<seed>/`
- Files: `eval_results*.json`, `eval_predictions.json`

---

## F) Sync Everything to Google Drive

- [ ] Sync results + logs
  - `bash scripts/sync_to_gdrive.sh`

**Uploads:**
- `experiments/results/` → `gdrive:Hebrew_Idiom_Detection/results/`
- `experiments/logs/` → `gdrive:Hebrew_Idiom_Detection/logs/`

---

## G) Download to Local (PyCharm)

Choose based on needs:

- [ ] Best model weights only
  - `bash scripts/download_best_checkpoints.sh`

- [ ] Training metrics only (lightweight)
  - `bash scripts/download_results_for_analysis.sh`

- [ ] Evaluation JSONs
  - `bash scripts/download_evaluation_results.sh`

---

## H) Local Analysis (Before Deep Evaluation Write-up)

Run in this order:

- [ ] Aggregate fine-tuning results
  - `python src/analyze_finetuning_results.py`
  - Outputs: `experiments/results/analysis/finetuning_summary.csv/md`

- [ ] Generalization gap analysis
  - `python src/analyze_generalization.py`
  - Outputs: `experiments/results/analysis/generalization/*`

- [ ] Error categorization
  - `python scripts/categorize_all_errors.py`
  - Updates: all `eval_predictions.json` with `error_category`

- [ ] Error distribution report
  - `python src/analyze_error_distribution.py`
  - Outputs: `experiments/results/analysis/error_analysis/*`

- [ ] Per-idiom F1 analysis
  - `python scripts/analyze_per_idiom_f1.py`
  - Outputs: `experiments/results/analysis/per_idiom_f1/*`

- [ ] Statistical tests
  - `python scripts/statistical_tests.py`
  - Outputs: `experiments/results/analysis/statistical_tests/*`

- [ ] Optuna study analysis (optional but recommended)
  - `python src/analyze_optuna_results.py --db_dir experiments/results/optuna_studies`
  - Outputs: `experiments/results/hpo_analysis/*`

---

## I) Final Evaluation + Figures

- [ ] Regenerate charts/tables if needed
  - `python src/analyze_finetuning_results.py --create_figures`
  - `python src/analyze_generalization.py --create_figures`

- [ ] Check outputs under:
  - `experiments/results/analysis/`
  - `paper/figures/`
  - `paper/tables/`

---

## J) Sanity Checks

- [ ] Confirm training outputs exist
  - `python src/audit_results.py`

- [ ] Count evaluation files
  - `find experiments/results/evaluation -name "eval_results*.json" | wc -l`

---

## K) What Gets Saved Where (Quick Reference)

- HPO DBs: `experiments/results/optuna_studies/`
- HPO trials: `experiments/results/hpo/`
- Best params: `experiments/results/best_hyperparameters/`
- Training runs: `experiments/results/full_fine-tuning/`
- Evaluations: `experiments/results/evaluation/`
- Analysis: `experiments/results/analysis/`
- Figures: `paper/figures/`
- Tables: `paper/tables/`
