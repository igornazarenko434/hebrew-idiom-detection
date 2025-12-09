# Helper Scripts for VAST.ai Training
## Hebrew Idiom Detection Project

This directory contains automation scripts for managing persistent volume workflows and running experiments on VAST.ai GPU instances.

**Last Updated:** 2025-12-08

---

## 📁 Scripts Overview

| Script | Purpose | When to Use | Priority |
|--------|---------|-------------|----------|
| `setup_volume.sh` | Initialize persistent volume (ONE TIME) | First time setup on cheap instance | ⭐⭐⭐ Critical (once) |
| `instance_bootstrap.sh` | Quick setup for new instances | Every time you rent a new instance | ⭐⭐⭐ Critical (every session) |
| `run_all_hpo.sh` | Run all HPO studies (10 studies) | Mission 4.5 | ⭐⭐⭐ Recommended |
| `run_all_experiments.sh` | Run final training (30 runs) | Mission 4.6 | ⭐⭐⭐ Recommended |
| `sync_to_gdrive.sh` | Upload results to Google Drive | After training completes | ⭐⭐⭐ Critical |
| `download_from_gdrive.sh` | Download dataset from Google Drive | Used by setup_volume.sh | ⭐⭐ Helper |

---

## 🚀 Quick Start Workflow

### **Phase 1: One-Time Setup (30 minutes)**

This creates a persistent volume that survives across all instances.

```bash
# 1. Create storage volume on Vast.ai website
#    - Go to https://vast.ai/console/storage/
#    - Click "Create Storage"
#    - Name: hebrew-idiom-volume
#    - Size: 100 GB
#    - Click "Create"

# 2. Rent CHEAP instance for setup (ANY GPU, <$0.20/hour)
#    - Attach volume at /workspace during rental
#    - SSH in

# 3. Download and run setup script
cd /root
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git temp_repo
cp temp_repo/scripts/setup_volume.sh .
rm -rf temp_repo

bash setup_volume.sh
# This takes 20-30 minutes and installs:
# - Python environment
# - All dependencies
# - Dataset
# - Project code
# - rclone authentication

# 4. DESTROY setup instance (keep volume!)
exit  # Exit SSH
# In Vast.ai console: Destroy instance, KEEP volume
```

### **Phase 2: Every Training Session (1 minute)**

Now you can rent instances and be training in under 2 minutes!

```bash
# 1. Rent GPU instance (RTX 4090, attach volume at /workspace)

# 2. SSH in
ssh -p <PORT> root@<IP>

# 3. Bootstrap (ONLY command needed!)
bash /workspace/project/scripts/instance_bootstrap.sh
# Takes ~30 seconds, pulls latest code, activates environment

# 4. Train
cd /workspace/project
python src/idiom_experiment.py --mode full_finetune --model_id onlplab/alephbert-base --task cls --device cuda

# OR run HPO
bash scripts/run_all_hpo.sh

# OR run all experiments
bash scripts/run_all_experiments.sh

# 5. Sync to Google Drive
bash scripts/sync_to_gdrive.sh

# 6. Destroy instance (volume stays safe!)
exit
```

---

## 📋 Detailed Script Documentation

### 1. `setup_volume.sh` ⭐ RUN ONCE

**Purpose:** Initialize persistent volume with complete training environment

**What it does:**
1. Creates directory structure on volume: `/workspace/{env,data,project,cache,config}`
2. Installs system dependencies (Python 3.10, git, curl, etc.)
3. Creates Python virtual environment on volume (persistent!)
4. Installs PyTorch with CUDA support
5. Clones GitHub repository to volume
6. Installs all Python dependencies from requirements.txt
7. Downloads dataset to volume
8. Configures rclone for Google Drive sync
9. Sets up environment variables
10. Pre-downloads HuggingFace models (optional)
11. Verifies everything is ready

**Usage:**
```bash
# On temporary setup instance
bash setup_volume.sh
```

**Time:** 20-30 minutes (one-time only!)

**Requirements:**
- Vast.ai instance with volume attached at `/workspace`
- Internet connection
- Will prompt for Google Drive authentication

**Output:**
- ✅ Volume fully configured and ready
- ✅ All dependencies installed
- ✅ Dataset downloaded
- ✅ rclone authenticated
- ✅ Can destroy setup instance

**Important:**
- Only run this ONCE when creating the volume
- After completion, destroy the instance but KEEP the volume
- Volume will be reused for all future training sessions

---

### 2. `instance_bootstrap.sh` ⭐ RUN EVERY SESSION

**Purpose:** Quick setup for new instances using existing volume

**What it does:**
1. Verifies volume is mounted at `/workspace`
2. Checks volume was initialized (setup_volume.sh completed)
3. Checks GPU availability
4. Symlinks rclone config from volume to instance
5. Symlinks .env file from volume
6. Activates Python environment from volume
7. Pulls latest code from GitHub
8. Verifies dataset exists
9. Shows you're ready to train

**Usage:**
```bash
# On any new instance with volume attached
bash /workspace/project/scripts/instance_bootstrap.sh
```

**Time:** ~30 seconds

**Requirements:**
- Volume must be attached at `/workspace`
- setup_volume.sh must have been run previously

**Output:**
- ✅ Environment activated
- ✅ Latest code pulled
- ✅ Ready to train immediately
- Shows quick start commands

**What it DOESN'T do:**
- ❌ Doesn't install anything (uses volume)
- ❌ Doesn't download dataset (already on volume)
- ❌ Doesn't configure rclone (uses volume config)

---

### 3. `run_all_hpo.sh` ⭐ MISSION 4.5

**Purpose:** Run hyperparameter optimization for all model-task combinations

**What it does:**
1. Runs HPO for 5 models × 2 tasks = 10 studies
2. Each study runs 15 trials (configured in hpo_config.yaml)
3. Saves best hyperparameters to `experiments/results/best_hyperparameters/`
4. Saves Optuna databases to `experiments/results/optuna_studies/`
5. Optionally syncs to Google Drive after each study

**Usage:**
```bash
cd /workspace/project
bash scripts/run_all_hpo.sh
```

**Time:** 50-75 GPU hours (depends on GPU speed)

**Cost:** ~$20-30 on Vast.ai @ $0.40/hour

**Models optimized:**
- onlplab/alephbert-base
- dicta-il/alephbertgimmel-base
- dicta-il/dictabert
- bert-base-multilingual-cased
- xlm-roberta-base

**Tasks:**
- cls (sequence classification)
- span (token classification)

**Output:**
- 10 best hyperparameter JSON files
- 10 Optuna study databases
- Complete logs in `experiments/logs/`

**Recommendation:** Use screen/tmux for this:
```bash
screen -S hpo
bash scripts/run_all_hpo.sh
# Detach: Ctrl+A then D
```

---

### 4. `run_all_experiments.sh` ⭐ MISSION 4.6

**Purpose:** Run final training with best hyperparameters (cross-seed validation)

**What it does:**
1. Loads best hyperparameters from Mission 4.5
2. Trains each model-task combination with 3 different seeds
3. Total: 5 models × 2 tasks × 3 seeds = 30 training runs
4. Saves results to `experiments/results/full_finetune/`
5. Optionally syncs to Google Drive after each model

**Usage:**
```bash
cd /workspace/project
bash scripts/run_all_experiments.sh
```

**Time:** 10-15 GPU hours

**Cost:** ~$4-6 on Vast.ai @ $0.40/hour

**Seeds:** 42, 123, 456 (for statistical analysis)

**Output:**
- 30 trained models with checkpoints
- 30 training_results.json files
- Complete logs in `experiments/logs/`
- Ready for statistical analysis (mean ± std across seeds)

---

### 5. `sync_to_gdrive.sh` ⭐ CRITICAL

**Purpose:** Upload training results to Google Drive (automated backup)

**What it syncs:**
- `experiments/results/` → `gdrive:Hebrew_Idiom_Detection/results/`
- `experiments/logs/` → `gdrive:Hebrew_Idiom_Detection/logs/`
- `models/` → `gdrive:Hebrew_Idiom_Detection/models/` (optional with --with-models)

**Usage:**

Basic sync (results and logs only):
```bash
cd /workspace/project
bash scripts/sync_to_gdrive.sh
```

With model checkpoints (large files):
```bash
bash scripts/sync_to_gdrive.sh --with-models
```

**Requirements:**
- rclone must be configured (done by setup_volume.sh)
- Volume must have `/workspace/config/.rclone.conf`

**Sync behavior:**
- Uses `--update` flag: Only uploads newer files
- Uses 4 parallel transfers (fast)
- Shows progress bars
- Skips .DS_Store, __pycache__, etc.

**When to use:**
- ✅ After each training run
- ✅ After HPO completes
- ✅ **ALWAYS before destroying instance** (critical!)
- ✅ Periodically during long runs

**Verification:**
```bash
# Check what was uploaded
rclone ls gdrive:Hebrew_Idiom_Detection/results/
```

---

### 6. `download_from_gdrive.sh`

**Purpose:** Download dataset from Google Drive

**Usage:**
```bash
bash scripts/download_from_gdrive.sh
```

**What it downloads:**
- Main dataset CSV (4800 sentences)
- Verifies split files exist in repo

**Note:** This is automatically called by setup_volume.sh. You rarely need to run this manually.

---

## 🎯 Complete Workflow Examples

### **Example 1: First Time Setup**

```bash
# === ON YOUR MAC ===
# 1. Commit latest code
cd ~/PycharmProjects/Final_Project_NLP/
git add .
git commit -m "Update training scripts"
git push origin main

# === ON VAST.AI WEBSITE ===
# 2. Create storage volume
#    Name: hebrew-idiom-volume
#    Size: 100 GB

# 3. Rent cheap instance
#    Any GPU, <$0.20/hour
#    Attach volume at /workspace

# === ON VAST.AI INSTANCE ===
# 4. SSH and setup
ssh -p <PORT> root@<IP>
cd /root
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git temp_repo
cp temp_repo/scripts/setup_volume.sh .
rm -rf temp_repo
bash setup_volume.sh
# Wait 20-30 minutes...

# 5. Exit and destroy instance (KEEP VOLUME!)
exit

# Volume is now ready for all future sessions!
```

### **Example 2: Quick Training Session**

```bash
# === ON YOUR MAC ===
# 1. Make code changes
git add .
git commit -m "Fix training bug"
git push origin main

# === ON VAST.AI WEBSITE ===
# 2. Rent GPU instance
#    RTX 4090, attach volume at /workspace

# === ON VAST.AI INSTANCE ===
# 3. SSH and bootstrap
ssh -p <PORT> root@<IP>
bash /workspace/project/scripts/instance_bootstrap.sh
# Takes 30 seconds...

# 4. Train
cd /workspace/project
python src/idiom_experiment.py --mode full_finetune --model_id onlplab/alephbert-base --task cls --device cuda

# 5. Sync
bash scripts/sync_to_gdrive.sh

# 6. Destroy instance
exit

# Done! Volume persists for next session.
```

### **Example 3: Run Full HPO (Mission 4.5)**

```bash
# === ON VAST.AI ===
# 1. Rent instance, attach volume
ssh -p <PORT> root@<IP>
bash /workspace/project/scripts/instance_bootstrap.sh

# 2. Start screen session (for long run)
screen -S hpo

# 3. Run HPO
cd /workspace/project
bash scripts/run_all_hpo.sh
# This will take 50-75 hours

# 4. Detach screen
# Press: Ctrl+A then D

# 5. Disconnect (training continues)
exit

# === LATER (CHECK PROGRESS) ===
ssh -p <PORT> root@<IP>
screen -r hpo  # Resume
# Or check logs:
tail -f /workspace/project/experiments/logs/hpo_batch_*.log

# === WHEN COMPLETE ===
# Sync results
bash /workspace/project/scripts/sync_to_gdrive.sh

# Destroy instance
exit
```

---

## 📊 Volume Structure

After setup_volume.sh completes:

```
/workspace/                                    # Volume root
├── env/                                       # Python 3.10 + all packages (5 GB)
├── data/                                      # Dataset (3 MB)
│   └── splits/
├── project/                                   # Git repository
│   ├── experiments/
│   │   ├── configs/
│   │   └── results/                           # Training outputs (grows)
│   │       ├── zero_shot/
│   │       ├── full_finetune/
│   │       ├── hpo/
│   │       ├── optuna_studies/
│   │       └── best_hyperparameters/
│   ├── scripts/
│   └── src/
├── cache/huggingface/                         # Model cache (10 GB)
└── config/                                    # Persistent configs
    ├── .env
    └── .rclone.conf
```

---

## ⚠️ Critical Reminders

### **1. ALWAYS Attach Volume at /workspace**

When renting instances:
- ✅ Click "Storage" → "Attach storage volume"
- ✅ Select: `hebrew-idiom-volume`
- ✅ Mount point: `/workspace`
- ❌ DON'T use `/mnt/volume` or other paths

### **2. ALWAYS Sync Before Destroying**

```bash
# ❌ WRONG (You lose everything!)
python src/idiom_experiment.py ...
exit  # ← Results lost!

# ✅ CORRECT
python src/idiom_experiment.py ...
bash scripts/sync_to_gdrive.sh  # ← Backup to Google Drive
# Wait for sync to complete
exit  # ← Now safe
```

### **3. Use screen/tmux for Long Runs**

```bash
# Start screen
screen -S training

# Run training
bash scripts/run_all_hpo.sh

# Detach (training continues)
# Press: Ctrl+A then D

# Disconnect SSH (safe)
exit

# Reconnect later
ssh -p <PORT> root@<IP>
screen -r training
```

### **4. Bootstrap Script is NOT Setup Script**

- `setup_volume.sh` = Run ONCE to create volume (20-30 min)
- `instance_bootstrap.sh` = Run EVERY session (30 seconds)

Don't confuse them!

---

## 🐛 Troubleshooting

### **Problem: Volume not mounted**

```bash
# Check if mounted
df -h | grep /workspace

# If not mounted:
# - Destroy instance
# - When renting, attach volume BEFORE clicking "Rent"
# - Mount point must be /workspace
```

### **Problem: bootstrap.sh fails "Volume not initialized"**

```bash
# You skipped setup_volume.sh!
# Run it once:
bash /workspace/project/scripts/setup_volume.sh
```

### **Problem: "Latest code not pulled"**

```bash
# On your Mac:
git push origin main

# On Vast.ai:
cd /workspace/project
git pull origin main
```

### **Problem: rclone sync fails**

```bash
# Check rclone config exists
cat /workspace/config/.rclone.conf

# Test connection
rclone lsd gdrive:

# If fails, re-authenticate:
rclone config reconnect gdrive:
# Save to volume:
cp ~/.config/rclone/rclone.conf /workspace/config/.rclone.conf
```

### **Problem: Out of memory during training**

```bash
# Reduce batch size
# Edit: experiments/configs/training_config.yaml
batch_size: 8  # Was 16

# Or use gradient accumulation
gradient_accumulation_steps: 2  # Effective batch = 16
```

### **Problem: GPU not detected**

```bash
# Check GPU
nvidia-smi

# Check PyTorch
source /workspace/env/bin/activate
python -c "import torch; print(torch.cuda.is_available())"

# If False:
# - Check Vast.ai instance has GPU
# - Check CUDA version matches PyTorch
```

---

## 📝 Script Modification Workflow

If you need to update these scripts:

```bash
# === ON YOUR MAC ===
# 1. Edit scripts
cd ~/PycharmProjects/Final_Project_NLP/scripts/
# Edit instance_bootstrap.sh or other scripts

# 2. Commit and push
git add scripts/
git commit -m "Update bootstrap script"
git push origin main

# === ON VAST.AI ===
# 3. Pull changes
cd /workspace/project
git pull origin main

# 4. Updated scripts ready to use
bash scripts/instance_bootstrap.sh
```

---

## 📚 Related Documentation

- **PATH_REFERENCE.md** - Complete path reference guide
- **VAST_AI_PERSISTENT_VOLUME_GUIDE.md** - Detailed volume workflow
- **VAST_AI_QUICK_START.md** - Quick reference guide
- **TRAINING_OUTPUT_ORGANIZATION.md** - Output structure guide

---

## ✅ Checklist

**Before Mission 4.5 (HPO):**
- [ ] Volume created and initialized (setup_volume.sh completed)
- [ ] Tested instance_bootstrap.sh on new instance
- [ ] Tested sync_to_gdrive.sh (files appear in Google Drive)
- [ ] Tested quick training run (100 samples, 1 epoch)
- [ ] Ready to run run_all_hpo.sh

**During Training:**
- [ ] Using screen/tmux for long runs
- [ ] Monitoring logs periodically
- [ ] Syncing results after each major milestone

**After Training:**
- [ ] Synced all results to Google Drive
- [ ] Verified files uploaded
- [ ] Destroyed instance (kept volume)

---

**Last Updated:** 2025-12-08
**Workflow:** Persistent Volume Architecture
**Status:** Production Ready ✅
