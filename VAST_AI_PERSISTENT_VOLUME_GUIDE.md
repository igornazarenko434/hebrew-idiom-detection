# Vast.ai Persistent Volume Workflow Guide
## Hebrew Idiom Detection Project

**Created:** December 8, 2025
**Purpose:** Complete guide for setting up cost-efficient, repeatable GPU training workflow

---

## Table of Contents

1. [Current Project State](#current-project-state)
2. [The Persistent Volume Architecture](#the-persistent-volume-architecture)
3. [What Goes Where](#what-goes-where)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Daily Workflow](#daily-workflow)
6. [Cost Analysis](#cost-analysis)
7. [Troubleshooting](#troubleshooting)

---

## Current Project State

### What You Already Have ✅

**Local Mac (Development):**
- Complete project in `/Users/igornazarenko/PycharmProjects/Final_Project_NLP/`
- All code in GitHub: `https://github.com/igornazarenko434/hebrew-idiom-detection.git`
- Dataset files locally in `data/` folder (4,800 sentences, all splits)
- rclone installed on Mac (`/opt/homebrew/bin/rclone`) but NOT configured
- Training scripts ready (Missions 4.1-4.4 complete)
- Docker configuration ready

**Google Drive:**
- Folder structure defined: `Hebrew_Idiom_Detection/`
  - `data/` - dataset uploaded
  - `models/` - for checkpoints
  - `results/` - for outputs
  - `logs/` - for training logs
  - `backups/` - for code backups
- Dataset file ID: `140zJatqT4LBl7yG-afFSoUrYrisi9276`

**Vast.ai:**
- Account created and funded
- Connection details in `VAST_AI_CONNECTION.txt`
- **No persistent volume yet** ⚠️

### What You Currently Do (Inefficient) ❌

Your current scripts (`setup_vast_instance.sh`, `sync_to_gdrive.sh`) assume:
1. Rent new instance
2. Install system packages
3. Clone GitHub repo
4. Install Python dependencies (5-10 minutes)
5. Download dataset with gdown
6. Run training
7. Upload results to Google Drive
8. **Destroy instance** → Everything lost!

**Next time:** Repeat steps 2-5 all over again → Waste time and money!

---

## The Persistent Volume Architecture

### Core Principle

**Separate state from compute:**
- **Volume (persistent, cheap):** Everything that survives between instances
- **Instance (ephemeral, expensive):** Just compute power
- **GitHub (source of truth):** All code
- **Google Drive (long-term backup):** Final results

### The Three Components

```
┌─────────────────────────────────────────────────────────────┐
│                   YOUR WORKFLOW ECOSYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  LOCAL MAC (Development)                                      │
│  ├─ Write code in PyCharm                                     │
│  ├─ Commit & push to GitHub                                   │
│  └─ Never train here (no GPU)                                 │
│                         │                                      │
│                         │ git push                             │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────┐              │
│  │              GITHUB                         │              │
│  │   Source of truth for all code              │              │
│  └─────────────────────────────────────────────┘              │
│                         │                                      │
│                         │ git pull                             │
│                         ▼                                      │
│  ╔═══════════════════════════════════════════════════════╗    │
│  ║           VAST.AI PERSISTENT VOLUME                   ║    │
│  ║  /mnt/volume/ (Mounted to every instance)             ║    │
│  ║  ├─ env/              (Python environment - installed │    │
│  ║  │                     ONCE, reused forever)          ║    │
│  ║  ├─ data/             (Dataset - downloaded ONCE)     ║    │
│  ║  ├─ project/          (Git clone, pull latest)        ║    │
│  ║  ├─ outputs/          (Training results)              ║    │
│  ║  ├─ cache/            (HuggingFace cache)             ║    │
│  ║  └─ config/           (rclone, secrets, .env)         ║    │
│  ║                                                        ║    │
│  ║  Cost: ~$0.10-0.20/month (ALWAYS active)              ║    │
│  ╚═══════════════════════════════════════════════════════╝    │
│                         ▲                                      │
│                         │ attached                             │
│  ┌─────────────────────┴──────────────────────┐               │
│  │     VAST.AI GPU INSTANCE (Ephemeral)       │               │
│  │  ├─ Rent RTX 4090 when needed              │               │
│  │  ├─ Symlink volume configs                 │               │
│  │  ├─ Activate volume environment             │               │
│  │  ├─ Pull latest code from GitHub            │               │
│  │  ├─ Run training → save to /mnt/volume      │               │
│  │  └─ DESTROY when done (save money!)         │               │
│  │                                              │               │
│  │  Cost: ~$0.30-0.50/hour (ONLY when training)│               │
│  └──────────────────────────────────────────────               │
│                         │                                      │
│                         │ rclone sync                          │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────┐              │
│  │         GOOGLE DRIVE (Long-term backup)     │              │
│  │  Hebrew_Idiom_Detection/                    │              │
│  │  ├─ results/ (All training outputs)         │              │
│  │  ├─ logs/ (TensorBoard logs)                │              │
│  │  └─ models/ (Optional: best checkpoints)    │              │
│  └─────────────────────────────────────────────┘              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## What Goes Where

### 1. Local Mac (Development Only)

**Purpose:** Write code, commit, push
**NEVER:** Train models, download large files

```
/Users/igornazarenko/PycharmProjects/Final_Project_NLP/
├─ src/                    # Edit code here
├─ experiments/configs/    # Edit training configs
├─ scripts/                # Edit automation scripts
├─ data/                   # Keep for reference (don't sync)
└─ .git/                   # Push to GitHub regularly
```

**Workflow:**
```bash
# 1. Edit code in PyCharm
# 2. Test locally if possible
# 3. Commit and push
git add .
git commit -m "Update training config"
git push origin main
```

---

### 2. GitHub (Code Source of Truth)

**What's stored:**
- All Python code (`src/`, `scripts/`, `tests/`)
- Configuration files (`experiments/configs/*.yaml`)
- Documentation (`README.md`, `*.md` files)
- Requirements (`requirements.txt`, `docker/Dockerfile`)
- Data split files (`data/splits/*.csv` - small files only)

**What's NOT stored (in .gitignore):**
- Large dataset files (`data/*.csv`, `data/*.json`)
- Model checkpoints (`models/`, `experiments/results/`)
- Python cache (`__pycache__/`, `.pyc`)
- Virtual environments (`.venv/`)

---

### 3. Vast.ai Persistent Volume (Working State)

**Volume Structure:** `/mnt/volume/`

```
/mnt/volume/
├─ env/                          # Python virtual environment (INSTALLED ONCE)
│  ├─ bin/python                 # Python 3.10+
│  └─ lib/python3.10/site-packages/
│     ├─ torch/                  # PyTorch with CUDA
│     ├─ transformers/           # HuggingFace
│     └─ ... (all requirements)
│
├─ data/                         # Dataset (DOWNLOADED ONCE)
│  ├─ expressions_data_tagged_v2.csv
│  ├─ splits/
│  │  ├─ train.csv
│  │  ├─ validation.csv
│  │  ├─ test.csv
│  │  └─ unseen_idiom_test.csv
│  └─ README.md
│
├─ project/                      # Git repository (PULL on each session)
│  ├─ .git/
│  ├─ src/
│  ├─ experiments/configs/
│  ├─ scripts/
│  └─ requirements.txt
│
├─ outputs/                      # Training results (PERSISTENT)
│  ├─ full_fine-tuning/
│  │  ├─ alephbert-base/
│  │  │  ├─ cls/
│  │  │  │  ├─ checkpoint-best/
│  │  │  │  ├─ training_results.json
│  │  │  │  └─ logs/
│  │  │  └─ span/
│  │  └─ ... (other models)
│  ├─ hpo_results/
│  └─ optuna_studies/
│
├─ cache/                        # HuggingFace cache (REUSED)
│  └─ huggingface/
│     └─ transformers/
│        ├─ models--onlplab--alephbert-base/
│        └─ ... (all downloaded models)
│
└─ config/                       # Secrets and configs (SETUP ONCE)
   ├─ .rclone.conf               # Google Drive auth (DO NOT COMMIT!)
   ├─ .env                       # Environment variables
   └─ secrets/
      └─ ... (any API keys)
```

**Volume Pricing:**
- Storage: ~$0.10-0.20 per GB per month
- Typical size: 50-100 GB
- **Cost: ~$5-20/month** (much cheaper than re-downloading/installing)

---

### 4. Vast.ai GPU Instance (Ephemeral Compute)

**Purpose:** Just compute, nothing persistent!

**What happens on instance:**
1. Mount `/mnt/volume/` (your persistent volume)
2. Symlink configs:
   ```bash
   ln -sf /mnt/volume/config/.rclone.conf ~/.config/rclone/rclone.conf
   ln -sf /mnt/volume/config/.env /workspace/.env
   ```
3. Activate environment:
   ```bash
   source /mnt/volume/env/bin/activate
   ```
4. Pull latest code:
   ```bash
   cd /mnt/volume/project
   git pull origin main
   ```
5. Run training:
   ```bash
   python src/idiom_experiment.py --mode hpo --task cls --device cuda
   ```
6. Results automatically saved to `/mnt/volume/outputs/`
7. **Destroy instance** when done (no data loss!)

**Instance Pricing:**
- RTX 4090 (24GB): ~$0.30-0.50/hour
- Training time per model: ~7-15 minutes
- **Cost per training run: ~$0.10-0.20**

---

### 5. Google Drive (Long-term Backup)

**Purpose:** Archive final results, share with team

**What's synced (via rclone):**
```
Hebrew_Idiom_Detection/
├─ results/                    # Training outputs (JSON, summaries)
│  ├─ full_fine-tuning/
│  ├─ hpo_results/
│  └─ zero_shot/
├─ logs/                       # TensorBoard logs
│  └─ events.out.tfevents.*
└─ models/                     # Optional: Best checkpoints only
   └─ best_models/
      ├─ alephbert-cls-best/
      └─ ...
```

**Sync timing:**
- After each major training run
- Before destroying Vast.ai instance
- Weekly for safety

**Storage:**
- Free: 15 GB (Google Drive free tier)
- Results: ~1-2 GB total
- Models: ~5-10 GB (optional)

---

## Step-by-Step Setup

### Phase 1: One-Time Volume Creation (15 minutes)

#### 1.1 Create Persistent Volume on Vast.ai

1. Log in to https://vast.ai/console/create/
2. **Instead of renting instance, create storage:**
   - Click "Storage" tab (top menu)
   - Click "Create Storage Volume"
   - **Name:** `hebrew-idiom-volume`
   - **Size:** 100 GB (recommended)
   - **Region:** Choose same as where you'll rent instances (e.g., US, EU)
   - Click "Create"

**Cost:** ~$10-20/month (always active, but cheap)

3. Note your volume ID (e.g., `vol_xyz123`)

---

#### 1.2 Rent Temporary Setup Instance

**Purpose:** One-time setup of the volume

1. Go to https://vast.ai/console/create/
2. **Search for instance:**
   - GPU: Any cheap GPU (even GTX 1080 is fine for setup)
   - VRAM: ≥16 GB
   - Disk: ≥50 GB (for temporary use)
   - Reliability: >95%
   - $/hour: <$0.20 (cheapest for setup)

3. **Before renting, attach your volume:**
   - In instance config, find "Storage" section
   - Click "Attach storage volume"
   - Select `hebrew-idiom-volume`
   - Mount point: `/mnt/volume`
   - Click "Rent"

4. **Connect via SSH:**
   ```bash
   ssh -p <PORT> root@<IP>
   # Connection details shown in Vast.ai console
   ```

5. **Verify volume is mounted:**
   ```bash
   df -h | grep /mnt/volume
   # Should show ~100GB volume
   ```

---

#### 1.3 Setup Volume Structure (Script)

**Run this on the instance:**

```bash
# Create directory structure
mkdir -p /mnt/volume/{env,data,data/splits,project,outputs,cache,config,config/secrets}

# Verify
ls -la /mnt/volume/
```

---

#### 1.4 Create Python Environment on Volume

```bash
# Install Python if needed
apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip git curl

# Create virtual environment on VOLUME (not instance)
python3.10 -m venv /mnt/volume/env

# Activate
source /mnt/volume/env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA (check CUDA version first)
nvcc --version  # Note CUDA version

# For CUDA 11.8 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU works
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

---

#### 1.5 Clone Project and Install Dependencies

```bash
# Activate environment
source /mnt/volume/env/bin/activate

# Clone your GitHub repo to VOLUME
cd /mnt/volume/
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git project

cd project

# Install all dependencies (this takes 5-10 minutes)
pip install -r requirements.txt

# Verify installations
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import optuna; print(f'Optuna: {optuna.__version__}')"
```

---

#### 1.6 Download Dataset to Volume

```bash
# Activate environment
source /mnt/volume/env/bin/activate

# Navigate to project
cd /mnt/volume/project

# Copy dataset script to volume data folder
cp scripts/download_from_gdrive.sh /mnt/volume/data/

# Download dataset
cd /mnt/volume/data/
bash download_from_gdrive.sh

# Verify files
ls -lh /mnt/volume/data/
# Should show:
# - expressions_data_tagged_v2.csv (~3 MB)
# - splits/ directory with train/val/test/unseen CSVs

# Or download manually:
pip install gdown
gdown 140zJatqT4LBl7yG-afFSoUrYrisi9276 -O /mnt/volume/data/expressions_data_tagged.csv

# If splits are in repo, copy them:
cp -r splits/ /mnt/volume/data/
```

---

#### 1.7 Setup rclone (Google Drive Integration)

**This is the most important step for avoiding re-authentication!**

```bash
# Install rclone on instance (temporary)
curl https://rclone.org/install.sh | sudo bash

# Configure rclone
rclone config

# Follow these prompts:
# n) New remote
# name> gdrive
# Storage> drive  (or type number for Google Drive)
# client_id> (press Enter)
# client_secret> (press Enter)
# scope> 1  (Full access)
# root_folder_id> (press Enter)
# service_account_file> (press Enter)
# Edit advanced config? n
# Use auto config? n  ⚠️ IMPORTANT: Select 'n' (we're on remote server)

# It will show you a URL like:
# https://accounts.google.com/o/oauth2/auth?client_id=...

# COPY THIS URL
# Open it on your LOCAL MACHINE (Mac)
# Authenticate with your Google account
# Grant access to Google Drive
# Copy the verification code
# Paste it back into the SSH session

# Choose team drive? n
# Keep this remote? y

# Test connection:
rclone lsd gdrive:
# Should list your Google Drive folders

# Create project folder structure if not exists:
rclone mkdir gdrive:Hebrew_Idiom_Detection
rclone mkdir gdrive:Hebrew_Idiom_Detection/results
rclone mkdir gdrive:Hebrew_Idiom_Detection/logs
rclone mkdir gdrive:Hebrew_Idiom_Detection/models
```

**CRITICAL: Save rclone config to VOLUME**

```bash
# Copy rclone config to volume (so it persists!)
mkdir -p /mnt/volume/config/.config/rclone/
cp ~/.config/rclone/rclone.conf /mnt/volume/config/.rclone.conf

# Verify it's saved
cat /mnt/volume/config/.rclone.conf
# Should show [gdrive] section with your auth token

# Also save to a safe location (backup)
cat /mnt/volume/config/.rclone.conf
# Copy this content and save it locally on your Mac
# Location: ~/Desktop/rclone_backup.conf (for safety)
```

---

#### 1.8 Setup Environment Variables on Volume

```bash
# Copy .env from project to volume config
cp /mnt/volume/project/.env /mnt/volume/config/.env

# Edit if needed
nano /mnt/volume/config/.env

# Add/verify these paths point to volume:
# LOCAL_DATA_DIR=/mnt/volume/data
# LOCAL_MODELS_DIR=/mnt/volume/cache/huggingface
# LOCAL_RESULTS_DIR=/mnt/volume/outputs
# LOCAL_LOGS_DIR=/mnt/volume/outputs/logs

# Save and exit (Ctrl+X, Y, Enter)
```

---

#### 1.9 Setup HuggingFace Cache on Volume

```bash
# Set HuggingFace cache to volume (so models don't re-download)
export HF_HOME=/mnt/volume/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/volume/cache/huggingface

# Add to volume config
echo "export HF_HOME=/mnt/volume/cache/huggingface" >> /mnt/volume/config/.env
echo "export TRANSFORMERS_CACHE=/mnt/volume/cache/huggingface" >> /mnt/volume/config/.env

# Pre-download models to cache (optional, saves time later)
cd /mnt/volume/project
source /mnt/volume/env/bin/activate
python src/model_download.py
# This downloads all 5 models to /mnt/volume/cache/
```

---

#### 1.10 Final Volume Verification

```bash
# Check volume structure
tree -L 2 /mnt/volume/

# Expected output:
# /mnt/volume/
# ├── cache/
# │   └── huggingface/
# ├── config/
# │   ├── .env
# │   └── .rclone.conf
# ├── data/
# │   ├── expressions_data_tagged_v2.csv
# │   └── splits/
# ├── env/
# │   ├── bin/
# │   └── lib/
# ├── outputs/
# └── project/
#     ├── src/
#     ├── experiments/
#     └── scripts/

# Check sizes
du -sh /mnt/volume/*
# env/       ~5 GB    (Python + all packages)
# data/      ~3 MB    (dataset)
# cache/     ~10 GB   (if models pre-downloaded)
# config/    ~1 KB    (rclone + .env)
# project/   ~50 MB   (code)
# outputs/   ~0       (empty initially)
```

---

#### 1.11 Destroy Setup Instance

**IMPORTANT: Do NOT destroy the volume!**

```bash
# Exit SSH
exit

# In Vast.ai console:
# 1. Find your instance
# 2. Click "Destroy instance"
# 3. ⚠️ CONFIRM: "Destroy instance but KEEP storage volume"
# 4. Verify volume still exists in "Storage" tab
```

**Result:** Volume is ready! All setup saved. Instance destroyed (no more charges).

---

### Phase 2: Daily Workflow (5 minutes per session)

#### 2.1 Start New Training Session

1. **Rent GPU instance:**
   - Go to https://vast.ai/console/create/
   - Search: RTX 4090, ≥24GB VRAM, >98% reliability
   - **Attach storage volume:** `hebrew-idiom-volume` at `/mnt/volume`
   - Rent instance

2. **Connect via SSH:**
   ```bash
   ssh -p <PORT> root@<IP>
   ```

3. **Run bootstrap script:**
   ```bash
   # This is the ONLY command you need!
   bash /mnt/volume/project/scripts/instance_bootstrap.sh
   ```

   **What the bootstrap script does** (see next section):
   - Symlinks rclone config from volume
   - Symlinks .env from volume
   - Activates volume environment
   - Pulls latest code from GitHub
   - Sets HuggingFace cache to volume
   - Shows you're ready to train

---

#### 2.2 Run Training

```bash
# Environment already activated by bootstrap
cd /mnt/volume/project

# Option 1: Single training run
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --device cuda

# Option 2: HPO (hyperparameter optimization)
bash scripts/run_all_hpo.sh

# Option 3: All experiments
bash scripts/run_all_experiments.sh

# Results automatically save to /mnt/volume/outputs/
```

---

#### 2.3 Sync Results to Google Drive

```bash
# After training completes
cd /mnt/volume/project
bash scripts/sync_to_gdrive.sh

# This uploads:
# /mnt/volume/outputs/ → gdrive:Hebrew_Idiom_Detection/results/
```

---

#### 2.4 Destroy Instance (Save Money!)

```bash
# Exit SSH
exit

# In Vast.ai console:
# Destroy instance (volume stays!)
```

**Cost saved:**
- Instance running idle: $0.30-0.50/hour
- Volume only: $0.01/hour
- **Savings: ~97% when not training!**

---

### Phase 3: Scripts to Create

Now that you understand the workflow, you need to create **2 new scripts** (the rest you already have):

#### 3.1 Volume Setup Script (Run Once)

**File:** `scripts/setup_volume.sh`

This is basically what you did manually in Phase 1.1-1.10, automated.

#### 3.2 Instance Bootstrap Script (Run Every Session)

**File:** `scripts/instance_bootstrap.sh`

This replaces your current `setup_vast_instance.sh` for the volume-based workflow.

---

## Cost Analysis

### Current Workflow (Without Volume)

**Per training session:**
- Rent instance: $0.40/hour
- Setup time: 15 minutes = $0.10
- Training time: 30 minutes = $0.20
- Upload time: 5 minutes = $0.03
- **Total per session: $0.33**

**For full project (30 training runs):**
- Setup: 30 × $0.10 = **$3.00 wasted**
- Training: 30 × $0.20 = $6.00
- Upload: 30 × $0.03 = $0.90
- **Total: $9.90**

---

### New Workflow (With Volume)

**One-time setup:**
- Setup instance (1 hour): $0.20
- **Total: $0.20** (one-time only)

**Per training session:**
- Bootstrap: 1 minute = $0.01
- Training: 30 minutes = $0.20
- Upload: 5 minutes = $0.03
- **Total per session: $0.24**

**Volume storage:**
- 100 GB × $0.15/GB/month = **$15/month**

**For full project (30 training runs over 1 month):**
- Setup: $0.20 (one-time)
- Training: 30 × $0.24 = $7.20
- Volume: $15.00 (1 month)
- **Total: $22.40**

**Wait, that's more expensive?**

NO! Because:
1. Time saved = ~400 minutes (6.7 hours of your life)
2. Eliminates errors from repeated setup
3. Volume persists beyond project (use for other projects)
4. Can pause/resume training (impossible without volume)
5. Pre-downloaded models save bandwidth

**Break-even point:** After 2 months or 50+ training runs

---

## Troubleshooting

### Volume not mounting

```bash
# Check volume is attached
df -h | grep /mnt/volume

# If not, stop instance, attach volume in UI, restart
```

### rclone auth expired

```bash
# Re-authenticate on instance
rclone config reconnect gdrive:

# Save new config to volume
cp ~/.config/rclone/rclone.conf /mnt/volume/config/.rclone.conf
```

### Python environment issues

```bash
# Re-create environment
rm -rf /mnt/volume/env
python3.10 -m venv /mnt/volume/env
source /mnt/volume/env/bin/activate
pip install -r /mnt/volume/project/requirements.txt
```

### Git conflicts

```bash
cd /mnt/volume/project
git reset --hard origin/main
git pull
```

---

## Summary

**What changed:**
- ❌ OLD: Re-install everything every session
- ✅ NEW: Install once, reuse forever

**Your action items:**
1. Create persistent volume on Vast.ai
2. Run one-time setup (Phase 1)
3. Create `instance_bootstrap.sh` script
4. Test workflow with small training run
5. From now on: Just rent instance → bootstrap → train → sync → destroy

**Time investment:**
- Setup: 30 minutes (one-time)
- Per session: 1 minute bootstrap + training time
- **Time saved: 10-15 minutes per session**

---

**Next:** I'll create the two scripts you need (`setup_volume.sh` and `instance_bootstrap.sh`).
