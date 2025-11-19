# Helper Scripts for VAST.ai Training

This directory contains automation scripts for running experiments on VAST.ai GPU instances.

---

## 📁 Scripts Overview

| Script | Purpose | When to Use | Priority |
|--------|---------|-------------|----------|
| `setup_vast_instance.sh` | Automate VAST.ai instance setup | First thing after SSH into new instance | ⭐⭐⭐ Critical |
| `download_from_gdrive.sh` | Download dataset from Google Drive | After cloning repo on VAST.ai | ⭐⭐⭐ Critical |
| `sync_to_gdrive.sh` | Upload results to Google Drive | After training completes | ⭐⭐⭐ Critical |

---

## 🚀 Quick Start Workflow

### **On VAST.ai Instance (Fresh Setup)**

```bash
# 1. SSH into VAST.ai instance
ssh -p <port> root@<host>

# 2. Clone repository
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
cd hebrew-idiom-detection

# 3. Run setup script (installs everything)
bash scripts/setup_vast_instance.sh

# 4. Train your model
python src/idiom_experiment.py --mode full_finetune --model_id onlplab/alephbert-base --task cls --device cuda

# 5. Sync results to Google Drive (before terminating instance!)
bash scripts/sync_to_gdrive.sh
```

---

## 📋 Detailed Script Documentation

### 1. `setup_vast_instance.sh`

**Purpose:** One-command setup of a fresh VAST.ai GPU instance

**What it does:**
1. Updates system packages (apt-get)
2. Installs essential tools (git, wget, curl, vim)
3. Upgrades pip
4. Clones your GitHub repository
5. Installs Python dependencies from requirements.txt
6. Downloads dataset from Google Drive
7. Verifies GPU availability
8. Shows quick start commands

**Usage:**
```bash
bash scripts/setup_vast_instance.sh
```

**Time:** ~5-10 minutes (depending on internet speed)

**Configuration:**
- Edit `GITHUB_REPO_URL` in the script if your repo URL changes
- No other configuration needed

**Output:**
- ✅ All dependencies installed
- ✅ Dataset downloaded
- ✅ GPU verified
- ✅ Ready to train

---

### 2. `download_from_gdrive.sh`

**Purpose:** Download dataset splits from Google Drive

**What it downloads:**
- `data/expressions_data_tagged.csv` (main dataset, 4800 rows)
- Verifies split files exist in repo:
  - `data/splits/train.csv` (3456 samples)
  - `data/splits/validation.csv` (432 samples)
  - `data/splits/test.csv` (432 samples, in-domain)
  - `data/splits/unseen_idiom_test.csv` (480 samples, zero-shot)

**Usage:**
```bash
bash scripts/download_from_gdrive.sh
```

**Requirements:**
- `gdown` installed (auto-installed by script if missing)

**Configuration:**
- File IDs are hardcoded (from your .env file)
- No changes needed

**Output:**
- Downloads dataset to `data/` directory
- Verifies row counts (should be 4800 + header)
- Checks split files exist

**Troubleshooting:**
- If download fails: Check Google Drive sharing settings (should be "Anyone with link")
- If split files missing: Push them to GitHub repo first

---

### 3. `sync_to_gdrive.sh`

**Purpose:** Upload training results, logs, and model checkpoints to Google Drive (automated with rclone)

**What it syncs:**
- `experiments/results/` → `gdrive:Hebrew_Idiom_Detection/results/`
- `experiments/logs/` → `gdrive:Hebrew_Idiom_Detection/logs/`
- `models/` → `gdrive:Hebrew_Idiom_Detection/models/` (optional, use `--with-models` flag)

**Usage:**

Basic (sync results and logs only):
```bash
bash scripts/sync_to_gdrive.sh
```

With model checkpoints (slower, larger files):
```bash
bash scripts/sync_to_gdrive.sh --with-models
```

**First-Time Setup (Required):**

The script uses `rclone` which must be configured once per VAST.ai instance:

```bash
# 1. Install rclone
curl https://rclone.org/install.sh | sudo bash

# 2. Configure Google Drive
rclone config

# Follow prompts:
# - Choose: n (new remote)
# - Name: gdrive
# - Storage: drive (Google Drive)
# - client_id: (press Enter)
# - client_secret: (press Enter)
# - scope: 1 (Full access)
# - root_folder_id: (press Enter)
# - service_account_file: (press Enter)
# - Edit advanced config: n
# - Use auto config: n (you're on remote server)
#
# Then follow authentication steps:
# - Open URL in browser on your LOCAL machine
# - Login to Google
# - Copy verification code
# - Paste back in terminal

# 3. Verify configuration
rclone lsd gdrive:
# Should list your Google Drive folders

# 4. Create project folder (first time only)
rclone mkdir gdrive:Hebrew_Idiom_Detection
rclone mkdir gdrive:Hebrew_Idiom_Detection/results
rclone mkdir gdrive:Hebrew_Idiom_Detection/logs
rclone mkdir gdrive:Hebrew_Idiom_Detection/models

# 5. Now you can use the sync script
bash scripts/sync_to_gdrive.sh
```

**Configuration:**
- Google Drive paths are hardcoded (match your .env structure)
- No changes needed

**Output:**
- Shows sync progress with file names and sizes
- Reports total data synced
- Confirms upload to Google Drive

**When to use:**
- ✅ After each training run completes
- ✅ After each HPO study completes
- ✅ **ALWAYS before terminating VAST.ai instance** (otherwise you lose everything!)

**Sync Options:**
- `--update`: Only uploads files that are newer locally (faster)
- `--verbose`: Shows detailed file list
- `--progress`: Shows upload progress bars
- `--transfers 4`: Uses 4 parallel uploads (faster)

**Troubleshooting:**
- **Error: rclone not found** → Run installation command above
- **Error: gdrive remote not configured** → Run `rclone config` setup
- **Files not syncing** → Check `rclone lsd gdrive:` to verify connection
- **Slow uploads** → Increase `--transfers` (use 8 for faster upload)

---

## 🎯 Typical Mission 4.4 Workflow

### **First Time VAST.ai Setup:**

```bash
# 1. Rent VAST.ai instance (RTX 3090/4090, 24GB+ VRAM)
# 2. SSH into instance
ssh -p <port> root@<host>

# 3. Run setup script
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
cd hebrew-idiom-detection
bash scripts/setup_vast_instance.sh

# 4. Configure rclone (one-time, 5 minutes)
curl https://rclone.org/install.sh | sudo bash
rclone config
# Follow prompts to add Google Drive as 'gdrive'

# 5. Test sync
bash scripts/sync_to_gdrive.sh
# Should upload any existing results

# 6. Test training (small subset, 5 minutes)
python src/idiom_experiment.py --mode full_finetune --model_id onlplab/alephbert-base --task cls --num_train_samples 100 --device cuda

# 7. Sync test results
bash scripts/sync_to_gdrive.sh

# 8. Check Google Drive to verify files uploaded
```

### **Subsequent Training Sessions:**

```bash
# 1. SSH into same instance (or new instance)
ssh -p <port> root@<host>

# 2. If new instance, run setup
cd hebrew-idiom-detection
bash scripts/setup_vast_instance.sh

# 3. If rclone not configured, configure it
# (Skip if already configured on this instance)

# 4. Run training
python src/idiom_experiment.py --mode full_finetune --config experiments/configs/training_config.yaml --task cls --device cuda

# 5. Sync results
bash scripts/sync_to_gdrive.sh

# 6. Terminate instance (done!)
```

---

## 📊 Mission 4.5 (HPO) Workflow

For Mission 4.5, you need to run 10 HPO studies (5 models × 2 tasks).

**Option A: Manual (10 commands):**

```bash
# Connect to VAST.ai
ssh -p <port> root@<host>
cd hebrew-idiom-detection

# Run each HPO study manually
python src/idiom_experiment.py --mode hpo --model_id onlplab/alephbert-base --task cls --config experiments/configs/hpo_config.yaml --device cuda
python src/idiom_experiment.py --mode hpo --model_id onlplab/alephbert-base --task span --config experiments/configs/hpo_config.yaml --device cuda
# ... repeat for all 5 models × 2 tasks

# Sync all results
bash scripts/sync_to_gdrive.sh
```

**Option B: Batch Script (Recommended for Mission 4.5+):**

Create `scripts/run_all_hpo.sh` to automate all 10 runs (see PRD Section 10.1).

---

## ⚠️ Important Notes

### **ALWAYS Sync Before Terminating:**

**❌ WRONG (You will lose everything!):**
```bash
# Train model
python src/idiom_experiment.py ...
# Terminate instance immediately
# ← Results lost forever!
```

**✅ CORRECT:**
```bash
# Train model
python src/idiom_experiment.py ...

# Sync to Google Drive
bash scripts/sync_to_gdrive.sh
# Wait for upload to complete

# Verify files uploaded
rclone ls gdrive:Hebrew_Idiom_Detection/results/

# NOW it's safe to terminate
```

### **Use screen/tmux for Long Training:**

Training can take hours. Use `screen` to keep it running if SSH disconnects:

```bash
# Start screen session
screen -S training

# Run training inside screen
python src/idiom_experiment.py --mode hpo ...

# Detach: Press Ctrl+A then D
# Disconnect SSH (training keeps running)

# Reconnect later
ssh -p <port> root@<host>
screen -r training  # Resume session
```

---

## 🐛 Troubleshooting

### **Problem: Dataset download fails**

```bash
# Check gdown is installed
pip install gdown

# Try manual download
gdown 140zJatqT4LBl7yG-afFSoUrYrisi9276 -O data/expressions_data_tagged.csv

# If still fails, check Google Drive sharing permissions
```

### **Problem: rclone authentication fails**

```bash
# Remove old config
rm ~/.config/rclone/rclone.conf

# Reconfigure from scratch
rclone config

# Choose 'gdrive' as name, follow all prompts
```

### **Problem: GPU not detected**

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# If false, check VAST.ai instance has GPU allocated
```

### **Problem: Out of memory during training**

```bash
# Reduce batch size in config
# Edit experiments/configs/training_config.yaml
# Change: batch_size: 16 → batch_size: 8

# Or use gradient accumulation
# Change: gradient_accumulation_steps: 1 → 2
```

---

## 📝 Script Maintenance

If you need to modify these scripts:

1. **Edit locally in PyCharm**
2. **Commit and push to GitHub**
   ```bash
   git add scripts/
   git commit -m "Update VAST.ai scripts"
   git push
   ```
3. **Pull on VAST.ai instance**
   ```bash
   cd hebrew-idiom-detection
   git pull
   ```

---

## 📚 Additional Resources

- **VAST.ai Documentation:** https://vast.ai/docs/
- **rclone Documentation:** https://rclone.org/docs/
- **gdown Documentation:** https://github.com/wkentaro/gdown
- **Project PRD:** `../FINAL_PRD_Hebrew_Idiom_Detection.md`
- **Missions Guide:** `../STEP_BY_STEP_MISSIONS.md`

---

## ✅ Quick Reference

**Before starting Mission 4.4:**
- [x] All 3 scripts created
- [x] Scripts are executable (chmod +x)
- [x] GitHub repository up to date
- [x] VAST.ai account active with credit

**Mission 4.4 Checklist:**
- [ ] Rent VAST.ai instance
- [ ] Run `setup_vast_instance.sh`
- [ ] Configure rclone (one-time)
- [ ] Test training on small subset
- [ ] Test sync to Google Drive
- [ ] Verify GPU works
- [ ] Ready for Mission 4.5 (HPO)

**After each training:**
- [ ] Run `sync_to_gdrive.sh`
- [ ] Verify files uploaded to Google Drive
- [ ] Safe to terminate instance

---

**Last Updated:** November 9, 2025
**Mission:** 4.4 - VAST.ai Training Environment Setup
**Status:** Ready to use ✅
