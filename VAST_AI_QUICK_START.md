# Vast.ai Quick Start Guide
## Hebrew Idiom Detection - Persistent Volume Workflow

**For the impatient:** Complete setup in 30 minutes, then train in 1-minute sessions forever.

---

## One-Time Setup (30 minutes)

### 1. Setup rclone on Your Local Mac (5 minutes)

**You mentioned you already have rclone but haven't configured it. Let's configure it once on your Mac:**

```bash
# Verify rclone is installed
which rclone
# Should show: /opt/homebrew/bin/rclone

# Configure Google Drive
rclone config

# Follow prompts:
# n (new remote)
# name: gdrive
# storage: drive
# client_id: (Enter)
# client_secret: (Enter)
# scope: 1
# auto config: y  ← YES on your Mac (it will open browser)
# Authenticate in browser
# Confirm: y

# Test it works
rclone lsd gdrive:
# Should show your Google Drive folders

# Create project folder
rclone mkdir gdrive:Hebrew_Idiom_Detection
rclone mkdir gdrive:Hebrew_Idiom_Detection/results
rclone mkdir gdrive:Hebrew_Idiom_Detection/logs
rclone mkdir gdrive:Hebrew_Idiom_Detection/models
```

**Save your rclone config (important!):**

```bash
# Backup your config
cp ~/.config/rclone/rclone.conf ~/Desktop/rclone_backup.conf

# You'll upload this to the Vast.ai volume later
```

---

### 2. Create Persistent Volume on Vast.ai (2 minutes)

1. Go to https://vast.ai/console/storage/
2. Click **"Create Storage"**
3. Settings:
   - Name: `hebrew-idiom-volume`
   - Size: `100 GB`
   - Region: `Any` (or same as where you'll rent GPUs)
4. Click **"Create"**
5. **Note the volume ID** (shown in list)

**Cost:** ~$10-15/month (always active, but cheap)

---

### 3. Rent Temporary Setup Instance (20-30 minutes)

**Find cheapest instance for setup:**

1. Go to https://vast.ai/console/create/
2. Search filters:
   - GPU: Any (even GTX 1080 Ti)
   - VRAM: ≥16 GB
   - Disk: ≥50 GB
   - Reliability: >95%
   - Sort by: **Price (lowest first)**
3. **Before renting:**
   - Click **"Storage"** button
   - Select **"Attach storage volume"**
   - Choose: `hebrew-idiom-volume`
   - Mount point: `/mnt/volume`
4. Click **"Rent"** (should be <$0.20/hour)

**Connect via SSH:**

```bash
# Use connection command from Vast.ai console
ssh -p <PORT> root@<IP>

# Example (replace with your details):
# ssh -p 12345 root@123.456.789.0
```

**Run setup script:**

```bash
# Download your rclone config from Mac first
# On your Mac, run:
# scp -P <PORT> ~/Desktop/rclone_backup.conf root@<IP>:/tmp/rclone.conf

# On the Vast.ai instance:
cd /root

# Clone your repo temporarily to get the setup script
git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git temp_repo
cp temp_repo/scripts/setup_volume.sh /root/
rm -rf temp_repo

# Run the setup
bash setup_volume.sh

# This will:
# 1. Create directory structure on volume
# 2. Install Python 3.10 + venv
# 3. Install PyTorch with CUDA
# 4. Clone your GitHub repo to /mnt/volume/project
# 5. Install all dependencies (5-10 min)
# 6. Download dataset to /mnt/volume/data
# 7. Setup rclone (you'll manually authenticate)
# 8. Configure environment variables
# 9. Optionally pre-download models (~10 GB)

# When prompted for rclone:
# - Follow the authentication flow
# - Open URL on your Mac
# - Copy verification code back

# Or manually copy your rclone config:
mkdir -p /mnt/volume/config/
cp /tmp/rclone.conf /mnt/volume/config/.rclone.conf
```

**Destroy setup instance (KEEP VOLUME!):**

```bash
# Exit SSH
exit

# In Vast.ai console:
# 1. Find your instance
# 2. Click "Destroy"
# 3. ⚠️ CONFIRM: Destroy instance but KEEP storage volume
# 4. Verify volume still shows in "Storage" tab
```

**Result:** Your volume is now ready with everything installed!

---

## Daily Workflow (1-2 minutes to start)

### Every Training Session

**1. Rent GPU Instance (1 min)**

- Go to https://vast.ai/console/create/
- Search: RTX 4090, ≥24GB VRAM, >98% reliability
- **Attach storage:** `hebrew-idiom-volume` at `/mnt/volume`
- Rent (~$0.30-0.50/hour)

**2. Connect & Bootstrap (1 min)**

```bash
# SSH to instance
ssh -p <PORT> root@<IP>

# Run bootstrap (automatically sets up everything)
bash /mnt/volume/project/scripts/instance_bootstrap.sh

# This takes ~30 seconds and:
# - Symlinks rclone config from volume
# - Activates Python environment from volume
# - Pulls latest code from GitHub
# - Verifies dataset
# - Shows you're ready to train
```

**3. Start Training**

**Option A: Quick test (5 min)**

```bash
cd /mnt/volume/project
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --num_train_samples 100 \
  --num_epochs 1 \
  --device cuda
```

**Option B: Single model full training (15-30 min)**

```bash
cd /mnt/volume/project
python src/idiom_experiment.py \
  --mode full_finetune \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/training_config.yaml \
  --device cuda
```

**Option C: HPO for one model (4-8 hours)**

```bash
cd /mnt/volume/project
python src/idiom_experiment.py \
  --mode hpo \
  --model_id onlplab/alephbert-base \
  --task cls \
  --config experiments/configs/hpo_config.yaml \
  --device cuda
```

**Option D: Run all HPO studies (50-75 hours)**

```bash
# Use screen/tmux for long runs
screen -S training

cd /mnt/volume/project
bash scripts/run_all_hpo.sh

# Detach: Ctrl+A then D
# Reattach: screen -r training
```

**4. Sync Results to Google Drive**

```bash
cd /mnt/volume/project
bash scripts/sync_to_gdrive.sh

# This uploads:
# - /mnt/volume/outputs/ → gdrive:Hebrew_Idiom_Detection/results/
```

**5. Destroy Instance**

```bash
# Exit SSH
exit

# In Vast.ai console: Destroy instance
# Volume automatically detaches and stays safe
```

---

## Cost Breakdown

### Setup (One-Time)

| Item | Cost |
|------|------|
| Setup instance (1 hour @ $0.20/hr) | $0.20 |
| **Total one-time** | **$0.20** |

### Ongoing

| Item | Cost |
|------|------|
| Volume storage (100GB @ $0.15/GB/mo) | $15/month |
| Instance idle (volume only, no instance) | $0/hour |
| Training (RTX 4090 @ $0.40/hr) | $0.40/hour |

### Example Project Costs

**Phase 4 & 5 (Your current missions):**

| Task | Time | Instance Cost | Volume Cost | Total |
|------|------|---------------|-------------|-------|
| HPO (10 studies × 5-7 hrs) | 60 hrs | 60 × $0.40 = $24 | - | $24 |
| Final training (30 runs × 15 min) | 7.5 hrs | 7.5 × $0.40 = $3 | - | $3 |
| Volume (1 month) | - | - | $15 | $15 |
| **Total for project** | - | - | - | **~$42** |

**Compare to repeating setup each time:**
- 30 sessions × 15 min setup × $0.40/hr = **$3 wasted**
- Plus: Hours of your time, potential errors

**Break-even:** After ~5 sessions or 1 month

---

## What's Stored Where

### Local Mac

```
/Users/igornazarenko/PycharmProjects/Final_Project_NLP/
├── src/                    ← Edit code here
├── experiments/configs/    ← Edit configs here
├── .git/                   ← Push to GitHub
└── (don't train here)
```

### GitHub

- All code (source of truth)
- Configuration files
- Documentation
- Small data files (splits)

### Vast.ai Volume (`/mnt/volume/`)

```
/mnt/volume/
├── env/                    ← Python + all packages (5 GB)
├── data/                   ← Dataset + splits (3 MB)
├── project/                ← Git clone (auto-updated)
├── outputs/                ← Training results (grows)
├── cache/                  ← Downloaded models (10 GB)
└── config/
    ├── .rclone.conf        ← Google Drive auth
    └── .env                ← Environment variables
```

### Google Drive

```
Hebrew_Idiom_Detection/
├── results/                ← Synced from /mnt/volume/outputs/
├── logs/                   ← TensorBoard logs
└── models/                 ← Optional: best checkpoints
```

---

## Troubleshooting

### "Volume not found"

**Problem:** Forgot to attach volume when renting instance

**Fix:**
```bash
# Can't fix on running instance
# Must destroy and rent new instance
# Make sure to click "Storage" and attach volume BEFORE renting
```

### "rclone not configured"

**Problem:** rclone config missing from volume

**Fix:**
```bash
# On instance:
rclone config
# Or copy from your Mac:
# scp -P <PORT> ~/Desktop/rclone_backup.conf root@<IP>:/mnt/volume/config/.rclone.conf
```

### "Python packages missing"

**Problem:** Dependencies not installed on volume

**Fix:**
```bash
source /mnt/volume/env/bin/activate
pip install -r /mnt/volume/project/requirements.txt
```

### "Dataset not found"

**Problem:** Dataset not downloaded to volume

**Fix:**
```bash
cd /mnt/volume/data
pip install gdown
gdown 140zJatqT4LBl7yG-afFSoUrYrisi9276 -O expressions_data_tagged_v2.csv

# Copy splits from repo
cp -r /mnt/volume/project/data/splits/ /mnt/volume/data/
```

### "Git pull fails"

**Problem:** Local changes conflict with remote

**Fix:**
```bash
cd /mnt/volume/project
git reset --hard origin/main
git pull
```

---

## Pro Tips

### Use screen/tmux for Long Training

```bash
# Start screen session
screen -S training

# Run your training
python src/idiom_experiment.py ...

# Detach: Ctrl+A then D
# SSH disconnects don't affect training!

# Reattach later
screen -r training
```

### Monitor Training Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor logs (if saved to file)
tail -f /mnt/volume/outputs/training.log

# TensorBoard (on local Mac after sync)
tensorboard --logdir experiments/results/
```

### Save Money

1. **Destroy instances immediately after training**
   - Don't leave running idle
   - Volume stays safe

2. **Use spot instances**
   - Vast.ai has cheaper "interruptible" instances
   - Save ~30-50%
   - Risk: Instance can be terminated

3. **Pre-download models**
   - During setup, download all models
   - Saves time and bandwidth later

4. **Batch your training**
   - Rent instance once, run multiple experiments
   - Use `screen` to run overnight

---

## Next Steps

1. ✅ **Configure rclone on your Mac** (5 minutes)
2. ✅ **Create persistent volume** on Vast.ai (2 minutes)
3. ✅ **Rent setup instance** and run `setup_volume.sh` (30 minutes)
4. ✅ **Destroy setup instance**, verify volume intact
5. 🚀 **Start training:** Rent GPU instance → bootstrap → train → sync → destroy

**First training run:** Try a quick test (100 samples) to verify everything works!

---

## Support

If you encounter issues:

1. Check troubleshooting section above
2. Review `VAST_AI_PERSISTENT_VOLUME_GUIDE.md` for detailed explanations
3. Verify volume is attached in Vast.ai console
4. Check volume contents: `ls -la /mnt/volume/`

**Emergency backup recovery:**

```bash
# If you lose volume, you still have:
# 1. Code in GitHub
# 2. rclone config on your Mac (~/Desktop/rclone_backup.conf)
# 3. Dataset on Google Drive
# 4. Results on Google Drive

# Just re-run setup_volume.sh on a new volume
```

---

**Remember:** The volume is your persistent workspace. Instances are just temporary compute. Never store anything important on the instance itself!

**Happy Training! 🚀**
