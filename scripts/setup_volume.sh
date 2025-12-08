#!/bin/bash
# ============================================================================
# Vast.ai Persistent Volume Setup Script (RUN ONCE)
# ============================================================================
# Purpose: Initialize persistent volume with complete environment
# Usage: bash scripts/setup_volume.sh
# Time: ~20-30 minutes (one-time only!)
# Run this on: Temporary Vast.ai instance with volume attached at /workspace
# ============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
VOLUME_PATH="/workspace"
GITHUB_REPO_URL="https://github.com/igornazarenko434/hebrew-idiom-detection.git"
PYTHON_VERSION="3.10"
# Google Drive File ID for dataset
DATASET_FILE_ID="140zJatqT4LBl7yG-afFSoUrYrisi9276"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║   Vast.ai Persistent Volume Setup              ║"
echo "║   Hebrew Idiom Detection Project               ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: This script sets up your volume ONE TIME${NC}"
echo -e "${YELLOW}⚠️  After completion, this volume will persist across all instances${NC}"
echo ""

# ============================================================================
# STEP 0: Pre-flight Checks
# ============================================================================

echo -e "${CYAN}Step 0/11: Pre-flight checks${NC}"
echo "=============================="
echo ""

# Check volume is mounted (basic check)
if [ ! -d "$VOLUME_PATH" ]; then
    echo -e "${RED}✗ ERROR: Volume path $VOLUME_PATH not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Volume path exists at $VOLUME_PATH${NC}"

# Check writable
if [ ! -w "$VOLUME_PATH" ]; then
    echo -e "${RED}✗ ERROR: Volume not writable${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Volume is writable${NC}"

# Show disk space
df -h "$VOLUME_PATH" | tail -1
echo ""

read -p "$(echo -e ${YELLOW}Continue with setup? This will create directories and install packages. [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

echo ""

# ============================================================================
# STEP 1: Create Directory Structure
# ============================================================================

echo -e "${CYAN}Step 1/11: Creating directory structure on volume${NC}"
echo "=================================================="
echo ""

mkdir -p "$VOLUME_PATH"/{env,data,data/splits,project,cache,config,config/secrets}

# Note: 'outputs' directory not needed as results go into project/experiments/results

# Create cache subdirectories
mkdir -p "$VOLUME_PATH"/cache/huggingface/{transformers,datasets}

echo "Created directories:"
ls -la "$VOLUME_PATH"

echo -e "${GREEN}✓ Directory structure created${NC}"
echo ""

# ============================================================================
# STEP 2: Install System Dependencies
# ============================================================================

echo -e "${CYAN}Step 2/11: Installing system dependencies${NC}"
echo "==========================================="
echo ""

# Update package lists
apt-get update -qq

# Install essential tools
apt-get install -y -qq \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    tree \
    build-essential

echo -e "${GREEN}✓ System dependencies installed${NC}"
echo "  Python: $(python3 --version)"
echo "  Git: $(git --version | head -1)"
echo ""

# ============================================================================
# STEP 3: Create Python Virtual Environment on Volume
# ============================================================================

echo -e "${CYAN}Step 3/11: Creating Python virtual environment${NC}"
echo "==============================================="
echo -e "${YELLOW}This may take 2-3 minutes...${NC}"
echo ""

# Create venv on VOLUME (not on instance)
python${PYTHON_VERSION} -m venv "$VOLUME_PATH/env"

# Activate environment
source "$VOLUME_PATH/env/bin/activate"

# Upgrade pip
pip install --upgrade pip -q

echo -e "${GREEN}✓ Virtual environment created at $VOLUME_PATH/env${NC}"
echo "  Python: $(python --version)"
echo "  Pip: $(pip --version | cut -d' ' -f2)"
echo ""

# ============================================================================
# STEP 4: Install PyTorch with CUDA Support
# ============================================================================

echo -e "${CYAN}Step 4/11: Installing PyTorch with CUDA${NC}"
echo "========================================"
echo ""

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo -e "${YELLOW}⚠️  nvcc not found, assuming CUDA 11.8${NC}"
    CUDA_VERSION="11.8"
fi

# Install PyTorch based on CUDA version
if [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "Installing PyTorch for CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8"* ]] || [[ "$CUDA_VERSION" == "11."* ]]; then
    echo "Installing PyTorch for CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch with default CUDA support..."
    pip install torch torchvision torchaudio
fi

# Verify GPU is available
echo ""
echo "Verifying GPU availability:"
python -c "
import torch
print(f'  ✓ PyTorch version: {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ CUDA version: {torch.version.cuda}')
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('  ⚠️  WARNING: CUDA not available!')
"

echo -e "${GREEN}✓ PyTorch installed with CUDA support${NC}"
echo ""

# ============================================================================
# STEP 5: Clone GitHub Repository
# ============================================================================

echo -e "${CYAN}Step 5/11: Cloning GitHub repository${NC}"
echo "====================================="
echo ""

cd "$VOLUME_PATH"

if [ -d "$VOLUME_PATH/project/.git" ]; then
    echo -e "${YELLOW}Project already exists, pulling latest changes...${NC}"
    cd "$VOLUME_PATH/project"
    git pull
    cd "$VOLUME_PATH"
else
    echo "Cloning from: $GITHUB_REPO_URL"
    git clone "$GITHUB_REPO_URL" "$VOLUME_PATH/project"
fi

echo -e "${GREEN}✓ Repository cloned to $VOLUME_PATH/project${NC}"
echo ""

# ============================================================================
# STEP 6: Install Python Dependencies
# ============================================================================

echo -e "${CYAN}Step 6/11: Installing Python dependencies${NC}"
echo "==========================================="
echo -e "${YELLOW}This may take 5-10 minutes...${NC}"
echo ""

cd "$VOLUME_PATH/project"

# Install all requirements
pip install -r requirements.txt

# Verify key installations
echo ""
echo "Verifying installations:"
python -c "import transformers; print(f'  ✓ Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'  ✓ Datasets: {datasets.__version__}')"
python -c "import optuna; print(f'  ✓ Optuna: {optuna.__version__}')"
python -c "import pandas; print(f'  ✓ Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'  ✓ Scikit-learn: {sklearn.__version__}')"

echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# ============================================================================
# STEP 7: Download Dataset to Volume
# ============================================================================

echo -e "${CYAN}Step 7/11: Downloading dataset${NC}"
echo "==============================="
echo ""

cd "$VOLUME_PATH/data"

# Check if dataset already exists
if [ -f "expressions_data_tagged_v2.csv" ] || [ -f "expressions_data_tagged.csv" ]; then
    echo -e "${YELLOW}Dataset already exists, skipping download${NC}"
else
    echo "Downloading dataset from Google Drive..."
    pip install gdown -q
    gdown "$DATASET_FILE_ID" -O expressions_data_tagged_v2.csv
    echo -e "${GREEN}✓ Dataset downloaded${NC}"
fi

# Copy split files from repo if they exist
if [ -d "$VOLUME_PATH/project/data/splits" ]; then
    echo "Copying split files from repository..."
    cp -r "$VOLUME_PATH/project/data/splits/"* "$VOLUME_PATH/data/splits/"
    echo -e "${GREEN}✓ Split files copied${NC}"
fi

# Verify dataset
echo ""
echo "Dataset files:"
ls -lh "$VOLUME_PATH/data/"
echo ""
echo "Split files:"
ls -lh "$VOLUME_PATH/data/splits/" 2>/dev/null || echo "  No splits found"

echo -e "${GREEN}✓ Dataset ready${NC}"
echo ""

# ============================================================================
# STEP 8: Setup rclone (Google Drive)
# ============================================================================

echo -e "${CYAN}Step 8/11: Setting up rclone (Google Drive integration)${NC}"
echo "======================================================="
echo ""

# Install rclone
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

echo -e "${GREEN}✓ rclone installed: $(rclone --version | head -1)${NC}"
echo ""

# Check if already configured
if [ -f "$VOLUME_PATH/config/.rclone.conf" ]; then
    echo -e "${YELLOW}rclone config already exists on volume${NC}"
    echo "Using existing configuration"
    mkdir -p ~/.config/rclone/
    ln -sf "$VOLUME_PATH/config/.rclone.conf" ~/.config/rclone/rclone.conf

    # Test connection
    if rclone lsd gdrive: &>/dev/null; then
        echo -e "${GREEN}✓ rclone authentication valid${NC}"
    else
        echo -e "${YELLOW}⚠️  rclone auth may be expired, re-authenticating...${NC}"
        rclone config reconnect gdrive:
        cp ~/.config/rclone/rclone.conf "$VOLUME_PATH/config/.rclone.conf"
    fi
else
    echo -e "${MAGENTA}═══════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}   MANUAL STEP REQUIRED: Configure rclone     ${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════${NC}"
    echo ""
    echo "I will now launch rclone configuration."
    echo ""
    echo -e "${YELLOW}Follow these steps:${NC}"
    echo "  1. Choose: ${GREEN}n${NC} (new remote)"
    echo "  2. Name: ${GREEN}gdrive${NC}"
    echo "  3. Storage: ${GREEN}drive${NC} (Google Drive)"
    echo "  4. client_id: ${GREEN}(press Enter)${NC}"
    echo "  5. client_secret: ${GREEN}(press Enter)${NC}"
    echo "  6. scope: ${GREEN}1${NC} (Full access)"
    echo "  7. root_folder_id: ${GREEN}(press Enter)${NC}"
    echo "  8. service_account_file: ${GREEN}(press Enter)${NC}"
    echo "  9. Edit advanced config: ${GREEN}n${NC}"
    echo "  10. Use auto config: ${GREEN}n${NC} ⚠️  IMPORTANT!"
    echo ""
    echo "Then:"
    echo "  - Copy the URL shown"
    echo "  - Open it on your ${GREEN}LOCAL MACHINE${NC} (Mac)"
    echo "  - Authenticate with Google"
    echo "  - Copy verification code"
    echo "  - Paste it back here"
    echo ""
    read -p "Press Enter to start rclone config..."

    rclone config

    # Save config to volume
    echo ""
    echo "Saving rclone config to volume..."
    mkdir -p "$VOLUME_PATH/config/.config/rclone/"
    cp ~/.config/rclone/rclone.conf "$VOLUME_PATH/config/.rclone.conf"

    # Test connection
    echo ""
    echo "Testing Google Drive connection..."
    if rclone lsd gdrive: &>/dev/null; then
        echo -e "${GREEN}✓ rclone configured and working${NC}"
    else
        echo -e "${RED}✗ rclone test failed${NC}"
        echo "Please check configuration"
        exit 1
    fi
fi

# Create Google Drive folder structure
echo ""
echo "Creating Google Drive folders..."
rclone mkdir gdrive:Hebrew_Idiom_Detection 2>/dev/null || true
rclone mkdir gdrive:Hebrew_Idiom_Detection/results 2>/dev/null || true
rclone mkdir gdrive:Hebrew_Idiom_Detection/logs 2>/dev/null || true
rclone mkdir gdrive:Hebrew_Idiom_Detection/models 2>/dev/null || true
rclone mkdir gdrive:Hebrew_Idiom_Detection/backups 2>/dev/null || true

echo -e "${GREEN}✓ Google Drive integration ready${NC}"
echo ""

# ============================================================================
# STEP 9: Setup Environment Variables
# ============================================================================

echo -e "${CYAN}Step 9/11: Configuring environment variables${NC}"
echo "==============================================="
echo ""

# Create .env file on volume
cat > "$VOLUME_PATH/config/.env" << 'EOF'
# Environment Variables for Hebrew Idiom Detection Project
# This file is stored on PERSISTENT VOLUME
# Auto-generated by setup_volume.sh

# HF Cache
HF_HOME=/workspace/cache/huggingface
TRANSFORMERS_CACHE=/workspace/cache/huggingface

# Dataset metadata
DATASET_NAME=Hebrew-Idioms-4800
DATASET_VERSION=2.0
DATASET_TOTAL_SENTENCES=4800

# Google Drive paths
GDRIVE_PROJECT_FOLDER=Hebrew_Idiom_Detection
GDRIVE_RESULTS_FOLDER=Hebrew_Idiom_Detection/results
GDRIVE_LOGS_FOLDER=Hebrew_Idiom_Detection/logs

# Project settings
RANDOM_SEED=42
EOF

echo -e "${GREEN}✓ Environment variables configured${NC}"
echo "  Saved to: $VOLUME_PATH/config/.env"
echo ""

# ============================================================================
# STEP 10: Pre-download Models (Optional)
# ============================================================================

echo -e "${CYAN}Step 10/11: Pre-downloading transformer models${NC}"
echo "==============================================="
echo ""

read -p "$(echo -e ${YELLOW}Download all 5 models now? \(~10 GB, saves time later\) [y/N]: ${NC})" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading models to cache..."
    export HF_HOME="$VOLUME_PATH/cache/huggingface"
    export TRANSFORMERS_CACHE="$VOLUME_PATH/cache/huggingface"

    cd "$VOLUME_PATH/project"
    source "$VOLUME_PATH/env/bin/activate"

    if [ -f "src/model_download.py" ]; then
        python src/model_download.py
        echo -e "${GREEN}✓ Models downloaded to cache${NC}"
    else
        echo -e "${YELLOW}model_download.py not found, skipping${NC}"
    fi
else
    echo "Skipped. Models will download on first use."
fi

echo ""

# ============================================================================
# STEP 11: Final Verification
# ============================================================================

echo -e "${CYAN}Step 11/11: Final verification${NC}"
echo "==============================="
echo ""

# Show volume structure
echo "Volume structure:"
ls -la "$VOLUME_PATH"
echo ""

# Show sizes
echo "Volume usage:"
du -sh "$VOLUME_PATH"/*
echo ""

# Verify all key components
echo "Component verification:"

# Python environment
if [ -f "$VOLUME_PATH/env/bin/python" ]; then
    echo -e "  ${GREEN}✓${NC} Python environment"
else
    echo -e "  ${RED}✗${NC} Python environment"
fi

# Dataset
if [ -f "$VOLUME_PATH/data/expressions_data_tagged_v2.csv" ] || [ -f "$VOLUME_PATH/data/expressions_data_tagged.csv" ]; then
    echo -e "  ${GREEN}✓${NC} Dataset"
else
    echo -e "  ${RED}✗${NC} Dataset"
fi

# Project code
if [ -d "$VOLUME_PATH/project/.git" ]; then
    echo -e "  ${GREEN}✓${NC} Project code"
else
    echo -e "  ${RED}✗${NC} Project code"
fi

# rclone config
if [ -f "$VOLUME_PATH/config/.rclone.conf" ]; then
    echo -e "  ${GREEN}✓${NC} rclone config"
else
    echo -e "  ${RED}✗${NC} rclone config"
fi

# .env file
if [ -f "$VOLUME_PATH/config/.env" ]; then
    echo -e "  ${GREEN}✓${NC} Environment variables"
else
    echo -e "  ${RED}✗${NC} Environment variables
fi

echo ""

# ============================================================================
# SETUP COMPLETE
# ============================================================================

echo "╔════════════════════════════════════════════════╗"
echo "║                                                ║"
echo "║        ✓ VOLUME SETUP COMPLETE!               ║"
echo "║                                                ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

echo -e "${GREEN}Your persistent volume is ready!${NC}"
echo ""
echo "Volume path: $VOLUME_PATH"
echo ""

echo "Next steps:"
echo "  1. ${YELLOW}Destroy this setup instance${NC} (in Vast.ai console)"
echo "  2. ${GREEN}Confirm:${NC} 'Destroy instance but KEEP storage volume'"
echo "  3. Rent a new GPU instance when ready to train"
echo "  4. Attach the volume at /workspace"
echo "  5. Run: ${CYAN}bash /workspace/project/scripts/instance_bootstrap.sh${NC}"
echo "  6. Start training!"
echo ""

echo -e "${MAGENTA}═══════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}   IMPORTANT: Save rclone config backup!      ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════${NC}"
echo ""
echo "On your ${GREEN}LOCAL MAC${NC}, run this command:"
echo ""
echo -e "${CYAN}scp -P <PORT> root@<IP>:/workspace/config/.rclone.conf ~/Desktop/rclone_backup.conf${NC}"
echo ""
echo "This is your backup in case the volume is lost."
echo ""

echo -e "${BLUE}Happy training! 🚀${NC}"
echo ""