#!/bin/bash
# ============================================================================
# VAST.ai Instance Setup Script
# ============================================================================
# Purpose: Automate setup of a fresh VAST.ai GPU instance
# Usage: bash setup_vast_instance.sh
# Time: ~5-10 minutes
# ============================================================================
# This script:
# 1. Updates system packages
# 2. Installs Python dependencies
# 3. Clones GitHub repository
# 4. Downloads dataset from Google Drive
# 5. Verifies GPU availability
# 6. Shows you're ready to train
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration (MODIFY THESE)
GITHUB_REPO_URL="https://github.com/igornazarenko434/hebrew-idiom-detection.git"
PROJECT_DIR="hebrew-idiom-detection"

echo ""
echo "========================================"
echo "  VAST.ai Instance Setup"
echo "  Hebrew Idiom Detection Project"
echo "========================================"
echo ""

# ============================================================================
# STEP 1: System Information
# ============================================================================

echo -e "${CYAN}Step 1/7: System Information${NC}"
echo "-----------------------------"

echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version 2>&1 || echo 'Not found')"
echo "Pip version: $(pip3 --version 2>&1 | head -1 || echo 'Not found')"

echo ""

# ============================================================================
# STEP 2: Update System Packages
# ============================================================================

echo -e "${CYAN}Step 2/7: Updating system packages${NC}"
echo "-----------------------------------"

if command -v apt-get &> /dev/null; then
    echo "Updating apt package lists..."
    apt-get update -qq || true

    echo "Installing essential tools (git, wget, curl, vim)..."
    apt-get install -y -qq git wget curl vim || true

    echo -e "${GREEN}✓ System packages updated${NC}"
elif command -v yum &> /dev/null; then
    echo "Updating yum packages..."
    yum update -y -q || true
    yum install -y -q git wget curl vim || true
    echo -e "${GREEN}✓ System packages updated${NC}"
else
    echo -e "${YELLOW}⚠ Package manager not found, skipping system updates${NC}"
fi

echo ""

# ============================================================================
# STEP 3: Upgrade pip
# ============================================================================

echo -e "${CYAN}Step 3/7: Upgrading pip${NC}"
echo "-----------------------"

pip3 install --upgrade pip -q
echo -e "${GREEN}✓ pip upgraded${NC}"
echo "Pip version: $(pip3 --version)"

echo ""

# ============================================================================
# STEP 4: Clone GitHub Repository
# ============================================================================

echo -e "${CYAN}Step 4/7: Cloning GitHub repository${NC}"
echo "------------------------------------"

# Check if directory already exists
if [ -d "${PROJECT_DIR}" ]; then
    echo -e "${YELLOW}⚠ Directory ${PROJECT_DIR} already exists${NC}"
    echo "Options:"
    echo "  1. Remove and re-clone (fresh start)"
    echo "  2. Pull latest changes (keep existing)"
    echo "  3. Skip (use existing)"
    echo ""
    read -p "Choose option (1/2/3): " option

    case $option in
        1)
            echo "Removing existing directory..."
            rm -rf "${PROJECT_DIR}"
            echo "Cloning repository..."
            git clone "${GITHUB_REPO_URL}" "${PROJECT_DIR}"
            echo -e "${GREEN}✓ Repository cloned (fresh)${NC}"
            ;;
        2)
            echo "Pulling latest changes..."
            cd "${PROJECT_DIR}"
            git pull
            cd ..
            echo -e "${GREEN}✓ Repository updated${NC}"
            ;;
        3)
            echo -e "${BLUE}ℹ Using existing repository${NC}"
            ;;
        *)
            echo -e "${YELLOW}Invalid option, using existing repository${NC}"
            ;;
    esac
else
    echo "Cloning from: ${GITHUB_REPO_URL}"
    git clone "${GITHUB_REPO_URL}" "${PROJECT_DIR}"
    echo -e "${GREEN}✓ Repository cloned${NC}"
fi

echo ""

# Change to project directory
cd "${PROJECT_DIR}"
echo "Changed to project directory: $(pwd)"

echo ""

# ============================================================================
# STEP 5: Install Python Dependencies
# ============================================================================

echo -e "${CYAN}Step 5/7: Installing Python dependencies${NC}"
echo "-----------------------------------------"

if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    echo -e "${YELLOW}This may take 3-5 minutes...${NC}"

    # Install with progress
    pip3 install -r requirements.txt

    echo -e "${GREEN}✓ Dependencies installed${NC}"

    # Verify key packages
    echo ""
    echo "Verifying installations:"
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
    python3 -c "import optuna; print(f'  Optuna: {optuna.__version__}')"
    python3 -c "import pandas; print(f'  Pandas: {pandas.__version__}')"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

echo ""

# ============================================================================
# STEP 6: Download Dataset from Google Drive
# ============================================================================

echo -e "${CYAN}Step 6/7: Downloading dataset from Google Drive${NC}"
echo "-------------------------------------------------"

if [ -f "scripts/download_from_gdrive.sh" ]; then
    echo "Running download script..."
    bash scripts/download_from_gdrive.sh
else
    echo -e "${YELLOW}⚠ download_from_gdrive.sh not found${NC}"
    echo "Downloading manually with gdown..."

    # Install gdown if needed
    pip3 install gdown -q

    # Create directories
    mkdir -p data data/splits

    # Download dataset
    echo "Downloading main dataset..."
    gdown 140zJatqT4LBl7yG-afFSoUrYrisi9276 -O data/expressions_data_tagged.csv

    echo -e "${GREEN}✓ Dataset downloaded${NC}"
    echo -e "${YELLOW}Note: Split files should be in GitHub repository${NC}"
fi

echo ""

# ============================================================================
# STEP 7: Verify GPU Availability
# ============================================================================

echo -e "${CYAN}Step 7/7: Verifying GPU availability${NC}"
echo "-------------------------------------"

# Check NVIDIA driver
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver Info:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
fi

# Check PyTorch GPU
echo "PyTorch GPU Check:"
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  GPU name: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('  ⚠ WARNING: CUDA not available!')
"

echo ""

# ============================================================================
# SETUP COMPLETE
# ============================================================================

echo "========================================"
echo -e "${GREEN}✓ VAST.ai Setup Complete!${NC}"
echo "========================================"
echo ""

echo -e "${BLUE}Project Directory:${NC} $(pwd)"
echo -e "${BLUE}Python Version:${NC} $(python3 --version)"
echo -e "${BLUE}GPU Status:${NC} $(python3 -c 'import torch; print("✓ Available" if torch.cuda.is_available() else "✗ Not Available")')"

echo ""
echo "========================================"
echo "READY TO TRAIN!"
echo "========================================"
echo ""

echo "Quick Start Commands:"
echo ""

echo "1. Test training on small subset (5 minutes):"
echo -e "${YELLOW}   python src/idiom_experiment.py --mode full_finetune --model_id onlplab/alephbert-base --task cls --num_train_samples 100 --device cuda${NC}"
echo ""

echo "2. Run full training with config:"
echo -e "${YELLOW}   python src/idiom_experiment.py --mode full_finetune --config experiments/configs/training_config.yaml --task cls --device cuda${NC}"
echo ""

echo "3. Run HPO (hyperparameter optimization):"
echo -e "${YELLOW}   python src/idiom_experiment.py --mode hpo --model_id onlplab/alephbert-base --task cls --config experiments/configs/hpo_config.yaml --device cuda${NC}"
echo ""

echo "4. After training, sync results to Google Drive:"
echo -e "${YELLOW}   bash scripts/sync_to_gdrive.sh${NC}"
echo ""

echo -e "${BLUE}Tip:${NC} Use 'screen' or 'tmux' to keep training running if SSH disconnects:"
echo "   screen -S training"
echo "   python src/idiom_experiment.py ..."
echo "   # Press Ctrl+A then D to detach"
echo "   # Reconnect with: screen -r training"
echo ""

echo -e "${GREEN}Happy Training! 🚀${NC}"
echo ""
