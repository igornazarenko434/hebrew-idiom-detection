#!/bin/bash
# ============================================================================
# Vast.ai Instance Bootstrap Script (RUN EVERY SESSION)
# ============================================================================
# Purpose: Quickly prepare a new instance using persistent volume
# Usage: bash /mnt/volume/project/scripts/instance_bootstrap.sh
# Time: ~1 minute
# Prerequisites: Volume setup completed (setup_volume.sh)
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
VOLUME_PATH="/mnt/volume"
PROJECT_PATH="$VOLUME_PATH/project"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║   Vast.ai Instance Bootstrap                   ║"
echo "║   Hebrew Idiom Detection Project               ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# STEP 0: Pre-flight Checks
# ============================================================================

echo -e "${CYAN}Step 0/8: Pre-flight checks${NC}"
echo "============================"
echo ""

# Check volume is mounted
if [ ! -d "$VOLUME_PATH" ]; then
    echo -e "${RED}✗ ERROR: Volume not mounted at $VOLUME_PATH${NC}"
    echo ""
    echo "Make sure you attached your persistent volume when renting this instance!"
    echo ""
    echo "In Vast.ai:"
    echo "  1. Search for instances"
    echo "  2. Before clicking 'Rent', click 'Storage' button"
    echo "  3. Attach your volume: hebrew-idiom-volume"
    echo "  4. Mount point: /mnt/volume"
    echo "  5. Then rent the instance"
    exit 1
fi

echo -e "${GREEN}✓ Volume mounted at $VOLUME_PATH${NC}"

# Check volume was set up
if [ ! -f "$VOLUME_PATH/config/.env" ]; then
    echo -e "${RED}✗ ERROR: Volume not initialized${NC}"
    echo ""
    echo "You need to run setup_volume.sh first!"
    echo "See: VAST_AI_PERSISTENT_VOLUME_GUIDE.md"
    exit 1
fi

echo -e "${GREEN}✓ Volume is initialized${NC}"

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo -e "${YELLOW}⚠️  WARNING: nvidia-smi not available${NC}"
    echo "This instance may not have a GPU!"
else
    echo -e "${GREEN}✓ GPU detected${NC}"
fi

echo ""

# ============================================================================
# STEP 1: System Information
# ============================================================================

echo -e "${CYAN}Step 1/8: System information${NC}"
echo "============================="
echo ""

echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo ""

if command -v nvidia-smi &>/dev/null; then
    echo "GPU info:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}No GPU detected${NC}"
fi

echo ""

# ============================================================================
# STEP 2: Symlink rclone Config
# ============================================================================

echo -e "${CYAN}Step 2/8: Linking rclone configuration${NC}"
echo "========================================"
echo ""

# Create rclone config directory
mkdir -p ~/.config/rclone/

# Symlink from volume
if [ -f "$VOLUME_PATH/config/.rclone.conf" ]; then
    ln -sf "$VOLUME_PATH/config/.rclone.conf" ~/.config/rclone/rclone.conf
    echo -e "${GREEN}✓ rclone config linked${NC}"

    # Verify
    if rclone listremotes | grep -q "gdrive:"; then
        echo -e "${GREEN}✓ Google Drive remote available${NC}"
    else
        echo -e "${YELLOW}⚠️  Google Drive remote not found${NC}"
        echo "You may need to reconfigure rclone"
    fi
else
    echo -e "${YELLOW}⚠️  rclone config not found on volume${NC}"
    echo "Google Drive sync will not work"
fi

echo ""

# ============================================================================
# STEP 3: Symlink Environment Variables
# ============================================================================

echo -e "${CYAN}Step 3/8: Loading environment variables${NC}"
echo "=========================================="
echo ""

# Symlink .env to project root
if [ -f "$VOLUME_PATH/config/.env" ]; then
    ln -sf "$VOLUME_PATH/config/.env" "$PROJECT_PATH/.env"
    echo -e "${GREEN}✓ Environment variables linked${NC}"

    # Load variables
    set -a
    source "$VOLUME_PATH/config/.env"
    set +a

    echo "  HF_HOME: $HF_HOME"
    echo "  DATA_DIR: $LOCAL_DATA_DIR"
    echo "  RESULTS_DIR: $LOCAL_RESULTS_DIR"
else
    echo -e "${YELLOW}⚠️  .env file not found on volume${NC}"
fi

# Export critical environment variables
export HF_HOME="$VOLUME_PATH/cache/huggingface"
export TRANSFORMERS_CACHE="$VOLUME_PATH/cache/huggingface"
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"

echo ""

# ============================================================================
# STEP 4: Activate Python Environment
# ============================================================================

echo -e "${CYAN}Step 4/8: Activating Python environment${NC}"
echo "=========================================="
echo ""

if [ -f "$VOLUME_PATH/env/bin/activate" ]; then
    source "$VOLUME_PATH/env/bin/activate"
    echo -e "${GREEN}✓ Python environment activated${NC}"
    echo "  Python: $(python --version)"
    echo "  Location: $(which python)"
else
    echo -e "${RED}✗ ERROR: Python environment not found${NC}"
    echo "Run setup_volume.sh first!"
    exit 1
fi

echo ""

# ============================================================================
# STEP 5: Update Project Code from GitHub
# ============================================================================

echo -e "${CYAN}Step 5/8: Updating project code from GitHub${NC}"
echo "=============================================="
echo ""

cd "$PROJECT_PATH"

# Check git status
if [ -d .git ]; then
    echo "Current branch: $(git branch --show-current)"

    # Fetch latest
    echo "Fetching latest changes..."
    git fetch origin

    # Check if we're behind
    BEHIND=$(git rev-list HEAD..origin/main --count 2>/dev/null || echo "0")

    if [ "$BEHIND" -gt 0 ]; then
        echo -e "${YELLOW}Found $BEHIND new commits, pulling...${NC}"
        git pull origin main
        echo -e "${GREEN}✓ Code updated${NC}"
    else
        echo -e "${GREEN}✓ Code is up to date${NC}"
    fi

    # Show last commit
    echo ""
    echo "Latest commit:"
    git log -1 --oneline
else
    echo -e "${YELLOW}⚠️  Not a git repository${NC}"
fi

echo ""

# ============================================================================
# STEP 6: Verify Dataset
# ============================================================================

echo -e "${CYAN}Step 6/8: Verifying dataset${NC}"
echo "============================"
echo ""

DATA_DIR="$VOLUME_PATH/data"

if [ -f "$DATA_DIR/expressions_data_tagged_v2.csv" ] || [ -f "$DATA_DIR/expressions_data_tagged.csv" ]; then
    echo -e "${GREEN}✓ Dataset found${NC}"
    ls -lh "$DATA_DIR"/*.csv | head -3
else
    echo -e "${RED}✗ Dataset not found${NC}"
    echo "Expected: $DATA_DIR/expressions_data_tagged_v2.csv"
fi

echo ""

# Check splits
if [ -d "$DATA_DIR/splits" ]; then
    SPLIT_COUNT=$(ls -1 "$DATA_DIR/splits"/*.csv 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $SPLIT_COUNT split files${NC}"
else
    echo -e "${YELLOW}⚠️  No splits directory${NC}"
fi

echo ""

# ============================================================================
# STEP 7: Install/Update Dependencies (if needed)
# ============================================================================

echo -e "${CYAN}Step 7/8: Checking Python dependencies${NC}"
echo "========================================"
echo ""

# Check if requirements.txt changed
if [ -f requirements.txt ]; then
    # Quick check of key packages
    if python -c "import transformers, torch, optuna" 2>/dev/null; then
        echo -e "${GREEN}✓ Core dependencies available${NC}"
        echo "  Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
        echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
        echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
    else
        echo -e "${YELLOW}⚠️  Some dependencies missing, installing...${NC}"
        pip install -r requirements.txt -q
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  requirements.txt not found${NC}"
fi

echo ""

# ============================================================================
# STEP 8: Final Status Check
# ============================================================================

echo -e "${CYAN}Step 8/8: Final status check${NC}"
echo "============================="
echo ""

# Verify everything is ready
echo "Component status:"

# Python environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "  ${GREEN}✓${NC} Python environment active"
else
    echo -e "  ${RED}✗${NC} Python environment NOT active"
fi

# GPU
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')")
    echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME (${GPU_MEM} GB)"
else
    echo -e "  ${YELLOW}⚠${NC}  GPU not available"
fi

# Dataset
if [ -f "$DATA_DIR/expressions_data_tagged_v2.csv" ] || [ -f "$DATA_DIR/expressions_data_tagged.csv" ]; then
    echo -e "  ${GREEN}✓${NC} Dataset ready"
else
    echo -e "  ${RED}✗${NC} Dataset missing"
fi

# rclone
if command -v rclone &>/dev/null && rclone listremotes | grep -q "gdrive:"; then
    echo -e "  ${GREEN}✓${NC} Google Drive sync ready"
else
    echo -e "  ${YELLOW}⚠${NC}  Google Drive sync not configured"
fi

# Disk space
echo ""
echo "Disk usage:"
df -h "$VOLUME_PATH" | tail -1

echo ""

# ============================================================================
# SETUP COMPLETE
# ============================================================================

echo "╔════════════════════════════════════════════════╗"
echo "║                                                ║"
echo "║          ✓ INSTANCE READY!                    ║"
echo "║                                                ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

echo -e "${GREEN}You can now start training!${NC}"
echo ""
echo "Quick start commands:"
echo ""

echo -e "${YELLOW}1. Test run (quick verification):${NC}"
echo "   cd $PROJECT_PATH"
echo "   python src/idiom_experiment.py \\"
echo "     --mode full_finetune \\"
echo "     --model_id onlplab/alephbert-base \\"
echo "     --task cls \\"
echo "     --config experiments/configs/training_config.yaml \\"
echo "     --num_train_samples 100 \\"
echo "     --num_epochs 1 \\"
echo "     --device cuda"
echo ""

echo -e "${YELLOW}2. Full training (single model):${NC}"
echo "   cd $PROJECT_PATH"
echo "   python src/idiom_experiment.py \\"
echo "     --mode full_finetune \\"
echo "     --model_id onlplab/alephbert-base \\"
echo "     --task cls \\"
echo "     --config experiments/configs/training_config.yaml \\"
echo "     --device cuda"
echo ""

echo -e "${YELLOW}3. Hyperparameter optimization (recommended):${NC}"
echo "   cd $PROJECT_PATH"
echo "   bash scripts/run_all_hpo.sh"
echo ""

echo -e "${YELLOW}4. Run all experiments (30 runs):${NC}"
echo "   cd $PROJECT_PATH"
echo "   bash scripts/run_all_experiments.sh"
echo ""

echo -e "${CYAN}After training:${NC}"
echo "  1. Sync results to Google Drive:"
echo "     bash scripts/sync_to_gdrive.sh"
echo ""
echo "  2. Destroy instance (saves money!):"
echo "     exit  # close SSH"
echo "     # Then destroy in Vast.ai console"
echo ""

echo -e "${MAGENTA}═══════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}   Use tmux/screen for long training runs!    ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════${NC}"
echo ""
echo "Start a persistent session:"
echo "  screen -S training"
echo "  # or"
echo "  tmux new -s training"
echo ""
echo "Then run your training command."
echo "Detach: Ctrl+A then D (screen) or Ctrl+B then D (tmux)"
echo "Reattach: screen -r training or tmux attach -t training"
echo ""

echo -e "${BLUE}Happy Training! 🚀${NC}"
echo ""

# Auto-navigate to project directory
cd "$PROJECT_PATH"
echo -e "${GREEN}Working directory: $(pwd)${NC}"
echo ""

# Show git status
if [ -d .git ]; then
    echo "Git status:"
    git status -s || true
    echo ""
fi

# Reminder about environment
echo -e "${CYAN}Environment is activated. You can now run Python scripts.${NC}"
echo ""

# Start a new shell with the environment
exec bash
