#!/bin/bash
# ============================================================================
# Batch Runner for All HPO Studies
# ============================================================================
# Purpose: Run hyperparameter optimization for all model-task combinations
# Usage: bash scripts/run_all_hpo.sh
# Time: ~50-75 GPU hours (15 trials × 10 combinations)
# Cost: ~$20-30 on VAST.ai
# ============================================================================
# Mission 4.5: Hyperparameter Optimization for All Models
# Runs 10 HPO studies: 5 models × 2 tasks
# Each study runs 15 trials (configured in hpo_config.yaml)
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "  Batch HPO Runner"
echo "  Hebrew Idiom Detection Project"
echo "  Mission 4.5"
echo "========================================"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model IDs (HuggingFace model identifiers)
MODELS=(
    "onlplab/alephbert-base"
    "dicta-il/alephbertgimmel-base"
    "dicta-il/dictabert"
    "dicta-il/neodictabert"
    "bert-base-multilingual-cased"
    "xlm-roberta-base"
)

# Tasks
TASKS=(
    "cls"      # Sequence classification (Task 1)
    "span"     # Token classification (Task 2)
)

# Configuration file
HPO_CONFIG="experiments/configs/hpo_config.yaml"

# Device
DEVICE="cuda"

# Number of trials per HPO study (from config)
N_TRIALS=15

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo -e "${CYAN}Pre-flight Checks${NC}"
echo "-------------------"

# Check if config file exists
if [ ! -f "${HPO_CONFIG}" ]; then
    echo -e "${RED}✗ Config file not found: ${HPO_CONFIG}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Config file found: ${HPO_CONFIG}${NC}"

# Check if data splits exist
if [ ! -f "data/splits/train.csv" ] || [ ! -f "data/splits/validation.csv" ]; then
    echo -e "${RED}✗ Data splits not found in data/splits/${NC}"
    echo "Run: bash scripts/download_from_gdrive.sh"
    exit 1
fi
echo -e "${GREEN}✓ Data splits found${NC}"

# Check GPU availability
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null
if [ $? -eq 0 ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}✓ GPU available: ${GPU_NAME}${NC}"
else
    echo -e "${RED}✗ GPU not available!${NC}"
    echo "HPO requires GPU. Please use VAST.ai or local GPU."
    exit 1
fi

# Check if idiom_experiment.py exists
if [ ! -f "src/idiom_experiment.py" ]; then
    echo -e "${RED}✗ Training script not found: src/idiom_experiment.py${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Training script found${NC}"

echo ""

# ============================================================================
# CALCULATE TOTAL EXPERIMENTS
# ============================================================================

TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#TASKS[@]}))
TOTAL_TRIALS=$((TOTAL_EXPERIMENTS * N_TRIALS))

echo -e "${BLUE}Experiment Configuration:${NC}"
echo "  Models: ${#MODELS[@]}"
echo "  Tasks: ${#TASKS[@]}"
echo "  Total HPO studies: ${TOTAL_EXPERIMENTS}"
echo "  Trials per study: ${N_TRIALS}"
echo "  Total trials: ${TOTAL_TRIALS}"
echo ""

# Estimate time (rough estimate: 20-30 min per trial)
MIN_MINUTES=$((TOTAL_TRIALS * 20))
MAX_MINUTES=$((TOTAL_TRIALS * 30))
echo -e "${YELLOW}Estimated time: ${MIN_MINUTES}-${MAX_MINUTES} minutes ($(($MIN_MINUTES/60))-$(($MAX_MINUTES/60)) hours)${NC}"
echo -e "${YELLOW}Estimated cost on VAST.ai: \$20-\$30 @ \$0.40/hr${NC}"
echo ""

# ============================================================================
# CONFIRMATION
# ============================================================================

echo -e "${MAGENTA}Models to optimize:${NC}"
for model in "${MODELS[@]}"; do
    echo "  - ${model}"
done
echo ""

echo -e "${MAGENTA}Tasks to optimize:${NC}"
for task in "${TASKS[@]}"; do
    echo "  - ${task}"
done
echo ""

read -p "Start HPO batch run? This will take several hours. (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# ============================================================================
# CREATE RESULTS DIRECTORY
# ============================================================================

mkdir -p experiments/results/hpo
mkdir -p experiments/logs


# ============================================================================
# MAIN LOOP
# ============================================================================

echo "========================================"
echo "Starting HPO Batch Run"
echo "========================================"
echo ""

# Start time
START_TIME=$(date +%s)

# Counter
CURRENT=0
FAILED=0

# Log file
LOG_FILE="experiments/logs/hpo_batch_$(date +%Y%m%d_%H%M%S).log"
echo "Batch HPO run started at $(date)" > "${LOG_FILE}"

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        CURRENT=$((CURRENT + 1))

        # Model short name (for display)
        MODEL_SHORT=$(echo "${model}" | sed 's/.*\///')

        echo ""
        echo "========================================"
        echo -e "${CYAN}HPO Study ${CURRENT}/${TOTAL_EXPERIMENTS}${NC}"
        echo "========================================"
        echo -e "${BLUE}Model:${NC} ${model}"
        echo -e "${BLUE}Task:${NC}  ${task}"
        echo -e "${BLUE}Trials:${NC} ${N_TRIALS}"
        echo ""

        # Log
        echo "=== HPO Study ${CURRENT}/${TOTAL_EXPERIMENTS}: ${MODEL_SHORT} | ${task} ===" >> "${LOG_FILE}"
        echo "Started: $(date)" >> "${LOG_FILE}"

        # Run HPO
        echo "Running HPO..."
        echo "Command: python src/idiom_experiment.py --mode hpo --model_id ${model} --task ${task} --config ${HPO_CONFIG} --device ${DEVICE}"
        echo ""

        # Execute (capture exit code)
        if python3 src/idiom_experiment.py \
            --mode hpo \
            --model_id "${model}" \
            --task "${task}" \
            --config "${HPO_CONFIG}" \
            --device "${DEVICE}" \
            2>&1 | tee -a "${LOG_FILE}"; then

            echo ""
            echo -e "${GREEN}✓ HPO completed: ${MODEL_SHORT} | ${task}${NC}"
            echo "Completed: $(date)" >> "${LOG_FILE}"
            echo "" >> "${LOG_FILE}"

        else
            echo ""
            echo -e "${RED}✗ HPO failed: ${MODEL_SHORT} | ${task}${NC}"
            echo "FAILED: $(date)" >> "${LOG_FILE}"
            echo "" >> "${LOG_FILE}"
            FAILED=$((FAILED + 1))
        fi

        # Show progress
        REMAINING=$((TOTAL_EXPERIMENTS - CURRENT))
        echo ""
        echo "Progress: ${CURRENT}/${TOTAL_EXPERIMENTS} complete, ${REMAINING} remaining"

        # Estimate remaining time
        ELAPSED=$(($(date +%s) - START_TIME))
        if [ ${CURRENT} -gt 0 ]; then
            AVG_TIME=$((ELAPSED / CURRENT))
            EST_REMAINING=$((AVG_TIME * REMAINING / 60))
            echo "Estimated time remaining: ~${EST_REMAINING} minutes"
        fi

        echo ""
        echo "----------------------------------------"

        # Optional: Sync to Google Drive after each study
        if command -v rclone &> /dev/null && rclone listremotes | grep -q "gdrive:"; then
            echo "Syncing results to Google Drive..."
            bash scripts/sync_to_gdrive.sh 2>&1 | tail -5
        fi

    done
done

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "========================================"
echo -e "${GREEN}✓ HPO Batch Run Complete!${NC}"
echo "========================================"
echo ""
echo "Summary:"
echo "  Total studies: ${TOTAL_EXPERIMENTS}"
echo "  Successful: $((TOTAL_EXPERIMENTS - FAILED))"
echo "  Failed: ${FAILED}"
echo "  Total time: ${HOURS}h ${MINUTES}m"
echo "  Log file: ${LOG_FILE}"
echo ""

# Show results directories
echo "Results saved to:"
echo "  experiments/results/hpo/"
echo "  experiments/results/best_hyperparameters/"
echo ""

# List best hyperparameters found
if [ -d "experiments/results/best_hyperparameters" ]; then
    echo "Best hyperparameters found:"
    ls -1 experiments/results/best_hyperparameters/ | grep .json | while read f; do
        echo "  - ${f}"
    done
    echo ""
fi

# Final sync to Google Drive
if command -v rclone &> /dev/null && rclone listremotes | grep -q "gdrive:"; then
    echo "Final sync to Google Drive..."
    bash scripts/sync_to_gdrive.sh
    echo ""
fi

echo "Next steps:"
echo "1. Review best hyperparameters in experiments/results/best_hyperparameters/"
echo "2. Analyze Optuna studies (can use optuna-dashboard)"
echo "3. Proceed to Mission 4.6: Final training with best hyperparameters"
echo "   bash scripts/run_all_experiments.sh"
echo ""

if [ ${FAILED} -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warning: ${FAILED} studies failed. Check log: ${LOG_FILE}${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}All HPO studies completed successfully! 🎉${NC}"
echo ""
