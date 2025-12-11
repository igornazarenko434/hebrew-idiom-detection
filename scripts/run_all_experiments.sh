#!/bin/bash
# ============================================================================
# Batch Runner for Final Training with Best Hyperparameters
# ============================================================================
# Purpose: Run final training for all model-task combinations with best hyperparameters
# Usage: bash scripts/run_all_experiments.sh
# Time: ~10-15 GPU hours (30 training runs)
# Cost: ~$4-6 on VAST.ai
# ============================================================================
# Mission 4.6: Final Training with Best Hyperparameters
# Runs 30 training experiments: 5 models × 2 tasks × 3 seeds
# Uses best hyperparameters from Mission 4.5 (HPO)
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
echo "  Batch Final Training Runner"
echo "  Hebrew Idiom Detection Project"
echo "  Mission 4.6"
echo "========================================"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Try to activate virtual environment
if [ -f "/workspace/env/bin/activate" ]; then
    source /workspace/env/bin/activate
    echo -e "${GREEN}✓ Activated environment: /workspace/env${NC}"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ Activated environment: .venv${NC}"
else
    echo -e "${YELLOW}⚠ Warning: No virtual environment found. Assuming system python has dependencies.${NC}"
fi

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model IDs (HuggingFace model identifiers)
MODELS=(
    #"onlplab/alephbert-base"
    #"dicta-il/alephbertgimmel-base"
    #"dicta-il/dictabert"
    "bert-base-multilingual-cased"
    "xlm-roberta-base"
)

# Tasks
TASKS=(
    "cls"      # Sequence classification (Task 1)
    "span"     # Token classification (Task 2)
)

# Seeds for cross-seed validation (PRD Section 5.3)
SEEDS=(42 123 456)

# Configuration file
TRAIN_CONFIG="experiments/configs/training_config.yaml"

# Best hyperparameters directory (from Mission 4.5)
BEST_PARAMS_DIR="experiments/results/best_hyperparameters"

# Device
DEVICE="cuda"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo -e "${CYAN}Pre-flight Checks${NC}"
echo "-------------------"

# Check if config file exists
if [ ! -f "${TRAIN_CONFIG}" ]; then
    echo -e "${RED}✗ Config file not found: ${TRAIN_CONFIG}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Config file found: ${TRAIN_CONFIG}${NC}"

# Check if best hyperparameters directory exists
if [ ! -d "${BEST_PARAMS_DIR}" ]; then
    echo -e "${RED}✗ Best hyperparameters directory not found: ${BEST_PARAMS_DIR}${NC}"
    echo "You must run Mission 4.5 (HPO) first!"
    echo "Run: bash scripts/run_all_hpo.sh"
    exit 1
fi
echo -e "${GREEN}✓ Best hyperparameters directory found${NC}"

# Count best hyperparameter files
BEST_PARAMS_COUNT=$(ls -1 "${BEST_PARAMS_DIR}"/*.json 2>/dev/null | wc -l)
EXPECTED_COUNT=$((${#MODELS[@]} * ${#TASKS[@]}))

if [ ${BEST_PARAMS_COUNT} -lt ${EXPECTED_COUNT} ]; then
    echo -e "${YELLOW}⚠ Warning: Expected ${EXPECTED_COUNT} hyperparameter files, found ${BEST_PARAMS_COUNT}${NC}"
    echo "Available hyperparameter files:"
    ls -1 "${BEST_PARAMS_DIR}"/*.json 2>/dev/null || echo "  None"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Found ${BEST_PARAMS_COUNT} hyperparameter files${NC}"
fi

# Check if data splits exist
if [ ! -f "data/splits/train.csv" ] || [ ! -f "data/splits/validation.csv" ] || [ ! -f "data/splits/test.csv" ]; then
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
    echo "Training requires GPU. Please use VAST.ai or local GPU."
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

TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#TASKS[@]} * ${#SEEDS[@]}))

echo -e "${BLUE}Experiment Configuration:${NC}"
echo "  Models: ${#MODELS[@]}"
echo "  Tasks: ${#TASKS[@]}"
echo "  Seeds: ${#SEEDS[@]}"
echo "  Total training runs: ${TOTAL_EXPERIMENTS}"
echo ""

# Estimate time (rough estimate: 20-30 min per run)
MIN_MINUTES=$((TOTAL_EXPERIMENTS * 20))
MAX_MINUTES=$((TOTAL_EXPERIMENTS * 30))
echo -e "${YELLOW}Estimated time: ${MIN_MINUTES}-${MAX_MINUTES} minutes ($(($MIN_MINUTES/60))-$(($MAX_MINUTES/60)) hours)${NC}"
echo -e "${YELLOW}Estimated cost on VAST.ai: \$4-\$6 @ \$0.40/hr${NC}"
echo ""

# ============================================================================
# CONFIRMATION
# ============================================================================

echo -e "${MAGENTA}Models to train:${NC}"
for model in "${MODELS[@]}"; do
    echo "  - ${model}"
done
echo ""

echo -e "${MAGENTA}Tasks to train:${NC}"
for task in "${TASKS[@]}"; do
    echo "  - ${task}"
done
echo ""

echo -e "${MAGENTA}Seeds for cross-validation:${NC}"
for seed in "${SEEDS[@]}"; do
    echo "  - ${seed}"
done
echo ""

read -p "Start final training batch run? This will take several hours. (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# ============================================================================
# CREATE RESULTS DIRECTORY
# ============================================================================

mkdir -p experiments/results/full_fine-tuning
mkdir -p experiments/logs

# ============================================================================
# MAIN LOOP
# ============================================================================

echo "========================================"
echo "Starting Final Training Batch Run"
echo "========================================"
echo ""

# Start time
START_TIME=$(date +%s)

# Counter
CURRENT=0
FAILED=0

# Log file
LOG_FILE="experiments/logs/training_batch_$(date +%Y%m%d_%H%M%S).log"
echo "Batch training run started at $(date)" > "${LOG_FILE}"

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        # Model short name (for display and file lookup)
        MODEL_SHORT=$(echo "${model}" | sed 's/.*\///')

        # Look for best hyperparameters file
        BEST_PARAMS_FILE="${BEST_PARAMS_DIR}/best_params_${MODEL_SHORT}_${task}.json"

        if [ ! -f "${BEST_PARAMS_FILE}" ]; then
            echo -e "${YELLOW}⚠ Warning: Best hyperparameters not found for ${MODEL_SHORT} | ${task}${NC}"
            echo "  Expected: ${BEST_PARAMS_FILE}"
            echo "  Skipping this model-task combination."
            echo ""
            continue
        fi

        # Run for each seed
        for seed in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))

            echo ""
            echo "========================================"
            echo -e "${CYAN}Training Run ${CURRENT}/${TOTAL_EXPERIMENTS}${NC}"
            echo "========================================"
            echo -e "${BLUE}Model:${NC}  ${model}"
            echo -e "${BLUE}Task:${NC}   ${task}"
            echo -e "${BLUE}Seed:${NC}   ${seed}"
            echo -e "${BLUE}Using:${NC}  ${BEST_PARAMS_FILE}"
            echo ""

            # Log
            echo "=== Training Run ${CURRENT}/${TOTAL_EXPERIMENTS}: ${MODEL_SHORT} | ${task} | seed=${seed} ===" >> "${LOG_FILE}"
            echo "Started: $(date)" >> "${LOG_FILE}"

            # Load best hyperparameters (read JSON)
            LEARNING_RATE=$(python3 -c "import json; print(json.load(open('${BEST_PARAMS_FILE}'))['learning_rate'])" 2>/dev/null || echo "2e-5")
            BATCH_SIZE=$(python3 -c "import json; print(json.load(open('${BEST_PARAMS_FILE}'))['batch_size'])" 2>/dev/null || echo "16")
            NUM_EPOCHS=$(python3 -c "import json; print(json.load(open('${BEST_PARAMS_FILE}'))['num_epochs'])" 2>/dev/null || echo "5")
            WARMUP_RATIO=$(python3 -c "import json; print(json.load(open('${BEST_PARAMS_FILE}'))['warmup_ratio'])" 2>/dev/null || echo "0.1")
            WEIGHT_DECAY=$(python3 -c "import json; print(json.load(open('${BEST_PARAMS_FILE}'))['weight_decay'])" 2>/dev/null || echo "0.01")
            GRAD_ACCUM=$(python3 -c "import json; print(json.load(open('${BEST_PARAMS_FILE}'))['gradient_accumulation_steps'])" 2>/dev/null || echo "1")

            echo "Best hyperparameters:"
            echo "  learning_rate: ${LEARNING_RATE}"
            echo "  batch_size: ${BATCH_SIZE}"
            echo "  num_epochs: ${NUM_EPOCHS}"
            echo "  warmup_ratio: ${WARMUP_RATIO}"
            echo "  weight_decay: ${WEIGHT_DECAY}"
            echo "  gradient_accumulation_steps: ${GRAD_ACCUM}"
            echo ""

            # Run training
            OUTPUT_DIR="experiments/results/full_fine-tuning/${MODEL_SHORT}/${task}/seed_${seed}"
            
            # Check if already done
            if [ -f "${OUTPUT_DIR}/training_results.json" ]; then
                echo -e "${GREEN}✓ Skipping completed run: ${MODEL_SHORT} | ${task} | seed=${seed}${NC}"
                continue
            fi

            echo "Running training..."
            echo "Command: python src/idiom_experiment.py --mode full_finetune --model_id ${model} --task ${task} --config ${TRAIN_CONFIG} --device ${DEVICE} --seed ${seed} --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --num_epochs ${NUM_EPOCHS} --warmup_ratio ${WARMUP_RATIO} --weight_decay ${WEIGHT_DECAY} --gradient_accumulation_steps ${GRAD_ACCUM} --output_dir ${OUTPUT_DIR}"
            echo ""

            # Execute (capture exit code)
            if python3 src/idiom_experiment.py \
                --mode full_finetune \
                --model_id "${model}" \
                --task "${task}" \
                --config "${TRAIN_CONFIG}" \
                --device "${DEVICE}" \
                --seed "${seed}" \
                --learning_rate "${LEARNING_RATE}" \
                --batch_size "${BATCH_SIZE}" \
                --num_epochs "${NUM_EPOCHS}" \
                --warmup_ratio "${WARMUP_RATIO}" \
                --weight_decay "${WEIGHT_DECAY}" \
                --gradient_accumulation_steps "${GRAD_ACCUM}" \
                --output_dir "${OUTPUT_DIR}" \
                2>&1 | tee -a "${LOG_FILE}"; then

                echo ""
                echo -e "${GREEN}✓ Training completed: ${MODEL_SHORT} | ${task} | seed=${seed}${NC}"
                echo "Completed: $(date)" >> "${LOG_FILE}"
                echo "" >> "${LOG_FILE}"

            else
                echo ""
                echo -e "${RED}✗ Training failed: ${MODEL_SHORT} | ${task} | seed=${seed}${NC}"
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

        done  # End seeds loop

        # Optional: Sync to Google Drive after each model-task (all seeds)
        if command -v rclone &> /dev/null && rclone listremotes | grep -q "gdrive:"; then
            echo "Syncing results to Google Drive..."
            bash scripts/sync_to_gdrive.sh 2>&1 | tail -5
        fi

    done  # End tasks loop
done  # End models loop

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "========================================"
echo -e "${GREEN}✓ Final Training Batch Run Complete!${NC}"
echo "========================================"
echo ""
echo "Summary:"
echo "  Total training runs: ${TOTAL_EXPERIMENTS}"
echo "  Successful: $((TOTAL_EXPERIMENTS - FAILED))"
echo "  Failed: ${FAILED}"
echo "  Total time: ${HOURS}h ${MINUTES}m"
echo "  Log file: ${LOG_FILE}"
echo ""

# Show results directories
echo "Results saved to:"
echo "  experiments/results/full_fine-tuning/"
echo ""

# List trained models
if [ -d "experiments/results/full_fine-tuning" ]; then
    echo "Trained models:"
    find experiments/results/full_fine-tuning -name "training_results.json" | while read f; do
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
echo "1. Review training results in experiments/results/full_fine-tuning/"
echo "2. Calculate mean ± std across 3 seeds for each model-task"
echo "3. Perform statistical tests (paired t-test)"
echo "4. Proceed to Mission 5: LLM Evaluation"
echo "5. Proceed to Mission 7: Comprehensive Analysis"
echo ""

if [ ${FAILED} -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warning: ${FAILED} training runs failed. Check log: ${LOG_FILE}${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}All training runs completed successfully! 🎉${NC}"
echo ""
echo "You now have:"
echo "  - ${TOTAL_EXPERIMENTS} trained models"
echo "  - Cross-seed validation results (3 seeds per model-task)"
echo "  - Ready for statistical analysis and paper writing"
echo ""
