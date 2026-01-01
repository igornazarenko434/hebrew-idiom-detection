#!/bin/bash
# ============================================================================
# Batch Evaluation Runner (Seen & Unseen Test Sets)
# ============================================================================
# Purpose: Evaluate ALL trained models (all seeds) on both test sets.
# This generates the data needed for "Generalization Gap" analysis.
# Usage: bash scripts/run_evaluation_batch.sh
# ============================================================================

set -e

# Configuration
MODELS_DIR="experiments/results/full_fine-tuning"
EVAL_BASE_DIR="experiments/results/evaluation"
SEEN_TEST_DATA="data/splits/test.csv"
UNSEEN_TEST_DATA="data/splits/unseen_idiom_test.csv"
DEVICE="cuda"

echo "========================================"
echo "  Batch Evaluation Runner"
echo "  Mission 4.7 / Phase 7"
echo "========================================"
echo ""

# Check if models directory exists
if [ ! -d "${MODELS_DIR}" ]; then
    echo "Error: Models directory not found: ${MODELS_DIR}"
    exit 1
fi

# Find all model files (recursively)
# Structure: .../model_name/task/seed_X/model.safetensors
echo "Scanning for trained models..."
MODEL_FILES=$(find "${MODELS_DIR}" -name "model.safetensors" | sort)
COUNT=$(echo "${MODEL_FILES}" | wc -l)

if [ -z "${MODEL_FILES}" ]; then
    echo "No models found to evaluate."
    exit 1
fi

echo "Found ${COUNT} models."
echo ""

CURRENT=0

for model_path in ${MODEL_FILES}; do
    # Skip intermediate checkpoints (folders named checkpoint-XXX)
    if [[ "${model_path}" == *"checkpoint-"* ]]; then
        continue
    fi

    CURRENT=$((CURRENT + 1))
    
    # Extract metadata from path
    # Path is: experiments/results/full_fine-tuning/MODEL/TASK/SEED/model.safetensors
    # dirname gets folder: .../SEED
    MODEL_DIR=$(dirname "${model_path}")
    
    # Extract parts
    SEED_DIR_NAME=$(basename "${MODEL_DIR}") # seed_42
    
    # Strict check: Ensure we are inside a seed folder
    if [[ "${SEED_DIR_NAME}" != "seed_"* ]]; then
        echo "Skipping non-seed directory: ${SEED_DIR_NAME}"
        continue
    fi

    TASK_DIR_NAME=$(basename $(dirname "${MODEL_DIR}")) # cls or span
    MODEL_NAME=$(basename $(dirname $(dirname "${MODEL_DIR}"))) # alephbert-base
    
    echo "---------------------------------------------------"
    echo "Processing [${CURRENT}/${COUNT}]: ${MODEL_NAME} | ${TASK_DIR_NAME} | ${SEED_DIR_NAME}"
    echo "---------------------------------------------------"

    # Define tasks
    # We must run evaluation for the specific task the model was trained on
    TASK_TYPE="${TASK_DIR_NAME}" 

    # -----------------------------
    # 1. Evaluate on SEEN Test Set
    # -----------------------------
    OUT_DIR_SEEN="${EVAL_BASE_DIR}/seen_test/${MODEL_NAME}/${TASK_TYPE}/${SEED_DIR_NAME}"
    EVAL_PREDICTIONS_SEEN="${OUT_DIR_SEEN}/eval_predictions.json" # Check for this file
    
    echo "  > Evaluating on SEEN test set..."
    
    if [ -f "${EVAL_PREDICTIONS_SEEN}" ]; then
        echo -e "    ${GREEN}✓ Skipping completed evaluation for SEEN test set.${NC}"
    else
        python src/idiom_experiment.py \
            --mode evaluate \
            --model_checkpoint "${MODEL_DIR}" \
            --data "${SEEN_TEST_DATA}" \
            --task "${TASK_TYPE}" \
            --device "${DEVICE}" \
            --output_dir "${OUT_DIR_SEEN}" > /dev/null
        echo "    Saved to: ${OUT_DIR_SEEN}"
    fi

    # -----------------------------
    # 2. Evaluate on UNSEEN Test Set
    # -----------------------------
    OUT_DIR_UNSEEN="${EVAL_BASE_DIR}/unseen_test/${MODEL_NAME}/${TASK_TYPE}/${SEED_DIR_NAME}"
    EVAL_PREDICTIONS_UNSEEN="${OUT_DIR_UNSEEN}/eval_predictions.json" # Check for this file
    
    echo "  > Evaluating on UNSEEN test set..."
    
    if [ -f "${EVAL_PREDICTIONS_UNSEEN}" ]; then
        echo -e "    ${GREEN}✓ Skipping completed evaluation for UNSEEN test set.${NC}"
    else
        python src/idiom_experiment.py \
            --mode evaluate \
            --model_checkpoint "${MODEL_DIR}" \
            --data "${UNSEEN_TEST_DATA}" \
            --task "${TASK_TYPE}" \
            --device "${DEVICE}" \
            --output_dir "${OUT_DIR_UNSEEN}" > /dev/null
        echo "    Saved to: ${OUT_DIR_UNSEEN}"
    fi
    
done

echo "========================================"
echo "All evaluations complete!"
echo "Results saved in: ${EVAL_BASE_DIR}"
# Sync to Google Drive if available
if command -v rclone &> /dev/null && rclone listremotes | grep -q "gdrive:"; then
    echo "Syncing results to Google Drive..."
    bash scripts/sync_to_gdrive.sh
else
    echo "Don't forget to run: bash scripts/sync_to_gdrive.sh"
fi
echo "========================================"
