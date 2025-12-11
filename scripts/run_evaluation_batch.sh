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
    CURRENT=$((CURRENT + 1))
    
    # Extract metadata from path
    # Path is: experiments/results/full_fine-tuning/MODEL/TASK/SEED/model.safetensors
    # dirname gets folder: .../SEED
    MODEL_DIR=$(dirname "${model_path}")
    
    # Extract parts
    SEED_DIR_NAME=$(basename "${MODEL_DIR}") # seed_42
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
    echo "  > Evaluating on SEEN test set..."
    
    python src/idiom_experiment.py \
        --mode evaluate \
        --model_checkpoint "${MODEL_DIR}" \
        --data "${SEEN_TEST_DATA}" \
        --task "${TASK_TYPE}" \
        --device "${DEVICE}" \
        --output_dir "${OUT_DIR_SEEN}" > /dev/null
        
    echo "    Saved to: ${OUT_DIR_SEEN}"

    # -----------------------------
    # 2. Evaluate on UNSEEN Test Set
    # -----------------------------
    OUT_DIR_UNSEEN="${EVAL_BASE_DIR}/unseen_test/${MODEL_NAME}/${TASK_TYPE}/${SEED_DIR_NAME}"
    echo "  > Evaluating on UNSEEN test set..."
    
    python src/idiom_experiment.py \
        --mode evaluate \
        --model_checkpoint "${MODEL_DIR}" \
        --data "${UNSEEN_TEST_DATA}" \
        --task "${TASK_TYPE}" \
        --device "${DEVICE}" \
        --output_dir "${OUT_DIR_UNSEEN}" > /dev/null

    echo "    Saved to: ${OUT_DIR_UNSEEN}"
    echo ""
    
done

echo "========================================"
echo "All evaluations complete!"
echo "Results saved in: ${EVAL_BASE_DIR}"
echo "Don't forget to run: bash scripts/sync_to_gdrive.sh"
echo "========================================"
