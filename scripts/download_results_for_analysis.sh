#!/bin/bash
# ============================================================================
# Download Training Results (Lightweight)
# ============================================================================
# Purpose: Download only JSON metrics and logs for analysis
# Skips large model weights to save time and bandwidth
# Usage: bash scripts/download_results_for_analysis.sh
# ============================================================================

set -e

# Configuration
GDRIVE_PATH="gdrive:Hebrew_Idiom_Detection/results/full_fine-tuning"
LOCAL_PATH="experiments/results/full_fine-tuning"

echo "========================================"
echo "  Download Training Results (Analysis Only)"
echo "========================================"
echo ""

# Check rclone
if ! command -v rclone &> /dev/null; then
    echo "Error: rclone not found."
    exit 1
fi

echo "Downloading JSON files from Google Drive..."
echo "Source: ${GDRIVE_PATH}"
echo "Dest:   ${LOCAL_PATH}"
echo ""

# Use filter to include only json files and exclude everything else
# This avoids downloading 15GB of model weights
rclone copy "${GDRIVE_PATH}" "${LOCAL_PATH}" \
    --include "*.json" \
    --include "trainer_state.json" \
    --include "training_results.json" \
    --include "eval_results*.json" \
    --verbose \
    --transfers 8

echo ""
echo "✅ Download complete."
echo "You can now run: python src/analyze_finetuning_results.py"
