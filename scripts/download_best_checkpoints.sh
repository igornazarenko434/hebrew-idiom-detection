#!/bin/bash
# ============================================================================
# Download Best Model Checkpoints Only (STRICT MODE)
# ============================================================================
# Purpose: Efficiently download ONLY the final/best model weights.
#          Uses strict filtering to avoid "recursive folder" size explosions.
#
# Files included: .json, .txt, .bin, .safetensors (at root of seed folder)
# Files excluded: checkpoint-X folders, logs, nested duplicate folders
#
# Usage: bash scripts/download_best_checkpoints.sh
# ============================================================================

set -e

# Configuration
GDRIVE_PATH="gdrive:Hebrew_Idiom_Detection/results/full_fine-tuning"
LOCAL_PATH="experiments/results/full_fine-tuning"

echo "========================================"
echo "  Download Best Model Checkpoints"
echo "  (STRICT MODE: Whitelist only)"
echo "========================================"
echo ""

# Check rclone
if ! command -v rclone &> /dev/null; then
    echo "Error: rclone not found."
    exit 1
fi

echo "Source: ${GDRIVE_PATH}"
echo "Dest:   ${LOCAL_PATH}"
echo ""
echo "Downloading ONLY essential model files [.json, .txt, .bin, .safetensors]..."
echo "Ignoring logs, intermediate checkpoints, and nested folders."

# Strict rclone copy
# 1. Include only needed file extensions
# 2. Exclude the known recursive folder 'full_fine-tuning'
# 3. Exclude checkpoints and logs
rclone copy "${GDRIVE_PATH}" "${LOCAL_PATH}" \
    --include "*.json" \
    --include "*.txt" \
    --include "*.bin" \
    --include "*.safetensors" \
    --exclude "checkpoint-*/" \
    --exclude "logs/" \
    --exclude "full_fine-tuning/**" \
    --verbose \
    --transfers 16 \
    --update

echo ""
echo "========================================"
echo "✅ Download complete."
echo "Only the best model weights were downloaded."
echo "You can now run: bash scripts/run_evaluation_batch.sh"
echo "========================================
