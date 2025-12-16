#!/bin/bash
# ============================================================================
# Download Best Model Checkpoints Only
# ============================================================================
# Purpose: Efficiently download ONLY the final/best model weights.
# Skips intermediate 'checkpoint-X' folders and logs to save time/space.
#
# Downloads structure:
# experiments/results/full_fine-tuning/MODEL/TASK/SEED/
#   ├── model.safetensors  (The best model)
#   ├── config.json
#   ├── tokenizer.json
#   └── ...
#
# Usage: bash scripts/download_best_checkpoints.sh
# ============================================================================

set -e

# Configuration
GDRIVE_PATH="gdrive:Hebrew_Idiom_Detection/results/full_fine-tuning"
LOCAL_PATH="experiments/results/full_fine-tuning"

echo "========================================"
echo "  Download Best Model Checkpoints"
echo "  (Optimized: Skips intermediate steps)"
echo "========================================"
echo ""

# Check rclone
if ! command -v rclone &> /dev/null;
    then
    echo "Error: rclone not found."
    exit 1
fi

echo "Source: ${GDRIVE_PATH}"
echo "Dest:   ${LOCAL_PATH}"
echo ""
echo "Downloading..."

# We use 'rclone copy' with filters to:
# 1. Exclude 'checkpoint-N' directories (intermediate saves)
# 2. Exclude 'logs' directories (TensorBoard logs)
# 3. Include everything else (the model files at the seed root)
rclone copy "${GDRIVE_PATH}" "${LOCAL_PATH}" \
    --exclude "checkpoint- નથી"
    --exclude "logs/"
    --exclude "full_fine-tuning/"
    --verbose \
    --transfers 16 \
    --update

echo ""
echo "========================================"
echo "✅ Download complete."
echo "Only the best model weights were downloaded."
echo "You can now run: bash scripts/run_evaluation_batch.sh"
echo "========================================