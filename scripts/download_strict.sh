#!/bin/bash
# ============================================================================
# Download Best Model Checkpoints (STRICT MODE)
# ============================================================================
# Purpose: Download ONLY essential model files. Ignores all folders.
#          Solves the "recursive folder" size explosion issue.
#
# Files included: .json, .txt, .bin, .safetensors
# ============================================================================

set -e

# Configuration
GDRIVE_PATH="gdrive:Hebrew_Idiom_Detection/results/full_fine-tuning"
LOCAL_PATH="experiments/results/full_fine-tuning"

echo "========================================"
echo "  Download Checkpoints (STRICT MODE)"
echo "========================================"

# Check rclone
if ! command -v rclone &> /dev/null; then
    echo "Error: rclone not found."
    exit 1
fi

echo "Source: ${GDRIVE_PATH}"
echo "Dest:   ${LOCAL_PATH}"
echo ""
echo "Downloading ONLY [.json, .txt, .bin, .safetensors]..."

# Whitelist approach: Only include specific files needed for loading models
rclone copy "${GDRIVE_PATH}" "${LOCAL_PATH}" \
    --include "*.json" \
    --include "*.txt" \
    --include "*.bin" \
    --include "*.safetensors" \
    --exclude "checkpoint-*/" \
    --exclude "logs/" \
    --verbose \
    --transfers 16 \
    --update

echo ""
echo "========================================"
echo "✅ Strict download complete."
echo "========================================"
