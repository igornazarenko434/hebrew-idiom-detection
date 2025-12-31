#!/bin/bash
# ============================================================================
# Download Evaluation Results (Seen vs Unseen)
# ============================================================================
# Purpose: Download evaluation JSONs for generalization analysis
# Usage: bash scripts/download_evaluation_results.sh
# ============================================================================

set -e

# Configuration
GDRIVE_PATH="gdrive:Hebrew_Idiom_Detection/results/evaluation"
LOCAL_PATH="experiments/results/evaluation"

echo "========================================"
echo "  Download Evaluation Results"
echo "========================================"
echo ""

# Check rclone
if ! command -v rclone &> /dev/null;
    then
    echo "Error: rclone not found."
    exit 1
fi

echo "Downloading JSON files from Google Drive..."
echo "Source: ${GDRIVE_PATH}"
echo "Dest:   ${LOCAL_PATH}"
echo ""

# Download everything in evaluation folder (it's mostly JSONs anyway)
rclone copy "${GDRIVE_PATH}" "${LOCAL_PATH}" \
    --include "*.json" \
    --verbose \
    --transfers 8

echo ""
echo "✅ Download complete."
echo "You can now run: python src/analyze_generalization.py"
