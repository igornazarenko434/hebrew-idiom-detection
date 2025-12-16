#!/bin/bash
# ============================================================================
# Download Trained Checkpoints from Google Drive
# ============================================================================
# Purpose: Download full model checkpoints (weights) for evaluation
# WARNING: This can download a lot of data (GBs)!
# Usage: bash scripts/download_checkpoints.sh
# ============================================================================

set -e

# Configuration
GDRIVE_PATH="gdrive:Hebrew_Idiom_Detection/results/full_fine-tuning"
LOCAL_PATH="experiments/results/full_fine-tuning"

echo "========================================"
echo "  Download Trained Checkpoints"
echo "========================================"
echo ""

# Check rclone
if ! command -v rclone &> /dev/null; then
    echo "Error: rclone not found."
    echo "Please run this on the Vast.ai instance after setup."
    exit 1
fi

echo -e "\033[1;33m⚠️  WARNING: This will download ALL trained model checkpoints.\033[0m"
echo "Ensure you have enough disk space on your volume."
echo ""
read -p "Continue? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Downloading checkpoints from Google Drive..."
echo "Source: ${GDRIVE_PATH}"
echo "Dest:   ${LOCAL_PATH}"
echo ""

# Create destination
mkdir -p "${LOCAL_PATH}"

# Download all files including large weights
# We use --update to skip existing matching files
rclone copy "${GDRIVE_PATH}" "${LOCAL_PATH}" \
    --verbose \
    --transfers 8 \
    --update

echo ""
echo "✅ Download complete."
echo "You can now run evaluation scripts on trained models."
