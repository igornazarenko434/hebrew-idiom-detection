#!/bin/bash
# ============================================================================
# Download Dataset from Google Drive to VAST.ai Instance
# ============================================================================
# Purpose: Download train/validation/test splits from Google Drive
# Usage: bash scripts/download_from_gdrive.sh
# Requirements: gdown installed (pip install gdown)
# ============================================================================

set -e  # Exit on error

echo "========================================"
echo "Downloading Dataset from Google Drive"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo -e "${RED}Error: gdown is not installed${NC}"
    echo "Installing gdown..."
    pip install gdown
fi

# Google Drive File IDs (from .env file)
DATASET_CSV_FILE_ID="140zJatqT4LBl7yG-afFSoUrYrisi9276"

# Create data directory if it doesn't exist
mkdir -p data
mkdir -p data/splits

echo ""
echo "Step 1/3: Downloading main dataset (CSV)..."
echo "-------------------------------------------"

# Download main CSV dataset
gdown ${DATASET_CSV_FILE_ID} -O data/expressions_data_tagged.csv

# Verify download
if [ -f "data/expressions_data_tagged.csv" ]; then
    file_size=$(wc -c < "data/expressions_data_tagged.csv")
    echo -e "${GREEN}✓ Main dataset downloaded successfully${NC}"
    echo "  File size: ${file_size} bytes"

    # Count rows (should be 4800 + 1 header)
    row_count=$(wc -l < data/expressions_data_tagged.csv)
    expected_rows=4801

    if [ "$row_count" -eq "$expected_rows" ]; then
        echo -e "${GREEN}✓ Row count verified: $row_count rows (4800 + header)${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Expected $expected_rows rows, got $row_count${NC}"
    fi
else
    echo -e "${RED}✗ Failed to download main dataset${NC}"
    exit 1
fi

echo ""
echo "Step 2/3: Checking for split files..."
echo "--------------------------------------"

# Check if split files exist (they should be in GitHub repo)
if [ -f "data/splits/train.csv" ] && [ -f "data/splits/validation.csv" ] && [ -f "data/splits/test.csv" ]; then
    echo -e "${GREEN}✓ Split files found (from repository)${NC}"

    # Count samples in each split
    train_count=$(($(wc -l < data/splits/train.csv) - 1))
    val_count=$(($(wc -l < data/splits/validation.csv) - 1))
    test_count=$(($(wc -l < data/splits/test.csv) - 1))

    echo "  Train: ${train_count} samples"
    echo "  Validation: ${val_count} samples"
    echo "  Test: ${test_count} samples"

    # Verify expected counts
    if [ "$train_count" -eq 3840 ] && [ "$val_count" -eq 480 ] && [ "$test_count" -eq 480 ]; then
        echo -e "${GREEN}✓ Split counts verified (3840/480/480)${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Expected 3840/480/480, got ${train_count}/${val_count}/${test_count}${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Split files not found in repository${NC}"
    echo "Note: Split files should be in your GitHub repository."
    echo "If missing, you'll need to run dataset_splitting.py locally and push to GitHub."
fi

echo ""
echo "Step 3/3: Verification"
echo "----------------------"

# List downloaded files
echo "Downloaded files:"
ls -lh data/*.csv 2>/dev/null || echo "  No CSV files in data/"
ls -lh data/splits/*.csv 2>/dev/null || echo "  No split files in data/splits/"

echo ""
echo "========================================"
echo -e "${GREEN}✓ Dataset download complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Verify GPU is available: python -c 'import torch; print(torch.cuda.is_available())'"
echo "2. Start training: python src/idiom_experiment.py --mode full_finetune --config experiments/configs/training_config.yaml"
echo ""
