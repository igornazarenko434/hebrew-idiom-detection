#!/bin/bash
# ============================================================================
# Sync Results to Google Drive (Automated with rclone)
# ============================================================================
# Purpose: Upload model checkpoints, results, and logs to Google Drive
# Usage: bash scripts/sync_to_gdrive.sh
# Requirements: rclone installed and configured (see setup instructions below)
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "Syncing Results to Google Drive"
echo "========================================"

# ============================================================================
# FIRST TIME SETUP (Run this section only once)
# ============================================================================
# If rclone is not configured yet, follow these steps:
#
# 1. Install rclone:
#    curl https://rclone.org/install.sh | sudo bash
#
# 2. Configure Google Drive:
#    rclone config
#    - Choose: n (new remote)
#    - Name: gdrive
#    - Storage: drive (Google Drive)
#    - client_id: (press Enter to skip)
#    - client_secret: (press Enter to skip)
#    - scope: 1 (Full access)
#    - root_folder_id: (press Enter to skip)
#    - service_account_file: (press Enter to skip)
#    - Edit advanced config: n
#    - Use auto config: n (because you're on a remote server)
#    - Follow the instructions to authenticate via browser on your local machine
#    - Paste the verification code back into the terminal
#
# 3. Test connection:
#    rclone lsd gdrive:
#    (Should list your Google Drive folders)
#
# 4. Create project folder in Google Drive (if not exists):
#    rclone mkdir gdrive:Hebrew_Idiom_Detection
#    rclone mkdir gdrive:Hebrew_Idiom_Detection/results
#    rclone mkdir gdrive:Hebrew_Idiom_Detection/logs
#    rclone mkdir gdrive:Hebrew_Idiom_Detection/models
#
# ============================================================================

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}Error: rclone is not installed${NC}"
    echo ""
    echo "Install rclone by running:"
    echo "  curl https://rclone.org/install.sh | sudo bash"
    echo ""
    echo "Then configure it with:"
    echo "  rclone config"
    echo "  (Choose: gdrive as remote name, Google Drive as storage type)"
    exit 1
fi

# Check if rclone is configured with gdrive remote
if ! rclone listremotes | grep -q "gdrive:"; then
    echo -e "${RED}Error: rclone 'gdrive' remote not configured${NC}"
    echo ""
    echo "Configure rclone with:"
    echo "  rclone config"
    echo ""
    echo "Follow the prompts to add Google Drive as 'gdrive' remote."
    echo "See script header for detailed instructions."
    exit 1
fi

echo -e "${GREEN}✓ rclone is installed and configured${NC}"
echo ""

# ============================================================================
# SYNC CONFIGURATION
# ============================================================================

# Google Drive paths (matching .env file structure)
GDRIVE_BASE="gdrive:Hebrew_Idiom_Detection"
GDRIVE_RESULTS="${GDRIVE_BASE}/results"
GDRIVE_LOGS="${GDRIVE_BASE}/logs"
GDRIVE_MODELS="${GDRIVE_BASE}/models"

# Local paths
LOCAL_RESULTS="experiments/results"
LOCAL_LOGS="experiments/logs"
LOCAL_MODELS="models"

# Sync options
# --update: Skip files that are newer on destination
# --verbose: Show files being transferred
# --progress: Show progress during transfer
# --transfers 4: Use 4 parallel transfers
# --checkers 8: Use 8 parallel checkers
# --exclude: Patterns to exclude

RCLONE_OPTS="--update --verbose --progress --transfers 4 --checkers 8"

# Exclude patterns (don't sync these)
EXCLUDE_PATTERNS="--exclude .DS_Store --exclude __pycache__/ --exclude *.pyc --exclude .ipynb_checkpoints/"

echo "Sync Configuration:"
echo "  Google Drive: ${GDRIVE_BASE}"
echo "  Local Results: ${LOCAL_RESULTS}"
echo "  Local Logs: ${LOCAL_LOGS}"
echo "  Local Models: ${LOCAL_MODELS}"
echo ""

# ============================================================================
# CREATE GOOGLE DRIVE FOLDERS (if they don't exist)
# ============================================================================

echo "Step 1/4: Ensuring Google Drive folders exist..."
echo "-------------------------------------------------"

rclone mkdir "${GDRIVE_BASE}" 2>/dev/null || true
rclone mkdir "${GDRIVE_RESULTS}" 2>/dev/null || true
rclone mkdir "${GDRIVE_LOGS}" 2>/dev/null || true
rclone mkdir "${GDRIVE_MODELS}" 2>/dev/null || true

echo -e "${GREEN}✓ Google Drive folders ready${NC}"
echo ""

# ============================================================================
# SYNC RESULTS
# ============================================================================

echo "Step 2/4: Syncing results..."
echo "----------------------------"

if [ -d "${LOCAL_RESULTS}" ]; then
    echo "Uploading: ${LOCAL_RESULTS}/ → ${GDRIVE_RESULTS}/"

    rclone copy "${LOCAL_RESULTS}/" "${GDRIVE_RESULTS}/" ${RCLONE_OPTS} ${EXCLUDE_PATTERNS}

    echo -e "${GREEN}✓ Results synced${NC}"
else
    echo -e "${YELLOW}⚠ No results directory found (${LOCAL_RESULTS})${NC}"
fi

echo ""

# ============================================================================
# SYNC LOGS
# ============================================================================

echo "Step 3/4: Syncing logs..."
echo "-------------------------"

if [ -d "${LOCAL_LOGS}" ]; then
    echo "Uploading: ${LOCAL_LOGS}/ → ${GDRIVE_LOGS}/"

    rclone copy "${LOCAL_LOGS}/" "${GDRIVE_LOGS}/" ${RCLONE_OPTS} ${EXCLUDE_PATTERNS}

    echo -e "${GREEN}✓ Logs synced${NC}"
else
    echo -e "${YELLOW}⚠ No logs directory found (${LOCAL_LOGS})${NC}"
fi

echo ""

# ============================================================================
# SYNC MODELS (OPTIONAL - Models are large, sync only if requested)
# ============================================================================

echo "Step 4/4: Syncing model checkpoints..."
echo "---------------------------------------"

# Only sync models if explicitly requested (to save time/bandwidth)
if [ "$1" == "--with-models" ]; then
    if [ -d "${LOCAL_MODELS}" ]; then
        echo "Uploading: ${LOCAL_MODELS}/ → ${GDRIVE_MODELS}/"
        echo -e "${YELLOW}⚠ Warning: This may take a while (models are large)${NC}"

        rclone copy "${LOCAL_MODELS}/" "${GDRIVE_MODELS}/" ${RCLONE_OPTS} ${EXCLUDE_PATTERNS}

        echo -e "${GREEN}✓ Models synced${NC}"
    else
        echo -e "${YELLOW}⚠ No models directory found (${LOCAL_MODELS})${NC}"
    fi
else
    echo -e "${BLUE}ℹ Skipping model checkpoints (use --with-models to include)${NC}"
    echo "  Note: Models are large. Usually only sync results and logs."
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "========================================"
echo -e "${GREEN}✓ Sync to Google Drive complete!${NC}"
echo "========================================"
echo ""

# Show what was synced
echo "Synced directories:"
rclone size "${GDRIVE_RESULTS}" 2>/dev/null | grep "Total size:" || echo "  Results: Unknown size"
rclone size "${GDRIVE_LOGS}" 2>/dev/null | grep "Total size:" || echo "  Logs: Unknown size"

echo ""
echo "View in Google Drive:"
echo "  https://drive.google.com/drive/folders/Hebrew_Idiom_Detection"
echo ""
echo "To verify sync:"
echo "  rclone ls ${GDRIVE_RESULTS}"
echo ""
echo "Usage notes:"
echo "  - Run this script after each training/HPO run"
echo "  - Run before terminating VAST.ai instance"
echo "  - Use --with-models flag to also sync model checkpoints (slower)"
echo "    Example: bash scripts/sync_to_gdrive.sh --with-models"
echo ""
