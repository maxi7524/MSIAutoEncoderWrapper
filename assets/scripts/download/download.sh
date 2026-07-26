#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

### ============================================
### wget for test (PXD001283) - bladder
### ============================================

DATA_DIR='../../data'
# DEPRECATED 
# URL='ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2014/11/PXD001283'

# mkdir -p $DATA_DIR
# echo "Download files from ebi..."
# wget -P $DATA_DIR $URL/*

### ============================================
### wget for results on drive
### ============================================

# Google drive folder ID
FOLDER_ID="137s4wKLL4A6KfUGowqAlTIHkd2H-N_R4"

# Download all files from the Google Drive folder into data/
# Requires: gdown (pip install gdown)
echo "Download files from google drive..."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$DATA_DIR"

# Enable safe globbing (no match -> empty list)
shopt -s nullglob

# extract zip files
# Extract all ZIP files and remove them afterwards
for zip_file in "$DATA_DIR"/*.7z; do
    7z x "$zip_file" -o"$DATA_DIR" -y
    rm "$zip_file"
done

# rename files 


# Disable nullglob to avoid side effects
shopt -u nullglob