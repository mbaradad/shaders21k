#!/bin/bash

set -e  # Exit on error

SHADER_DIR="shader_codes"
ZIP_FILE="all_codes.zip"
DOWNLOADS_DIR="downloads"
GDRIVE_ID="1kIiBdeW9CEIfRlYOYTuxfWvN036k3Iig"
GDRIVE_URL="https://drive.google.com/uc?export=download&id=${GDRIVE_ID}"

echo "=========================================="
echo "Shaders21k Download Script"
echo "=========================================="
echo ""

# Create directories if they don't exist
mkdir -p "$SHADER_DIR"
mkdir -p "$DOWNLOADS_DIR"

# Check if codes are already extracted
if [ -d "$SHADER_DIR" ] && [ "$(ls -A $SHADER_DIR)" ]; then
    echo "Shader codes already exist in $SHADER_DIR/"
    read -p "Do you want to re-download and extract? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download. Exiting."
        exit 0
    fi
    rm -rf "$SHADER_DIR"/*
fi

# Check if zip file already exists in downloads/
ZIP_PATH=""
if [ -f "$DOWNLOADS_DIR/$ZIP_FILE" ]; then
    echo "Found $ZIP_FILE in $DOWNLOADS_DIR/"
    echo "Verifying file integrity..."
    
    if unzip -t "$DOWNLOADS_DIR/$ZIP_FILE" > /dev/null 2>&1; then
        echo "✓ File is valid. Using existing download."
        ZIP_PATH="$DOWNLOADS_DIR/$ZIP_FILE"
    else
        echo "✗ File is corrupted. Will re-download."
        rm -f "$DOWNLOADS_DIR/$ZIP_FILE"
    fi
fi

# If we don't have a valid zip file, try to download
if [ -z "$ZIP_PATH" ]; then
    echo ""
    echo "Attempting to download $ZIP_FILE..."
    echo ""
    
    download_success=false
    
    # Function to try downloading from Google Drive using gdown
    try_gdown() {
        if command -v gdown &> /dev/null; then
            echo "Attempting download using gdown..."
            if gdown "$GDRIVE_ID" -O "$DOWNLOADS_DIR/$ZIP_FILE" 2>&1; then
                return 0
            fi
        else
            echo "gdown not found. (Install with: pip install gdown)"
        fi
        return 1
    }
    
    # Function to try downloading using curl
    try_curl() {
        echo "Attempting download using curl..."
        if curl -L -o "$DOWNLOADS_DIR/$ZIP_FILE" "$GDRIVE_URL" 2>&1; then
            # Check if file is not empty and not HTML error page
            if [ -s "$DOWNLOADS_DIR/$ZIP_FILE" ]; then
                file_type=$(file -b "$DOWNLOADS_DIR/$ZIP_FILE" | cut -d' ' -f1)
                if [ "$file_type" = "Zip" ]; then
                    return 0
                fi
            fi
        fi
        rm -f "$DOWNLOADS_DIR/$ZIP_FILE"
        return 1
    }
    
    # Function to try downloading using wget
    try_wget() {
        echo "Attempting download using wget..."
        if wget --no-check-certificate -O "$DOWNLOADS_DIR/$ZIP_FILE" "$GDRIVE_URL" 2>&1; then
            # Check if file is not empty and not HTML error page
            if [ -s "$DOWNLOADS_DIR/$ZIP_FILE" ]; then
                file_type=$(file -b "$DOWNLOADS_DIR/$ZIP_FILE" | cut -d' ' -f1)
                if [ "$file_type" = "Zip" ]; then
                    return 0
                fi
            fi
        fi
        rm -f "$DOWNLOADS_DIR/$ZIP_FILE"
        return 1
    }
    
    # Try different download methods
    if try_gdown; then
        download_success=true
    elif try_curl; then
        download_success=true
    elif try_wget; then
        download_success=true
    fi
    
    # If automatic download failed, prompt for manual download
    if [ "$download_success" = false ] || [ ! -f "$DOWNLOADS_DIR/$ZIP_FILE" ] || [ ! -s "$DOWNLOADS_DIR/$ZIP_FILE" ]; then
        echo ""
        echo "=========================================="
        echo "AUTOMATIC DOWNLOAD FAILED"
        echo "=========================================="
        echo ""
        echo "Please download the file manually:"
        echo ""
        echo "1. Visit this link:"
        echo "   https://drive.google.com/file/d/1kIiBdeW9CEIfRlYOYTuxfWvN036k3Iig/view?usp=sharing"
        echo ""
        echo "2. Download the 'all_codes.zip' file"
        echo ""
        echo "3. Place it in: $(pwd)/$DOWNLOADS_DIR/"
        echo ""
        echo "4. Press Enter to continue once the file is ready"
        echo ""
        
        read -p "Press Enter once you've placed $ZIP_FILE in $DOWNLOADS_DIR/..."
        
        if [ ! -f "$DOWNLOADS_DIR/$ZIP_FILE" ]; then
            echo ""
            echo "ERROR: $ZIP_FILE not found in $DOWNLOADS_DIR/"
            echo "Exiting."
            exit 1
        fi
    fi
    
    ZIP_PATH="$DOWNLOADS_DIR/$ZIP_FILE"
fi

# Verify the zip file is valid
echo ""
echo "Verifying zip file integrity..."
if ! unzip -t "$ZIP_PATH" > /dev/null 2>&1; then
    echo "ERROR: $ZIP_FILE is corrupted or invalid."
    echo "Please download it again manually and place it in $DOWNLOADS_DIR/"
    rm -f "$ZIP_PATH"
    exit 1
fi

echo "✓ Zip file is valid"

# Check the structure of the zip file to determine extraction strategy
echo ""
echo "Checking zip file structure..."
TEMP_LIST=$(unzip -l "$ZIP_PATH" | head -20)

# Extract the zip file intelligently
echo "Extracting shader codes..."

# First, extract to a temporary location
TEMP_DIR=$(mktemp -d)
unzip -q -o "$ZIP_PATH" -d "$TEMP_DIR"

# Check if there's a single top-level directory named shader_codes
if [ -d "$TEMP_DIR/shader_codes" ] && [ $(ls -A "$TEMP_DIR" | wc -l) -eq 1 ]; then
    echo "Detected nested shader_codes directory, flattening structure..."
    # Move contents directly to target directory
    mv "$TEMP_DIR/shader_codes"/* "$SHADER_DIR/"
else
    echo "Moving files to $SHADER_DIR/..."
    # Move everything to target directory
    mv "$TEMP_DIR"/* "$SHADER_DIR/"
fi

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "SUCCESS!"
echo "=========================================="
echo "Shader codes extracted to: $SHADER_DIR/"
echo "Total files: $(find $SHADER_DIR -type f | wc -l)"
echo ""
echo "Note: Downloaded file kept in $DOWNLOADS_DIR/ for future use"
echo ""
