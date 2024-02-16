#!/bin/bash

# Base URL for downloading videos
BASE_URL="http://data.csail.mit.edu/synthetic_training/shaders21k/shaders21k_videos"

# Loop through each line in the 'all_videos' file
while IFS= read -r VIDEO_PATH; do
    # Extract the directory path from VIDEO_PATH
    DIRECTORY_PATH=$(dirname "$VIDEO_PATH")
    
    # Create the directory structure if it does not exist
    mkdir -p "videos/$DIRECTORY_PATH"
    
    # Extract the file name from VIDEO_PATH
    FILE_NAME=$(basename "$VIDEO_PATH")
    
    echo "Downloading $VIDEO_PATH"
    
    # Download the video file into the correct directory
    wget -O "videos/$VIDEO_PATH" "$BASE_URL/$VIDEO_PATH"
done < scripts/download/all_videos