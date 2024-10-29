#!/bin/bash

# Check for git-lfs installation
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs not found. Please install it from https://git-lfs.github.com/ and rerun this script."
    exit 1
fi

# Initialize git-lfs
git lfs install

# Clone the repository
git clone https://huggingface.co/microsoft/OmniParser

# Move specified folders to /weights
mkdir -p /weights
mv OmniParser/icon_caption_blip2 /weights/
mv OmniParser/icon_caption_florence /weights/
mv OmniParser/icon_detect /weights/

# Clean up cloned repo
rm -rf OmniParser

# Run convert_safetensor_to_pt.py in /weights
if [[ -f /weights/convert_safetensor_to_pt.py ]]; then
    python3 /weights/convert_safetensor_to_pt.py
else
    echo "convert_safetensor_to_pt.py not found in /weights."
    exit 1
fi

echo "Folders moved to /weights and conversion script executed successfully."
