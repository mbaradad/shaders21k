#!/bin/bash
echo "This script does not work anymore, as codes have been put offline"
mkdir -p shader_codes

wget -O all_codes.zip http://data.csail.mit.edu/synthetic_training/shaders21k/all_codes.zip
yes | unzip all_codes.zip
rm all_codes.zip
