#!/bin/bash
mkdir -p shader_codes

wget -O shader_codes/all_codes.zip http://data.csail.mit.edu/synthetic_training/shaders21k/all_codes.zip
unzip shader_codes/all_codes.zip
rm shader_codes/all_codes.zip