#!/bin/bash
mkdir -p generators

models=('shaders21k-256x256-gamma-5' \
        'shaders21k-512x512-gamma-40' \
        )

for MODEL in ${models[@]}
do
    echo "Downloading StyleGAN model $MODEL"
    mkdir -p generators/$MODEL
wget -O generators/$MODEL/network-snapshot-025000.pkl http://data.csail.mit.edu/synthetic_training/shaders21k/models/generators/$MODEL/network-snapshot-025000.pkl
done
