#!/bin/bash
mkdir -p encoders

datasets=('shaders1k' \
          'shaders21k' \
          'shaders21k_6mixup' \
          'shaders21k_stylegan' \
          'shaders21k_4mixup_live' \
           )

for DATASET in ${datasets[@]}
do
    echo "Downloading $DATASET"
    wget -O encoders/$DATASET/checkpoint_0199.pth.tar http://data.csail.mit.edu/synthetic_training/shaders21k/models/encoders/$DATASET/checkpoint_0199.pth.tar
done