#!/bin/bash
mkdir -p data

datasets=('shaders1k' \
          'shaders21k' \
          'shaders21k_6mixup' \
          'shaders21k_stylegan' \
           )

for DATASET in ${datasets[@]}
do
    echo "Downloading $DATASET"
    wget -O data/$DATASET.zip http://data.csail.mit.edu/synthetic_training/shaders21k/zipped_data/$DATASET.zip
    yes | unzip data/$DATASET.zip -d data/$DATASET
    rm data/$DATASET.zip
done