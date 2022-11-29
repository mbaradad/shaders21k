#!/usr/bin/env bash

python dataset_generation.py --gpu 0 --mixup 6 &
python dataset_generation.py --gpu 0 --mixup 6 &
python dataset_generation.py --gpu 0 --mixup 6 &
python dataset_generation.py --gpu 0 --mixup 6 &
python dataset_generation.py --gpu 1 --mixup 6 &
python dataset_generation.py --gpu 1 --mixup 6 &
python dataset_generation.py --gpu 1 --mixup 6 &
python dataset_generation.py --gpu 1 --mixup 6 &
python dataset_generation.py --gpu 2 --mixup 6 &
python dataset_generation.py --gpu 2 --mixup 6 &
python dataset_generation.py --gpu 2 --mixup 6 &
python dataset_generation.py --gpu 2 --mixup 6 &
python dataset_generation.py --gpu 3 --mixup 6 &
python dataset_generation.py --gpu 3 --mixup 6 &
python dataset_generation.py --gpu 3 --mixup 6 &
python dataset_generation.py --gpu 3 --mixup 6 &
wait


image_generation/stylegan2/generate_large_scale_dataset_stylegan2_mixup_6.sh