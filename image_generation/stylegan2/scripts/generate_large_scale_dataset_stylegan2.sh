#!/usr/bin/env bash

# to generate samples from a GAN
python dataset_generation.py --gpu 0 &
python dataset_generation.py --gpu 0 &
python dataset_generation.py --gpu 0 &
python dataset_generation.py --gpu 0 &
python dataset_generation.py --gpu 1 &
python dataset_generation.py --gpu 1 &
python dataset_generation.py --gpu 1 &
python dataset_generation.py --gpu 1 &
python dataset_generation.py --gpu 2 &
python dataset_generation.py --gpu 2 &
python dataset_generation.py --gpu 2 &
python dataset_generation.py --gpu 2 &
python dataset_generation.py --gpu 3 &
python dataset_generation.py --gpu 3 &
python dataset_generation.py --gpu 3 &
python dataset_generation.py --gpu 3 &
wait