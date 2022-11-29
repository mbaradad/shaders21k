#!/usr/bin/env bash
python train.py --outdir=./shaders-training-runs --gpus=4 --gamma=100 --augpipe=bgcfnc --metrics fid50k \
    --data=/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/datasets_dumped/twigl_mix_none_256/train