# Procedural Image Programs for Representation Learning

This repo contains code and datasets for the paper _Procedural Image Programs for Representation Learning_
 (NeurIPS 2022). 

<p align="center">
  <img width="100%" src="https://mbaradad.github.io/shaders21k/images/teaser.png">
</p>

[[Project page](https://mbaradad.github.io/shaders21k)] 
[[Paper](https://arxiv.org/abs/2211.16412)]

# Requirements
For the main training logic, the requirements can be installed 
```pip install requirements.txt```. OpenCV is also required, which can be just installed with conda, from conda-forge channel:
```conda install -c conda-forge opencv```.

To render with the shaders with OpenGL and GPU, NVIDIA cards supporting CUDA should be able to render by default. 

# Download data and models
To download data and models, run the appropriate script (X=datasets/encoders/stylegan_generators) as:
```
./scripts/download/download_$X.sh
```
IMPORTANT: All datasets we provide are unsorted (as used for contrastive methods). The folder structure does not reflect classes/shader ids, and it is only for convenience 
(e.g. faster naviagation of the folder structure on network disks). 
To train with shader id as class (e.g. for training classification with cross entropy) the folder structure cannot be used, and you will need to use the rendering funcitonalities we provide
to generate a dataset with N images per class. 

Additional datasets used in our previous paper [Learning to See by Looking at Noise](https://github.com/mbaradad/learning_with_noise)
can be similarly downloaded with the scripts from that repository.

# Data generation
The shader generation has been tested with TitanRTX with CUDA 12.4 and drivers 550.90.07. Older drivers and CUDA should work, but issues have been found with the current implementation with 3090.


The main rendering functionality for shaders is under ```image_generation/shaders/renderer_moderngl.py```. 
This script can be used to render data (see the __main__ in the script).

If you want to generate data for a single shader (as used in the paper, which generates samples at random), you can use the utility ```image_generation/shaders/generate_data_single_shader.py```, for example as:

```
python image_generation/shaders/generate_data_single_shader.py --shader-file shader_codes/shadertoy/W/Wd2XWy.fragment --n-samples 105000 --resolution 256 --output-path shader_images/Wd2XWy
```

Additionally, we provide an utility to generate a single video for a given shader, under```image_generation/shaders/generate_video_single_shader.py```, which can be analogously used as:

```
python image_generation/shaders/generate_video_single_shader.py --shader-file shader_codes/shadertoy/W/Wd2XWy.fragment --duration 30 --output-path Wd2XWy.mp4 --fps 24 --resolution 512
```


To generate from StyleGAN, first download the GAN models (or train them yourself with the datasets available), and use:

```
image_generation/stylegan2/generate_large_scale_dataset_stylegan2_mixup_6.sh
```


# Training
The main training scripts borrow most of their logic from 
[SupCon](https://github.com/HobbitLong/SupContrast),
[MoCo_v2](https://github.com/facebookresearch/moco),
and [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch),
with minor modifications.

### Live training
To generate data while training, see moco/main_train.py, with parameters ```--dataset-type shaders/shader_list```. Shader codes and datasets can be downloaded using the scripts under ```./download/download_shader_codes.sh``` and ```./download/download_datasets.sh```

# Citation
```
@inproceedings{baradad2022procedural,
        title={Procedural Image Programs for Representation Learning},
        author={Manel Baradad and Chun-Fu Chen and Jonas Wulff and Tongzhou Wang and Rogerio Feris and Antonio Torralba and Phillip Isola},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=wJwHTgIoE0P}
}
```


