# Procedural Image Programs for Representation Learning

This repo contains code and scripts to download datasets for the paper _Procedural Image Programs for Representation Learning_
 (NeurIPS 2022). On a future release, we will provide easier access to the shader codes, but for now we require contacting the authors for access. Rendered data and models are already available.

<p align="center">
  <img width="100%" src="https://mbaradad.github.io/shaders21k/images/teaser.png">
</p>

[[Project page](https://mbaradad.github.io/shaders21k)] 
[[Paper](https://openreview.net/pdf?id=wJwHTgIoE0P)]

# Requirements
For the main training logic, the requirements can be installed 
```pip install requirements.txt```. OpenCV is also required, which can be just installed with conda, from conda-forge channel:
```conda install -c conda-forge opencv```.

To render with the shaders with OpenGL and GPU, NVIDIA cards supporting CUDA should be able to render by default. 

# Download data and models
To download data and models, run the appropriate script under ```scripts```.

Additional datasets used in our previous paper [Learning to See by Looking at Noise](https://github.com/mbaradad/learning_with_noise)
can be similarly downloaded with the scripts from  the repo. 

# Data generation

The main rendering functionality for shaders is under ```image_generation/shaders/renderer_moderngl.py```. 
To either render the samples or to train live using moco, the shader codes are required. Contact the authors for access to the original codes. 

To generate from StyleGAN, first download the GAN models (or train them yourself with the datasets available), and use
```image_generation/stylegan2/generate_large_scale_dataset_stylegan2_mixup_6.sh```.

# Training
The main training scripts borrow most of their logic from 
[SupCon](https://github.com/HobbitLong/SupContrast),
[MoCo_v2](https://github.com/facebookresearch/moco),
and [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch),
with minor modifications.

### Live training
To generate data while training, see moco/main_train.py, with parameters ```--dataset-type shaders/shader_list```. Access to the shaders and the shader_list must be requested to the authors for now. 


## Access to the shader codes
The shader codes can be downloaded using the id's under shader_ids/. 
Additionally, we provide them pre-packed by completing the following form:



