# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

# modification over generate.py

import sys
sys.path.append('../..')

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from dataset.samples_mixer import *
# from my_python_utils.common_utils import *

from torchvision.datasets.folder import pil_loader

def get_folder_and_filename(base_folder, i):
  folder = base_folder + '/' + str((i // 1000) * 1000).zfill(10)
  filename = '{}/{}.jpg'.format(folder, str(i).zfill(6))

  return folder, filename

import legacy

LARGE_SCALE_SAMPLES=1300000

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl',
              default='/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/image_generation/stylegan2/shaders-21k-best-gamma/00000-shaders21k_256x256-auto4-gamma5-bgcfnc/network-snapshot-025000.pkl',
              help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--gpu', help='Gpu to use', type=int, default=1, show_default=True)
@click.option('--mixup', help='n samples to mix', type=int, default=1, show_default=True)
@click.option('--outdir', help='output_directory', type=str, default='/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/datasets_post_icml/stylegan2_shaders21k', show_default=True)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    gpu: int,
    mixup: int,
    outdir: str
):
    assert mixup > 0, "mixup parameter must be > 0"

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:' + str(gpu))
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if mixup > 1:
        outdir += '_mixup_convex_uniform_n_samples_' + str(mixup)
    os.makedirs(outdir, exist_ok=True)

    # Labels.n
    label = torch.zeros([1, G.c_dim], device=device)
    assert G.c_dim == 0, "Only implemented for unconditional!"

    seeds = list(range(LARGE_SCALE_SAMPLES))

    import random
    random.shuffle(seeds)

    if mixup > 1:
        samples_mixer = ConvexSamplesMixer(mixup, convex_combination_type='uniform', dirichlet_alpha=1.0)

    # Generate images.
    for seed_idx, seed in enumerate(tqdm(seeds)):
        folder, filename = get_folder_and_filename(outdir, seed)
        error = False
        if os.path.exists(filename):
            try:
                if os.path.exists(filename):
                    # open image, if it fails create again
                    img = pil_loader(filename)
                    assert img.size == (256, 256)
                    continue
            except:
                print("Error in an image {}, will regenerate!".format(seed))
                error = True
                pass

        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        def generate_image(seed):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            return img

        if mixup == 1:
            img = generate_image(seed)
            pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        else:
            mixup_seeds = [seed + i * LARGE_SCALE_SAMPLES for i in range(mixup)]
            imgs = [generate_image(s) for s in mixup_seeds]

            pil_imgs = [PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB') for img in imgs]
            mixed_img, _ = samples_mixer.mix_samples(pil_imgs, [0 for _ in range(mixup)])
            pil_img = mixed_img

        os.makedirs(folder, exist_ok=True)
        if os.path.exists(filename) and not error:
            continue
        pil_img.save(filename)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
