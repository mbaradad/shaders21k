import numpy as np
from PIL import Image

class SamplesMixer():
  def __init__(self, n_samples_mix):
    self.n_samples_mix = n_samples_mix
    assert self.n_samples_mix > 1, "Mix sample mode is activated, but n_mixes is < 2, (n_mixes = {})".format(self.n_samples_mix)

  def get_n_samples(self):
    return self.n_samples_mix

  def mix_samples(self):
    raise Exception("Not implemented")

  def mix_program_is(self, program_is):
    return np.array(program_is)

class ConvexSamplesMixer(SamplesMixer):
  def __init__(self, n_samples_mix, convex_combination_type='non-uniform', dirichlet_alpha=1.0, **kwargs):
    super().__init__(n_samples_mix)

    convex_combination_type_choices = ['uniform', 'non-uniform', 'dirichlet']
    assert convex_combination_type in convex_combination_type_choices, "Invalid mix-sampling: {}, choices: {}".format(convex_combination_type, convex_combination_type_choices)
    self.mix_sampling = convex_combination_type
    self.dirichlet_alpha = dirichlet_alpha

  def mix_samples(self, pil_images, program_is):
    if self.mix_sampling == 'non-uniform':
      # first naive approach which is neither uniform nor clear form Dirichlet, not sure if it has a name or a Dirichlet equivalent
      mixing_probabilities = np.random.uniform(0, 1, self.n_samples_mix)
      mixing_probabilities = mixing_probabilities / mixing_probabilities.sum()
    else:
      if self.mix_sampling == 'uniform':
        alphas = [1 for _ in range(self.n_samples_mix)]
      else:
        alphas = [self.dirichlet_alpha for _ in range(self.n_samples_mix)]
      mixing_probabilities = np.random.dirichlet(alphas)


    mixed_image = np.array(pil_images[0]) * mixing_probabilities[0]
    for mix_i in range(1, self.n_samples_mix):
      current_pil_image = pil_images[mix_i]
      mixed_image += np.array(current_pil_image) * mixing_probabilities[mix_i]

    mixed_image = np.array(mixed_image, dtype='uint8')
    pil_image = Image.fromarray(mixed_image)

    """
    # implementation with PIL.Image.blend. Not finished, check it is correct.
    sum_mix = mixing_probabilities[0]
    for i in range(len(pil_images) - 1):
      mix_prob = mixing_probabilities[i + 1]
      # if 0, returns first image, if 1 returns second image
      mixed_image = Image.blend(pil_images[i], pil_images[i + 1],  mix_prob / (sum_mix + mix_prob))
      sum_mix += mix_prob
    pil_image = mixed_image
    """

    return pil_image, self.mix_program_is(program_is)

class CutMixSamplesMixer(SamplesMixer):
  def __init__(self, n_samples_mix, alpha=1.0, **kwargs):
    super().__init__(n_samples_mix)
    self.alpha = alpha

  # https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py

  def add_one_cut(self, canvas, image):
    assert type(canvas) is np.ndarray and type(image) is np.ndarray, "Both canvas and the image should be a np.array."

  def mix_samples(self, pil_images, program_is):
    mixed_image = np.array(pil_images[0])

    for pil_image in pil_images:
      lam = np.random.beta(self.alpha, self.alpha)
      current_image = np.array(pil_image)

      image_h, image_w = current_image.shape[:-1]
      cx = np.random.uniform(0, image_w)
      cy = np.random.uniform(0, image_h)
      w = image_w * np.sqrt(1 - lam)
      h = image_h * np.sqrt(1 - lam)
      x0 = int(np.round(max(cx - w / 2, 0)))
      x1 = int(np.round(min(cx + w / 2, image_w)))
      y0 = int(np.round(max(cy - h / 2, 0)))
      y1 = int(np.round(min(cy + h / 2, image_h)))

      mixed_image[y0:y1, x0:x1] = current_image[y0:y1, x0:x1]

    pil_image = Image.fromarray(mixed_image)
    return pil_image, self.mix_program_is(program_is)

class ConvexCutMixSamplesMixer(SamplesMixer):
  def __init__(self, n_samples_mix, **kwargs):
    super().__init__(n_samples_mix)
    self.cutmix_mixer = CutMixSamplesMixer(n_samples_mix, **kwargs)
    self.convex_mixer = ConvexSamplesMixer(n_samples_mix, **kwargs)

  def mix_samples(self, pil_images, program_is):
    # first produce cut mixes, by convining with a white image.
    white_pil_image = Image.fromarray(np.zeros((pil_images[0].height, pil_images[0].width, 3), dtype='uint8'))
    cut_pil_images = [pil_images[0]]
    for pil_image in pil_images[1:]:
      cut_pil_images.append(self.cutmix_mixer.mix_samples([white_pil_image, pil_image], program_is)[0])

    mixed_pil_image, program_i = self.convex_mixer.mix_samples(cut_pil_images, program_is)
    return mixed_pil_image, program_i


def get_sample_mixer(opt):
  if opt.mixing_type == 'none':
    return None
  return mixername_to_class[opt.mixing_type](**opt.__dict__)

mixername_to_class = {'convex': ConvexSamplesMixer,
                      'cutmix': CutMixSamplesMixer,
                      'convex_cutmix': ConvexCutMixSamplesMixer }