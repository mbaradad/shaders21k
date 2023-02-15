from utils import *

from image_generation.shaders.glsl_utils import *
from image_generation.shaders.on_the_fly_moderngl_shader import ModernGLOnlineDataset
from image_generation.shaders.renderer_moderngl import RendererModernGL

def generate_dataset_with_n_fragments(output_path, total_samples, shader_file, resolution, gpu, overwrite):
  assert os.path.exists(shader_file), "Fragment path {} does not exist!"

  os.makedirs(output_path, exist_ok=True)

  dataset = ModernGLOnlineDataset([shader_file],
                                  resolution=resolution,
                                  max_queue_size=total_samples * 2,
                                  n_samples=-1,
                                  parallel=True,
                                  gpus=[gpu],
                                  virtual_dataset_size=10000,
                                  sample_mixer=None,
                                  transform_before=None)

  p_bar = tqdm(total=total_samples)

  n_generated = 0
  while n_generated < total_samples:
    image_folder, file = RendererModernGL.get_folder_and_filename(output_path, n_generated)

    if not os.path.exists(file) or overwrite:
      os.makedirs(image_folder, exist_ok=True)
      image, program_i = dataset.__getitem__(0)
      cv2_imwrite(tonumpy(image) * 255, file)

    p_bar.update(1)
    n_generated += 1

  p_bar.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # default args for small scale with alexnet, same as training with noise paper

  parser.add_argument("--shader-file", type=str, default='/data/vision/torralba/movies_sfm/releases/shaders21k/shader_codes/shadertoy/l/ltVfWG.fragment')
  parser.add_argument("--n-samples", type=int, default=105000)
  parser.add_argument("--resolution", type=int, default=256)
  parser.add_argument("--output-path", type=str, default='/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/datasets_glsl_top_shaders_small_scale/ltVfWG')
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--overwrite", type=str2bool, default="False")


  args = parser.parse_args()

  generate_dataset_with_n_fragments(args.output_path,
                                    args.n_samples,
                                    args.shader_file,
                                    args.resolution,
                                    args.gpu,
                                    overwrite=args.overwrite)
