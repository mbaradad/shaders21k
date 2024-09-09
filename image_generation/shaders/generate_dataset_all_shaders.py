from utils import *

from image_generation.shaders.glsl_utils import *
from image_generation.shaders.on_the_fly_moderngl_shader import ModernGLOnlineDataset
from image_generation.shaders.renderer_moderngl import RendererModernGL

def generate_dataset_with_n_fragments_shader_list(output_path, total_samples, shader_file_list, shader_codes_path, resolution, gpu, overwrite):
  assert type(shader_file_list) == list, "Shader file list must be a list of paths to shader files"
  shader_file_list = [os.path.join(shader_codes_path, shader_file) for shader_file in shader_file_list]
  for shader_file in tqdm(shader_file_list):
    assert os.path.exists(shader_file), "Fragment path {} does not exist!".format(shader_file)

  os.makedirs(output_path, exist_ok=True)

  dataset = ModernGLOnlineDataset(shader_file_list,
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

    if n_generated % 1000 == 0:
      print("Generated {} images".format(n_generated))

  p_bar.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # default args for small scale with alexnet, same as training with noise paper

  parser.add_argument("--shaders-file", type=str, default='/vision-nfs/torralba/u/mbaradad/movies_sfm/releases/shaders21k/shader_codes/shaders_list')
  parser.add_argument("--shader-codes-path", type=str, default='/vision-nfs/torralba/u/mbaradad/movies_sfm/releases/shaders21k')

  parser.add_argument("--n-shaders", type=int, default=-1)
  parser.add_argument("--n-samples", type=int, default=1300000)
  parser.add_argument("--resolution", type=int, default=512)
  parser.add_argument("--output-path", type=str, default='/tmp/shaders21k_test')
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--overwrite", type=str2bool, default="False")

  args = parser.parse_args()
  
  shader_files_list = read_text_file_lines(args.shaders_file)
  if args.n_shaders > 0:
    shader_files_list = random.sample(shader_files_list, args.n_shaders)
  
  generate_dataset_with_n_fragments_shader_list(args.output_path,
                                                args.n_samples,
                                                shader_files_list,
                                                args.shader_codes_path,
                                                args.resolution,
                                                args.gpu,
                                                overwrite=args.overwrite)
