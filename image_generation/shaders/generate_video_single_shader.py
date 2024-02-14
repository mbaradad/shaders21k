from utils import *

from image_generation.shaders.glsl_utils import *
from image_generation.shaders.renderer_moderngl import RendererModernGL, get_program_from_shader_path

def generate_video_single_shader(shader_path, output_file, duration, resolution, fps, gpu):
  assert os.path.exists(shader_path), "Fragment path {} does not exist!"

  program = get_program_from_shader_path(shader_path)
  total_samples = duration * fps
  renderer = RendererModernGL([program], 
                              n_images_to_generate=total_samples, 
                              fps=fps,
                              resolution=resolution,
                              temporal_sampling='continuous', 
                              gpu=gpu,
                              max_failed_samples=0)

  imgs, _ = renderer.render()
  # dump images into output_file, and assert that this is ends with .mp4
  assert output_file.endswith('.mp4'), "Output file must end with .mp4"
  video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (resolution, resolution))
  for img in imgs:
    video_writer.write(tonumpy(img).transpose((1,2,0)))
  video_writer.release()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # default args for small scale with alexnet, same as training with noise paper

  parser.add_argument("--shader-file", type=str, default='/data/vision/torralba/movies_sfm/releases/shaders21k/shader_codes/shadertoy/l/ltVfWG.fragment')
  parser.add_argument("--duration", type=int, default=20, help="Duration of the video in seconds")
  parser.add_argument("--resolution", type=int, default=512)
  parser.add_argument("--fps", type=int, default=24)
  parser.add_argument("--output-path", type=str, default='/tmp/ltVfWG.mp4')
  parser.add_argument("--gpu", type=int, default=0)
  

  args = parser.parse_args()

  generate_video_single_shader(args.shader_file,
                               args.output_path,
                               args.duration,
                               args.resolution,
                               args.fps,
                               args.gpu)
