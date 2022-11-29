from utils import *

from shaders21k_private.image_generation.shaders.programs.shadertoy_program import *
from shaders21k_private.image_generation.shaders.programs.twigl_program import *

import multiprocessing as mp
import os
from tqdm import tqdm

from PIL import Image

assert 'LD_LIBRARY_PATH' in os.environ.keys(), \
       "LD_LIBRARY_PATH not set, you need to set it to point to annaconda to use this package (for example:\n" \
       "export LD_LIBRARY_PATH=$HOME/anaconda3/lib"

# to make this work in parallel, on different threads all OpenGL packages must be imported after spawining the multiprocess.
# solution from here: https://groups.google.com/g/pyglet-users/c/_wqyOiU2rN4
class RendererModernGL():
  def __init__(self, programs, n_images_to_generate=100, fps=10,
               debug=False, completed_file=None, resolution=256,
               temporal_sampling='continuous', gpu=0,
               max_failed_samples=50):
    if not 'MainProcess' in mp.current_process().name:
      opengl_packages = ['moderngl', 'pyglet']
      if any([k in sys.modules for k in opengl_packages]):
        raise Exception("Importing opengl from the main thread may cause problems when this is also launched from child process (for example if the generator is in a dataloading class). "
                        "Remove imports or do not use this package in the main thread: " + str([k for k in opengl_packages if k in sys.modules]))
    assert temporal_sampling in ['continuous', 'random'] or type(temporal_sampling) is list
    if type(temporal_sampling) is list:
      assert len(temporal_sampling) == n_images_to_generate
    self.temporal_sampling_mode = temporal_sampling
    import moderngl
    self.moderngl = moderngl
    self.max_failed_samples = max_failed_samples

    self.resolution = resolution
    try:
      self.ctx = moderngl.create_context(standalone=True, backend='egl', require=OPENGL_REQUIRED_VERSION, device_index=gpu)
      self.fbo = self.ctx.simple_framebuffer((self.resolution, self.resolution), components=4)
      self.fbo.use()

    except Exception as e:
      print("Typical errors are as follows:")
      print("If you see that libpython... is failing, probably you need to set LD_LIBRARY_PATH to the conda lib path: {}".format('/data/vision/torralba/movies_sfm/home/anaconda3/lib'))
      print("Currently echo $LD_LIBRARY_PATH: {}.".format(os.environ['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in os.environ.keys() else ''))
      print("")
      print("Else you may require a lower version of opengl, current is {} 330 or higher should work".format(OPENGL_REQUIRED_VERSION))
      raise e

    self.n_images_to_generate = n_images_to_generate
    self.n_frames_generated = 0

    self.completed_file = completed_file
    self.debug = debug

    self.fps = fps

    self.frag_shader_codes = []
    self.compiled_progs = []
    self.vaos = []
    if not type(programs) is list:
      programs = [programs]
    self.programs = programs

    last_fragment_failed_name = None
    last_exception = None

    def compile_single_program(i):
      program = self.programs[i]
      try:
        frag_shader = program.get_frag_shader_code()
        compiled_prog = self.ctx.program(vertex_shader=vertex_shader_moderngl, fragment_shader=frag_shader)
        vao = self.ctx.simple_vertex_array(compiled_prog, self.ctx.buffer(vertices_moderngl), 'in_vert')
      except Exception as e:
        print("Failed program: ".format(program.get_name()))
        print("With exception: ")
        print(e)
        raise e

      return frag_shader, compiled_prog, vao

    compiled_info = process_in_parallel_or_not(compile_single_program, range(len(self.programs)), parallel=False)

    self.frag_shader_codes = [k[0] for k in compiled_info]
    self.compiled_progs = [k[1] for k in compiled_info]
    self.vaos = [k[2] for k in compiled_info]

    if len(self.frag_shader_codes) != len(self.programs):
      self.release()

      raise Exception("Failed to compile some shaders: {} of {}\n"
                      "Last fragment failed: {} with exception:\n"
                      "{}".format(len(self.frag_shader_codes), len(self.programs), last_fragment_failed_name, last_exception))
    else:
      if len(self.frag_shader_codes) > 1:
        print("Correctly compiled {} fragments and prepared to start rendering.".format(len(self.frag_shader_codes)))

  def get_uniforms(self, program_i):
    return self.compiled_progs[program_i]._members

  def set_uniforms(self, program_i, uniforms):
    for uniform_name, uniform_value in uniforms.items():
      self.compiled_progs[program_i][uniform_name] = uniform_value


  def update_uniform(self, program_i, uniform_name, uniform_value):
    if not self.compiled_progs[program_i].get(uniform_name, None) is None:
      self.compiled_progs[program_i][uniform_name] = uniform_value

  def get_timestep_uniform_str(self, program_i):
    if self.modes[program_i][0] == 'classic':
      timestep_str = 'time'
    else:
      # use compact uniform names
      timestep_str = 't'
    return timestep_str

  def update_uniform_time(self, program_i, n_frame):
    # sample t in the current bin, taking into account fps
    if self.temporal_sampling_mode == 'continuous':
      timestep = np.random.uniform(n_frame / self.fps, (n_frame + 1) / self.fps)
    elif self.temporal_sampling_mode == 'random':
      timestep = np.random.uniform(0.1, 10)
    elif type(self.temporal_sampling_mode) is list:
      timestep = self.temporal_sampling_mode[n_frame]
    else:
      raise Exception("not impolemented")

    timestep_str = self.programs[program_i].get_timestep_uniform_str()
    self.update_uniform(program_i, timestep_str, timestep)

  def update_uniforms(self, program_i):
    uniforms = self.programs[program_i].get_uniforms(self.resolution, self.n_frames_generated, self.fps)

    for uniform_name, uniform_value in uniforms.items():
      self.update_uniform(program_i, uniform_name, uniform_value)

    self.update_uniform_time(program_i, self.n_frames_generated)

  # saves into the folder
  def render_to_folder(self, folder, show_progress=False, store_uniforms=False):
    return self.__render__(store_folder=folder, store=True, show_progress=show_progress, store_uniforms=store_uniforms)

  def render(self, show_progress=False, fragment_indices=None):
    self.rendered_images = []
    self.rendered_fragment_indices = []
    self.__render__(store=False, show_progress=show_progress, shader_indices=fragment_indices)
    return self.rendered_images, self.rendered_fragment_indices

  @staticmethod
  def get_folder_and_filename(base_folder, i):
    return get_folder_and_filename(base_folder, i)

  def release(self):
    self.fbo.release()
    self.ctx.release()

  def should_filter(self, image):
    return False

  def __render__(self, store_folder='', store=False, show_progress=False, store_uniforms=False, shader_indices=None):
    self.n_frames_generated = 0
    if not shader_indices is None:
      assert len(shader_indices) == self.n_images_to_generate

    if show_progress:
      pbar = tqdm(total=self.n_images_to_generate)
    if store:
      print("Generating fragment to folder {}".format(store_folder))

    filtered = 0
    # Draw Function
    while True:
      # sample program
      if shader_indices is None:
        program_i = np.random.randint(0, len(self.compiled_progs))
      else:
        program_i = shader_indices[self.n_frames_generated]

      self.update_uniforms(program_i)
      self.vaos[program_i].render(mode=self.moderngl.TRIANGLES)

      image = Image.frombytes('RGB', (self.resolution, self.resolution), self.fbo.read(components=3))
      image = image.transpose(Image.FLIP_TOP_BOTTOM)
      image = np.array(image).transpose((2, 0, 1))

      if self.should_filter(image) and filtered < self.max_failed_samples:
        filtered += 1
      else:
        filtered = 0
      if store:
        folder, screenshot_file = self.get_folder_and_filename(store_folder, self.n_frames_generated)
        os.makedirs(folder, exist_ok=True)
        cv2_imwrite(image, screenshot_file)
        if store_uniforms:
          uniforms_to_save = dict([(k,float(v)) for k,v in self.cube.uniforms.items() if k.startswith('v_')])
          np.savez_compressed(screenshot_file[:-4], **uniforms_to_save)
      else:
        # To view as data array
        self.rendered_images.append(image)
        self.rendered_fragment_indices.append(program_i)

      self.n_frames_generated += 1
      if show_progress:
        pbar.update(1)

      if self.n_frames_generated >= self.n_images_to_generate or (not self.completed_file is None and os.path.exists(self.completed_file)):
        if show_progress:
          pbar.close()

        return

def get_program_from_shader_path(f):
  if 'shadertoy' in f:
    program = get_shadertoy_program_from_shader_path(f)
  elif 'twigl' in f:
    program = get_twigl_program_from_shader_path(f)
  else:
    raise Exception("Could not identify file as Shadertoy or Twigl: " + f)
  return program

def get_renderer_from_fragments_file(fragments_files, precompile_in_parallel, renderer_kwargs=dict(), subsample_n=-1):
  if not type(fragments_files) is list:
    fragments_files = [fragments_files]

  shader_files = []
  for fragments_file in fragments_files:
    shader_files.extend(read_text_file_lines(fragments_file))

  if subsample_n > 0:
    assert len(shader_files) >= subsample_n
    shader_files = random.sample(shader_files, subsample_n)

  print("Loading programs...")
  programs = process_in_parallel_or_not(get_program_from_shader_path, shader_files, False)

  if precompile_in_parallel:
    print("Creating renderers in parallel, to cache compiled programs and allow faster sequential loading")
    def create_renderer_for_some(some_programs):
      RendererModernGL(some_programs, **renderer_kwargs)
    process_in_parallel_or_not(create_renderer_for_some, chunk_list(programs, 200), True)

  print("Loading renderer...")
  renderer = RendererModernGL(programs, **renderer_kwargs)

  return renderer


if __name__ == '__main__':
  shadertoy_only = False
  performance_to_shader = 'shader_codes/shaders_by_performance'
  shader_by_perf = [a.split(': ')[-1] for a in read_text_file_lines(performance_to_shader)]
  if shadertoy_only:
    shader_by_perf = [a for a in shader_by_perf if len(a) < 24]

  all_shaders = read_text_file_lines('shader_codes/shaders_list')
  for i in tqdm(range(100)):
    top_perf = [k.split(': ')[-1] for k in all_shaders if shader_by_perf[-i] in k][0]

    program = get_program_from_shader_path(top_perf)

    renderer = RendererModernGL([program])

    images, ids = renderer.render()
    from my_python_utils.common_utils import *
    gif_path = imshow(images, gif=True, title=str(i) + '_' + top_perf, env='dumped_shaders')