from image_generation.shaders.glsl_utils import *

# to make this work in parallel, on different threads all OpenGL packages must be imported after spawning the multiprocess.
# solution from here: https://groups.google.com/g/pyglet-users/c/_wqyOiU2rN4
class ShadertoyProgram():
  def __init__(self, fragment_shader_code, name):
    self.name = name
    self.frag_shader = self._get_full_fragment_code(fragment_shader_code)

  def get_frag_shader_code(self):
    return self.frag_shader

  def get_name(self):
    return self.name

  def get_timestep_uniform_str(self):
    timestep_str = 'iTime'
    return timestep_str

  def get_uniforms(self, resolution, n_frames_generated, fps):
    resolution_str = 'iResolution'
    mouse_str = 'iMouse'
    frame_n_str = 'iFrame'
    backbuffer = 'b'

    date = 'iDate'
    sample_rate = 'iSampleRate'
    time_delta = 'iTimeDelta'
    frame_rate = 'iFrameRate'

    uniforms = dict()
    uniforms[resolution_str] = (resolution * 1.0, resolution * 1.0, resolution * 1.0)
    uniforms[mouse_str] = (0.0, 0.0, 0.0, 0.0)
    uniforms[frame_n_str] = n_frames_generated
    uniforms[backbuffer] = 0

    uniforms[date] = (3, 10, 3, 2021)
    uniforms[sample_rate] = 1 / fps
    uniforms[time_delta] = 1 / fps
    uniforms[frame_rate] = fps

    return uniforms

  def _get_full_fragment_code(self, fragment_shader_code):
    return fragment_shader_header + '\n\n\n' + fragment_shader_code


def get_shadertoy_code_from_shader_path(shader_path):
  shader_data = load_shader(shader_path)

  if len(shader_data['renderpass']) != 1:
    return shader_path, "More than 1 renderpass", False

  shader_code = shader_data['renderpass'][0]['code']

  return shader_path, shader_code, True

def get_shadertoy_program_from_shader_path(shader_path):
  shader_code = get_shadertoy_code_from_shader_path(shader_path)[1]
  name = shader_path.split('/')[-1].split('.')[0]
  return ShadertoyProgram(shader_code, name)


def get_shader_path_from_id(shader_id):
  return '{}/{}/{}.fragment'.format(SHADER_TOY_DUMPED_SHADERS, shader_id[0], shader_id)