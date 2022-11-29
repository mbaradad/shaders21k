from shaders21k_private.image_generation.shaders.glsl_utils import *
import string

from utils import *

class TwiglProgram():
  def __init__(self, shader_code, mode, name):
      self.name = name

      self.mode = mode
      self.mode_name = mode[0]
      assert self.mode_name in ['classic', 'geek', 'geeker', 'geekest']

      if type(mode[1]) is str:
        assert mode[1] in ['True', 'False']
        mode[1] = eval(mode[1])
      if type(mode[2]) is str:
        assert mode[2] in ['True', 'False']
        mode[2] = eval(mode[2])
      if type(mode[3]) is str:
        assert mode[3] in ['True', 'False']
        mode[3] = eval(mode[3])

      self.frag_shader = self._get_full_shader_code(shader_code, mode)

  def get_frag_shader_code(self):
    return self.frag_shader

  def get_name(self):
    return self.name

  def get_timestep_uniform_str(self):
    if self.mode == 'classic':
      timestep_str = 'time'
    else:
      # use compact uniform names
      timestep_str = 't'
    return timestep_str

  def get_uniforms(self, resolution, n_frames_generated, fps):
    if self.mode == 'classic':
      # use classic uniform strings
      resolution_str = 'resolution'
      mouse_str = 'mouse'
      frame_n_str = 'frame'
      backbuffer = 'backbuffer'
    else:
      # use compact uniform names
      resolution_str = 'r'
      mouse_str = 'm'
      frame_n_str = 'f'
      backbuffer = 'b'

    uniforms = dict()

    uniforms[resolution_str] = (resolution * 1.0, resolution * 1.0)
    uniforms[mouse_str] = (0.0, 0.0)
    uniforms[frame_n_str] = n_frames_generated
    uniforms[backbuffer] = 0

    return uniforms

  @staticmethod
  def preprocess_shader_code(shader_code):
    # initialize unitialized integers. There are some twigl shaders that
    for letter in list(string.ascii_lowercase):
      int_string = 'int {};'.format(letter)
      int_string_0_init = 'int {} = 0;'.format(letter)
      shader_code = shader_code.replace(int_string, int_string_0_init)
    return shader_code

  @staticmethod
  def preprocess_shader_code_ternary_operator(shader_code):
    # escapes ternary operator if it has a += assignment, which compiles without parenthesis if there are assignments in the 2nd and/or 3rd input in twigl.app but not with standard c
    # see: https://stackoverflow.com/questions/65891926/assignment-in-ternary-operator
    if not '?' in shader_code:
      return shader_code
    i = 0
    final_shader_code = ''
    while i < len(shader_code):
      if shader_code[i] != '?':
        final_shader_code += shader_code[i]
        i += 1
      else:
        # see if we match with : and ;, then add parenthesis before and after
        j = i + 1
        while shader_code[j] != ':' and j < len(shader_code):
          j += 1
        k = j + 1
        l_brack = 0
        while k < len(shader_code):
          if l_brack == 0 and (shader_code[k] == ';' or (shader_code[k] == ')')):
            break
          if shader_code[k] == '(':
            l_brack += 1
          if shader_code[k] == ')':
            l_brack -= 1
          k += 1
        if k < len(shader_code):
          final_shader_code += '?'
          if '=' in shader_code[i + 1:j]:
            final_shader_code += '(' + shader_code[i + 1:j] + ')'
          else:
            final_shader_code += shader_code[i + 1:j]
          final_shader_code += ':'
          if '=' in shader_code[j + 1:k]:
            final_shader_code += '(' + shader_code[j + 1:k] + ')'
          else:
            final_shader_code += shader_code[j + 1:k]
          final_shader_code += shader_code[k]
        i = k + 1

    return final_shader_code

  def _get_full_shader_code(self, fragment_shader_code, mode):
    n, r, i, o = get_mode(*mode[:3])
    shader_code = self.preprocess_shader_code(fragment_shader_code)
    if mode[3]:
      shader_code = self.preprocess_shader_code_ternary_operator(shader_code)
    return n + r + i + shader_code + o


def get_twigl_program_from_shader_path(shader_path):
  fragment_name = shader_path.split('/')[-1].split('.')[0]

  fragment_code_file = '{}/fragments/{}.fragment'.format(TWIGL_FRAGMENTS_WITH_MODE_MODERNGL, fragment_name)
  fragment_mode_file = '{}/modes/{}.fragment'.format(TWIGL_FRAGMENTS_WITH_MODE_MODERNGL, fragment_name)

  shader_code = read_text_file(fragment_code_file)
  mode = read_text_file_lines(fragment_mode_file)

  return TwiglProgram(shader_code, mode, fragment_name)