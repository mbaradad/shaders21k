from utils import *
import json

import hashlib

N_SAMPLES_SMALL_SCALE=100000
N_SAMPLES_LARGE_SCALE=1300000

OPENGL_REQUIRED_VERSION = 440

BASE_DIR='/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models'
SELENIUM_MINED_GLSL_FRAGMENTS='{}/selenium_dumped_fragments'.format(BASE_DIR)
SELENIUM_MINED_GLSL_BITLY_FRAGMENTS='{}/selenium_dumped_bitly_fragments'.format(BASE_DIR)
SELENIUM_MINED_GLSL_USERNAMES='{}/selenium_dumped_usernames'.format(BASE_DIR)

API_GLSL_BITLY_SCRIPTS_FOLDER='{}/api_dumped_bitly_fragments'.format(BASE_DIR)
API_GLSL_SCRIPTS_FOLDER='{}/api_dumped_fragments'.format(BASE_DIR)
BITLY_GLSL_FRAGMENTS_FOLDER='{}/bitly_dumped_fragments'.format(BASE_DIR)

SHADER_TOY_DUMPED_SHADERS='{}/shadertoy_fragments/dumped_shaders'.format(BASE_DIR)

BITLY_PROCESSED_URLS='{}/bitly_processed_urls'.format(BASE_DIR)

IMAGES_GLSL_GENERATED='{}/images_glsl_generated'.format(BASE_DIR)
IMAGES_PER_SHADER=IMAGES_GLSL_GENERATED + '/images_per_fragment'
IMAGES_PER_SHADER_SHADERTOY=IMAGES_GLSL_GENERATED + '/images_per_fragment_shadertoy'
IMAGES_PER_FRAGMENT_TARS_PATH=IMAGES_GLSL_GENERATED + '/shader_images_compressed'


TWIGL_FRAGMENTS_WITH_MODE_PYGLET='{}/twigl_fragments_with_mode'.format(BASE_DIR)
TWIGL_FRAGMENTS_WITH_MODE_MODERNGL='{}/twigl_fragments_with_mode_moderngl'.format(BASE_DIR)

GOOD_SHADERS_FILE=TWIGL_FRAGMENTS_WITH_MODE_MODERNGL + '/good_fragments'
GOOD_SHADERS_FILE_TWIGL_WITH_PERFORMANCE=TWIGL_FRAGMENTS_WITH_MODE_MODERNGL + '/good_fragments_i100_10k_examples'

SHADERS_FILE_TWIGL_AND_SHADERTOY_WITH_PREDICTED_PERFORMANCE=BASE_DIR + '/activations_performance_predictor/shaders_icml_image_folder_mix_samples_6_samples_checkpoint_0199.pth.tar/computed_performance'

GOOD_SHADERS_FILE_SHADERTOY=SHADER_TOY_DUMPED_SHADERS + '/good_fragments'

MIXED_GOOD_SHADERS_FILE=TWIGL_FRAGMENTS_WITH_MODE_MODERNGL + '/good_fragments_mixed_twigl_shadertoy'

TWIGL_FRAGMENTS_WITH_MANUAL_SWEEP_PATH='{}/twigl_fragments_with_mode_moderngl/fragments_with_sweep_manual'.format(BASE_DIR)
TWIGL_FRAGMENTS_WITH_AUTO_SWEEP_PATH='{}/twigl_fragments_with_mode_moderngl/fragments_with_sweep_automatic'.format(BASE_DIR)

SWEEPS_1_CLASS_PER_FRAGMENT_PATH=IMAGES_GLSL_GENERATED + '/sweeps_1_class_per_fragment'
SWEEPS_MULTICLASS_PER_FRAGMENT_PATH=IMAGES_GLSL_GENERATED + '/sweeps_multiclass_per_fragment'

DATASETS_GLSL_GENERATED='{}/final_glsl_datasets'.format(BASE_DIR)
COMPRESSED_FINAL_GLSL_IMAGES='{}/compressed_glsl_images'.format(DATASETS_GLSL_GENERATED)

DATASET_SWEEP_MULTICLASS_PATH=DATASETS_GLSL_GENERATED + '/dataset_sweep_multiclass'

DATASET_SWEEP_MULTICLASS_PATH=DATASETS_GLSL_GENERATED + '/dataset_sweep_multiclass'
DATASET_SWEEP_SINGLE_CLASS_PATH=DATASETS_GLSL_GENERATED + '/dataset_sweep_singleclass'

COMPILATION_ERRORS_FOLDER=IMAGES_GLSL_GENERATED + '/images_per_fragment_compilation_errors'

SUPCON_ENCODERS_PATH='/data/vision/torralba/movies_sfm/home/no_training_cnn/contrastive_image_models/sup_contrast/encoders'

GLSL_FIGS_OUTPUT_DIR='/data/vision/torralba/movies_sfm/home/no_training_cnn/visualizations/glsl_scripts_paper/figures'

RENDERING_TIME_PER_SHADER='{}/rendering_time_per_shader'.format(BASE_DIR)

LARGE_SCALE_STYLEGAN_ORIENTED='/data/vision/torralba/movies_sfm/home/releases/noiselearning/raw_data/large_scale/stylegan-oriented/train'

IMAGENET_PATH='/data/vision/torralba/datasets/imagenet_pytorch'

DUMPED_DATASETS_PATH='{}/datasets_dumped'.format(BASE_DIR)


def mode_to_string_mode(mode):
  s = mode[0]
  if mode[1]:
    s+= ' (300) es'
  if mode[2]:
    s+= ' (MRT)'
  if mode[3]:
    s+= ' with ternary op. (a?b:c) preprocessing'
  return s

def get_stylegan_texture(i):
  assert i <= 1300000, "Not enough stylegan textures, requested {} but only 1.3M available".format(i)
  folder = LARGE_SCALE_STYLEGAN_ORIENTED + '/' + str((i // 1000)).zfill(5)
  image_path = '{}/{}.jpg'.format(folder, str(i).zfill(8))
  image = cv2_imread(image_path)
  return image

def load_shader(shader_path):
  out_fp = open(shader_path, "r")
  shader_dict = json.load(out_fp)
  out_fp.close()

  shaderId = shader_dict['info']['id']
  shader_dict['info']['url'] = 'https://www.shadertoy.com/view/' + shaderId

  return shader_dict

def get_generated_images_path(shader_code_path ,resolution):
  assert shader_code_path.endswith('.fragment'), "shader_code_path={} is not a shader code as it does not end with .fragment".format(shader_code_path)

  is_shadertoy = 'shadertoy' in shader_code_path
  if is_shadertoy:
    datasets_glsl_generated_path = IMAGES_PER_SHADER_SHADERTOY
  else:
    datasets_glsl_generated_path = IMAGES_PER_SHADER
  os.makedirs(datasets_glsl_generated_path, exist_ok=True)

  fragment_name = shader_code_path.split('/')[-1].split('.')[0]
  # if is shader_toy, add subfolder starting with first two letters, as there are many more (20k) and ls is slow with that many files per folder
  folder_path = datasets_glsl_generated_path + '/resolution_{}'.format(resolution) + '/' + (fragment_name[:2] + '/' if is_shadertoy else '') + fragment_name

  return folder_path


def get_random_stylegan_texture():
  image = None
  while image is None:
    i = random.randint(0, 105000)
    folder = LARGE_SCALE_STYLEGAN_ORIENTED + '/' + str((i // 1000) * 1000).zfill(8)
    image_path = '{}/{}.jpg'.format(folder, str(i).zfill(8))
    try:
      image = cv2_imread(image_path)
    except:
      pass
  return image

def get_folder_and_filename(base_folder, i):
  folder = base_folder + '/' + str((i // 1000) * 1000).zfill(10)
  filename = '{}/{}.jpg'.format(folder, str(i).zfill(6))

  return folder, filename

def get_image_paths(folder_path, n_samples):
  return [get_folder_and_filename(folder_path, i)[1] for i in range(n_samples)]

def get_good_shaders(shader_name_only=True):
  shaders = read_text_file_lines(GOOD_SHADERS_FILE)
  assert len(shaders) == 1089, "There should be 1089 on the latest version for ICML paper"

  if shader_name_only:
    shaders = [k.split('/')[-1].replace('.fragment','') for k in shaders]

  # sort alphabetically, which is random as the name is a hash of the code
  shaders.sort()
  return shaders

def get_shaders_sorted_by_performance():
  good_shaders_file_with_preformance = GOOD_SHADERS_FILE + '_i100_10k_examples'
  if os.path.exists(good_shaders_file_with_preformance):
    shaders_with_performance = [k.split(',') for k in read_text_file_lines(good_shaders_file_with_preformance)]
    shaders_with_performance = [(float(k[0]), k[1]) for k in shaders_with_performance]
    shaders_with_performance.sort(key=lambda x: -1 * x[0])
  else:
    shaders = get_good_shaders(shader_name_only=True)
    print("Loading per shader information.")
    shaders_with_performance = []
    for s in tqdm(shaders):
      things_dir = SUPCON_ENCODERS_PATH + '/SimCLR/single_fragments/{}/SimCLR_resnet18_lr_0.015_decay_0.0001_bsz_128_temp_0.07_trial_0_cosine_100'.format(s)
      acc = -1
      if os.path.exists(things_dir):
        computed_files = listdir(things_dir, prepend_folder=False)
        for file in computed_files:
          if is_accuracy_100_file(file, n_samples=10000):
            acc = float(read_text_file_lines(things_dir + '/' + file)[0].split('accuracy: ')[-1])

      if acc == -1:
        raise Exception("Accuracy for shader {} not computed".format(s))
      shaders_with_performance.append((acc, s))
    shaders_with_performance.sort(key=lambda x: -1 * x[0])
    write_text_file_lines(['{},{}'.format(k[0],k[1]) for k in shaders_with_performance], good_shaders_file_with_preformance)
  return shaders_with_performance




def get_mixed_good_shaders():
  if not os.path.exists(MIXED_GOOD_SHADERS_FILE):
    shadertoy_good_fragments = read_text_file_lines(GOOD_SHADERS_FILE_SHADERTOY)
    twigl_good_fragments = read_text_file_lines(GOOD_SHADERS_FILE)

    mixed_good_shaders = shadertoy_good_fragments + twigl_good_fragments

    rand_state = random.getstate()
    random.seed(1337)
    random.shuffle(mixed_good_shaders)
    random.setstate(rand_state)

    write_text_file_lines(mixed_good_shaders, MIXED_GOOD_SHADERS_FILE)
    return mixed_good_shaders
  else:
    mixed_good_shaders = read_text_file_lines(MIXED_GOOD_SHADERS_FILE)
    return mixed_good_shaders


def get_n_shaders_sweep_datasets(use_shadertoy=False):
  n_fragments_dataset = [1,2,4,8,16,32, 64] + list(range(128,1089, 128))
  if use_shadertoy:
    '''
    shadertoy_and_twigl_shaders = get_mixed_good_shaders()
    n_shaders = len(shadertoy_and_twigl_shaders)
    # remove last one, as is arbitrary for twigl length
    increment = 1024
    while n_fragments_dataset[-1] + increment < n_shaders:
      n_fragments_dataset.append(n_fragments_dataset[-1] + increment)

    if n_fragments_dataset[-1] != n_shaders:
      n_fragments_dataset.append(n_shaders)
    '''
    # hardcoded to reduce the number from the original
    n_fragments_dataset = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 21083]
  else:
    n_fragments_dataset += [1089]
  return n_fragments_dataset


def list_top_users(good_fragments_only=False):
  usernames = listdir(API_GLSL_SCRIPTS_FOLDER, type='folder', prepend_folder=True)
  if good_fragments_only:
    good_fragments = [k.split('/')[-1] + '.fragment' for k in read_text_file_lines(GOOD_SHADERS_FILE)]
  scripts_per_user = list()
  for user_folder in usernames:
    username = user_folder.split('/')[-1].split('_')[-1]
    if username == 'usernames':
      continue
    if good_fragments_only:
      n_scripts = len([k for k in listdir(user_folder) if k in good_fragments])
    else:
      n_scripts = len(listdir(user_folder))
    scripts_per_user.append((n_scripts, username))

  sorted_scripts = sorted([k for k in scripts_per_user if k[0] > 0], key=lambda x: -1 * x[0])

  return sorted_scripts


glsl_words = ['float', 'vec', 'while', 'define', 'mat', 'FC.', 'gl_FragCoord.', 'gl_FragColor.', 'precision highp float']
def is_glsl(text):
    if 'No results' in text:
        return False
    for word in glsl_words:
        if word in text:
            return True
    return False

valid_start_words = ['float', 'vec', 'for', 'while', '#define', 'mat', 'precision', 'void', "#version", "o.", "int "]
def starts_with_valid_word(text):
  for w in valid_start_words:
    if text.startswith(w):
      return True
  else:
    return False

def get_hash_string(fragment_text):
  return hashlib.md5(fragment_text.encode()).hexdigest()

def parse_tweet_glsl(text, min_length_characters=1):
  invalid_words = ["Processing"]
  for w in invalid_words:
    if w in text:
      return False, ""
  glsl_program = str(text)
  if is_glsl(text):
    # remove everything from the end so that it ends in a valid program (either ends with ; or }
    while len(glsl_program) > 0:
      if not glsl_program[-1] in [';', '}']:
        glsl_program = glsl_program[:-1]
      else:
        break

    # remove everything from the begining until one of the valid_start_words appears
    while len(glsl_program) > 0:
      if not starts_with_valid_word(glsl_program):
        glsl_program = glsl_program[1:]
      else:
        break

    if len(glsl_program) < min_length_characters:
      valid_glsl = False
    else:
      valid_glsl = True
  else:
    valid_glsl = False
  return valid_glsl, glsl_program


# TODO: try this simpler version, which seems to be the one used in twigl:


vert_shader = """
 #version 300 es

 layout(location = 0) in vec3 vertexPosition;
 uniform mat4 projection_matrix, view_matrix, model_matrix;
 out vec4 vVertex;

 void main()
 {
     vVertex = model_matrix * vec4(vertexPosition, 1.0);
     gl_Position = projection_matrix * view_matrix * vVertex;
 }
 """

vertex_shader_moderngl_shadertoy="""
#version 430
in vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""


fragment_shader_header="""
#version 430
#define HW_PERFORMANCE 1
uniform vec3      iResolution;
uniform float     iTime;
uniform float     iChannelTime[4];
uniform vec4      iMouse;
uniform vec4      iDate;
uniform float     iSampleRate;
uniform vec3      iChannelResolution[4];
uniform int       iFrame;
uniform float     iTimeDelta;
uniform float     iFrameRate;
uniform sampler2D iChannel0;
uniform struct {
  sampler2D sampler;
  vec3  size;
  float time;
  int   loaded;
}iCh0;
uniform sampler2D iChannel1;
uniform struct {
  sampler2D sampler;
  vec3  size;
  float time;
  int   loaded;
}iCh1;
uniform sampler2D iChannel2;
uniform struct {
  sampler2D sampler;
  vec3  size;
  float time;
  int   loaded;
}iCh2;
uniform sampler2D iChannel3;
uniform struct {
  sampler2D sampler;
  vec3  size;
  float time;
  int   loaded;
}iCh3;


void mainImage( out vec4 c, in vec2 f );
void st_assert( bool cond );
void st_assert( bool cond, int v );
out vec4 shadertoy_out_color;
void st_assert( bool cond, int v ) {if(!cond){if(v==0)shadertoy_out_color.x=-1.0;else if(v==1)shadertoy_out_color.y=-1.0;else if(v==2)shadertoy_out_color.z=-1.0;else shadertoy_out_color.w=-1.0;}}
void st_assert( bool cond        ) {if(!cond)shadertoy_out_color.x=-1.0;}
void main( void ){
shadertoy_out_color = vec4(1.0,1.0,1.0,1.0);
vec4 color = vec4(0.0,0.0,0.0,1.0);
mainImage( color, gl_FragCoord.xy );
if(shadertoy_out_color.x<0.0) color=vec4(1.0,0.0,0.0,1.0);
if(shadertoy_out_color.y<0.0) color=vec4(0.0,1.0,0.0,1.0);
if(shadertoy_out_color.z<0.0) color=vec4(0.0,0.0,1.0,1.0);
if(shadertoy_out_color.w<0.0) color=vec4(1.0,1.0,0.0,1.0);
shadertoy_out_color = vec4(color.xyz,1.0);}
"""


vertices_moderngl = np.array([
     -1.0, -1.0,
     1.0, -1.0,
     -1.0, 1.0,
     1.0,  -1.0,
     1.0, 1.0,
     -1.0, 1.0],
    dtype='f4',
)

vertex_shader_moderngl="""
#version 330
in vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""


def get_classic(three_hundred=False, MRT=False):
  if three_hundred:
    from image_generation.shaders.programs.twigl_modes.classic_300 import n, r, i, o
  elif MRT:
    from image_generation.shaders.programs.twigl_modes.classic_mrt import n, r, i, o
  else:
    from image_generation.shaders.programs.twigl_modes.classic import n, r, i, o
  return n, r, i, o

def get_geek(three_hundred=False, MRT=False):
  if three_hundred:
    from image_generation.shaders.programs.twigl_modes.geek_300 import n, r, i, o
  elif MRT:
    from image_generation.shaders.programs.twigl_modes.geek_mrt import n, r, i, o
  else:
    from image_generation.shaders.programs.twigl_modes.geek import n, r, i, o
  return n, r, i, o

def get_geeker(three_hundred=False, MRT=False):
  if three_hundred:
    from image_generation.shaders.programs.twigl_modes.geeker_300 import n, r, i, o
  elif MRT:
    from image_generation.shaders.programs.twigl_modes.geeker_mrt import n, r, i, o
  else:
    from image_generation.shaders.programs.twigl_modes.geeker import n, r, i, o
  return n, r, i, o

def get_geekest(three_hundred=False, MRT=False):
  if three_hundred:
    from image_generation.shaders.programs.twigl_modes.geekest_300 import n, r, i, o
  elif MRT:
    from image_generation.shaders.programs.twigl_modes.geekest_mrt import n, r, i, o
  else:
    from image_generation.shaders.programs.twigl_modes.geekest import n, r, i, o
  return n, r, i, o


def get_modes_by_priority():
  # geekest 300 first, as they seem to be the most common, then classic 300, and the rest doesn't seem to matter much
  modes = [('geekest', True, False),
           ('classic', True, False),
           ('classic', False, False),
           ('geekest', False, False),
           ('geek', True, False),
           ('geeker', True, False),
           ('geek', False, False),
           ('geeker', False, False),
           # finally al mrt
           ('classic', False, True),
           ('geekest', False, True),
           ('geek', False, True),
           ('geeker', False, True)
   ]

  for m in get_all_available_modes():
    assert m in modes, "Mode {} not found in modes by priority"

  return modes

def get_mode(mode, three_hundred, MRT):
  assert type(three_hundred) is bool and type(three_hundred) is bool, "modes should be booleans"
  if mode == 'classic':
    return get_classic(three_hundred, MRT)
  elif mode == 'geek':
    return get_geek(three_hundred, MRT)
  elif mode == 'geeker':
    return get_geeker(three_hundred, MRT)
  elif mode == 'geekest':
    return get_geekest(three_hundred, MRT)

def get_all_available_modes():
  mode_names = ['classic', 'geek', 'geeker', 'geekest']
  modes = [(k, False, False) for k in mode_names]
  modes.extend([(k, True, False) for k in mode_names])
  modes.extend([(k, False, True) for k in mode_names])

  return modes


def get_text_from_tweet(tweet):
  if hasattr(tweet, 'retweeted_status'):
    if hasattr(tweet.retweeted_status, 'full_text'):
      text = tweet.retweeted_status.full_text
  elif hasattr(tweet, 'extended_tweet'):
    text = tweet.extended_tweet["full_text"]
  elif hasattr(tweet, 'full_text'):
    text = tweet.full_text
  else:
    text = tweet.text
  return html.unescape(text)

import html


def find_all_fragment_images(fragment_path):
  folders = listdir(fragment_path, prepend_folder=True)
  all_images = []
  for f in folders:
    all_images.extend(listdir(f, prepend_folder=True, extension='jpg'))
  return all_images


def is_accuracy_100_file(file, n_samples=10000):
  if not 'accuracy' in file or not 'imagenet100' in file:
    return False
  if n_samples == -1:
    return not '_n_samples' in file
  else:
    return '_n_samples_{}'.format(n_samples) in file


def get_rendering_time_file(shader_code_path, n_samples):
  fragment_name = shader_code_path.split('/')[-1]

  output_rendering_time_path = RENDERING_TIME_PER_SHADER + '/n_samples_' + str(n_samples) + '/' + fragment_name[:2]
  os.makedirs(output_rendering_time_path, exist_ok=True)
  output_rendering_time_file = output_rendering_time_path + '/' + fragment_name

  return output_rendering_time_file


# GENERATION UTILS

TOTAL_SMALL_SCALE_SAMPLES = 130000
TOTAL_LARGE_SCALE_SAMPLES = 1300000
FPS=4
RESOLUTION_SMALL_SCALE=256
RESOLUTION_LARGE_SCALE=384


def get_generation_info(opt):
  if opt.top_performing:
    # sorted by performance
    assert not opt.use_shadertoy, "Performance not available for shadertoy shaders"
    good_fragments_paths = get_good_shaders(shader_name_only=False)
    fragment_name_to_path = dict([(k.split('/')[-1].replace('.fragment', ''), k) for k in good_fragments_paths])
    fragments_by_performance = get_shaders_sorted_by_performance()
    good_fragments_paths = [fragment_name_to_path[k[1]] for k in fragments_by_performance]
  else:
    if opt.use_shadertoy:
      # sorted at random, but fixed
      good_fragments_paths = get_mixed_good_shaders()
    else:
      # sorted alphabetically
      good_fragments_paths = get_good_shaders(shader_name_only=False)
      good_fragments_paths.sort()

  n_fragments_dataset = get_n_shaders_sweep_datasets(use_shadertoy=opt.use_shadertoy)

  if opt.total_samples != -1:
    total_samples = opt.total_samples
    resolution = RESOLUTION_LARGE_SCALE
  else:
    if opt.large_scale:
      total_samples = int(TOTAL_LARGE_SCALE_SAMPLES * 1.3)
      resolution = RESOLUTION_LARGE_SCALE
    else:
      total_samples = int(TOTAL_SMALL_SCALE_SAMPLES * 1.3)
      resolution = RESOLUTION_SMALL_SCALE
  if opt.resolution != -1:
    resolution = opt.resolution

  return n_fragments_dataset, good_fragments_paths, total_samples, resolution


def read_text_file_lines(filename, stop_at=-1):
  lines = list()
  with open(filename, 'r') as f:
    for line in f:
      if stop_at > 0 and len(lines) >= stop_at:
        return lines
      lines.append(line.replace('\n',''))
  return lines

def write_text_file_lines(lines, file):
  assert type(lines) is list, "Lines should be a list of strings"
  with open(file, 'w') as file_handler:
    for item in lines:
      file_handler.write("%s\n" % item)

def write_text_file(text, filename):
  with open(filename, "w") as file:
    file.write(text)

def read_text_file(filename):
  text_file = open(filename, "r")
  data = text_file.read()
  text_file.close()
  return data

def float2str(float, prec=2):
  return ("{0:." + str(prec) + "f}").format(float)
