from utils import *
import multiprocessing as mp

from torch.utils.data import Dataset

from image_generation.shaders.renderer_moderngl import RendererModernGL, get_program_from_shader_path
from image_generation.shaders.glsl_utils import *
import torchvision.transforms as transforms
from image_generation.shaders.samples_mixer import *

from multiprocessing import Lock
import time

# from https://www.quora.com/In-Python-is-it-possible-to-decide-which-class-to-inherit-during-runtime
def get_moderngl_worker_class(cls):
  assert cls in [RendererModernGL]
  class RendererWorker(cls):
    def __init__(self, results_queue,
                 n_samples,
                 time_to_sleep_if_full=5,
                 consistency_check=False,
                 consistency_check_its=10000,
                 *args, **kwargs):
      self.results_queue = results_queue
      self.n_samples = n_samples
      self.time_to_sleep_if_full = time_to_sleep_if_full
      # whether to check for

      self.consistency_check = consistency_check
      self.consistency_check_its = consistency_check_its
      self.consistency_image_data = None

      super().__init__(*args, **kwargs)
      self.__render__()

    def check_consistency(self):
      # render a prefixed image, and check that it's always the same.
      # sometimes, when rendering and training with the same gpus the images produced are corrupted after a while.
      # This tests heuristically that this is not happening

      if self.consistency_image_data is None:
        program_i = np.random.randint(0, len(self.programs))
        self.update_uniforms(program_i)

        uniforms = self.programs[program_i].get_uniforms(self.resolution, self.n_frames_generated, self.fps)
        timestep_str = self.programs[program_i].get_timestep_uniform_str()
        uniforms[timestep_str] = 13.37
        self.consistency_image_data = dict(program_i=program_i, uniforms=uniforms)

        image_to_compare = None
      else:
        program_i = self.consistency_image_data['program_i']
        uniforms = self.consistency_image_data['uniforms']
        image_to_compare = self.consistency_image_data['image']

      for uniform_name, uniform_value in uniforms.items():
        self.update_uniform(program_i, uniform_name, uniform_value)

      self.vaos[program_i].render(mode=self.moderngl.TRIANGLES)

      pil_image = Image.frombytes('RGB', (self.resolution, self.resolution), self.fbo.read(components=3))
      pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)

      generated_image = np.array(pil_image).transpose((2,0,1))

      if image_to_compare is None:
        self.consistency_image_data['image'] = generated_image
      else:
        if not np.allclose(image_to_compare, generated_image, atol=1e-1):
          print("ERROR IN THE GENERATOR!! Some samples are different when using the same parameters. This is probably an issue with multithreading and/or pytorch + OpenGL.")
          print("Bad program name: " + self.programs[program_i].name)
          print("After having generated {} samples".format(self.n_frames_generated))
          raise Exception("Error with on on_the_fly_moderngl_shader.py")


    # same logic as parent class but with sleep when queue is full
    def __render__(self):
      self.n_frames_generated = 0

      # Draw Function
      pil_images = []
      program_is = []
      while True:
        if self.n_frames_generated % self.consistency_check_its == 0:
          self.check_consistency()

        if self.n_images_to_generate != -1 and self.n_frames_generated >= self.n_images_to_generate or self.results_queue.full():
          time.sleep(self.time_to_sleep_if_full)
          # self.results_queue.qsize()
          continue
        program_i = np.random.randint(0, len(self.programs))

        self.update_uniforms(program_i)
        self.vaos[program_i].render(mode=self.moderngl.TRIANGLES)

        image = Image.frombytes('RGB', (self.resolution, self.resolution), self.fbo.read(components=3))
        pil_image = image.transpose(Image.FLIP_TOP_BOTTOM)

        pil_images.append(pil_image)
        program_is.append(program_i)

        if len(pil_images) == self.n_samples:
          assert len(pil_images)
          self.results_queue.put((pil_images, program_is))
          pil_images = []
          program_is = []

        self.n_frames_generated += 1
  return RendererWorker


def moderngl_worker_func(results_queue, n_samples, *args, **kwargs):
  TwiglWorkerClass = get_moderngl_worker_class(RendererModernGL)
  sgn_worker = TwiglWorkerClass(results_queue, n_samples, *args, **kwargs)
  sgn_worker.__render__()

class ModernGLOnlineDataset(Dataset):
  def __init__(self, shader_paths,
               transform=transforms.ToTensor(),
               resolution=256,
               max_queue_size=3000,
               n_samples=10000,
               parallel=True,
               gpus=[0],
               virtual_dataset_size=10000,
               sample_mixer=None,
               transform_before=None,
               base_shaders_path='',
               workers_per_gpu=1,
               consistency_check=False,
               consistency_check_its=10000):

    self.n_samples = n_samples
    self.virtual_dataset_size = virtual_dataset_size

    self.sample_mixer = sample_mixer
    if not self.sample_mixer is None:
      max_queue_size = int(max_queue_size / self.sample_mixer.get_n_samples())
    self.queue = mp.Queue(max_queue_size)

    self.transform_before = transform_before
    self.workers_per_gpu = workers_per_gpu
    assert workers_per_gpu == 1, "More than worker per gpu can generate problems"

    assert type(gpus) is list and all([type(k) is int for k in gpus])

    if n_samples != -1:
      assert n_samples % len(gpus) == 0, "Requested samples {} has to be divisible by number of gpus {} for simplicity".format(n_samples, len(gpus))

    self.transform = transform
    self.shader_codes = []
    if not type(shader_paths) is list:
      shader_paths = [shader_paths]

    self.programs = []

    print("Reading shaders")
    for shader_path in tqdm(shader_paths):
      shader_path = base_shaders_path + shader_path
      self.programs.append(get_program_from_shader_path(shader_path))

    self.mp_manager = mp.Manager()
    self.lock = Lock()

    self.generated_images = self.mp_manager.list()

    kwargs_dict = dict(results_queue=self.queue,
                       n_samples=1 if sample_mixer is None else sample_mixer.get_n_samples(),
                       programs=self.programs,
                       n_images_to_generate=-1,
                       resolution=resolution,
                       temporal_sampling='random',
                       gpu=0,
                       consistency_check=consistency_check,
                       consistency_check_its=consistency_check_its)

    self.generation_process = []
    if parallel:
      for gpu in gpus:
        for _ in range(workers_per_gpu):
          kwargs_dict['gpu'] = gpu
          process = mp.Process(target=moderngl_worker_func, kwargs=kwargs_dict)
          process.start()
          self.generation_process.append(process)
    else:
      print("Generating {} images, this can take a while!".format(n_samples))
      assert max_queue_size >= n_samples, "When not in parallel mode, qsize ({}) must be >= n_sampels ({})"
      moderngl_worker_func(**kwargs_dict)

    # just one class
    self.class_to_idx = dict([(i,i) for i in range(len(self.shader_codes))])

  def __len__(self):
    if self.n_samples == -1:
      return self.virtual_dataset_size
    else:
      return self.n_samples

  def __getitem__(self, index):
    if self.n_samples == -1:
      get_item = self.queue.get
    else:
      get_item = lambda: self.__getitem_finite__(np.random.randint(0, self.__len__()))
    try:
      if self.sample_mixer is None:
        pil_image, program_i = get_item()
        pil_image = pil_image[0]
        program_i = program_i[0]
      else:
        n_samples_to_mix = self.sample_mixer.get_n_samples()
        pil_images, program_is = get_item()

        assert len(pil_images) == n_samples_to_mix and len(pil_images) == len(program_is), "The queue didn't return the expected number of samples! Check the twigl generator."

        if not self.transform_before is None:
          # apply the transform before
          pil_images = [self.transform_before(k) for k in pil_images]
        pil_image, program_i = self.sample_mixer.mix_samples(pil_images, program_is)

    except Exception as e:
      print(e)
      raise e

    if not self.transform is None:
      image = self.transform(pil_image)
    else:
      image = pil_image
    return image, program_i

  def __del__(self):
    # kill processes whent he class is deleted
    for p in self.generation_process:
          p.terminate()

  def __getitem_finite__(self, index):
    self.lock.acquire()
    if len(self.generated_images) >= self.n_samples:
      pil_image, program_i = self.generated_images[index]
    else:
      pil_image, program_i = self.queue.get()
      self.generated_images.append((pil_image, program_i))
      if len(self.generated_images) >= self.n_samples and not self.queue is None:
        for p in self.generation_process:
          p.terminate()
        del self.queue
        self.queue = None
        self.generated_images = self.generated_images[:self.n_samples]
    self.lock.release()

    return pil_image, program_i
