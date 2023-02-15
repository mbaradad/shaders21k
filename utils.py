import os
import torch
from tqdm import tqdm
from p_tqdm import p_map
import numpy as np
import cv2
import sys
import random
import argparse

from PIL import Image


def listdir(folder, prepend_folder=False, extension=None, type=None):
  assert type in [None, 'file', 'folder'], "Type must be None, 'file' or 'folder'"
  files = [k for k in os.listdir(folder) if (True if extension is None else k.endswith(extension))]
  if type == 'folder':
    files = [k for k in files if os.path.isdir(folder + '/' + k)]
  elif type == 'file':
    files = [k for k in files if not os.path.isdir(folder + '/' + k)]
  if prepend_folder:
    files = [folder + '/' + f for f in files]
  return files


def checkpoint_can_be_loaded(checkpoint):
  try:
    torch.load(checkpoint, map_location=torch.device('cpu'))
    return True
  except Exception as e:
    print(e)
    return False


def read_text_file_lines(filename, stop_at=-1):
  lines = list()
  with open(filename, 'r') as f:
    for line in f:
      if stop_at > 0 and len(lines) >= stop_at:
        return lines
      lines.append(line.replace('\n',''))
  return lines

def process_in_parallel_or_not(function, elements, parallel, use_pathos=False, num_cpus=-1):
  from pathos.multiprocessing import Pool
  if parallel:
    if use_pathos:
      pool = Pool()
      pool.apply(function, elements)

    if num_cpus > 0:
      return p_map(function, elements, num_cpus=num_cpus)
    else:
      return p_map(function, elements)
  else:
    returns = []
    for k in tqdm(elements):
      returns.append(function(k))

  return returns


def cv2_imwrite(im, file, normalize=False, jpg_quality=None):
  if len(im.shape) == 3 and im.shape[0] == 3 or im.shape[0] == 4:
    im = im.transpose(1, 2, 0)
  if normalize:
    im = (im - im.min())/(im.max() - im.min())
    im = np.array(255.0*im, dtype='uint8')
  if jpg_quality is None:
    # The default jpg quality seems to be 95
    if im.shape[-1] == 3:
      cv2.imwrite(file, im[:,:,::-1])
    else:
      raise Exception('Alpha not working correctly')
      im_reversed = np.concatenate((im[:,:,3:0:-1], im[:,:,-2:-1]), axis=2)
      cv2.imwrite(file, im_reversed)
  else:
    cv2.imwrite(file, im[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])


def chunk_list(seq, n_chunks):
  avg = len(seq) / float(n_chunks)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def chunk_list_max_len(seq, max_len):
  out = []
  last = 0

  while last < len(seq):
    out.append(seq[int(last):int(last + max_len)])
    last += max_len

  return out


def cv2_imread(file, return_BGR=False, read_alpha=False):
  im = None
  if read_alpha:
    try:
      im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    except:
      print("Failed to read alpha channel, will us standard imread!")
  if not read_alpha or im is None:
    im = cv2.imread(file)
  if im is None:
    raise Exception('Image {} could not be read!'.format(file))
  im = im.transpose(2,0,1)
  if return_BGR:
    return im
  if im.shape[0] == 4:
    return np.concatenate((im[:3][::-1], im[3:4]))
  else:
    return im[::-1, :, :]


def select_gpus(gpus_arg):
  #so that default gpu is one of the selected, instead of 0
  gpus_arg = str(gpus_arg)
  if len(gpus_arg) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
    gpus = list(range(len(gpus_arg.split(','))))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpus = []
  print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

  flag = 0
  for i in range(len(gpus)):
    for i1 in range(len(gpus)):
      if i != i1:
        if gpus[i] == gpus[i1]:
          flag = 1
  assert not flag, "Gpus repeated: {}".format(gpus)

  return gpus


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

def str2bool(v):
  assert type(v) is str
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')


def tonumpy(tensor):
  if type(tensor) is Image:
    return np.array(tensor).transpose((2,0,1))
  if type(tensor) is list:
    return np.array(tensor)
  if type(tensor) is np.ndarray:
    return tensor
  if tensor.requires_grad:
    tensor = tensor.detach()
  if type(tensor) is torch.autograd.Variable:
    tensor = tensor.data
  if tensor.is_cuda:
    tensor = tensor.cpu()
  return tensor.detach().numpy()