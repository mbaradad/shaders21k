import random

import os
import torch
from tqdm import tqdm
from p_tqdm import p_map
import numpy as np
import cv2

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
    a = torch.load(checkpoint, map_location=torch.device('cpu'))
  except Exception as e:
    print(e)
    return False
  return True


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
