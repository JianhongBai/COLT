import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import random
import numpy as np
import pickle
from matplotlib import pyplot as plt
from pdb import set_trace
# from data.utils import _gaussian_blur
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision import datasets
class Supervised_places365_Dataset(Dataset):

  def __init__(self, root, txt, transform=None, returnPath=False):
    self.img_path = []
    self.labels = []
    self.root = root
    self.transform = transform
    self.returnPath = returnPath
    self.txt = txt

    with open(txt) as f:
      for line in f:
        self.img_path.append(os.path.join(root, line.split()[0]))

  def __len__(self):
    return len(self.img_path)

  def __getitem__(self, index):

    path = self.img_path[index]

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    if self.transform is not None:
      sample = self.transform(sample)

    if not self.returnPath:
      return sample, -1, index
    else:
      return sample, -1, index, path.replace(self.root, '')


def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo)
  return dict


class Unsupervised_places365_Dataset(Supervised_places365_Dataset):
  def __init__(self, returnIdx=False, returnLabel=False, return_ood_flag=False, **kwds):
    super().__init__(**kwds)
    self.returnIdx = returnIdx
    self.returnLabel = returnLabel
    self.return_ood_flag = return_ood_flag

  def __getitem__(self, index):
    path = self.img_path[index]

    if not os.path.isfile(path):
      path = path + ".gz"

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    samples = [self.transform(sample), self.transform(sample)]
    if self.returnIdx and (not self.returnLabel):
      return torch.stack(samples), index
    elif (not self.returnIdx) and self.returnLabel:
      return torch.stack(samples), -1
    elif self.returnIdx and self.returnLabel:
      return torch.stack(samples), -1, index
    elif self.return_ood_flag:
      return torch.stack(samples), -1
    else:
      return torch.stack(samples)