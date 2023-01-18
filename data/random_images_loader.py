import numpy as np
import torch
from bisect import bisect_left
import random
class Supervised_RandomImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, data_num=50000, random_sample=False):
        self.data_num = data_num
        self.data = np.load('300K_random_images.npy').astype(np.uint8)
        self.transform = transform
        self.random_sample = random_sample
        if self.random_sample:
            id_list = list(range(300000))
            self.cifar_idxs = []
            self.id_no_cifar = [x for x in id_list if x not in self.cifar_idxs]
            self.id_sample = random.sample(self.id_no_cifar, data_num)

        self.labels = None

    def __getitem__(self, index):
        if self.random_sample:
            index = self.id_sample[index]

        img = self.data[index]
        sample = self.transform(img)
        return sample, -1, index

    def resample(self):
        if self.data_num == -1:
            self.id_sample = self.id_no_cifar
        else:
            self.id_sample = random.sample(self.id_no_cifar, self.data_num)

    def __len__(self):
        if not self.random_sample:
            return self.data.shape[0]
        else:
            return len(self.id_sample)


class Unsupervised_RandomImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, data_num=50000, return_ood_flag=False, random_sample=False):
        self.data_num = data_num
        self.return_ood_flag = return_ood_flag
        self.data = np.load('300K_random_images.npy').astype(np.uint8)
        self.transform = transform
        self.random_sample = random_sample
        if self.random_sample:
            id_list = list(range(300000))
            self.cifar_idxs = []
            self.id_no_cifar = [x for x in id_list if x not in self.cifar_idxs]
            self.id_sample = random.sample(self.id_no_cifar, data_num)

        self.labels = None

    def __getitem__(self, index):
        if self.random_sample:
            index = self.id_sample[index]
        img = self.data[index]
        samples = [self.transform(img), self.transform(img)]
        if self.return_ood_flag:
            return torch.stack(samples), -1
        return torch.stack(samples)

    def resample(self):
        if self.data_num == -1:
            self.id_sample = self.id_no_cifar
        else:
            self.id_sample = random.sample(self.id_no_cifar, self.data_num)

    def __len__(self):
        if not self.random_sample:
            return self.data.shape[0]
        else:
            return len(self.id_sample)