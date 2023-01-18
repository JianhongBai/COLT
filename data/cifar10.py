import torch
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np


class subsetCIFAR10(CIFAR10):
    def __init__(self, sublist, psudoClass, **kwds):
        super().__init__(**kwds)

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        if psudoClass is not None:
            self.targets = psudoClass


class CustomCIFAR10(CIFAR10):
    def __init__(self, sublist, return_targets=False, returnIdx=False, **kwds):
        super().__init__(**kwds)

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(10)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        self.return_targets = return_targets
        self.returnIdx = returnIdx
        id_targets = torch.tensor(self.targets)
        cls_ins_num_list = []
        for i in range(10):
            cls_ins_num = (id_targets == i).sum()
            cls_ins_num_list.append(cls_ins_num)
        cls_ins_num = torch.tensor(cls_ins_num_list)
        val, idx = cls_ins_num.sort()
        self.few_labels = idx[0:3]
        self.median_labels = idx[3:7]
        self.many_labels = idx[7:10]

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if self.return_targets and self.returnIdx:
            return imgs, torch.tensor(self.targets[idx]), idx
        if self.return_targets:
            return torch.stack(imgs), torch.tensor(self.targets[idx])
        if self.returnIdx:
            return torch.stack(imgs), idx
        return torch.stack(imgs)

class Supervised_CIFAR10(CIFAR10):
    def __init__(self, sublist, return_targets=False, returnIdx=False, **kwds):
        super().__init__(**kwds)
        self.txt = sublist
        self.return_targets = return_targets
        self.returnIdx = returnIdx
        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(10)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = self.transform(img)
        if self.return_targets and self.returnIdx:
            return imgs, torch.tensor(self.targets[idx]), idx
        if self.returnIdx:
            return imgs, idx
        return imgs

import torchvision
class Set_imbalance_ratio_CIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(Set_imbalance_ratio_CIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.val_idx = np.load("split/cifar10/valIdxList.npy")
        self.sample_list = []
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num - 500
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        targets_np[self.val_idx] = -1
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            self.sample_list.append(selec_idx)
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


