from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision
import math
import numpy as np
import torch
from torchvision import transforms
import random
import os.path as osp


def make_dataset(list_path, dataset_root=None):
    f = open(list_path, 'r')
    lines = f.readlines()
    f.close()

    img_list = []
    lb_list = []

    for l in lines:
        if dataset_root is None:
            img_list.append(l.strip().split(' ')[0])
        else:
            img_full_path = osp.join(dataset_root, l.strip().split(' ')[0])
            img_list.append(img_full_path)

        lb_list.append(int(l.strip().split(' ')[1]))

    return img_list, lb_list


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageSet(Dataset):
    def __init__(self, list_path, args=None, train=False, pseudo_lb=None, balanced=False, mode='RGB'):
        imgs, lb = make_dataset(list_path, args.dataset_root)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + list_path + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.args = args
        self.imgs = imgs    # list of str path
        self.lb = lb        # list of int64

        self.shared_cls_num = args.shared_class_num
        self.total_cls_num = args.total_class_num

        self.modify_lb = [i if i < self.shared_cls_num else self.shared_cls_num for i in self.lb]

        self.pseudo_lb = pseudo_lb  # None or a list

        if train:
            resize_size = 256
            crop_size = 224
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            resize_size = 256
            crop_size = 224
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.balance_sample = balanced
        self.cls_num = len(list(set(self.lb)))
        self.class_dict = self._get_class_dict()  # each cls's samples' id

    def _get_class_dict(self):
        class_dict = dict()
        for i, cur_lb in enumerate(self.lb):
            if not cur_lb in class_dict.keys():
                class_dict[cur_lb] = []
            class_dict[cur_lb].append(i)
        return class_dict


    def __getitem__(self, index):
        if self.balance_sample:
            # Balanced Sampling
            # Currently only support on Source dataset
            sample_class = random.randint(0, self.cls_num - 1)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img_path, lb, ori_lb = self.imgs[index], self.modify_lb[index], self.lb[index]

        if self.pseudo_lb is not None:
            pseudo_lb = self.pseudo_lb[index]

        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        #lb = int(lb)  # torch.int64 --> label
        #ori_lb = int(ori_lb)

        if self.pseudo_lb is None:
            return img, lb, ori_lb
        else:
            return img, lb, ori_lb, pseudo_lb

    def __len__(self):
        return len(self.imgs)
