import datetime
import time
import torch
import os
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os.path as osp
import torch.nn as nn
import torch
import heapq
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.num_class = num_class

        self.avg = np.zeros(num_class)
        self.sum = np.zeros(num_class)
        self.count = np.zeros(num_class)

    def reset(self):

        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, gt_lb, pred_lb, n=1):
        for i, value in enumerate(gt_lb):
            self.sum[value] += pred_lb[i] * n  # value: cls id; val[i]: if prediction correct: 1, else: 0
            self.count[value] += n
            self.avg[value] = self.sum[value] / self.count[value]  # acc of each class

class Logger(object):
    """Record Loss/Acc values throughout training"""

    def __init__(self, name, value_name_list, save_path, plot_scale=None):
        self.name = name
        self.value_dict = {}
        for v in value_name_list:
            self.value_dict[v] = []

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.system('mkdir -p ' + self.save_path)

        self.plot_scale = plot_scale

    def add_record(self, value_list):
        cnt = 0
        for k in self.value_dict.keys():
            self.value_dict[k].append(value_list[cnt])
            cnt += 1

    def save_record(self):
        key_list = []
        for v in self.value_dict.keys():
            x_len = len(self.value_dict[v])
            key_list.append(v)

        color_list = ['#800000', '#469990', '#911eb4', '#bfef45', '#42d4f4', '#ffd8b1', '#fffac8']
        x = [i+1 for i in range(x_len)]

        fig = plt.figure(1)

        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        if self.plot_scale is not None:
            axes.set_ylim(self.plot_scale)
        else:
            axes.set_ylim([-2, 5])

        for i in range(len(key_list)):
            cur_key = key_list[i]
            plt.plot(x, self.value_dict[cur_key], color=color_list[i % len(color_list)], linestyle='-', label=cur_key,
                     linewidth=2.3)

        plt.title(self.name)
        plt.legend(prop={'size': 8})

        plt.savefig(osp.join(self.save_path, self.name+'.png'))
        plt.close()

        torch.save(self.value_dict, osp.join(self.save_path, self.name+'.pt'))


def calculate_vsmap(knw_visual_feat, lb, knw_attri_mat, method='ESZSL'):
    """
    :param knw_visual_feat: tensor, cuda, [N_knw, 512]: X
    :param lb: list
    :param knw_attri_mat: tensor, cuda, [Cs, 85]:      S
    :return: V: tensor, cuda, [512, 85]
    """
    visual_feat = knw_visual_feat.cpu().numpy()
    attri_mat = knw_attri_mat.cpu().numpy()

    attri_gt = np.zeros((visual_feat.shape[0], attri_mat.shape[1]))  # [N_knw, 85]:   YS'
    for i in range(lb.size(0)):
        attri_gt[i] = attri_mat[lb[i]]

    if method == 'ESZSL':
        gamma = math.pow(10, 3)
        lamb = math.pow(10, 0.2)

        part_1 = LA.inv(np.matmul(visual_feat.transpose(), visual_feat) + gamma * np.eye(visual_feat.shape[1], visual_feat.shape[1]))
        part_2 = np.matmul(visual_feat.transpose(), attri_gt)
        part_3 = LA.inv(np.matmul(attri_mat.transpose(), attri_mat) + lamb * np.eye(attri_mat.shape[1], attri_mat.shape[1]))

        V = np.matmul(np.matmul(part_1, part_2), part_3)
        V = torch.tensor(V).cuda().float()  # 512 x 85

    elif method == 'linear':
        eps = 1e-6
        V = LA.inv(np.matmul(visual_feat.transpose(), visual_feat) + eps * np.eye(visual_feat.shape[1], visual_feat.shape[1]))
        V = np.matmul(np.matmul(V, visual_feat.transpose()), attri_gt)  # [512, 85]
        V = torch.tensor(V).cuda().float()  # 512 x 85

    return V
