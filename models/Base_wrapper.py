import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
import models.network as network
import models.propagation as propagation
from models.utils import AverageMeter, Logger
import torch.nn.functional as F
import torch.nn as nn
from sklearn import preprocessing
from sklearn.cluster import KMeans
import torch.optim as optim
import os.path as osp
import copy
import heapq
import seaborn as sns
import matplotlib.pyplot as plt


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


class Base_Solver:
    def __init__(self, args, s_loader_len, t_loader_len):
        self.args = args

        self.netF = network.ResBase(res_name=args.net)
        self.netB = network.feat_classifier_two(class_num=args.bottleneck,
                                                input_dim=self.netF.in_features, bottleneck_dim=1024)
        # If choose not to use netC for fine-tuning with margin, set args.pretrain_epochs=0.
        self.netC = network.ArcMarginProduct(in_features=args.bottleneck, out_features=args.shared_class_num,
                                             s=30.0, m=args.margin, easy_margin=False)
        self.net = nn.Sequential(self.netF, self.netB, self.netC)
        self.netA = network.AttributeProjector(in_feat=args.bottleneck, hidden_feat=256)  # 2 layers: 512->256->85
        if self.args.train_margin:
            if self.args.multi_unk:
                self.netC_all = network.ArcMarginProduct(in_features=args.bottleneck+85,
                                                         out_features=args.shared_class_num + args.unk_cluster_n,
                                                         s=30.0, m=args.margin, easy_margin=False)
            else:
                self.netC_all = network.ArcMarginProduct(in_features=args.bottleneck+85,
                                                         out_features=args.shared_class_num + 1,
                                                         s=30.0, m=args.margin, easy_margin=False)
        else:
            self.netC_all = network.feat_classifier_two(class_num=self.args.shared_class_num + 1,
                                                        input_dim=args.bottleneck + 85)
        self.netD = network.feat_classifier_two(class_num=2,
                                                input_dim=args.bottleneck+85)
        if args.parallel:
            self.net = torch.nn.DataParallel(self.net)
            self.netA = torch.nn.DataParallel(self.netA)
            self.netC_all = torch.nn.DataParallel(self.netC_all)
            self.netD = torch.nn.DataParallel(self.netD)

        self.net = self.net.cuda()
        self.netA = self.netA.cuda()
        self.netC_all = self.netC_all.cuda()
        self.netD = self.netD.cuda()

        self.AttributeProp = propagation.AttributePropagation()

        if args.use_pretrain:
            modelpath = args.output_dir_src + '/source_F.pt'
            self.netF.load_state_dict(torch.load(modelpath))
            modelpath = args.output_dir_src + '/source_B.pt'
            self.netB.load_state_dict(torch.load(modelpath))
            modelpath = args.output_dir_src + '/source_C.pt'
            self.netC.load_state_dict(torch.load(modelpath))

        param_group = []
        learning_rate = args.lr
        for k, v in self.netF.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate * 0.1}]
            #v.requires_grad = False

        for k, v in self.netB.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in self.netC.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]

        for k, v in self.netC_all.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in self.netD.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in self.netA.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]

        self.optimizer = optim.SGD(param_group)
        self.optimizer = op_copy(self.optimizer)

        self.max_iter = args.epochs * min(s_loader_len, t_loader_len)
        self.iter_num = 0

        self.estimate_meter = AverageMeter(self.args.shared_class_num+1)
        self.estimate_refine_meter = AverageMeter(self.args.shared_class_num+1)

        self.cls_meter = AverageMeter(self.args.shared_class_num+1)
        self.attri_meter = AverageMeter(self.args.total_class_num)

        plot_save_path = self.args.save_folder
        self.cluster_acc_logger = Logger(name='Cluster Acc', value_name_list=['OS*', 'UNK', 'H'],
                                         save_path=osp.join(plot_save_path), plot_scale=[0, 100])

        self.train_loss_logger = Logger(name='Train Loss', value_name_list=['Total', 'Cls', 'Align', 'Attri'],
                                        save_path=osp.join(plot_save_path))

        self.test_acc_logger_rec = Logger(name='Test Recog Acc', value_name_list=['OS*', 'UNK', 'H'],
                                          save_path=osp.join(plot_save_path), plot_scale=[0, 100])

        if self.args.revise_prototype == 'post':
            self.test_acc_logger_sem = Logger(name='Test Recover Acc',
                                              value_name_list=['S', 'U', 'H', 'S_rev', 'H_rev'],
                                              save_path=osp.join(plot_save_path), plot_scale=[0, 100])
        elif self.args.revise_prototype == 'no':
            self.test_acc_logger_sem = Logger(name='Test Recover Acc', value_name_list=['S', 'U', 'H'],
                                              save_path=osp.join(plot_save_path), plot_scale=[0, 100])

    def save_net(self, pretrain=False):
        model_save_path = self.args.save_folder
        if not os.path.exists(model_save_path):
            os.system('mkdir -p ' + model_save_path)

        if pretrain:
            torch.save(self.netF.state_dict(), osp.join(model_save_path, "net_F_pre.pt"))
            torch.save(self.netB.state_dict(), osp.join(model_save_path, "net_B_pre.pt"))
            torch.save(self.netC.state_dict(), osp.join(model_save_path, "net_C_pre.pt"))
        else:
            torch.save(self.netF.state_dict(), osp.join(model_save_path, "net_F.pt"))
            torch.save(self.netB.state_dict(), osp.join(model_save_path, "net_B.pt"))
            torch.save(self.netC.state_dict(), osp.join(model_save_path, "net_C.pt"))
            torch.save(self.netC_all.state_dict(), osp.join(model_save_path, "net_C_all.pt"))
            torch.save(self.netD.state_dict(), osp.join(model_save_path, "net_D.pt"))
            torch.save(self.netA.state_dict(), osp.join(model_save_path, "net_A.pt"))

        return

    def load_net(self, load_folder):
        modelpath = osp.join(load_folder, 'net_F.pt')
        self.netF.load_state_dict(torch.load(modelpath))

        modelpath = osp.join(load_folder, 'net_B.pt')
        self.netB.load_state_dict(torch.load(modelpath))

        modelpath = osp.join(load_folder, 'net_C_all.pt')
        self.netC_all.load_state_dict(torch.load(modelpath))

        modelpath = osp.join(load_folder, 'net_A.pt')
        self.netA.load_state_dict(torch.load(modelpath))

    @torch.no_grad()
    def extract_feat(self, extract_loader):
        self.netF.eval()
        feat_list = []
        lb_list = []
        for batch_idx, batch in enumerate(extract_loader):
            img = batch[0].cuda()
            lb = batch[1]   # modified lb: 0 -- Cs

            feat = self.netF(img)

            feat_list.append(feat)
            lb_list.append(lb)

        feat_list = torch.cat(feat_list, dim=0).cpu().numpy()
        lb_list = torch.cat(lb_list, dim=0).numpy()

        np.save(osp.join(self.args.feat_save_dir, self.args.feat_name+'_feature.npy'), feat_list)
        np.save(osp.join(self.args.feat_save_dir, self.args.feat_name+'_label.npy'), lb_list)

        return

    @torch.no_grad()
    def visualize_confusion_matrix(self, test_loader, attri_mat):
        self.netF.eval()
        self.netB.eval()
        self.netC_all.eval()
        self.netA.eval()

        self.cls_meter.reset()
        self.attri_meter.reset()

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        attri_pred_lb = []
        gt_lb = []
        for batch_idx, batch in enumerate(test_loader):
            img = batch[0].cuda()
            lb = batch[1]       # 0 -- Cs
            ori_lb = batch[2]   # 0 -- (Cs+Ct-1)

            feat = self.netB(self.netF(img))
            attri_pred = self.netA(feat)  # [bs, 85]
            feat_joint = torch.cat([feat, attri_pred], dim=1)  # [bs, 256+85]
            logits = self.netC_all(feat_joint)  # [bs, Cs+K]

            pred_lb = logits.argmax(dim=1)
            pred_lb[pred_lb >= self.args.shared_class_num] = self.args.shared_class_num
            self.cls_meter.update(lb.numpy(), (pred_lb.cpu().numpy()==lb.numpy()).astype(int))

            for i in range(pred_lb.size(0)):
                cos_sim = cos(attri_pred[i].view(1,-1), attri_mat)
                if pred_lb[i] < self.args.shared_class_num:
                    attri_cls_pred = cos_sim[:self.args.shared_class_num].argmax().item()
                else:
                    attri_cls_pred = cos_sim[self.args.shared_class_num:].argmax().item() + self.args.shared_class_num

                attri_pred_lb.append(attri_cls_pred)
                gt_lb.append(ori_lb[i].item())

        self.attri_meter.update(np.array(gt_lb), (np.array(gt_lb)==np.array(attri_pred_lb)).astype(int))

        print(self.cls_meter.avg)
        print(self.attri_meter.avg)

        cf_m = np.zeros((self.args.total_class_num, self.args.total_class_num), dtype=float)
        for i in range(len(attri_pred_lb)):
            cf_m[gt_lb[i], attri_pred_lb[i]] += 1.0
        for i in range(cf_m.shape[0]):
            cf_m[i] = (cf_m[i] / cf_m[i].sum()) * 100

        #
        ax = sns.heatmap(cf_m, annot=False, fmt='.2f', vmin=0, vmax=100, cbar=False, square=True)
        ax.xaxis.set_ticklabels(['0', '', '', '', '', '5', '', '', '', '', '10', '', '', '', '', '15', ''])
        ax.yaxis.set_ticklabels(['0', '', '', '', '', '5', '', '', '', '', '10', '', '', '', '', '15', ''])

        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')

        # Display the visualization of the Confusion Matrix.
        plt.savefig(osp.join(self.args.save_folder, 'cf_all.svg'))
        plt.close()

        ###
        cf_m_unk = cf_m[self.args.shared_class_num:, self.args.shared_class_num:]
        bx = sns.heatmap(cf_m_unk, annot=True, fmt='.2f', vmin=0, vmax=100, cbar=True, square=True)
        bx.xaxis.set_ticklabels(['10', '11', '12', '13', '14', '15', '16'])
        bx.yaxis.set_ticklabels(['10', '11', '12', '13', '14', '15', '16'])

        # Display the visualization of the Confusion Matrix.
        plt.savefig(osp.join(self.args.save_folder, 'cf_unk.svg'))
        plt.close()

        return

    @torch.no_grad()
    def eval_tgt_cluster(self, test_loader):
        self.netF.eval()
        self.netB.eval()

        cls_feat_dict = []
        general_unseen_feat = []
        for i in range(self.args.total_class_num):
            cls_feat_dict.append([])

        for batch_idx, batch in enumerate(test_loader):
            img = batch[0].cuda()
            lb = batch[1]  # 0 -- Cs
            ori_lb = batch[2]  # 0 -- (Cs+Ct-1)

            feat = self.netB(self.netF(img))
            for i in range(feat.size(0)):
                cls_feat_dict[ori_lb[i]].append(feat[i].unsqueeze(0))
                if lb[i] == self.args.shared_class_num:
                    general_unseen_feat.append(feat[i].unsqueeze(0))

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        general_unseen_feat = torch.cat(general_unseen_feat, dim=0)
        general_unseen_centroid = general_unseen_feat.mean(dim=0).unsqueeze(0)
        general_unseen_sigma = 0.5 * (1 - cos(general_unseen_feat, general_unseen_centroid).mean())

        cls_centroid_list = []  # Cs+Ck
        sigma_list = []         # Cs+Ck
        for i in range(len(cls_feat_dict)):
            cls_feat_dict[i] = torch.cat(cls_feat_dict[i], dim=0)
            centroid = cls_feat_dict[i].mean(dim=0).unsqueeze(0)
            cls_centroid_list.append(centroid)

            intra_dist = 0.5 * (1 - cos(cls_feat_dict[i], centroid).mean())
            sigma_list.append(intra_dist)

        ## Evaluate Cs+1 clusters
        db_1 = 0.
        for i in range(self.args.shared_class_num):
            factor_list = [1.0 * (sigma_list[i] + sigma_list[j]) /
                           (0.5 * (1 - cos(cls_centroid_list[i], cls_centroid_list[j]))) if i != j else 0
                           for j in range(self.args.shared_class_num)]
            factor_list.append(1.0 * (sigma_list[i] + general_unseen_sigma) /
                               (0.5 * (1 - cos(cls_centroid_list[i], general_unseen_centroid))))
            factor_list = torch.tensor(factor_list)

            db_1 += factor_list.max().item()

        factor_list = [1.0 * (general_unseen_sigma + sigma_list[j]) /
                       (0.5 * (1 - cos(general_unseen_centroid, cls_centroid_list[j])))
                       for j in range(self.args.shared_class_num)]
        factor_list = torch.tensor(factor_list)
        db_1 += factor_list.max().item()
        db_1 /= (self.args.shared_class_num + 1)

        ## Evaluate Cs+Ck clusters
        db_2 = 0.
        for i in range(self.args.total_class_num):
            factor_list = [1.0 * (sigma_list[i] + sigma_list[j]) /
                           (0.5 * (1 - cos(cls_centroid_list[i], cls_centroid_list[j]))) if i != j else 0
                           for j in range(self.args.total_class_num)]

            factor_list = torch.tensor(factor_list)

            db_2 += factor_list.max().item()
        db_2 /= self.args.total_class_num

        ## Evaluate Ck clusters
        db_3 = 0.
        for i in range(self.args.shared_class_num, self.args.total_class_num):
            factor_list = [1.0 * (sigma_list[i] + sigma_list[j]) /
                           (0.5 * (1 - cos(cls_centroid_list[i], cls_centroid_list[j]))) if i != j else 0
                           for j in range(self.args.shared_class_num, self.args.total_class_num)]

            factor_list = torch.tensor(factor_list)

            db_3 += factor_list.max().item()
        db_3 /= (self.args.total_class_num - self.args.shared_class_num)

        print("Davies–Bouldin index:")
        print("Cs+1: {:.3f} | Cs+Ck: {:.3f} | Ck: {:.3f}".format(db_1, db_2, db_3))

        return
