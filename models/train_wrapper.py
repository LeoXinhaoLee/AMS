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
from models.Base_wrapper import Base_Solver


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75, type='inv'):
    if type == 'inv':
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
    elif type == 'piece_wise':
        if iter_num <= max_iter // 2:
            decay = 1
        else:
            decay = 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True

    return optimizer


class Solver(Base_Solver):
    
    def __init__(self, args, s_loader_len, t_loader_len):
        super(Solver, self).__init__(args, s_loader_len, t_loader_len)


    def train_on_source(self, source_loader):
        self.net.train()
        for batch_idx, batch in enumerate(source_loader):
            self.optimizer.zero_grad()

            img = batch[0].cuda()
            lb = batch[1].cuda()

            feat = self.netB(self.netF(img))  # [bs, 512]
            logits, cosine = self.netC(feat, lb)  # [bs, shared_cls_num]

            pred = logits.argmax(dim=1)
            acc = pred.eq(lb).float().mean()

            loss_arcdce = F.cross_entropy(logits, lb)
            loss_arcreg = -cosine.mean()
            if self.args.pre_reg:
                loss = loss_arcdce + loss_arcreg
            else:
                loss_arcreg = 0.
                loss = loss_arcdce

            loss.backward()
            self.optimizer.step()

        print('Src loss: {:.2f}={:.2f}+{:.2f} | Acc: {:.2%}'.format(loss, loss_arcdce, loss_arcreg, acc))


    @torch.no_grad()
    def test_source_only(self, test_loader, attri_mat):
        self.netF.eval()
        self.netB.eval()
        self.netC.eval()
        self.netA.eval()

        self.attri_meter.reset()
        self.cls_meter.reset()

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        for batch_idx, batch in enumerate(test_loader):
            img = batch[0].cuda()
            lb = batch[1]
            ori_lb = batch[2]

            feat = self.netB(self.netF(img))
            attri_pred = self.netA(feat)  # [bs, 85]

            attri_pred_lb = []
            for i in range(attri_pred.size(0)):
                cur_attri_pred = attri_pred[i].repeat(attri_mat.size(0), 1)
                cos_sim = cos(cur_attri_pred, attri_mat)  # [all_cls_n, ]
                attri_cls_pred = cos_sim.argmax().item()
                attri_pred_lb.append(attri_cls_pred)

            attri_pred_lb = np.array(attri_pred_lb)
            self.attri_meter.update(ori_lb.numpy(), (attri_pred_lb == ori_lb.numpy()).astype(int))

            pred_logits = self.netC(feat)
            pred_lb = pred_logits.argmax(dim=1)
            self.cls_meter.update(lb.numpy(), (pred_lb.cpu().numpy() == lb.numpy()).astype(int))


        print('Attri Cls Acc >> S: {:.2%} | U: {:.2%}'.format(self.attri_meter.avg[:self.args.shared_class_num].mean(),
                                                              self.attri_meter.avg[self.args.shared_class_num:].mean()))
        print('S Acc: ', end='')
        for i in range(self.args.shared_class_num):
            print('{:.2%} '.format(self.attri_meter.avg[i]), end='')

        print('\nU Acc: ', end='')
        for i in range(self.args.shared_class_num, self.args.total_class_num):
            print('{:.2%} '.format(self.attri_meter.avg[i]), end='')
        print('\n')

        print('Cls Acc >> OS*: {:.2%} | UNK: {:.2%}'.format(self.cls_meter.avg[:self.args.shared_class_num].mean(),
                                                            self.cls_meter.avg[-1]))
        print('Acc: ', end='')
        for i in range(self.args.shared_class_num+1):
            print('{:.2%} '.format(self.cls_meter.avg[i]), end='')
        print('\n')

        return


    def k_means_clustering(self, all_feat, init_centroids=None):
        """
        :param all_feat: torch, cuda
        :param init_centroids: torch, cuda
        :return:np
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        tgt_pseudo_lb = []
        tgt_score = []  # the bigger, the more confident

        for i in range(all_feat.size(0)):
            cur_feat = all_feat[i].repeat(init_centroids.size(0), 1)  # [N_cls, d]
            cos_dist = 0.5 * (1 - cos(cur_feat, init_centroids))
            prob = F.softmax(-1.0 * cos_dist, dim=0)
            tgt_pseudo_lb.append(prob.argmax().item())  # int
            tgt_score.append(prob.max().item())

        tgt_score = np.array(tgt_score)
        tgt_pseudo_lb = np.array(tgt_pseudo_lb, dtype=int)

        return tgt_score, tgt_pseudo_lb


    def get_centroids(self, all_feat, all_lb, num_cls, empty_fill=None):
        """
        :param all_feat: np
        :param all_lb: np
        :param num_cls: int
        :return: np
        """
        cls_feat_dict = {}
        for i in range(num_cls):
            cls_feat_dict[i] = []

        for i in range(all_feat.shape[0]):
            cls_feat_dict[all_lb[i]].append(all_feat[i].reshape(1,-1))

        centroids = np.zeros((num_cls, all_feat.shape[1]))

        for i in range(num_cls):
            if len(cls_feat_dict[i]) > 0:
                centroids[i] = np.concatenate(cls_feat_dict[i], axis=0).mean(axis=0)
            else:
                centroids[i] = empty_fill[i]

        return centroids

    @torch.no_grad()
    def estimate_pseudo_label(self, source_loader, target_loader, attri_mat):
        print('Estimating Pseudo Labels...')
        self.estimate_meter.reset()
        self.estimate_refine_meter.reset()

        self.netF.eval()
        self.netB.eval()
        self.netC.eval()

        # Source Centroids
        source_cls_feat = {}
        for i in range(self.args.shared_class_num):
            source_cls_feat[i] = []

        all_src_feat = []
        all_src_gt_lb = []
        for batch_idx, batch in enumerate(source_loader):
            img = batch[0].cuda()
            lb = batch[1]

            feat = self.netB(self.netF(img))  # [bs, 256]
            if feat.dim() == 1:
                feat = feat.view(1, -1)

            all_src_feat.append(feat)
            all_src_gt_lb.append(lb)

            for i in range(lb.size()[0]):
                source_cls_feat[lb[i].item()].append(feat[i].view(1,-1))

        all_src_feat = torch.cat(all_src_feat, dim=0)
        all_src_gt_lb = torch.cat(all_src_gt_lb, dim=0).numpy()

        source_cls_centroids = torch.zeros((self.args.shared_class_num, self.args.bottleneck)).cuda()  # [shared_cls_n, 256]
        for k in source_cls_feat.keys():
            source_cls_centroids[k] = torch.cat(source_cls_feat[k], dim=0).mean(dim=0)  # [256,]

        # Target Centroids
        all_tgt_feat = []
        all_tgt_gt_lb = []
        for batch_idx, batch in enumerate(target_loader):
            img = batch[0].cuda()
            lb = batch[1]

            feat = self.netB(self.netF(img))  # [bs, 256]
            if feat.dim() == 1:
                feat = feat.view(1, -1)

            all_tgt_feat.append(feat)
            all_tgt_gt_lb.append(lb)

        all_tgt_feat = torch.cat(all_tgt_feat, dim=0)
        all_tgt_gt_lb = torch.cat(all_tgt_gt_lb, dim=0).numpy()

        # Inital clustering
        tgt_prob, tgt_pseudo_lb = self.k_means_clustering(all_tgt_feat, init_centroids=source_cls_centroids)

        if self.args.reject_unk == 'overall':
            thresh = tgt_prob.mean()
            tgt_pseudo_lb[tgt_prob<thresh] = self.args.shared_class_num  # modify as Unk

        elif self.args.reject_unk == 'class_wise':
            # Reject class-wise
            for i in range(self.args.shared_class_num):
                cur_cls_indice = (tgt_pseudo_lb == i)
                cur_cls_prob_mean = tgt_prob[cur_cls_indice].mean()
                cur_cls_rej_id = (tgt_prob[cur_cls_indice] < cur_cls_prob_mean)
                tgt_pseudo_lb[np.where(cur_cls_indice==True)[0][cur_cls_rej_id]] = self.args.shared_class_num

        ### Initial clustering result
        self.estimate_meter.update(all_tgt_gt_lb, (all_tgt_gt_lb==tgt_pseudo_lb).astype(int))

        target_cls_feat = {}
        for i in range(self.args.shared_class_num+1):
            target_cls_feat[i] = []

        for i in range(all_tgt_feat.size(0)):
            target_cls_feat[tgt_pseudo_lb[i]].append(all_tgt_feat[i].view(1,-1))

        target_knw_centroids = torch.zeros((self.args.shared_class_num, self.args.bottleneck)).cuda()  # [shared_cls_n, 256]

        for i in range(self.args.shared_class_num):
            if target_cls_feat[i] == []:
                target_cls_feat[i].append(source_cls_centroids[i].view(1,-1))

            target_knw_centroids[i] = torch.cat(target_cls_feat[i], dim=0).mean(dim=0)


        # Adjust shared class centroids
        # [shared_cls_n, 256]
        shared_cls_centroids = (1-self.args.alpha) * source_cls_centroids + self.args.alpha * target_knw_centroids
        shared_cls_centroids = shared_cls_centroids.cpu().numpy()

        # [K, 256]: Clustered K Centroids on T UNK
        tgt_unk_feat = torch.cat(target_cls_feat[self.args.shared_class_num], dim=0).cpu().numpy()
        tgt_unk_feat_norm = preprocessing.normalize(tgt_unk_feat, axis=1)
        unk_cls_lb = KMeans(n_clusters=self.args.unk_cluster_n).fit(tgt_unk_feat_norm).labels_  # 0-(K-1)

        unk_cls_feat_dict = {}
        for i in range(self.args.unk_cluster_n):
            unk_cls_feat_dict[i] = []

        for i in range(tgt_unk_feat.shape[0]):
            unk_cls_feat_dict[unk_cls_lb[i]].append(tgt_unk_feat[i].reshape(1,-1))

        unk_cls_centroids = np.zeros((self.args.unk_cluster_n, tgt_unk_feat.shape[1]), dtype=float)
        for i in range(unk_cls_centroids.shape[0]):
            unk_cls_centroids[i] = np.concatenate(unk_cls_feat_dict[i], axis=0).mean(axis=0)
        unk_cls_lb += self.args.shared_class_num  # shared_cls_n -- (shared_cls_n+K-1)


        # Second-time clustering refinement
        all_cls_centroids = np.concatenate([shared_cls_centroids, unk_cls_centroids], axis=0)
        all_tgt_lb = copy.deepcopy(tgt_pseudo_lb)
        all_tgt_lb[np.where(all_tgt_lb == self.args.shared_class_num)[0]] = unk_cls_lb

        if self.args.sec_ref == 'all':
            _, refined_lb = self.k_means_clustering(all_tgt_feat,
                                                    init_centroids=torch.tensor(all_cls_centroids).float().cuda())
            # Visual centroids
            refined_all_cls_centroids = self.get_centroids(all_tgt_feat.cpu().numpy(), refined_lb,
                                                           self.args.shared_class_num+self.args.unk_cluster_n,
                                                           empty_fill=all_cls_centroids)  # [shared_n+K, d]
            refined_centroids = copy.deepcopy(refined_all_cls_centroids)
            refined_centroids[:self.args.shared_class_num] = (1-self.args.alpha) * source_cls_centroids.cpu().numpy() + \
                                                             self.args.alpha * refined_centroids[:self.args.shared_class_num]
        else:
            # No second refinement
            refined_lb = all_tgt_lb
            refined_centroids = all_cls_centroids

        all_tgt_refined_lb = copy.deepcopy(refined_lb)
        all_tgt_refined_lb[all_tgt_refined_lb >= self.args.shared_class_num] = self.args.shared_class_num
        self.estimate_refine_meter.update(all_tgt_gt_lb, (all_tgt_gt_lb == all_tgt_refined_lb).astype(int))

        print('Estimation Done!')

        ### Pre-refinement performance
        os_star = self.estimate_meter.avg[:self.args.shared_class_num].mean()
        unk_acc = self.estimate_meter.avg[self.args.shared_class_num]
        h_score = 2 * os_star * unk_acc / (os_star + unk_acc)
        print('Pre-refinement >> OS*: {:.2%} | UNK: {:.2%} | H: {:.2%}'.format(os_star,unk_acc,h_score))
        print('Acc: ', end='')
        for k in range(self.estimate_meter.avg.shape[0]):
            print('{:.2%} '.format(self.estimate_meter.avg[k]), end='')
        print('\n')

        ### Post-refinement performance
        os_star = self.estimate_refine_meter.avg[:self.args.shared_class_num].mean()
        unk_acc = self.estimate_refine_meter.avg[self.args.shared_class_num]
        h_score = 2 * os_star * unk_acc / (os_star + unk_acc)
        print('Post-refinement >> OS*: {:.2%} | UNK: {:.2%} | H: {:.2%}'.format(os_star, unk_acc, h_score))
        print('Acc: ', end='')
        for k in range(self.estimate_refine_meter.avg.shape[0]):
            print('{:.2%} '.format(self.estimate_refine_meter.avg[k]), end='')
        print('\n')

        self.cluster_acc_logger.add_record([os_star * 100, unk_acc * 100, h_score * 100])

        return torch.tensor(refined_centroids).float(), refined_lb.tolist()

    def train_on_loader(self, source_loader, target_loader, centroids, attri_dict, first_flag=True):
        self.netF.train()
        self.netB.train()
        self.netA.train()
        self.netC_all.train()
        self.netD.train()

        total_batch = min(len(source_loader), len(target_loader))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for batch_idx, (batch_s, batch_t) in enumerate(zip(source_loader, target_loader)):

            self.optimizer.zero_grad()

            img_s = batch_s[0].cuda()
            lb_s = batch_s[1].cuda()         # 0--(shared_cls-1)

            img_t = batch_t[0].cuda()
            pseudo_lb_t = batch_t[3].cuda()  # 0--(shared_cls+K-1): estimated info, used for training
            #lb_t = batch_t[1].cuda()         # 0--shared_cls: gt info, only for testing

            feat_s = self.netB(self.netF(img_s))
            feat_t = self.netB(self.netF(img_t))

            #####
            # Partial Alignment
            #####

            # Source batch
            partial_align_loss_s = 0.
            for i in range(feat_s.size(0)):
                align_dist = 0.5 * (1 - cos(feat_s[i].view(1,-1), centroids))
                attract = align_dist[lb_s[i]]
                distract = -1.0 * (align_dist.sum() - attract) / (align_dist.size(0)-1)
                partial_align_loss_s = partial_align_loss_s + attract + distract
            partial_align_loss_s = partial_align_loss_s / feat_s.size(0)

            # Target batch
            partial_align_loss_t = 0.
            for i in range(feat_t.size(0)):
                align_dist = 0.5 * (1 - cos(feat_t[i].view(1,-1), centroids))
                attract = align_dist[pseudo_lb_t[i]]
                distract = -1.0 * (align_dist.sum() - attract) / (align_dist.size(0) - 1)
                partial_align_loss_t = partial_align_loss_t + attract + distract
            partial_align_loss_t = partial_align_loss_t / feat_t.size(0)

            partial_align_loss = partial_align_loss_s + partial_align_loss_t


            #####
            # Attribute Propagation
            #####

            att_pred_loss = torch.tensor([0.]).cuda()
            att_pred_loss_knw = torch.tensor([0.]).cuda()

            all_feat = torch.cat([feat_s, feat_t], dim=0)
            all_lb = torch.cat([lb_s, pseudo_lb_t])

            knw_feat = all_feat[all_lb < self.args.shared_class_num]
            knw_lb = all_lb[all_lb < self.args.shared_class_num]

            all_pred_attri = self.netA(all_feat)

            ## Knw Attri loss
            pred_attri_knw = all_pred_attri[all_lb < self.args.shared_class_num]
            prop_attri_knw = self.AttributeProp(knw_feat, pred_attri_knw)    # [Ns+Nts, 85]
            prop_attri_knw = prop_attri_knw.clamp(min=0., max=1.)            # Strictly proved, propogator with
                                                                             # row L-1 normalization will only guarantee
                                                                             # element <=1, but can't guarantee >=0
            knw_attri = torch.zeros((knw_lb.size(0), 85)).cuda()
            for i in range(knw_attri.size(0)):
                knw_attri[i] = attri_dict[knw_lb[i].item()]
            att_pred_loss_knw = F.binary_cross_entropy(prop_attri_knw.flatten(), knw_attri.flatten())

            att_pred_loss = att_pred_loss_knw


            #####
            # Visual-Semantic Joint Classification Training
            #####

            # All S: feat + gt attri & feat + pred attri
            feat_joint_s_gt = torch.zeros((feat_s.size(0), feat_s.size(1)+85)).cuda()
            feat_joint_s_pred = torch.zeros((feat_s.size(0), feat_s.size(1)+85)).cuda()

            attri_s_pred = self.netA(feat_s)  # [Ns, 85]

            for i in range(feat_joint_s_gt.size(0)):
                feat_joint_s_gt[i] = torch.cat([feat_s[i], attri_dict[lb_s[i].item()]])
                feat_joint_s_pred[i] = torch.cat([feat_s[i], attri_s_pred[i]])

            cls_loss_s = 0.
            if self.args.train_margin:
                logits_s_1, cosine_s_1 = self.netC_all(feat_joint_s_gt, lb_s)  # [Ns, Cs+K]
                logits_s_2, cosine_s_2 = self.netC_all(feat_joint_s_pred, lb_s)
                if self.args.train_reg:
                    cls_loss_s = F.cross_entropy(logits_s_1, lb_s) + F.cross_entropy(logits_s_2, lb_s) - \
                                 cosine_s_1.mean() - cosine_s_2.mean()
                else:
                    cls_loss_s = F.cross_entropy(logits_s_1, lb_s) + F.cross_entropy(logits_s_2, lb_s)
            else:
                logits_s_1 = self.netC_all(feat_joint_s_gt)  # [Ns, Cs+1]
                logits_s_2 = self.netC_all(feat_joint_s_pred)
                cls_loss_s = F.cross_entropy(logits_s_1, lb_s) + F.cross_entropy(logits_s_2, lb_s)

            # Target
            cls_loss_t = 0.

            # Knw T: feat + pseudo attri
            feat_joint_t_knw_pseudo = []
            lb_t_knw_pseudo = []
            for i in range(feat_t.size(0)):
                if pseudo_lb_t[i] < self.args.shared_class_num:
                    feat_joint_t_knw_pseudo.append(torch.cat([feat_t[i], attri_dict[pseudo_lb_t[i].item()]]).view(1,-1))
                    lb_t_knw_pseudo.append(pseudo_lb_t[i])

            if feat_joint_t_knw_pseudo is not []:
                feat_joint_t_knw_pseudo = torch.cat(feat_joint_t_knw_pseudo, dim=0)
                lb_t_knw_pseudo = torch.tensor(lb_t_knw_pseudo).cuda()

                if self.args.train_margin:
                    logits_t_1, cosine_t_1 = self.netC_all(feat_joint_t_knw_pseudo, lb_t_knw_pseudo)
                    if self.args.train_reg:
                        cls_loss_t += (F.cross_entropy(logits_t_1, lb_t_knw_pseudo) - cosine_t_1.mean())
                    else:
                        cls_loss_t += F.cross_entropy(logits_t_1, lb_t_knw_pseudo)
                else:
                    logits_t_1 = self.netC_all(feat_joint_t_knw_pseudo)
                    cls_loss_t += F.cross_entropy(logits_t_1, lb_t_knw_pseudo)

            # All T: feat + pred attri
            attri_t_pred = self.netA(feat_t)  # [Ns, 85]

            feat_joint_t_pred = torch.zeros((feat_t.size(0), feat_t.size(1)+85)).cuda()

            for i in range(feat_joint_t_pred.size(0)):
                feat_joint_t_pred[i] = torch.cat([feat_t[i], attri_t_pred[i]])

            if self.args.train_margin:
                if not self.args.multi_unk:
                    lb_t_all_pred = copy.deepcopy(pseudo_lb_t)
                    lb_t_all_pred[lb_t_all_pred >= self.args.shared_class_num] = self.args.shared_class_num

                    logits_t_2, cosine_t_2 = self.netC_all(feat_joint_t_pred, lb_t_all_pred)  # [Nt, Cs+K]
                    if self.args.train_reg:
                        cls_loss_t += (F.cross_entropy(logits_t_2, lb_t_all_pred) - cosine_t_2.mean())
                    else:
                        cls_loss_t += F.cross_entropy(logits_t_2, lb_t_all_pred)
                else:
                    # multi unk
                    if self.args.corresp:
                        if first_flag:
                            logits_t_2, cosine_t_2 = self.netC_all(feat_joint_t_pred, pseudo_lb_t)  # [Nt, Cs+K]
                            if self.args.train_reg:
                                cls_loss_t += (F.cross_entropy(logits_t_2, pseudo_lb_t) - cosine_t_2.mean())
                            else:
                                cls_loss_t += F.cross_entropy(logits_t_2, pseudo_lb_t)
                        else:
                            lb_t_all_pred = copy.deepcopy(pseudo_lb_t)
                            with torch.no_grad():
                                if feat_joint_t_pred[lb_t_all_pred >= self.args.shared_class_num].size(0) > 0:
                                    estim_logits_unk_t = self.netC_all(feat_joint_t_pred[lb_t_all_pred >= self.args.shared_class_num])
                                    if estim_logits_unk_t.dim() == 1:
                                        estim_logits_unk_t = estim_logits_unk_t.view(1,-1)  # later argmax along dim=1
                                    corresp_unk_lb = estim_logits_unk_t[:,self.args.shared_class_num:].argmax(dim=1) + \
                                                     self.args.shared_class_num
                                    lb_t_all_pred[lb_t_all_pred >= self.args.shared_class_num] = corresp_unk_lb

                            logits_t_2, cosine_t_2 = self.netC_all(feat_joint_t_pred, lb_t_all_pred)  # [Nt, Cs+K]
                            if self.args.train_reg:
                                cls_loss_t += (F.cross_entropy(logits_t_2, lb_t_all_pred) - cosine_t_2.mean())
                            else:
                                cls_loss_t += F.cross_entropy(logits_t_2, lb_t_all_pred)
                    else:
                        logits_t_2, cosine_t_2 = self.netC_all(feat_joint_t_pred, pseudo_lb_t)  # [Nt, Cs+K]
                        if self.args.train_reg:
                            cls_loss_t += (F.cross_entropy(logits_t_2, pseudo_lb_t) - cosine_t_2.mean())
                        else:
                            cls_loss_t += F.cross_entropy(logits_t_2, pseudo_lb_t)
            else:
                lb_t_all_pred = copy.deepcopy(pseudo_lb_t)  # if not deepcopy, modify lb_t_all_pred will change pseudo_lb_t
                lb_t_all_pred[lb_t_all_pred >= self.args.shared_class_num] = self.args.shared_class_num
                logits_t_2 = self.netC_all(feat_joint_t_pred)  # [Nt, K+1]
                cls_loss_t += F.cross_entropy(logits_t_2, lb_t_all_pred)

            cls_loss = cls_loss_s + cls_loss_t

            # D: discriminate Target domain knw vs. unk
            # Knw--1, Unk--0
            if not isinstance(feat_joint_t_knw_pseudo, list):
                # comparison between tensor and list should by `is`
                logits_t_knw_pseudo = self.netD(feat_joint_t_knw_pseudo)  # [Nts, 2]
                bi_lb_knw = torch.ones(logits_t_knw_pseudo.size(0), dtype=torch.int64).cuda()
                cls_loss += F.cross_entropy(logits_t_knw_pseudo, bi_lb_knw)

            logits_t_all_pred = self.netD(feat_joint_t_pred)  # [Nt, 2]
            bi_lb_all = torch.zeros(logits_t_all_pred.size(0), dtype=torch.int64).cuda()
            bi_lb_all[pseudo_lb_t < self.args.shared_class_num] = 1
            cls_loss += F.cross_entropy(logits_t_all_pred, bi_lb_all)

            ###
            total_loss = self.args.lambda1 * partial_align_loss + \
                         self.args.lambda2 * cls_loss + \
                         self.args.lambda3 * att_pred_loss

            #torch.cuda.empty_cache()
            total_loss.backward()
            self.optimizer.step()
            #torch.cuda.empty_cache()

            self.iter_num += 1
            lr_scheduler(self.optimizer, iter_num=self.iter_num, max_iter=self.max_iter, type=self.args.lr_scheduler)

            if (batch_idx+1) % 20 == 0:
                print('Batch: {:<3}/{} | Total Loss: {:.2f} | Cls Loss: {:.2f} | Partial Loss: {:2f} | Attri Loss: '
                      '{:.2f}'.format(
                    batch_idx, total_batch, total_loss.item(), self.args.lambda2 * cls_loss.item(),
                    self.args.lambda1 * partial_align_loss.item(), self.args.lambda3 * att_pred_loss.item()
                ))
                self.train_loss_logger.add_record([total_loss.item(), self.args.lambda2 * cls_loss.item(),
                                                   self.args.lambda1 * partial_align_loss.item(),
                                                   self.args.lambda3 * att_pred_loss.item()])
        return

    @torch.no_grad()
    def test_on_loader(self, test_loader, attri_mat):
        print('\nTesting...')

        self.netF.eval()
        self.netB.eval()
        self.netA.eval()
        self.netC.eval()
        self.netC_all.eval()
        self.netD.eval()

        self.cls_meter.reset()
        self.attri_meter.reset()

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        for batch_idx, batch in enumerate(test_loader):
            img = batch[0].cuda()
            lb = batch[1]       # 0 -- Cs
            ori_lb = batch[2]   # 0 -- (Cs+Ct-1)

            feat = self.netB(self.netF(img))
            attri_pred = self.netA(feat)  # [bs, 85]
            feat_joint = torch.cat([feat, attri_pred], dim=1)  # [bs, 256+85]
            logits = self.netC_all(feat_joint)  # [bs, K+1]

            pred_lb = logits.argmax(dim=1)
            pred_lb[pred_lb >= self.args.shared_class_num] = self.args.shared_class_num
            self.cls_meter.update(lb.numpy(), (pred_lb.cpu().numpy()==lb.numpy()).astype(int))

            attri_pred_lb = []
            for i in range(attri_pred.size(0)):
                cos_sim = cos(attri_pred[i].view(1,-1), attri_mat)  # [all_cls_n, ]
                if pred_lb[i] < self.args.shared_class_num:
                    attri_cls_pred = cos_sim[:self.args.shared_class_num].argmax().item()
                else:
                    attri_cls_pred = cos_sim[self.args.shared_class_num:].argmax().item() + self.args.shared_class_num

                attri_pred_lb.append(attri_cls_pred)

            attri_pred_lb = np.array(attri_pred_lb)
            self.attri_meter.update(ori_lb.numpy(), (attri_pred_lb==ori_lb.numpy()).astype(int))

        print('---Open Set Rocognition---')
        os_star = self.cls_meter.avg[:self.args.shared_class_num].mean()
        unk_acc = self.cls_meter.avg[self.args.shared_class_num]
        H_score = 2 * os_star * unk_acc / (os_star + unk_acc)
        print('OS*: {:.2%} | UNK: {:.2%} | H: {:.2%}'.format(os_star, unk_acc, H_score))
        print('Acc: ', end='')
        for k in range(self.cls_meter.avg.shape[0]):
            print('{:.2%} '.format(self.cls_meter.avg[k]), end='')

        self.test_acc_logger_rec.add_record([os_star*100, unk_acc*100, H_score*100])

        print('\n---Semantics Recovery---')
        seen_acc = self.attri_meter.avg[:self.args.shared_class_num].mean()
        unseen_acc = self.attri_meter.avg[self.args.shared_class_num:].mean()
        H_score = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
        print('Seen Acc: {:.2%} | Unseen Acc: {:.2%} | H: {:.2%}'.format(seen_acc, unseen_acc, H_score))
        print('S Acc: ', end='')
        for i in range(self.args.shared_class_num):
            print('{:.2%} '.format(self.attri_meter.avg[i]), end='')
        print('\nU Acc: ', end='')
        for i in range(self.args.shared_class_num, self.args.total_class_num):
            print('{:.2%} '.format(self.attri_meter.avg[i]), end='')
        print('\n')

        if self.args.revise_prototype == 'post':
            self.pre_rev = [seen_acc, unseen_acc, H_score]
        elif self.args.revise_prototype == 'no':
            self.test_acc_logger_sem.add_record([seen_acc*100, unseen_acc*100, H_score*100])

        self.cluster_acc_logger.save_record()
        self.train_loss_logger.save_record()
        self.test_acc_logger_rec.save_record()
        if self.args.revise_prototype == 'no':
            self.test_acc_logger_sem.save_record()

        return

    @torch.no_grad()
    def test_on_loader_get_feat_lb(self, test_loader):
        self.netF.eval()
        self.netB.eval()
        self.netA.eval()
        self.netC.eval()
        self.netC_all.eval()
        self.netD.eval()

        self.cls_meter.reset()
        self.attri_meter.reset()

        pred_lb_list = []
        feat_list = []
        ori_lb_list = []

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        all_attri_pred_list = []
        knw_visual_feat = []
        knw_lb_list = []

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

            #self.cls_meter.update(lb.numpy(), (pred_lb.cpu().numpy()==lb.numpy()).astype(int))
            # Already printed. Semantic revision does not affect visual classification result.

            if feat.dim() == 1:
                feat = feat.view(1,-1)
            if attri_pred.dim() == 1:
                attri_pred = attri_pred.view(1,-1)

            feat_list.append(feat)
            pred_lb_list.append(pred_lb)
            ori_lb_list.append(ori_lb)
            all_attri_pred_list.append(attri_pred)

            for i in range(attri_pred.size(0)):
                if pred_lb[i]< self.args.shared_class_num:
                    #knw_attri_pred.append(attri_pred[i].view(1, -1))
                    #knw_attri_gt.append(ori_knw_attri_mat[pred_lb[i]].view(1, -1))
                    knw_visual_feat.append(feat[i].view(1, -1))
                    knw_lb_list.append(pred_lb[i].item())

        return torch.cat(feat_list, dim=0), \
               torch.cat(pred_lb_list, dim=0), torch.cat(ori_lb_list, dim=0), \
               torch.cat(all_attri_pred_list, dim=0), \
               torch.cat(knw_visual_feat, dim=0), torch.tensor(knw_lb_list)


    def test_on_loader_semantics(self, t_feat, GA_attri_pred, pred_lb, ori_lb, V_0, attri_mat_ori):
        self.cls_meter.reset()
        self.attri_meter.reset()

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        attri_pred_post = torch.mm(t_feat, V_0)

        attri_pred_lb_post = []
        for i in range(attri_pred_post.size(0)):
            if pred_lb[i] >= self.args.shared_class_num:
                # no re-projection for unk cls
                attri_pred_post[i] = GA_attri_pred[i]

            sim_post = cos(attri_pred_post[i].view(1, -1), attri_mat_ori)

            if pred_lb[i] < self.args.shared_class_num:
                attri_cls_pred_post = sim_post[:self.args.shared_class_num].argmax().item()
            else:
                attri_cls_pred_post = sim_post[self.args.shared_class_num:].argmax().item() + self.args.shared_class_num

            attri_pred_lb_post.append(attri_cls_pred_post)

        attri_pred_lb_post = np.array(attri_pred_lb_post)

        self.attri_meter.reset()
        self.attri_meter.update(ori_lb.numpy(), (attri_pred_lb_post == ori_lb.numpy()).astype(int))
        print('---Semantics Recovery (post re-projection)---')
        seen_acc_post = self.attri_meter.avg[:self.args.shared_class_num].mean()
        unseen_acc_post = self.attri_meter.avg[self.args.shared_class_num:].mean()
        H_score_post = 2 * seen_acc_post * unseen_acc_post / (seen_acc_post + unseen_acc_post)
        print('Seen Acc: {:.2%} | Unseen Acc: {:.2%} | H: {:.2%}'.format(seen_acc_post, unseen_acc_post, H_score_post))
        print('S Acc: ', end='')
        for i in range(self.args.shared_class_num):
            print('{:.2%} '.format(self.attri_meter.avg[i]), end='')
        print('\nU Acc: ', end='')
        for i in range(self.args.shared_class_num, self.args.total_class_num):
            print('{:.2%} '.format(self.attri_meter.avg[i]), end='')
        print('\n')

        self.test_acc_logger_sem.add_record([self.pre_rev[0] * 100, self.pre_rev[1] * 100, self.pre_rev[2] * 100,
                                             seen_acc_post * 100, H_score_post * 100])
        self.test_acc_logger_sem.save_record()

        return
