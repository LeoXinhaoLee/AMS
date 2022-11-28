import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import random
import subprocess
import time
from dataset.data_list import ImageSet
from models import train_wrapper
import models.utils as utils
import datetime


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train(args):
    # Prepare Attribute Vectors for each class
    attri_mat = torch.tensor(np.load(args.attri_path)).float().cuda()  # [total_cls_n, 85]
    attri_mat_ori = attri_mat.clone()

    # Prepare Data
    #source_set = ImageSet(args.source_path, args, train=True)
    if args.src_balance:
        # default
        source_set = ImageSet(args.source_path, args, train=True, balanced=True)  # Balanced source sampler
    else:
        source_set = ImageSet(args.source_path, args, train=True)
    target_set = ImageSet(args.target_path, args, train=True)
    test_set = ImageSet(args.target_path, args, train=False)

    source_loader = DataLoader(source_set, batch_size=args.batchsize, shuffle=True,
                               num_workers=4, drop_last=True)
    target_loader = DataLoader(target_set, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)

    test_loader = DataLoader(test_set, batch_size=2*args.batchsize, shuffle=False,
                               num_workers=4, drop_last=False)


    # For pseudo label estimation
    # train=False, drop_last=False
    # Since no training is performed in estimation
    # batch_size can be set larger to speed up
    source_estimate_set = ImageSet(args.source_path, args, train=False)
    target_estimate_set = ImageSet(args.target_path, args, train=False)

    source_estimate_loader = DataLoader(source_estimate_set, batch_size=2*args.batchsize, shuffle=False,
                                       num_workers=4, drop_last=False)
    target_estimate_loader = DataLoader(target_estimate_set, batch_size=2*args.batchsize, shuffle=False,
                                       num_workers=4, drop_last=False)


    s_loader_len = len(source_loader)
    t_loader_len = len(target_loader)

    # Prepare Networks
    model = train_wrapper.Solver(args, s_loader_len, t_loader_len)

    if args.eval_cluster:
        print('\nEvaluating Clustering on Target (Pre fine-tuning)----------------')
        model.eval_tgt_cluster(test_loader)

    print('Source Only Training----------------')  # Fine-tuning with margin on Source domain
    for epoch in range(args.pretrain_epochs):
        model.train_on_source(source_loader)

    if args.eval_cluster:
        print('\nEvaluating Clustering on Target (Post fine-tuning)----------------')
        model.eval_tgt_cluster(test_loader)

    if args.save_model:
        model.save_net(pretrain=True)

    if not args.no_test_after_pretrain:
        print('\nTesting on Source----------------')
        source_test_set = ImageSet(args.source_path, args, train=False)
        source_test_loader = DataLoader(source_test_set, batch_size=args.batchsize, shuffle=False,
                                        num_workers=4, drop_last=False)
        model.test_source_only(source_test_loader, attri_mat[:args.shared_class_num])

        print('\nTesting on Target----------------')
        model.test_source_only(test_loader, attri_mat)


    for epoch in range(args.epochs):
        print('\n\nEpoch {}----------------'.format(epoch+1))
        # centroids: tensor [shared_class_n+K, 256]
        # refined_lb: list of pseudo labels(int) of target samples
        centroids, refined_lb = model.estimate_pseudo_label(source_estimate_loader, target_estimate_loader, attri_mat)
        centroids = centroids.cuda()

        target_set_pseudo = ImageSet(args.target_path, args, train=True, pseudo_lb=refined_lb)
        target_loader = DataLoader(target_set_pseudo, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)

        model.train_on_loader(source_loader, target_loader, centroids, attri_mat, first_flag=(epoch==0))

        if (epoch+1) % args.test_interval == 0 or epoch == 0:
            # visual classification Acc & semantics recovery Acc (w/o re-projection)
            model.test_on_loader(test_loader, attri_mat)

            if args.revise_prototype == 'post':
                # pseudo_lb: (tensor) 0 -- Cs; t_ori_lb: (tensor) 0 -- Cs+Ct-1; knw_lb: (tensor) 0 -- Cs-1
                all_t_feat, pseudo_lb, t_ori_lb, all_attri_pred, \
                knw_visual_feat, knw_lb = model.test_on_loader_get_feat_lb(test_loader)

                ###
                # Compare: ESZSL re-projection on T Seen cls
                ###
                # V_0: Xs -> Ks
                V_0 = utils.calculate_vsmap(knw_visual_feat, knw_lb, attri_mat_ori[:args.shared_class_num],
                                            method='ESZSL')  # V: [512, 85]
                # semantics recovery Acc (w re-projection)
                model.test_on_loader_semantics(all_t_feat, all_attri_pred, pseudo_lb, t_ori_lb, V_0, attri_mat_ori)

    if args.save_model:
        model.save_net()
        if args.revise_prototype == 'post' and 'V_0' in locals():
            torch.save(V_0.cpu(), osp.join(args.save_folder, 'W.pt'))

    return


def wait_for_GPU_avaliable(gpu_id):
    isVisited = False
    while True:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,nounits,noheader']).decode()
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        #available_memory = gpu_memory_map[int(gpu_id)]
        available_memory = 0
        gpu_list = gpu_id.split(',')
        for g in gpu_list:
            available_memory += gpu_memory_map[int(g)]

        # wait unless GPU memory is more than 8000
        if available_memory < 7000:
            if not isVisited:
                print("GPU full, wait...........")
                isVisited = True
            time.sleep(120)
            continue
        else:
            print("Empty GPU! Start process!")
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    working_dir = osp.dirname(osp.abspath(__file__))  # the dir of this .py file

    parser.add_argument('--source', type=str, default='R', choices=['R', 'P', 'A', '3D2', 'P0-2', 'P0-5',
                                                                    '3D20-9', '3D20-19', '3D20-29'])
    parser.add_argument('--target', type=str, default='A', choices=['R', 'P', 'A'])
    parser.add_argument('--datadir', type=str, default=osp.join(working_dir, 'data'))
    parser.add_argument('--dataset_root', type=str, default='/media/room/date/DataSets',
                        choices=['/media/room/date/DataSets', '/media/dzk/TP/data', '/home/ljj/data'])
    parser.add_argument('--outputdir', type=str, default=osp.join(working_dir,'exp'))
    parser.add_argument('--expname', type=str, default='exp0')
    parser.add_argument('--test_name', type=str, default='test')
    parser.add_argument('--log_mode', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--batchsize', type=int, default=32)  # default: 32
    parser.add_argument('--src_balance', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='inv', choices=['inv', 'piece_wise'])
    parser.add_argument('--use_pretrain', type=bool, default=False)
    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--no_test_after_pretrain', default=False, action='store_true')
    parser.add_argument('--eval_cluster', default=False, action='store_true')
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--parallel', default=False, action='store_true')

    parser.add_argument('--net', type=str, default='resnet50')
    parser.add_argument('--bottleneck', type=int, default=512, help="Gz feat dim")
    parser.add_argument('--alpha', type=float, default= 0.001, help="mix centroids")
    parser.add_argument('--lambda1', type=float, default= 0.1, help="Partial alignment")
    parser.add_argument('--lambda2', type=float, default= 0.1, help="(Arc+Reg) Classification")
    parser.add_argument('--lambda3', type=float, default= 1, help="Attribute prediction")
    parser.add_argument('--margin', type=float, default=0.2)

    parser.add_argument('--reject_unk', type=str, default='class_wise', choices=['overall', 'class_wise'])
    parser.add_argument('--sec_ref', type=str, default='all', choices=['all', 'no_sec_ref'])
    parser.add_argument('--unk_cluster_n', type=int, default=None)

    parser.add_argument('--pretrain_epochs', type=int, default=0, help="Whether to fine-tune with margin")
    parser.add_argument('--pre_reg', default=False, action='store_true')
    parser.add_argument('--train_margin', default=False, action='store_true')
    parser.add_argument('--multi_unk', default=False, action='store_true')
    parser.add_argument('--train_reg', default=False, action='store_true')
    parser.add_argument('--corresp', default=False, action='store_true')
    parser.add_argument('--revise_prototype', default='no', choices=['no', 'post'], help="Revise class attribute vectors")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    if args.seed is None:
        args.seed = random.randint(0, 1000)
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # args.save_folder = osp.join(args.outputdir, args.source + '2' + args.target, args.expname)
    args.save_folder = osp.join(args.outputdir, 'seed_' + str(args.seed), args.expname, args.source + '2' + args.target,
                                args.test_name)
    # e.g. ./exp/seed_0/exp1/R2AwA_17/BCE_cos
    if not os.path.exists(args.save_folder):
        os.system('mkdir -p ' + args.save_folder)

    if args.log_mode:
        stdout_path = osp.join(args.save_folder, 'log.txt')
        sys.stdout = open(stdout_path, "w")
        stderr_path = osp.join(args.save_folder, 'log_err.txt')
        sys.stderr = open(stderr_path, "w")

    print(print_args(args))

    if args.source == 'R' and args.target == 'A':
        args.source_path = osp.join(args.datadir, 'DomainNet', 'Real_src_0-9.txt')
        args.target_path = osp.join(args.datadir, 'AwA2', 'AwA2_tgt_0-16.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 10
        args.total_class_num = 17
    elif args.source == 'R' and args.target == 'P':
        args.source_path = osp.join(args.datadir, 'DomainNet', 'Real_src_0-9.txt')
        args.target_path = osp.join(args.datadir, 'DomainNet', 'Painting_tgt_0-16.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 10
        args.total_class_num = 17
    elif args.source == 'P' and args.target == 'A':
        args.source_path = osp.join(args.datadir, 'DomainNet', 'Painting_src_0-9.txt')
        args.target_path = osp.join(args.datadir, 'AwA2', 'AwA2_tgt_0-16.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 10
        args.total_class_num = 17
    elif args.source == 'P' and args.target == 'R':
        args.source_path = osp.join(args.datadir, 'DomainNet', 'Painting_src_0-9.txt')
        args.target_path = osp.join(args.datadir, 'DomainNet', 'Real_tgt_0-16.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 10
        args.total_class_num = 17
    elif args.source == 'P0-2' and args.target == 'R':
        args.source_path = osp.join(args.datadir, 'DomainNet', 'Painting_src_0-2.txt')
        args.target_path = osp.join(args.datadir, 'DomainNet', 'Real_tgt_0-16.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 3
        args.total_class_num = 17
    elif args.source == 'P0-5' and args.target == 'R':
        args.source_path = osp.join(args.datadir, 'DomainNet', 'Painting_src_0-5.txt')
        args.target_path = osp.join(args.datadir, 'DomainNet', 'Real_tgt_0-16.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 6
        args.total_class_num = 17
    elif args.source == 'A' and args.target == 'P':
        args.source_path = osp.join(args.datadir, 'AwA2', 'AwA2_src_0-9.txt')
        args.target_path = osp.join(args.datadir, 'DomainNet', 'Painting_tgt_0-16.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 10
        args.total_class_num = 17
    elif args.source == 'A' and args.target == 'R':
        args.source_path = osp.join(args.datadir, 'AwA2', 'AwA2_src_0-9.txt')
        args.target_path = osp.join(args.datadir, 'DomainNet', 'Real_tgt_0-16.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 10
        args.total_class_num = 17
    elif args.source == '3D2' and args.target == 'A':
        args.source_path = osp.join(args.datadir, '3D2', '3D2_src_0-39.txt')
        args.target_path = osp.join(args.datadir, 'AwA2', 'AwA2_tgt_0-49.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_50.npy')
        args.shared_class_num = 40
        args.total_class_num = 50
    elif args.source == '3D20-9' and args.target == 'A':
        args.source_path = osp.join(args.datadir, '3D2', '3D2_src_0-9.txt')
        args.target_path = osp.join(args.datadir, 'AwA2', 'AwA2_tgt_0-49.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_50.npy')
        args.shared_class_num = 10
        args.total_class_num = 50
    elif args.source == '3D20-19' and args.target == 'A':
        args.source_path = osp.join(args.datadir, '3D2', '3D2_src_0-19.txt')
        args.target_path = osp.join(args.datadir, 'AwA2', 'AwA2_tgt_0-49.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_50.npy')
        args.shared_class_num = 20
        args.total_class_num = 50
    elif args.source == '3D20-29' and args.target == 'A':
        args.source_path = osp.join(args.datadir, '3D2', '3D2_src_0-29.txt')
        args.target_path = osp.join(args.datadir, 'AwA2', 'AwA2_tgt_0-49.txt')
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_50.npy')
        args.shared_class_num = 30
        args.total_class_num = 50
    else:
        print('Transfer direction {} to {} not supported yet!'.format(args.source, args.target))
        exit(1)

    if (args.unk_cluster_n is None) or (args.unk_cluster_n < 1):
        args.unk_cluster_n = args.total_class_num - args.shared_class_num

    print(datetime.datetime.now())
    wait_for_GPU_avaliable(args.gpuid)

    start_t = time.time()

    train(args)

    end_t = time.time()
    duration = end_t - start_t

    print('\nTraining Wall Time: {:.0f} h, {:.0f} min, {:.0f} s'.format(duration//3600, (duration%3600)//60, duration%60))
    print(datetime.datetime.now())

    if args.log_mode:
        sys.stdout.close()
        sys.stderr.close()
