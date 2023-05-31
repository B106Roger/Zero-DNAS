# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import os
import sys
from datetime import datetime
import yaml
import torch
import numpy as np
import torch.nn as nn
import tqdm
import shutil
import torchvision

import _init_paths

# import timm packages
from timm.utils import CheckpointSaver, update_summary
# from timm.loss import LabelSmoothingCrossEntropy
# from timm.data import Dataset, create_loader
# from timm.models import resume_checkpoint

# import apex as distributed package otherwise we use torch.nn.parallel.distributed as distributed package
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    USE_APEX = True
except ImportError:
    from torch.nn.parallel import DataParallel as DDP
    USE_APEX = False

# import models and training functions
from lib.utils.flops_table import FlopsEst
from lib.core.train import train_epoch, validate, train_epoch_sensitive
from lib.models.structures.supernet import gen_supernet
from lib.models.PrioritizedBoard import PrioritizedBoard
from lib.models.MetaMatchingNetwork import MetaMatchingNetwork
from lib.config import DEFAULT_CROP_PCT, IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from lib.utils.util import parse_config_args, get_logger, \
    create_optimizer_supernet, create_supernet_scheduler
from lib.utils.datasets import create_dataloader
from lib.utils.kd_utils import FeatureAdaptation
from lib.models.blocks.yolo_blocks import Conv, ConvNP, BottleneckCSP, BottleneckCSP2, set_algorithm_type
from lib.utils.general import check_img_size, labels_to_class_weights, is_parallel, compute_loss, test, ModelEMA, build_foreground_mask, compute_sensitive_loss, plot_images
from lib.utils.torch_utils import select_device
from lib.utils.attentive_sampling import collect_samples
from scipy.special import softmax
from lib.models.AttentiveNasSampler import ArchSampler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=25)
matplotlib.rc('axes', titlesize=15)
import random

def config_backup(backup_path, args):
    os.makedirs(backup_path, exist_ok=True)
    shutil.copy(args.cfg,  os.path.join(backup_path, os.path.basename(args.cfg)))
    shutil.copy(args.hyp,  os.path.join(backup_path, os.path.basename(args.hyp)))
    shutil.copy(args.data, os.path.join(backup_path, os.path.basename(args.data)))
    shutil.copy('lib/core/train.py', os.path.join(backup_path, 'core_train.py'))
    shutil.copy('tools/train_dnas.py', os.path.join(backup_path, 'tools_train_dnas.py'))
    shutil.copy('lib/models/structures/supernet.py', os.path.join(backup_path, 'supernet.py'))
    shutil.copy('lib/models/blocks/yolo_blocks.py', os.path.join(backup_path, 'yolo_blocks.py'))
    shutil.copy('lib/models/builders/build_supernet.py', os.path.join(backup_path, 'build_supernet.py'))
    
    
    
    with open(os.path.join(backup_path, 'commandline.txt'), 'w') as f:
        f.writelines(' '.join(sys.argv))
    
task_dict = {
    'DNAS-25':     { 'GFLOPS': 25,  'PARAMS': 34.84, 'CHOICES': {'n_bottlenecks': [0, 1, 2], 'gamma': [0.25, 0.50, 0.75]}},
    'DNAS-35':     { 'GFLOPS': 35,  'PARAMS': 34.84, 'CHOICES': {'n_bottlenecks': [0, 1, 2], 'gamma': [0.25, 0.50, 0.75]}},
    'DNAS-45':     { 'GFLOPS': 45,  'PARAMS': 34.84, 'CHOICES': {'n_bottlenecks': [0, 1, 2], 'gamma': [0.25, 0.50, 0.75]}},  
    
    'NAS-SS': { 'GFLOPS': 5.7,  'PARAMS': 32.0, 'CHOICES': {'n_bottlenecks': [0, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-S':  { 'GFLOPS': 7.0,  'PARAMS': 36.0, 'CHOICES': {'n_bottlenecks': [0, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-M':  { 'GFLOPS': 9.0,  'PARAMS': 40.0, 'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS':    { 'GFLOPS': 11.9, 'PARAMS': 52.5, 'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-L':  { 'GFLOPS': 16.5, 'PARAMS': 70.2, 'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
}
task_name = 'DNAS-25'
FLOP_RESOLUTION = None
TASK_FLOPS      = task_dict[task_name]['GFLOPS']     # e.g TASK_FLOPS  = 5  means 50 GFLOPs
TASK_PARAMS     = task_dict[task_name]['PARAMS']     # e.g TASK_PARAMS = 32 means 32 million parameters.
SEARCH_SPACES   = task_dict[task_name]['CHOICES']
def main():
    args, cfg = parse_config_args('super net training')
    # resolve logging
    # output_dir = os.path.join(cfg.SAVE_PATH,
    #                           "{}-{}".format(datetime.now().strftime('%m%d-%H:%M:%S'),
    #                                          cfg.MODEL))
    FLOP_RESOLUTION = (None, 3, cfg.search_resolution, cfg.search_resolution)
    output_dir = os.path.join(cfg.SAVE_PATH, cfg.exp_name)
    output_bakup_dir = os.path.join(output_dir, 'config')
    config_backup(output_bakup_dir, args)

    if args.local_rank == 0:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        logger_path = os.path.join(output_dir, "train.log")
        with open(logger_path, 'w') as file:
            pass
        logger = get_logger(logger_path)
    else:
        logger = None

    # initialize distributed parameters
    device = select_device(args.device, batch_size=cfg.DATASET.BATCH_SIZE)
    cfg.NUM_GPU = torch.cuda.device_count()
    cfg.WORKERS = torch.cuda.device_count()
    
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = 1
    args.global_rank = -1
    if args.local_rank == 0:
        logger.info(
            'Training on Process %d with %d GPUs.',
                args.local_rank, cfg.NUM_GPU)

    # fix random seeds
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set search space argument
    set_algorithm_type('DNAS')
    BottleneckCSP.set_search_space(cfg.search_space.BOTTLENECK_CSP)
    BottleneckCSP2.set_search_space(cfg.search_space.BOTTLENECK_CSP2)
    # generate supernet
    print('SEARCH_SPACES', SEARCH_SPACES)
    model, sta_num, resolution = gen_supernet(
        flops_minimum=cfg.SUPERNET.FLOPS_MINIMUM,
        flops_maximum=cfg.SUPERNET.FLOPS_MAXIMUM,
        choices=SEARCH_SPACES,
        num_classes=cfg.DATASET.NUM_CLASSES,
        drop_rate=cfg.NET.DROPOUT_RATE,
        global_pool=cfg.NET.GP,
        resunit=cfg.SUPERNET.RESUNIT,
        dil_conv=cfg.SUPERNET.DIL_CONV,
        slice=cfg.SUPERNET.SLICE,
        verbose=cfg.VERBOSE,
        logger=logger,
        init_temp=5.0)
    
    # print(model)
    # initialize meta matching networks
    MetaMN = MetaMatchingNetwork(cfg)
    
    # number of choice blocks in supernet
    choice_num = model.choices # First bottlecsp
    if args.local_rank == 0:
        logger.info('Supernet created, param count: %.2f M', (
            sum([m.numel() for m in model.parameters()]) / 1e6))
        logger.info('resolution: %d', (cfg.DATASET.IMAGE_SIZE))
        logger.info('choice number: %d', (choice_num))

    #initialize prioritized board
    prioritized_board = PrioritizedBoard(cfg, CHOICE_NUM=choice_num, sta_num=sta_num, acc_gap=0.06)
    # print(model.blocks[1])
    prunable_module_type = (nn.BatchNorm2d, )
    prunable_modules = []
    CBL_idx = []
    for idx, module in enumerate(model.modules()):
        if isinstance(module, ConvNP):
            CBL_idx.append(idx)    
    # initialize flops look-up table
    model_est = FlopsEst(model, input_shape=(None, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE))
    if args.collect_samples > 0:
        collect_samples(args.collect_samples, model, prioritized_board, model_est)
        exit()

    # optionally resume from a checkpoint
    optimizer_state = None
    resume_epoch = None
    if cfg.AUTO_RESUME:
        optimizer_state, resume_epoch = resume_checkpoint(
            model, cfg.RESUME_PATH)

    # create optimizer and resume from checkpoint
    if args.resume_theta_training:
        print(f'Resuming training from: {args.resume_theta_training}')
        model.thetas_main = torch.load(args.resume_theta_training)
    optimizer, theta_optimizer = create_optimizer_supernet(cfg, model, USE_APEX)
    model.module.update_main() if is_parallel(model) else model.update_main()
    
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state['optimizer'])

    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [cfg.DATASET.IMAGE_SIZE] * 2]
    
    cuda = device.type != 'cpu'
    if cuda and torch.cuda.device_count() > 1 and args.local_rank != -1:
        model = torch.nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs. device: {device}')

    model = model.to(device)
    
    # create learning rate scheduler
    lr_scheduler, num_epochs = create_supernet_scheduler(cfg, optimizer)

    start_epoch = resume_epoch if resume_epoch is not None else 0
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: %d', num_epochs)

    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    

    # Trainloader
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    train_thetas_path = data_dict['train_thetas']

    test_path = data_dict['val']
    nc, names = (1, ['item']) if args.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, args.data)  # check
    
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    
    dataloader_thetas, dataset_thetas = create_dataloader(train_thetas_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect,
                                            world_size=args.world_size)
    mlc = np.concatenate(dataset_thetas.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader_thetas)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, args.data, nc - 1)

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset_thetas.labels, nc).to(device)  # attach class weights
    model.names = names
    ############################
    # MY CODE PART
    ############################
    is_ddp = is_parallel(model)
    # arch_theta=(num_of_stage, num_of_choices) 
    arch_theta = torch.cat([theta().reshape(1, -1) for theta in (model.module.thetas if is_ddp else model.thetas )], dim=0)
    num_stage  = arch_theta.shape[0]
    num_choice = arch_theta.shape[1]
    
    ########################################
    # Random Sample 10 Architecture
    ########################################
    arch_list = []
    for i in range(7):
        sample_theta = torch.zeros((num_stage, num_choice)).cuda()
        for j in range(num_stage):
            choice_idx = random.randint(0, num_choice-1)
            sample_theta[j, choice_idx] = 1.
        arch_list.append(sample_theta)
        ########################################
        # Log Architecture Result
        ########################################
        arch_idx = i
        arch = sample_theta
        depth_choices = len(SEARCH_SPACES['n_bottlenecks'])
        gamma_choices = len(SEARCH_SPACES['gamma'])
        choice = arch.cpu().detach().numpy().argmax(axis=-1)
        
        depth = np.array(SEARCH_SPACES['n_bottlenecks'])[choice // depth_choices]
        gamma = np.array(SEARCH_SPACES['gamma'])[choice % gamma_choices]
        print(f'[Arch {arch_idx}] Depth: {[str(i) for i in depth]}  Gamma: {[str(i) for i in gamma]}')
        ########################################
        # Calculate FLOPS and PARAMS
        ########################################
        architecture_info = {
            'arch_type': 'continuous',
            'arch': arch
        }
        flops  = model.module.calculate_flops_new(architecture_info, model_est.flops_dict)   if is_ddp else model.calculate_flops_new(architecture_info, model_est.flops_dict)
        params = model.module.calculate_params_new(architecture_info, model_est.params_dict) if is_ddp else model.calculate_params_new(architecture_info, model_est.params_dict)
        print(f'[Arch {arch_idx}] FLOPS: {flops/1e3}G   PARAMS: {params}M')
    print('arch_theta', arch_theta.shape)
    

    
    score_list = [[] for _ in range(len(arch_list))] 
    from pathlib import Path
    import cv2
    import argparse
    import matplotlib
    from dropblock import DropBlock2D  # pip install git+https://github.com/miguelvr/dropblock.git#egg=dropblock
    matplotlib.rc('font', size=15)
    SAMPLE_DATA_POINT=4
    
    drop_layer = DropBlock2D(block_size=10, drop_prob=0.5)
    ################################################################################
    # Experiment 1 : Relative ranking for a set of model on different images.
    ################################################################################
    for iter_idx, (uimgs, targets, paths, _) in enumerate(dataloader_thetas):
        if iter_idx < SAMPLE_DATA_POINT: continue
        elif iter_idx > SAMPLE_DATA_POINT: break
        
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        drop_mask = torch.ones((imgs.shape[0], 1, imgs.shape[2], imgs.shape[3])).cuda()
        drop_mask = drop_layer(drop_mask) 
        
        drop_mask = (drop_mask > 0.).float()
        imgs_aug  = drop_mask * imgs
        
        score_list = [[] for _ in range(len(arch_list))] 
        for arch_idx, arch in enumerate(arch_list):
            pred     = model.module(imgs, arch)      if is_ddp else model(imgs, arch)
            pred_aug = model.module(imgs_aug, arch)  if is_ddp else model(imgs_aug, arch)
            
            mask_sizes = [torch.tensor(p.shape[2:4]) for p in pred[0][1]]
            masks = build_foreground_mask(mask_sizes, cfg.DATASET.BATCH_SIZE, targets.to(device), model)  # scaled by batch_size
            
            sen_loss, loss_items = compute_sensitive_loss(pred[0][1], pred_aug[0][1], targets, masks)  # scaled by batch_size
            loss_items = loss_items.cpu().detach().numpy()
            score_list[arch_idx].append(loss_items[:2].reshape((2,1)))
        
        ##################################################################################
        ### Save Origin / Augment Image
        ### Save Mask Size
        ### Save Score (One picture with 1 image calculate on different architecture)
        ##################################################################################
        # if iter_idx < SAMPLE_DATA_POINT: 
        result = plot_images(images=imgs,       targets=targets, paths=paths, fname=str(Path(output_dir) / ('imgs_batch%g.jpg' % iter_idx)))
        result = plot_images(images=imgs_aug,   targets=targets, paths=paths, fname=str(Path(output_dir) / ('augs_batch%g.jpg' % iter_idx)))


        ################################################################################
        # Experiment 2 : Different initialize weights for model on the same images.
        ################################################################################
        init_fn_list = [
            ('init_yolo',      lambda: model._initialize_weights()),
            ('init_efficient', lambda: model._initialize_efficientnet()),
        ]
        sample_points = 25
        for fn_idx, (kind, init_fn) in enumerate(init_fn_list):
            score_list = [[] for _ in range(len(arch_list))] 
            for s_idx in tqdm.tqdm(range(sample_points)):
                init_fn()
                for arch_idx, arch in enumerate(arch_list):
                    pred     = model.module(imgs, arch)      if is_ddp else model(imgs, arch)
                    pred_aug = model.module(imgs_aug, arch)  if is_ddp else model(imgs_aug, arch)
                    ########################
                    mask_sizes = [torch.tensor(p.shape[2:4]) for p in pred[0][1]]
                    masks = build_foreground_mask(mask_sizes, cfg.DATASET.BATCH_SIZE, targets.to(device), model)  # scaled by batch_size
                    #######################
                    sen_loss, loss_items = compute_sensitive_loss(pred[0][1], pred_aug[0][1], targets, masks)  # scaled by batch_size
                    loss_items = loss_items.cpu().detach().numpy()
                    score_list[arch_idx].append(loss_items[:2].reshape((2,1)))

            # print(score_list)
            ######################################
            # Save Origin / Augment Image
            # Save Mask Size
            # Save Score (One picture with 1 image calculate on different architecture)
            ######################################
            if True:
                logdir = output_dir 
                    
                x1 = [i for i in range(len(arch_list))]
                x2 = [i for i in range(len(arch_list))]   # 因為長條圖寬度 0.4，所以位移距離為除以 2 為 0.2
                x3 = [i-0.125 for i in range(len(arch_list))]   # 因為長條圖寬度 0.4，所以位移距離為除以 2 為 0.2
                x_str = [f'{i:02d}' for i in range(len(arch_list))]
                ##################
                y1  = [i*1e-6 for i in range(-12,13,2)]
                y_str = [str(i) for i in range(-12,13,2)]
                print('Init Index', fn_idx)
                # score_list=(arch, sample_points, 2)
                fore_list = np.array(score_list)[:,:,0].reshape((len(arch_list), -1))
                back_list = np.array(score_list)[:,:,1].reshape((len(arch_list), -1))
                sen_list  = np.array(score_list).sum(axis=(-1,-2)).reshape((len(arch_list), -1))
                print(fore_list.shape)
                print(back_list.shape)
                print(sen_list.shape)
                ############################################################################################################
                pos = [i * 2 for i in range(1, len(arch_list) + 1)]
                for fig_data, fig_name in zip([fore_list, back_list, sen_list],['fore','back','sense']):
                    fig, ax = plt.subplots()
                    print(fig_data.shape, len(fig_data), len(pos))
                    VP = ax.boxplot([item for item in fig_data], 
                        # positions=pos, 
                        labels=x_str, showfliers =True,
                        widths=0.25, patch_artist=True, showmeans=False, 
                        medianprops={"color": "white", "linewidth": 0.25},
                        boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.25},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5})
                    # plt.yticks(y1, y_str)

                    
                    ax.set_xlabel('Arch')
                    # ax.set_ylabel('Sensitivity (1e-6)')
                    fig.tight_layout()
                    fig.savefig(str(Path(logdir) /f'score_{iter_idx}_{kind}_{fig_name}.jpg'))
                

if __name__ == '__main__':
    main()
