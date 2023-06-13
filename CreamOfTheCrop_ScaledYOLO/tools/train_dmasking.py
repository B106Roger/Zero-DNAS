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

import _init_paths

# import timm packages
from timm.utils import CheckpointSaver, update_summary

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
from lib.core.train import train_epoch_dnas, train_epoch_dnas_V2
from lib.models.structures.supernet import gen_supernet
from lib.utils.util import convert_lowercase, get_logger, \
    create_optimizer_supernet, create_supernet_scheduler, stringify_theta, write_thetas, export_thetas
from lib.utils.datasets import create_dataloader
from lib.utils.general import check_img_size, labels_to_class_weights, is_parallel, compute_loss, test, ModelEMA, random_testing
from lib.utils.torch_utils import select_device
from lib.config import cfg
import argparse
import random

def config_backup(config_bakup_dir, code_backup_dir, args):
    os.makedirs(config_bakup_dir, exist_ok=True)
    os.makedirs(code_backup_dir,  exist_ok=True)
    
    shutil.copy(args.cfg,  os.path.join(config_bakup_dir, os.path.basename(args.cfg)))
    shutil.copy(args.hyp,  os.path.join(config_bakup_dir, os.path.basename(args.hyp)))
    shutil.copy(args.data, os.path.join(config_bakup_dir, os.path.basename(args.data)))
    shutil.copy(args.model,os.path.join(config_bakup_dir, os.path.basename(args.model)))
    with open(os.path.join(config_bakup_dir, 'commandline.txt'), 'w') as f:
        f.writelines(' '.join(sys.argv))
        
    shutil.copy('lib/core/train.py',                      os.path.join(code_backup_dir, 'core_train.py'))
    shutil.copy(f'{__file__}',                            os.path.join(code_backup_dir, os.path.basename(__file__)))
    shutil.copy('lib/models/structures/supernet.py',      os.path.join(code_backup_dir, 'supernet.py'))
    shutil.copy('lib/models/blocks/yolo_blocks.py',       os.path.join(code_backup_dir, 'yolo_blocks.py'))
    shutil.copy('lib/models/blocks/yolo_blocks_search.py',os.path.join(code_backup_dir, 'yolo_blocks_search.py'))
    shutil.copy('lib/models/builders/build_supernet.py',  os.path.join(code_backup_dir, 'build_supernet.py'))
    shutil.copy('lib/utils/general.py',                   os.path.join(code_backup_dir, 'general.py'))

def parse_config_args(exp_name):
    parser = argparse.ArgumentParser(description=exp_name)
    ###################################################################################
    # Commonly Used Parameter !!
    ###################################################################################
    parser.add_argument('--cfg',  type=str, help='configuration of cream')
    parser.add_argument('--data', type=str, default='config/dataset/voc.yaml', help='data.yaml path')
    parser.add_argument('--hyp',  type=str, default='config/dataset/training/hyp.scratch.yaml', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--model',type=str, default='config/model/Search-YOLOv4-CSP.yaml', help='model path')
    parser.add_argument('--exp_name', type=str, default='exp', help="name of experiments")
    parser.add_argument('--nas', default='', type=str, help='NAS-Search-Space and hardware constraint combination')
    parser.add_argument('--pretrain_dir',  default='', type=str, help='pretrain model state dict')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    ###################################################################################
    
    
    ###################################################################################
    # Seldom Used Parameter
    ###################################################################################
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--collect-samples', type=int, default=0, help='Sample a lot of different architectures with corresponding flops, if not 0 then samples specified number and exits the programm')
    parser.add_argument('--collect-synflows', type=int, default=0, help='Sample a lot of different architectures with corresponding synflows, if not 0 then samples specified number and exits the programm')
    parser.add_argument('--resume-theta-training', default='', type=str, help='load pretrained thetas')
    ###################################################################################
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.exp_name = args.exp_name
    converted_cfg = convert_lowercase(cfg)

    return args, converted_cfg

task_dict = {
    'DNAS-25':     { 'GFLOPS': 25,  'PARAMS': None, },
    'DNAS-35':     { 'GFLOPS': 35,  'PARAMS': None, },
    'DNAS-45':     { 'GFLOPS': 45,  'PARAMS': None, },
    
    'DNAS-70':     { 'GFLOPS': 70,  'PARAMS': None, },
    'DNAS-60':     { 'GFLOPS': 60,  'PARAMS': None, },
    'DNAS-50':     { 'GFLOPS': 50,  'PARAMS': None, },
    'DNAS-40':     { 'GFLOPS': 40,  'PARAMS': None, },
    'DNAS-30':     { 'GFLOPS': 30,  'PARAMS': None, },
        
    'NAS-SS': { 'GFLOPS': 5.7,  'PARAMS': 32.0, }, #'CHOICES': {'n_bottlenecks': [0, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-S':  { 'GFLOPS': 7.0,  'PARAMS': 36.0, }, #'CHOICES': {'n_bottlenecks': [0, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-M':  { 'GFLOPS': 9.0,  'PARAMS': 40.0, }, #'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS':    { 'GFLOPS': 11.9, 'PARAMS': 52.5, }, #'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-L':  { 'GFLOPS': 16.5, 'PARAMS': 70.2, }, #'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
}

def main():
    args, cfg = parse_config_args('super net training')
    with open(args.model ) as f:
        model_args   = yaml.load(f, Loader=yaml.FullLoader)
    search_space = model_args['search_space']
    
    task_name = args.nas if args.nas != '' else 'DNAS-25'
    TASK_FLOPS      = task_dict[task_name]['GFLOPS']     # e.g TASK_FLOPS  = 5  means 50 GFLOPs
    TASK_PARAMS     = task_dict[task_name]['PARAMS']     # e.g TASK_PARAMS = 32 means 32 million parameters.
    SEARCH_SPACES   = model_args['search_space']
    USE_AMP         = False
    FLOP_RESOLUTION = (None, 3, cfg.search_resolution, cfg.search_resolution)
    
    output_dir = os.path.join(cfg.SAVE_PATH, cfg.exp_name)
    config_bakup_dir = os.path.join(output_dir, 'config')
    code_backup_dir  = os.path.join(output_dir, 'code')
    theta_dir        = 'thetas_weights' # thetas.pt
    model_dir        = os.path.join(output_dir, 'model')
    model_w_dir      = os.path.join(output_dir, 'model_weights')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(theta_dir, exist_ok=True)
    os.makedirs(model_w_dir, exist_ok=True)
    config_backup(config_bakup_dir, code_backup_dir, args)

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
    random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # generate supernet
    print('SEARCH_SPACES', SEARCH_SPACES)
    model, sta_num, resolution = gen_supernet(
        model_args,
        num_classes=cfg.DATASET.NUM_CLASSES,
        verbose=cfg.VERBOSE,
        logger=logger,
        init_temp=cfg.TEMPERATURE.INIT)
    
    # number of choice blocks in supernet
    # choice_num = model.choices # First bottlecsp
    if args.local_rank == 0:
        logger.info('Supernet created, param count: %.2f M', (
            sum([m.numel() for m in model.parameters()]) / 1e6))
        logger.info('resolution: %d', (cfg.DATASET.IMAGE_SIZE))

    # initialize flops look-up table
    model_est = FlopsEst(model, input_shape=(None, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE), search_space=SEARCH_SPACES)

    # create optimizer and resume from checkpoint
    if args.resume_theta_training:
        print(f'Resuming training from: {args.resume_theta_training}')
        model.thetas_main = torch.load(args.resume_theta_training)
    optimizer, theta_optimizer = create_optimizer_supernet(cfg, model, USE_APEX)
    model.module.update_main() if is_parallel(model) else model.update_main()
    
    # if optimizer_state is not None:
    #     optimizer.load_state_dict(optimizer_state['optimizer'])

    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [cfg.DATASET.IMAGE_SIZE] * 2]
    
    # convert model to distributed mode
    if cfg.BATCHNORM.SYNC_BN:
        try:
            if USE_APEX:
                model = convert_syncbn_model(model)
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if args.local_rank == 0:
                logger.info('Converted model to use Synchronized BatchNorm.')
        except Exception as exception:
            logger.info(
                'Failed to enable Synchronized BatchNorm. '
                'Install Apex or Torch >= 1.1 with Exception %s', exception)
    
    cuda = device.type != 'cpu'
    if cuda and torch.cuda.device_count() > 1 and args.local_rank != -1:
        model = torch.nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs. device: {device}')

    model = model.to(device)
    
    # create learning rate scheduler
    lr_scheduler, num_epochs = create_supernet_scheduler(cfg, optimizer)

    # start_epoch = resume_epoch if resume_epoch is not None else 0
    # if start_epoch > 0:
    #     lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: %d', num_epochs)

    # Hyper Parameter Config
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Dataset Config
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    train_weight_path = data_dict['train_weight']
    train_thetas_path = data_dict['train_thetas']

    test_path = data_dict['val']
    nc, names = (1, ['item']) if args.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, args.data)  # check
    
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset

    dataloader_weight, dataset_weight = create_dataloader(train_weight_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect,
                                            world_size=args.world_size)
    dataloader_thetas, dataset_thetas = create_dataloader(train_thetas_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect,
                                            world_size=args.world_size)
    mlc = np.concatenate(dataset_weight.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader_weight)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, args.data, nc - 1)


    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset_weight.labels, nc).to(device)  # attach class weights
    model.names = names
    print('[Info] cfg.TEMPERATURE.INIT', cfg.TEMPERATURE.INIT)
    print('[Info] cfg.TEMPERATURE.FINAL', cfg.TEMPERATURE.FINAL)

    ema = ModelEMA(model) if args.local_rank in [-1, 0] else None

    # Testloader
    if args.local_rank in [-1, 0]:
        # ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        # if ema is not None:
        #     ema.updates = start_epoch * nb // 1  # set EMA updates ***
        testloader = create_dataloader(test_path, imgsz_test, 16, gs, args, hyp=hyp, augment=False,
                                       cache=args.cache_images, rect=True, local_rank=-1, world_size=args.world_size)[0]
    
    start_epoch = 0
    MODEL_WEIGHT_NAME = os.path.join(args.pretrain_dir, f'model_{cfg.FREEZE_EPOCH}.pt') 
    EMA_WEIGHT_NAME   = os.path.join(args.pretrain_dir, f'ema_pretrained_{cfg.FREEZE_EPOCH}.pt') 
    OPTIMIZER_NAME    = os.path.join(args.pretrain_dir, f'optimizer_{cfg.FREEZE_EPOCH}.pt')
    if args.pretrain_dir != '':
        start_epoch = 40
        
        # Load Supernet MOdel
        model.load_state_dict(torch.load(MODEL_WEIGHT_NAME), strict=False)
        
        # Load EMA Model
        if os.path.exists(EMA_WEIGHT_NAME):
            ema.ema.load_state_dict(torch.load(EMA_WEIGHT_NAME))
        else:
            ema = ModelEMA(model) if args.local_rank in [-1, 0] else None
        ema.updates      = 40 * len(dataloader_weight)
        ema.updates_arch = 40 * len(dataloader_thetas)
        
        # Load Model Weights Optimizer Parameter
        if os.path.exists(OPTIMIZER_NAME):
            optimizer.load_state_dict(torch.load(OPTIMIZER_NAME))
        
        # Restore Learning Rate
        lr_scheduler.step(start_epoch)
        
        # Calculate mAP of pretrain weights
        _, _, map50, *other = test(
            data=args.data, batch_size=16, imgsz=416, save_json=False,
            model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
            single_cls=False, dataloader=testloader, save_dir=output_dir, logger=logger
        )
        
    # training scheme
    method = 'ver1'
    sample_temp = 1.0
    
    try:
        print('TASK_FLOPS', TASK_FLOPS)
        print('FREEZE_EPOCH', cfg.FREEZE_EPOCH)
        print('EPOCH', num_epochs)
        filename = os.path.join(model_dir, f'DNAS-current_f{TASK_FLOPS}.yaml')
        export_thetas(model.softmax_sampling(detach=True), model, model.model_args, filename)

        for epoch in range(start_epoch+1, num_epochs+1):
            model.train()
            thetas_enable = False if epoch <= cfg.FREEZE_EPOCH else True
            # if epoch <= cfg.FREEZE_EPOCH:
            #     thetas_enable = False
            #     model.temperature = sample_temp
            # elif epoch == cfg.FREEZE_EPOCH+1:
            #     thetas_enable = True
            #     model.temperature = cfg.TEMPERATURE.INIT
            # else:
            #     thetas_enable = True
            
            ####################################################
            # Update Model Method 2
            # For epoch in epochs:
            #     For iteration in iterations:
            #         update model weights
            #         update model architectures
            ####################################################
            if method == 'ver2':
                train_epoch_dnas_V2(model, dataloader_weight, dataloader_thetas, optimizer, theta_optimizer, cfg, device=device, 
                    task_flops=TASK_FLOPS, task_params=TASK_PARAMS, logger=logger, 
                    est=model_est, local_rank=args.local_rank, world_size=args.world_size, 
                    epoch=epoch, total_epoch=num_epochs, logdir=output_dir, is_gumbel=True, ema=ema
                )
            
            ####################################################
            # Update Model Method 1
            # For epoch in epochs:
            #     For iteration in iterations:
            #         update model weights
            #     For iteration in iteartions:       
            #         update model architectures
            ####################################################
            if method == 'ver1':
                # Train Architecture First
                if thetas_enable:
                    train_epoch_dnas(model, dataloader_thetas, theta_optimizer, cfg, device=device, 
                        task_flops=TASK_FLOPS, task_params=TASK_PARAMS, logger=logger, 
                        est=model_est, local_rank=args.local_rank, world_size=args.world_size, use_amp=USE_AMP,
                        epoch=epoch, total_epoch=num_epochs, logdir=output_dir, is_gumbel=True, ema=ema, warmup=False, description="architecture"
                    )
                
                # Train Network Parameter
                train_epoch_dnas(model, dataloader_weight, optimizer, cfg, device=device, 
                    task_flops=TASK_FLOPS, task_params=TASK_PARAMS, logger=logger, 
                    est=model_est, local_rank=args.local_rank, world_size=args.world_size, use_amp=USE_AMP,
                    epoch=epoch, total_epoch=num_epochs, logdir=output_dir, is_gumbel=True, ema=ema, description="weights"
                )
                

                if epoch == cfg.FREEZE_EPOCH :
                    torch.save(model.state_dict(),     os.path.join(output_dir, 'model_weights', f'model_pretrained_{epoch}.pt'))
                    torch.save(ema.ema.state_dict(),   os.path.join(output_dir, 'model_weights', f'ema_pretrained_{epoch}.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'model_weights', f'optimizer_{epoch}.pt'))
                    
            if thetas_enable:
                # temp_decay = np.exp(-0.045) # 0.9560
                # temp_decay = 0.2**(1/80)    # 0.9800
                temp_decay = (cfg.TEMPERATURE.FINAL/cfg.TEMPERATURE.INIT)**(1/(num_epochs-cfg.FREEZE_EPOCH)) # 0.9560

                ##############################################################
                # Reduce & Save Architecture Temperature
                ##############################################################
                if is_parallel(model):
                    # Save Architecture Temperature
                    write_thetas(output_dir, ema.ema.module.thetas_main, epoch, model.module.temperature)
                    torch.save(ema.ema.module.thetas_main, os.path.join(theta_dir, 'thetas.pt'))
                    # Save Archtiecture Config
                    filename = os.path.join(model_dir, f'DNAS-current_f{TASK_FLOPS}.yaml')
                    export_thetas(model.module.softmax_sampling(detach=True), model, model.model_args, filename)
                    # Reduce Architecture Temperature
                    model.module.temperature = model.module.temperature * temp_decay
                else:
                    # Save Architecture Temperature
                    write_thetas(output_dir, ema.ema.thetas_main, epoch, model.temperature)
                    torch.save(ema.ema.thetas_main, os.path.join(theta_dir, 'thetas.pt'))
                    # Save Archtiecture Config
                    filename = os.path.join(model_dir, f'DNAS-current_f{TASK_FLOPS}.yaml')
                    export_thetas(model.softmax_sampling(detach=True), model, model.model_args, filename)
                    # Reduce Architecture Temperature
                    model.temperature = model.temperature * temp_decay
                
                if False:
                    ##############################################################
                    # Calculate Discrete Model Architecture FLOPS
                    ##############################################################
                    is_ddp = is_parallel(model)
                    
                    discrete_arch = ema.ema.discretize_sampling()
                    discrete_str_arch = stringify_theta(discrete_arch)
                    architecture_info = {
                        'arch_type': 'continuous',
                        'arch': discrete_arch
                    }
                    flops = ema.ema.calculate_flops_new(architecture_info, model_est.flops_dict) if is_ddp else ema.ema.calculate_flops_new(architecture_info, model_est.flops_dict)
                    logger.info(f'EMA Discrete FLOPS : {flops:6.2f} Discrete Archtiecture : {discrete_str_arch[0]}')
                    ##############################################################
                    # Calculate Continuous Model Architecture FLOPS
                    ##############################################################
                    continuous_arch = ema.ema.softmax_sampling()
                    continuous_str_arch = stringify_theta(continuous_arch)
                    architecture_info = {
                        'arch_type': 'continuous',
                        'arch': continuous_arch
                    }
                    flops = ema.ema.module.calculate_flops_new(architecture_info, model_est.flops_dict) if is_ddp else ema.ema.calculate_flops_new(architecture_info, model_est.flops_dict)
                    logger.info(f'EMA Continous FLOPS : {flops:6.2f} Discrete Archtiecture : {continuous_str_arch[0]}')   
                    
                if True:
                    ##############################################################
                    # Calculate Discrete Model Architecture FLOPS
                    ##############################################################
                    is_ddp = is_parallel(model)
                    
                    discrete_arch = model.discretize_sampling()
                    discrete_str_arch = stringify_theta(discrete_arch, normalize=False)
                    architecture_info = {
                        'arch_type': 'continuous',
                        'arch': discrete_arch
                    }
                    flops = model.calculate_flops_new(architecture_info, model_est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, model_est.flops_dict)
                    logger.info(f'Model Discrete FLOPS : {flops:6.2f} Discrete Archtiecture : {discrete_str_arch}')
                    ##############################################################
                    # Calculate Continuous Model Architecture FLOPS
                    ##############################################################
                    continuous_arch = model.softmax_sampling()
                    continuous_str_arch = stringify_theta(continuous_arch, normalize=False)
                    architecture_info = {
                        'arch_type': 'continuous',
                        'arch': continuous_arch
                    }
                    flops = model.module.calculate_flops_new(architecture_info, model_est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, model_est.flops_dict)
                    logger.info(f'Model Continous FLOPS : {flops:6.2f} Discrete Archtiecture : {continuous_str_arch}')        
            
            lr_scheduler.step()
            _, _, map50, *other = test(
                data=args.data,
                batch_size=16,
                imgsz=416,
                save_json=False,
                model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                single_cls=False,
                dataloader=testloader,
                save_dir=output_dir,
                logger=logger
            )
            random_testing('end of testing')
            print()
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        logger.info(s)
        filename = os.path.join(model_dir, f'best_f{TASK_FLOPS}.yaml')
        export_thetas(model.softmax_sampling(detach=True), model, model.model_args, filename)

    except KeyboardInterrupt:
        pass



if __name__ == '__main__':
    main()
