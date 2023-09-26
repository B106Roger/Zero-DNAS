# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import cv2
import os
import sys
from datetime import datetime
import yaml
import torch
import numpy as np
import torch.nn as nn
import tqdm
import shutil
import scipy.stats as stats
# import _init_paths
import sys, json
sys.path.insert(0, 'lib')

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
from lib.core.izdnas import train_epoch_dnas, train_epoch_dnas_V2, train_epoch_zdnas
from lib.models.structures.supernet import gen_supernet
from lib.utils.util import convert_lowercase, get_logger, \
    create_optimizer_supernet, create_supernet_scheduler, stringify_theta, write_thetas, export_thetas
from lib.utils.datasets import create_dataloader
from lib.utils.general import check_img_size, labels_to_class_weights, is_parallel, compute_loss, test, ModelEMA, random_testing
from lib.utils.torch_utils import select_device
from lib.config import cfg
import argparse
import random
import glob
from natsort import natsorted
from itertools import combinations
import imageio
import matplotlib.pyplot as plt


def analyze_param(model_param, search_blocks_name):
    """
    analyze the model parameter in stage.
    """
    
    #############################
    # Searchable block name
    #############################
    # In YOLOv4-CSP, 4,6,8,10,16,21,25,29 are searchable structure
    # search_blocks_name = ['blocks.4', 'blocks.6', 'blocks.8', 'blocks.10', 'blocks.16', 'blocks.21', 'blocks.25', 'blocks.29']
    # In YOLOv7-CSP, 4,10,16,22,28,33,39,45 are searchable structure
    # search_blocks_name ['blocks.4', 'blocks.10', 'blocks.16', 'blocks.22', 'blocks.28', 'blocks.33', 'blocks.39', 'blocks.45']
    
    param_stats = {}
    for key in model_param.keys():
        name = '.'.join(key.split('.')[:2])
        if len(model_param[key].shape) == 4: # Suppose convlution kernel is 4 dimension vector
            if name not in param_stats.keys():
                param_stats[name] = model_param[key].flatten()
            else:
                param_stats[name] = torch.cat([param_stats[name], model_param[key].flatten()], dim=0)

    # Corresponding to the number of stage
    results = []
    print(param_stats.keys())
    for block_name in param_stats.keys():
        if block_name not in search_blocks_name: continue
        
        mean = param_stats[block_name].mean().detach().cpu().numpy().item()
        var  = param_stats[block_name].var().detach().cpu().numpy().item()
        min_val = param_stats[block_name].min().detach().cpu().numpy().item()
        max_val = param_stats[block_name].max().detach().cpu().numpy().item()
        print(f"{block_name}=({mean:.6f}, {var:.6f}, {min_val:.6f}, {max_val:.6f})")
        results.append({'block_name': block_name,'mean': mean, 'var': var, 'min': min_val, 'max': max_val})

    return results

def analyze_parameter_epoch_info(model, args, img_list, target_list):
    NUM_STAGES = 8
    epoch_model_list = ['model_init.pt', *[f'ema_pretrained_{epoch}.pt' for epoch in range(2,22,2)]]    
    # epoch_model_list = epoch_model_list[:4]
    
    exp_list      = natsorted(glob.glob(os.path.join('experiments','workspace','valid_exp',args.exp_series+'*')))
    # exp_list      = exp_list[:2]
    
    exp_name_list = [exp_name.split('/')[-1] for exp_name in exp_list]
    stats_folder  = os.path.join('experiments','workspace','statistics',args.exp_series)
    
    os.makedirs(stats_folder, exist_ok=True)
    for exp_name in exp_name_list:
        os.makedirs(os.path.join(stats_folder, exp_name), exist_ok=True)
    # writers = dict([(zc_name, imageio.get_writer(os.path.join(stats_folder,f'{zc_name}.mp4'), fps=4)) for zc_name in zc_function_list.keys()])
    
    print('[Roger] Statistics save path: ',      stats_folder)
    print('[Roger] List of model checkpoints: ', epoch_model_list)
    
    # store_img = (img_list[IMG_IDX] * 255.0).int().permute(0,2,3,1).cpu().numpy()[...,::-1]
    # cv2.imwrite(os.path.join(stats_folder, f'pair{IMG_IDX}_0.jpg'), store_img[0])
    # cv2.imwrite(os.path.join(stats_folder, f'pair{IMG_IDX}_1.jpg'), store_img[1])
    
    
    ####################################################
    # Prepare Data To Be Plot
    ####################################################
    data = [
        [{
            'mean': {'x':[],'y':[]},
            'var': {'x':[],'y':[]},
            'min': {'x':[],'y':[]},
            'max': {'x':[],'y':[]},
            
        } for i in range(NUM_STAGES)]
        for j in range(len(exp_list))
    ]
    
    searchable_block_name = [f'blocks.{block_id}' for block_id in model.searchable_block_idx ]
    print('searchable_block_name', searchable_block_name)
    for model_idx, (exp_path, exp_name) in enumerate(zip(exp_list, exp_name_list)): # For different rand seed model
        for epoch, model_file in enumerate(epoch_model_list):                       # For each epoch
            epoch *= 2 
            model_path = os.path.join(exp_path, 'model_weights', model_file)
            print(f'Loading {model_path}')
            load_param = torch.load(model_path)
            model.load_state_dict(load_param)
            
            ###############################################
            # Replaciable  Area
            # Analyze Parameter
            ###############################################
            stage_statistics = analyze_param(load_param, searchable_block_name)
            ###############################################
            
            # Store to data to facilitate plotting figure
            for stage_idx, stage_stats in enumerate(stage_statistics):
                for key in stage_stats.keys():
                    if key == 'block_name': continue
                    else:
                        data[model_idx][stage_idx][key]['y'].append(stage_stats[key])
                        data[model_idx][stage_idx][key]['x'].append(epoch)
            print(data[model_idx])
            
                    
        ###############################################
        # Plotting Figure Inidividualy
        ###############################################
        fig, axes = plt.subplots(4, 2, figsize=(8,10))
        for stage_idx in range(NUM_STAGES):
            row_idx = stage_idx // 2
            col_idx = stage_idx %  2
            print(f'data[{model_idx}][{stage_idx}]', data[model_idx][stage_idx])
            
            axes[row_idx, col_idx].plot(data[model_idx][stage_idx]['mean']['x'], data[model_idx][stage_idx]['mean']['y'], label='mean')
            axes[row_idx, col_idx].plot(data[model_idx][stage_idx]['var']['x'], data[model_idx][stage_idx]['var']['y'],   label='var')
            axes[row_idx, col_idx].set_title(f"Stage{stage_idx} Statistics")
            axes[row_idx, col_idx].legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(stats_folder, exp_name_list[model_idx], f'{exp_name}_param_analyze.jpg'), dpi=300)
        plt.close(fig)



    ###############################################
    # Plotting Figure Together
    ###############################################
    fig, axes = plt.subplots(4, 2, figsize=(16,20))
    for stage_idx in range(NUM_STAGES):
        for model_idx, exp_name in enumerate(exp_name_list):
            exp_name = exp_name[-3:]
            row_idx = stage_idx // 2
            col_idx = stage_idx %  2
            
            axes[row_idx, col_idx].plot(data[model_idx][stage_idx]['mean']['x'], data[model_idx][stage_idx]['mean']['y'], label=f'mean-{exp_name}')
            # axes[row_idx, col_idx].plot(data[model_idx][stage_idx]['var']['x'],  data[model_idx][stage_idx]['var']['y'],   label=f'var-{exp_name}')
            axes[row_idx, col_idx].set_title(f"Stage{stage_idx} Statistics")
            axes[row_idx, col_idx].legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, f'param_analyze.jpg'), dpi=300)
    plt.close(fig)

    with open(os.path.join(stats_folder, f'param_analyze.json'), 'w') as f:
        json.dump(data, f)

def parse_config_args(exp_name):
    parser = argparse.ArgumentParser(description=exp_name)
    ###################################################################################
    # Commonly Used Parameter !!
    ###################################################################################
    parser.add_argument('--cfg',  type=str, default='config/search/exp_v4.yaml',           help='configuration of cream')
    parser.add_argument('--data', type=str, default='config/dataset/voc_dnas.yaml',              help='data.yaml path')
    parser.add_argument('--hyp',  type=str, default='config/training/hyp.zerocost.yaml', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--model',type=str, default='config/model/Search-YOLOv4-CSP.yaml',       help='model path')
    parser.add_argument('--exp_series', type=str, default='exp_series', help="name of experiments")
    # parser.add_argument('--nas',  type=str, help='NAS-Search-Space and hardware constraint combination')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--zc', type=str, default='naswot', help='zero cost metrics')
    
    ###################################################################################
    
    
    # ###################################################################################
    # # Seldom Used Parameter
    # ###################################################################################
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    # parser.add_argument('--collect-samples', type=int, default=0, help='Sample a lot of different architectures with corresponding flops, if not 0 then samples specified number and exits the programm')
    # parser.add_argument('--collect-synflows', type=int, default=0, help='Sample a lot of different architectures with corresponding synflows, if not 0 then samples specified number and exits the programm')
    # parser.add_argument('--resume-theta-training', default='', type=str, help='load pretrained thetas')
    # ###################################################################################
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    converted_cfg = convert_lowercase(cfg)

    return args, converted_cfg

def main():
    args, cfg = parse_config_args('super net training')
    
    #######################################
    # Model Config
    #######################################
    with open(args.model ) as f:
        model_args   = yaml.load(f, Loader=yaml.FullLoader)
    search_space = model_args['search_space']

    # task_name = args.nas if args.nas != '' else 'DNAS-25'
    # TASK_FLOPS      = task_dict[task_name]['GFLOPS']     # e.g TASK_FLOPS  = 5  means 50 GFLOPs
    # TASK_PARAMS     = task_dict[task_name]['PARAMS']     # e.g TASK_PARAMS = 32 means 32 million parameters.
    SEARCH_SPACES   = model_args['search_space']
    USE_AMP         = False
    FLOP_RESOLUTION = (None, 3, cfg.search_resolution, cfg.search_resolution)
    
    output_dir = 'tmp'

    if args.local_rank == 0:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        logger_path = os.path.join(output_dir, "train.log")
        with open(logger_path, 'w') as file:
            pass
        logger = get_logger(logger_path)
    else:
        logger = None

    #######################################
    # Dataset Config
    #######################################
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_weight_path = data_dict['train_weight']
    train_thetas_path = data_dict['train_thetas']
    test_path         = data_dict['val']
    nc, names = (1, ['item']) if args.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, args.data)  # check

    # Number of class in data config would override other config
    if data_dict['nc'] != cfg.DATASET.NUM_CLASSES:
        logger.info(f"args.data with nc={data_dict['nc']} override the {args.cfg} cfg.DATASET.NUM_CLASSES={cfg.DATASET.NUM_CLASSES}")
        cfg.DATASET.NUM_CLASSES = data_dict['nc']
    if data_dict['nc'] != model_args['nc']:
        logger.info(f"args.data with nc={data_dict['nc']} override the {args.model} args.model['nc']={model_args['nc']}")
        model_args['nc'] = data_dict['nc']
        
    #######################################
    # Hyper Parameter Config
    #######################################
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset


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

    model, sta_num, resolution = gen_supernet(
        model_args,
        num_classes=cfg.DATASET.NUM_CLASSES,
        verbose=cfg.VERBOSE,
        logger=logger,
        init_temp=cfg.TEMPERATURE.INIT)

    # number of choice blocks in supernet
    if args.local_rank == 0:
        logger.info('Supernet created, param count: %.2f M', (
            sum([m.numel() for m in model.parameters()]) / 1e6))
        logger.info('resolution: %d', (cfg.DATASET.IMAGE_SIZE))

    # initialize flops look-up table
    model_est = FlopsEst(model, input_shape=(None, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE), search_space=SEARCH_SPACES, signature=args.model)

    optimizer, theta_optimizer = create_optimizer_supernet(cfg, model, USE_APEX)
    model.module.update_main() if is_parallel(model) else model.update_main()

    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [cfg.DATASET.IMAGE_SIZE] * 2]

    cuda = device.type != 'cpu'
    if cuda and torch.cuda.device_count() > 1 and args.local_rank != -1:
        model = torch.nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs. device: {device}')

    model = model.to(device)

    # dataloader_weight, dataset_weight = create_dataloader(train_weight_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
    #                                         cache=args.cache_images, rect=args.rect,
    #                                         world_size=args.world_size)
    
    # dataloader_thetas, dataset_thetas = create_dataloader(train_thetas_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
    #                                         cache=args.cache_images, rect=args.rect,
    #                                         world_size=args.world_size)
    
    dataloader_weight, dataset_weight = create_dataloader(train_thetas_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect,
                                            world_size=args.world_size)
    dataloader_thetas, dataset_thetas = create_dataloader(train_thetas_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect,
                                            world_size=args.world_size)
    
    img_pairs = []
    target_pairs = []
    for iter_idx, (uimgs, targets, paths, _) in enumerate(dataloader_weight):
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets  = targets.to(device)
        
        imgs=imgs
        targets=targets
        
        img_pairs.append(imgs)
        target_pairs.append(targets)
        
        if iter_idx == 10: break

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

    is_ddp = is_parallel(model)
    ##################################################################
    ### Choice a Zero-Cost Method
    ##################################################################  
    
    
    
    model.eval()  
    from lib.zero_proxy import naswot, snip
    PROXY_DICT = {
        'naswot': naswot.calculate_zero_cost_map2,
        'snip':   snip.calculate_zero_cost_map,
        
    }
    if args.zc not in PROXY_DICT.keys():
        raise Value(f"key {args.zc} is not registered in PROXY_DICT")
    zc_func = PROXY_DICT[args.zc]
    IMG_IDX = 0
    
    prob = model.softmax_sampling(detach=True)
    zc_function_list = {
        'naswot':  lambda arch_prob, idx, short_name=None: PROXY_DICT['naswot'](model, arch_prob, img_pairs[idx][:2], target_pairs[idx][:2], short_name=short_name),
        'snip':    lambda arch_prob, idx, short_name=None: PROXY_DICT['snip'](model, arch_prob, img_pairs[idx], target_pairs[idx], short_name=short_name),
        # 'synflow': lambda arch_prob: PROXY_DICT['synflow'](model, arch_prob, imgs, targets, None),
    }
    
    analyze_parameter_epoch_info(model,args,img_pairs, target_pairs)
    
if __name__ == '__main__':
    main()
