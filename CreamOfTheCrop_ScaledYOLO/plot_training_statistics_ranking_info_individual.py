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
import sys, json, copy
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


def analyze_ranking(model, zero_cost_func, image_idx):
    """
    analyze the model ranking in stage.
    """
    prob = model.softmax_sampling(detach=True)
    
    zc_map = zero_cost_func(prob, image_idx)
    num_stages = len(zc_map)
    
    # (num_stage, num_candidate)
    score_array = np.array([
        np.array([
            stage_map[key] for key in stage_map.keys()
        ]) for stage_map in zc_map
    ])
    
    # (num_stage, num_candidate, 1)
    rank_array  = np.expand_dims(np.argsort(score_array, axis=1), axis=-1)
    score_array = np.expand_dims(score_array, axis=-1)
    
    # Corresponding to the number of stage
    results = {
        'rank':  rank_array,
        'score': score_array
    }
    return results



def analyze_ranking_epoch_info(model, args, zero_cost_func):
    IMAGE_IDX = 0
    NUM_STAGES = len(model.searchable_block_idx)
    epoch_model_list = ['model_init.pt', *[f'ema_pretrained_{epoch}.pt' for epoch in range(2,22,2)]]    
    epoch_model_list = epoch_model_list[:10]
    
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
    
    # Overall Result
    data = {
        'tau': {'x': [], 'y': []},
        'var': {'x': [], 'y': []}
    }
    # Stage-Wise Result
    data2 = [
        {
            'tau': {'x': [], 'y': []},
            'var': {'x': [], 'y': []}
        } for i in range(NUM_STAGES)
    ]
    
    score_stats = [
        {
            'rank':  [],
            'score': [],
            'epoch': []
        }
        for j in range(len(exp_list))
    ]
    
    searchable_block_name = [f'blocks.{block_id}' for block_id in model.searchable_block_idx ]
    print('searchable_block_name', searchable_block_name)

    for epoch_idx, model_file in enumerate(epoch_model_list):                       # For each epoch
        epoch = epoch_idx * 2 
        for model_idx, (exp_path, exp_name) in enumerate(zip(exp_list, exp_name_list)): # For different rand seed model
            model_path = os.path.join(exp_path, 'model_weights', model_file)
            print(f'Loading {model_path}')
            model.load_state_dict(torch.load(model_path))
            
            ###############################################
            # Replaciable  Area
            # Analyze Parameter
            ###############################################
            stage_statistics = analyze_ranking(model, zero_cost_func, IMAGE_IDX)
            ###############################################
            
            for key in stage_statistics.keys():
                score_stats[model_idx][key].append(stage_statistics[key])
            score_stats[model_idx]['epoch'].append(epoch)
        
        # (num_stage, num_candidate, num_exp)
        rank_array  = np.concatenate([score_stats[m_idx]['rank'][epoch_idx]  for m_idx in range(len(exp_name_list))], axis=-1)
        score_array = np.concatenate([score_stats[m_idx]['score'][epoch_idx] for m_idx in range(len(exp_name_list))], axis=-1)

        num_exp = len(exp_name_list)
        stage_tau_list = []
        for stage_idx in range(NUM_STAGES):
            comb_tau_list = [] # length=>(8,)
            # Computation for Kendal's tau
            for (i,j) in combinations(list(range(num_exp)), 2):
                rank1 = rank_array[stage_idx, ..., i]
                rank2 = rank_array[stage_idx, ..., j]
                    
                coef, p_value = stats.kendalltau(rank1, rank2)
                comb_tau_list.append(coef)
            
            stage_tau = np.mean(comb_tau_list)
            stage_tau_list.append(stage_tau)
        
            data2[stage_idx]['tau']['y'].append(stage_tau)
            data2[stage_idx]['tau']['x'].append(epoch)
            
            data2[stage_idx]['var']['y'].append(np.mean(np.var(score_array[stage_idx], axis=-1), axis=-1).item())
            data2[stage_idx]['var']['x'].append(epoch)

        data['tau']['y'].append(np.mean(data2[stage_idx]['tau']['y']))
        data['tau']['x'].append(epoch)
        data['var']['y'].append(np.mean(data2[stage_idx]['var']['y']))
        data['var']['x'].append(epoch)


    
    ###############################################
    # Plotting Kendal tau for each stage
    ###############################################
    fig, axes = plt.subplots(4, 2, figsize=(16,20))
    for stage_idx in range(NUM_STAGES):
        ###################
        # Draw Figure
        ###################
        row_idx = stage_idx // 2
        col_idx = stage_idx %  2
        
        color = 'tab:red'
        axes[row_idx,  col_idx].plot(data2[stage_idx]['tau']['x'], data2[stage_idx]['tau']['y'], label=f'tau', color = color)
        axes[row_idx,  col_idx].set_xlabel('Epoch')
        axes[row_idx,  col_idx].set_ylabel('Kendals Tau', color = color)
        axes[row_idx,  col_idx].tick_params(axis='y', labelcolor=color)
        
        color = 'tab:blue'
        axes2 = axes[row_idx,  col_idx].twinx() 
        axes2.plot(data2[stage_idx]['var']['x'], data2[stage_idx]['var']['y'], label=f'var', color = color)
        axes2.set_ylabel('Zero Cost Score', color = color)
        axes2.tick_params(axis='y', labelcolor=color)
        axes[row_idx, col_idx].set_title(f"Stage{stage_idx} Statistics")
        
        line1, label1 = axes[row_idx, col_idx].get_legend_handles_labels()
        line2, label2 = axes2.get_legend_handles_labels()
        axes[row_idx, col_idx].legend(line1+line2, label1+label2, loc='center right')
        

    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, f'ranking_analyze_stage.jpg'), dpi=300)
    plt.close(fig)

    with open(os.path.join(stats_folder, f'ranking_analyze_stage.json'), 'w') as f:
        json.dump(data2, f)

    ##############################################
    # Plotting Kendal tau for model
    ##############################################    
    fig, axes = plt.subplots(1, 1, figsize=(8,10))
    axes2 = axes.twinx()

    color = 'tab:red'
    axes.plot(data['tau']['x'], data['tau']['y'], label='tau', color = color)
    axes.set_ylabel('Kendals Tau', color = color)
    axes.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:blue'   
    axes2.plot(data['var']['x'],  data['var']['y'],   label='var', color = color)
    axes2.set_ylabel('Zero Cost Score', color = color)
    axes2.tick_params(axis='y', labelcolor=color)
    
    axes.set_xlabel('Epoch')
    axes.set_title(f"Model Statistics")
    line1, label1 = axes.get_legend_handles_labels()
    line2, label2 = axes2.get_legend_handles_labels()
    axes.legend(line1+line2, label1+label2, loc='center right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, f'ranking_analyze_overall.jpg'), dpi=300)
    plt.close(fig)
    
    with open(os.path.join(stats_folder, f'ranking_analyze_overall.json'), 'w') as f:
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
    prob = model.softmax_sampling(detach=True)
    
    from lib.zero_proxy import naswot, snip
    PROXY_DICT = {
        'naswot': naswot.calculate_zero_cost_map2,
        'snip':   snip.calculate_zero_cost_map,    
    }
    zc_function_list = {
        'naswot':  lambda arch_prob, idx, short_name=None: PROXY_DICT['naswot'](model, arch_prob, img_pairs[idx][:2], target_pairs[idx][:2], short_name=short_name),
        'snip':    lambda arch_prob, idx, short_name=None: PROXY_DICT['snip'](model, arch_prob, img_pairs[idx], target_pairs[idx], short_name=short_name),
    }
    if args.zc not in PROXY_DICT.keys():
        raise Value(f"key {args.zc} is not registered in PROXY_DICT")
    
    analyze_ranking_epoch_info(model, args, zc_function_list[args.zc])
    
if __name__ == '__main__':
    main()
