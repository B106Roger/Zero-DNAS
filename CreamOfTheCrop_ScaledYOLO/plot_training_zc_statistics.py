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
import sys
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

def parse_config_args(exp_name):
    parser = argparse.ArgumentParser(description=exp_name)
    ###################################################################################
    # Commonly Used Parameter !!
    ###################################################################################
    parser.add_argument('--cfg',  type=str, default='config/search/exp_v4.yaml',           help='configuration of cream')
    parser.add_argument('--data', type=str, default='config/dataset/voc_dnas.yaml',              help='data.yaml path')
    parser.add_argument('--hyp',  type=str, default='config/training/hyp.zerocost.yaml', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--model',type=str, default='config/model/Search-YOLOv4-CSP.yaml',       help='model path')
    parser.add_argument('--exp', type=str, default='exp', help="name of experiments")
    parser.add_argument('--nas',  type=str, help='NAS-Search-Space and hardware constraint combination')
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

    task_name = args.nas if args.nas != '' else 'DNAS-25'
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
    
    try:
        model_list = ['model_init.pt', *[f'ema_pretrained_{epoch}.pt' for epoch in range(2,22,2)]]
        # model_list = ['model_init.pt', *[f'ema_pretrained_{epoch}.pt' for epoch in range(2,14,2)]]
        
        
        exp_list   = natsorted(glob.glob(os.path.join('experiments','workspace','valid_exp',args.exp)))
        stats_folder = os.path.join('experiments','workspace','valid_exp',args.exp,"analyze")
        
        os.makedirs(stats_folder, exist_ok=True)
        writers = dict([(zc_name, imageio.get_writer(os.path.join(stats_folder,f'{zc_name}.mp4'), fps=4)) for zc_name in zc_function_list.keys()])
        
        
        # model_list = model_list[4:]
        print(os.path.join('experiments','workspace','valid_exp',args.exp))
        print(model_list)
        f = open(os.path.join(stats_folder, f'log_data{IMG_IDX}.txt'), 'w')
        store_img = (img_pairs[IMG_IDX] * 255.0).int().permute(0,2,3,1).cpu().numpy()[...,::-1]
        cv2.imwrite(os.path.join(stats_folder, f'pair{IMG_IDX}_0.jpg'), store_img[0])
        cv2.imwrite(os.path.join(stats_folder, f'pair{IMG_IDX}_1.jpg'), store_img[1])
        
        f.write(f"{'model':28s} ")
        for i in range(8):
            f.write(f'stage{i:<5d} ')
        f.write(f'{"overall":10s}\n')
        
        

        
        # Param Plot
        
        ###################################################
        P_NORM=2
        norm_dict = {}
        params_list = []
        for model_file in model_list:
            for model_idx, exp_path in enumerate(exp_list):
                model_idx *= 2 
                model_path = os.path.join(exp_path, 'model_weights', model_file)
                print(f'Loading {model_path}')
                load_param = torch.load(model_path)
                model.load_state_dict(load_param)
                params_list.append(analyze_model(load_param))
                
                # ##################################################
                # # Calculate Zc Map And Zc Score
                # ##################################################
                # d_prob = model.module.softmax_sampling(detach=True) if is_ddp else model.softmax_sampling(detach=True)
                
                # prob = model.module.softmax_sampling() if is_ddp else model.softmax_sampling()
                # architecture_info = {
                #     'arch_type': 'continuous',
                #     'arch': prob
                # }
                # zc_map_list =       dict([(zc_name, zc_function(d_prob, IMG_IDX)) for zc_name, zc_function in zc_function_list.items()])
                # zc_map_short_list = dict([(zc_name, zc_function(d_prob, IMG_IDX, True))  for zc_name, zc_function in zc_function_list.items()])
                # for zc_name, zc_map in zc_map_short_list.items(): print(f'{zc_name} {zc_map}')

                # zc_scores = dict([(zc_name, model.calculate_zc(architecture_info, zc_map)) for zc_name, zc_map in zc_map_list.items()])
                # zc_loss_dict= dict([(zc_name, zc_score) for zc_name, zc_score in zc_scores.items()])
                
                # ##################################################
                # # Calculate Zero-DNAS Gradient Norm
                # ##################################################
                # for key, zc_loss_value in reversed(zc_loss_dict.items()):
                #     print(f'{key} Norm ({key}={zc_loss_value})')
                #     gradient1 = torch.autograd.grad(zc_loss_value, model.get_optimizer_parameter(), retain_graph=True)
                #     norm_list, avg_norm = theta_grad_norm_v2(gradient1, p=P_NORM)
                #     if key not in norm_dict: norm_dict[key] = []
                #     norm_dict[key].append((norm_list, avg_norm))
                # ##################################################
                # # Plot Zc Value for different epoch
                # ##################################################                
                # for zc_name in zc_function_list.keys():
                #     zc_map = zc_map_short_list[zc_name]
                #     fig = analyze_map_func2([{'naswot_map': zc_map}], model_file, None)
                #     fig.tight_layout()
                #     fig.canvas.draw()
                #     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                #     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                #     writers[zc_name].append_data(data)
                #     plt.close(fig)
                    


        # # Norm Plot
        # norm_fig, norm_axes = plt.subplots(1, 2)
        # print(len(norm_axes))
        # search_name = ['gamma', 'depth']
        # legend = []

        # # 创建两个空折线，一个在每个子图中
        # if True:
        #     data = {
        #         'naswot': {
        #             0: {'x':[],'y':[]},
        #             1: {'x':[],'y':[]},
        #         },
        #         'snip': {
        #             0: {'x':[],'y':[]},
        #             1: {'x':[],'y':[]},
        #         }
        #     }

        #     for zc_name, search_norm_list in norm_dict.items():
        #         legend.append(zc_name)
        #         for epoch_idx, epoch_norm_list in enumerate(search_norm_list):
        #             epoch_idx *= 2
        #             for s_idx, stage_norm_list in enumerate(epoch_norm_list):
        #                 data[zc_name][s_idx]['x'].append(epoch_idx)
        #                 data[zc_name][s_idx]['y'].append(np.mean(stage_norm_list))
            
        #     for zc_name, search_norm_list in norm_dict.items():
        #         for s_idx in range(2):
        #             norm_axes[s_idx].plot(data[zc_name][s_idx]['x'], data[zc_name][s_idx]['y'], label=zc_name)
                    
        #     for i in range(len(norm_axes)):
        #         norm_axes[i].set_title(f"L{P_NORM} Norm {search_name[i]}")
        #         norm_axes[i].set_xlabel("Epoch")
        #         norm_axes[i].legend()
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(stats_folder, f'norm_L{P_NORM}_analysis.jpg'), dpi=300)
        #     plt.close(fig)
        
        data = [
            {
                'mean': {'x':[],'y':[]},
                'var': {'x':[],'y':[]},
                'min': {'x':[],'y':[]},
                'max': {'x':[],'y':[]},
                
            } for i in range(8)
        ]
        # Norm Plot
        norm_fig, norm_axes = plt.subplots(4, 2, figsize=(8,10))
        for epoch_idx, epoch_stats in enumerate(params_list):
            epoch_idx *= 2
            for stage_idx, stage_stats_dict in enumerate(epoch_stats):
                row_idx = stage_idx // 2
                col_idx = stage_idx %  2
                data[stage_idx]['mean']['y'].append(stage_stats_dict['mean'])
                data[stage_idx]['mean']['x'].append(epoch_idx)
                data[stage_idx]['var']['y'].append(stage_stats_dict['var'])
                data[stage_idx]['var']['x'].append(epoch_idx)
                data[stage_idx]['min']['y'].append(stage_stats_dict['min_val'])
                data[stage_idx]['max']['y'].append(stage_stats_dict['max_val'])
                
        print(data)
        
        for stage_idx in range(8):
            row_idx = stage_idx // 2
            col_idx = stage_idx %  2
            norm_axes[row_idx, col_idx].plot(data[stage_idx]['mean']['x'], data[stage_idx]['mean']['y'], label='mean')
            norm_axes[row_idx, col_idx].plot(data[stage_idx]['var']['x'], data[stage_idx]['var']['y'], label='var')
            # norm_axes[row_idx, col_idx].plot(data[stage_idx]['var']['x'], data[stage_idx]['min']['y'], label='min')
            # norm_axes[row_idx, col_idx].plot(data[stage_idx]['var']['x'], data[stage_idx]['max']['y'], label='max')
            
            norm_axes[row_idx, col_idx].set_title(f"Stage{stage_idx} Statistics")
            norm_axes[row_idx, col_idx].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(stats_folder, 'param_analysis.jpg'), dpi=300)
        # plt.close(fig)

    except KeyboardInterrupt:
        pass

def theta_grad_norm_v2(gradients, p=1):
    n_search = 2
    print(f'{"key":20s}|', end='')
    for i in range(len(gradients)//n_search):
        stage_str = f'stage{i}'
        print(f"{stage_str:>12s} ", end='')
    print(f"{'average':>12s}")
    
    overall_norm_list = []
    for s_idx in range(n_search):
        head_str = f'search{s_idx}_{len(gradients[s_idx])}'
        print(f'{head_str:20s}|', end='')
        stage_norm_list = []
        for i in range(len(gradients)//n_search):
            norm = torch.norm(gradients[i*n_search+s_idx], p=p).detach().cpu().numpy()
            stage_norm_list.append(norm)
            print(f'{norm:12.8f} ',end='')
        print(f'{np.mean(stage_norm_list):12.8f} ')
        overall_norm_list.append(stage_norm_list)
    return overall_norm_list

def analyze_map_func2(arch_info_list, title, img_filename):
    # zc_maps1 = arch_info1['naswot_map']
    # zc_maps2 = arch_info2['naswot_map']
    # arch1    = arch_info1['arch']
    # arch2    = arch_info2['arch']
    write_img  = img_filename is not None
    fig, axes = plt.subplots(8)
    fig.suptitle(title)
    for stage_id in range(len(arch_info_list[0]['naswot_map'])):
        score_list = []
        rank_list = []
        
        keys = arch_info_list[0]['naswot_map'][stage_id].keys()
        
        for arch_id, arch_info in enumerate(arch_info_list):
            zc_map = arch_info['naswot_map'][stage_id]
            score  = np.array([zc_map[key] for key in keys])
            rank   = (-score ).argsort()[::-1]
            
            score_list.append(score)
            rank_list.append(rank)

        candidiate_num  = len(rank)
        comp_list  = score_list
        color_list = ['r', 'g', 'b', 'c', 'k', 'm']
        # arch_list  = [arch_info['arch'] for arch_info in arch_info_list]
        x = np.arange(candidiate_num) * 0.8
        
        with_val = 0.1
        for i, score_arr in enumerate(comp_list):
            axes[stage_id].bar(x - with_val* (i-len(comp_list)/2), height=score_arr, width=with_val, color=[color_list[i]]*candidiate_num, align='edge')
        #######################################
        # Basic Math Information
        #######################################
        margin = 0.2
        all_scores = np.concatenate(comp_list)
        center  = all_scores.mean()
        min_val = all_scores.min() - 0.05
        max_val = all_scores.max() + 0.05
        
        #######################################
        # Set Plot Style
        #######################################
        axes[stage_id].set_ylim([min_val, max_val])
        axes[stage_id].set_xticks(x, list(keys))
        axes[stage_id].set_ylabel(f'Depth={stage_id}')
        axes[stage_id].legend([f'Arch{stage}' for stage in range(len(comp_list))], labelcolor=color_list)
        
        for ii in range(3,11,4): axes[stage_id].axvline((ii+0.5)*0.8, color='black')
        arr_size  = (max_val-min_val)*with_val
        
        # Arrow Plot
        # for ii, (arch, color) in enumerate(zip(arch_list,color_list)):
        #     loc_idx = arch[stage_id]['gamma'].argmax().numpy() * 4 + arch[stage_id]['n_bottlenecks'].argmax().numpy()
        #     loc = x[loc_idx] - with_val * (ii-len(comp_list)/2) + with_val
        #     axes[stage_id].arrow(loc, min_val+arr_size, 0, -arr_size*0.6666, 
        #                          head_width=arr_size*0.8, head_length=arr_size*0.3333, color=color, edgecolor='black')

        # Rank Plot
        for ii, (rank, score, color) in enumerate(zip(rank_list,comp_list,color_list)):
            for iii in range(4):
                loc = x[rank[iii]] - with_val * (ii-len(comp_list)/2) #+ 0.024
                axes[stage_id].text(loc, score[rank[iii]]-arr_size*1.2, str(iii+1), color=color)

    fig.set_size_inches(15.5, 15.5)
    fig.tight_layout()
    if img_filename is not None: fig.savefig(img_filename)
    return fig

def analyze_model(model_param):
    param_stats = {}
    for key in model_param.keys():
        name = '.'.join(key.split('.')[:2])
        if len(model_param[key].shape) == 4:
            if name not in param_stats.keys():
                param_stats[name] = model_param[key].flatten()
            else:
                param_stats[name] = torch.cat([param_stats[name], model_param[key].flatten()], dim=0)
                
            # mean = model_param[key].mean().detach().cpu().numpy()
            # var  = model_param[key].var().detach().cpu().numpy()
            # print(f"{key}=({mean:.6f}, {var:.6f})", end=' ')

    results = []
    print(param_stats.keys())
    for block_name in param_stats.keys():
        if block_name not in ['blocks.4', 'blocks.6', 'blocks.8', 'blocks.10',\
            'blocks.16', 'blocks.21', 'blocks.25', 'blocks.29']:
            continue
        mean = param_stats[block_name].mean().detach().cpu().numpy()
        var  = param_stats[block_name].var().detach().cpu().numpy()
        min_val = param_stats[block_name].min().detach().cpu().numpy()
        max_val = param_stats[block_name].max().detach().cpu().numpy()
        print(f"{block_name}=({mean:.6f}, {var:.6f}, {min_val:.6f}, {max_val:.6f})")
        results.append({'block_name': block_name,'mean': mean, 'var': var, 'min_val': min_val, 'max_val': max_val})

    return results

if __name__ == '__main__':
    main()
