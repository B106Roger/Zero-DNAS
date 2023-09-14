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

def parse_config_args(exp_name):
    parser = argparse.ArgumentParser(description=exp_name)
    ###################################################################################
    # Commonly Used Parameter !!
    ###################################################################################
    parser.add_argument('--cfg',  type=str, default='config/search/exp_v4.yaml',           help='configuration of cream')
    parser.add_argument('--data', type=str, default='config/dataset/voc_dnas.yaml',              help='data.yaml path')
    parser.add_argument('--hyp',  type=str, default='config/training/hyp.zerocost.yaml', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--model',type=str, default='config/model/Search-YOLOv4-CSP.yaml',       help='model path')
    parser.add_argument('--exps', type=str, default='exp', help="name of experiments")
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

    is_ddp = is_parallel(model)
    ##################################################################
    ### Choice a Zero-Cost Method
    ##################################################################  
    stats_folder = os.path.join('experiments','workspace','valid_exp',f"stats_{args.exps}")
    img_pairs = []
    for iter_idx, (uimgs, targets, paths, _) in enumerate(dataloader_weight):
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets  = targets.to(device)
        
        imgs=imgs[:2]
        targets=targets[:2]
        img_pairs.append(imgs)

        store_img = (imgs * 255.0).int().permute(0,2,3,1).cpu().numpy()[...,::-1]
        cv2.imwrite(os.path.join(stats_folder, f'pair{iter_idx}_0.jpg'), store_img[0])
        cv2.imwrite(os.path.join(stats_folder, f'pair{iter_idx}_1.jpg'), store_img[1])
        
        if iter_idx == 10: break

    
    model.eval()  
    from lib.zero_proxy import naswot
    PROXY_DICT = {
        'naswot': naswot.calculate_zero_cost_map2,
    }
    if args.zc not in PROXY_DICT.keys():
        raise Value(f"key {args.zc} is not registered in PROXY_DICT")
    zc_func = PROXY_DICT[args.zc]
    wot_function = lambda arch_prob, idx: PROXY_DICT[args.zc](model, arch_prob, img_pairs[idx], targets, None)
    IMG_IDX = 1
    
    prob = model.softmax_sampling(detach=True)
    wot_function(prob, IMG_IDX)
    
    try:
        model_list = ['model_init.pt', *[f'ema_pretrained_{epoch}.pt' for epoch in range(2,22,2)]]
        exp_list   = natsorted(glob.glob(os.path.join('experiments','workspace','valid_exp',args.exps+'*')))
        
        os.makedirs(stats_folder, exist_ok=True)
        
        # model_list = model_list[4:]
        print(os.path.join('experiments','workspace','valid_exp',args.exps))
        print(model_list)

        for exp_path in exp_list:
            exp_name = exp_path.split('/')[-1]
            f = open(os.path.join(stats_folder, f'data_rank_{exp_name}.txt'), 'w')
            f.write(f"{'model':28s} ")
            for i in range(8):
                f.write(f'stage{i:<5d} ')
            f.write(f'{"overall":10s}\n')
            for model_file in model_list:
                zc_map_list = []
                zc_map_rank_list = []
                zc_map_correlation = []
                for IMG_IDX in range(10):
                    model_path = os.path.join(exp_path, 'model_weights', model_file)
                    print(f'Loading {model_path}')
                    model.load_state_dict(torch.load(model_path))
                    
                    zc_map = wot_function(prob, IMG_IDX)
                    num_stages = len(zc_map)
                    
                    zc_map_array = np.array([
                        np.array([
                            stage_map[key] for key in stage_map.keys()
                        ]) for stage_map in zc_map
                    ])
                    
                    zc_map_rank_array = np.argsort(zc_map_array, axis=1)
                
                    zc_map_list.append(np.expand_dims(zc_map_array, axis=-1))
                    zc_map_rank_list.append(np.expand_dims(zc_map_rank_array, axis=-1))
                
                # (num_stages, num_choices, num_seed)
                zc_map_list = np.concatenate(zc_map_list, axis=-1)
                zc_map_rank_list = np.concatenate(zc_map_rank_list, axis=-1)
                
                zc_score_var = np.var(zc_map_list, axis=-1)
                zc_rank_var  = np.var(zc_map_rank_list, axis=-1)
                
                zc_score_var = zc_score_var.mean(axis=-1)
                zc_rank_var = zc_rank_var.mean(axis=-1)
                
                stats_list = {
                    'kendalls_tau':  stats.kendalltau, #kendalltau(x, y, initial_lexsort=True)
                    'spearmans_rho': stats.spearmanr , #spearmanr(a, b=None, axis=0, nan_policy='propagate', alternative='two-sided')
                }
                
                for stage in range(len(zc_map_rank_list)):
                    stage_correlation = {}
                    num_choices = len(zc_map_rank_list[stage])
                    for stats_name, stats_func in stats_list.items():
                        num_seeds = len(zc_map_rank_list[stage][0])
                        comb_score_list = []
                        for (i,j) in combinations(list(range(num_seeds)), 2):
                            rank1 = zc_map_rank_list[stage, ..., i]
                            rank2 = zc_map_rank_list[stage, ..., j]
                            
                            coef, p_value = stats_func(rank1, rank2)
                            comb_score_list.append(coef)
                        comb_score = np.mean(comb_score_list)
                        stage_correlation[stats_name] = comb_score
                    
                    zc_map_correlation.append(stage_correlation)
                        

                f.write(f'{model_file:26s}\n')
                f.write(f'{"score var":26s} ')
                for stage_var in zc_score_var:
                    f.write(f'{stage_var:10.6f} ')
                f.write(f'{zc_score_var.mean():10.6f}\n')
                
                for stats_name in stats_list.keys():
                    f.write(f'{stats_name:26s} ')
                    tmp_list = []
                    for  stage_var in zc_map_correlation:
                        f.write(f'{stage_var[stats_name]:10.6f} ')
                        tmp_list.append(stage_var[stats_name])
                    f.write(f'{np.mean(tmp_list):10.6f}\n')
                
                f.write('*'*100+'\n')
                f.flush()
    except KeyboardInterrupt:
        pass



if __name__ == '__main__':
    main()
