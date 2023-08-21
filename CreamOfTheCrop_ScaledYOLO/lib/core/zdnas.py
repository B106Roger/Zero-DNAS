# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

from pathlib import Path
import numpy as np
import time, os
import torchvision
import torch.nn.functional as F
import itertools
import random as rd
import copy
import operator
from tqdm import tqdm
from torch.cuda import amp
from scipy.special import softmax
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.utils.kd_utils import compute_loss_KD
from lib.utils.synflow import sum_arr_tensor
from lib.zero_proxy import snip, synflow, naswot, grasp
from datetime import datetime


PROXY_DICT = {
    # 'snip'  :  snip.calculate_snip,
    # 'synflow': synflow.calculate_synflow,
    'naswot' : naswot.calculate_zero_cost_map2,
    # 'grasp': grasp.calculate_grasp
}


#######################################
# Search Zero-Cost Aging Evolution
#######################################
def train_epoch_zdnas(epoch, model, zc_func, theta_optimizer, cfg, device, task_flops, 
                     est=None, logger=None, local_rank=0, prefix='', logdir='./'):
    batch_size = cfg.DATASET.BATCH_SIZE
    is_ddp = is_parallel(model)
    nn_model = model.module if is_ddp else model
    
    naive_model  = model.module if is_ddp else model
    search_space = naive_model.search_space
    temperature = nn_model.temperature
    
    # Average Meter
    avg_params = AverageMeter()
    avg_flops  = AverageMeter()
    avg_floss  = AverageMeter()
    avg_zcloss = AverageMeter()
    total      = AverageMeter()
    
    
    alpha = 0.01 #0.01 # 0.03        # for flops_loss
    beta  = 0.01         # for params_loss
    gamma = 0.03 # 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    num_iter = 2880
    if local_rank in [-1, 0]:
        logger.info(('%10s' * 8) % ('Epoch', 'gpu_num', 'Param', 'FLOPS', 'f_loss', 'zc_loss', 'total', 'temp'))
        pbar = tqdm(range(num_iter), total=num_iter, bar_format='{l_bar}{bar:5}{r_bar}')  # progress bar
    
    f=open(os.path.join(logdir, 'train.txt'), 'a')
    for iter_idx in pbar:
        if iter_idx % 50 == 0: 
            arch_prob = model.module.softmax_sampling(temperature, detach=True) if is_ddp else model.softmax_sampling(temperature, detach=True)
            zc_map = zc_func(arch_prob)
            f.write(f'[{epoch}-{iter_idx:04d}] {str(arch_prob)}\n')
            f.write(f'[{epoch}-{iter_idx:04d}] {str(zc_map)}\n')
            
        ##########################################################
        # Calculate Basic Information (FLOPS, Params, ZC_Score)
        ##########################################################
        gumbel_prob = model.module.gumbel_sampling(temperature) if is_ddp else model.gumbel_sampling(temperature)
        architecture_info = {
            'arch_type': 'continuous',
            'arch': gumbel_prob
        }
         
        zc_score = nn_model.calculate_zc(architecture_info, zc_map)
        output_flops  = nn_model.calculate_flops_new (architecture_info, est.flops_dict) / 1e3
        output_params = nn_model.calculate_params_new(architecture_info, est.params_dict)
        
        #########################################
        # Calculate Loss
        #########################################
        squared_error_flops = (output_flops - task_flops) ** 2
        # squared_error_params = (output_params - task_params) ** 2
        
        flops_loss  = squared_error_flops * alpha
        # params_loss = squared_error_params * beta
        zc_loss     = zc_score * gamma
        
        loss = zc_loss + flops_loss

        #########################################
        # Calculate Loss
        #########################################
        theta_optimizer.zero_grad()
        loss.backward()
        theta_optimizer.step()
        
        
        # Update Average Meter
        avg_params.update(output_params.item(), 1)
        avg_flops.update(output_flops.item(), 1)
        avg_floss.update(squared_error_flops.item(), 1)
        avg_zcloss.update(zc_score.item(), 1)
        total.update(loss.item(), 1)
        
        # Print
        if local_rank in [-1, 0]:
            ni = iter_idx
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%3d/%3d' % (epoch,cfg.EPOCHS), mem, avg_params.avg, avg_flops.avg, avg_floss.avg, \
                    avg_zcloss.avg, total.avg, temperature)

            date_time = datetime.now().strftime('%m/%d %I:%M:%S %p') + ' | '
            pbar.set_description(date_time + s)
            
    ##############################################################
    # Print Continuous FLOP Value
    ##############################################################
    arch_prob = model.module.softmax_sampling(temperature) if is_ddp else model.softmax_sampling(temperature)
    architecture_info = {
        'arch_type': 'continuous',
        'arch': arch_prob
    }
    output_flops  = model.calculate_flops_new (architecture_info, est.flops_dict) / 1e3
    output_params = model.calculate_params_new(architecture_info, est.params_dict)
    zc_score = model.calculate_zc(architecture_info, zc_map)
    print(f'Continuous Current FLOPS: {output_flops:.2f}G   Params: {output_params:.2f}M   ZC: {zc_score}')
    
    ##############################################################
    # Print Discrete FLOP Value
    ##############################################################
    arch_prob = model.module.discretize_sampling() if is_ddp else model.discretize_sampling()
    architecture_info = {
        'arch_type': 'continuous',
        'arch': arch_prob
    }
    output_flops  = model.calculate_flops_new (architecture_info, est.flops_dict) / 1e3
    output_params = model.calculate_params_new(architecture_info, est.params_dict)
    zc_score = model.calculate_zc(architecture_info, zc_map)
    print(f'Discrete Current FLOPS: {output_flops:.2f}G   Params: {output_params:.2f}M   ZC: {zc_score}')
            
    logger.info(s)
    return nn_model.thetas_main
    
        