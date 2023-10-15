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
import random
from datetime import datetime
from tqdm import tqdm
from torch.cuda import amp
from scipy.special import softmax
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.utils.kd_utils import compute_loss_KD
from lib.utils.synflow import sum_arr_tensor
from lib.zero_proxy import snip
from lib.utils.general import random_testing


class DefaultEnter():
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(f"exc_type : {exc_type}")
        # print(f"exc_val  : {exc_val}")
        # print(f"exc_tb   : {exc_tb}")
        pass

def flop_mean_square_error(nn_model, estimator, arch_info, target_flops):
    """
    model: nn.Module
    arch_info: dict
    target_flops: GFLOPS
    """
    output_flops = nn_model.calculate_flops_new (arch_info, estimator.flops_dict) / 1e3
    flop_loss    = (output_flops - target_flops) ** 2
    
    return output_flops, flop_loss 

def param_mean_square_error(nn_model, estimator, arch_info, target_params):
    """
    model: nn.Module
    arch_info: dict
    target_params: M Params
    """
    output_params = nn_model.calculate_params_new (arch_info, estimator.params_dict)
    params_loss    = (output_params - target_params) ** 2
    
    return output_params, params_loss 

##############################
# Train DNAS
##############################
def train_step_dnas_V2(model, input, targets, is_gumbel, est, task_flops, task_params, device):
    alpha = 0.1        # for flops_loss
    beta = 0.01         # for params_loss
    gamma = 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    eta = 0.01          # for regularization loss, default 0.01
    is_ddp = is_parallel(model)
    temperature = model.module.temperature if is_ddp else model.temperature

    if is_gumbel:
        gumbel_prob = model.module.gumbel_sampling(temperature) if is_ddp else model.gumbel_sampling(temperature)
    else:
        gumbel_prob = model.module.softmax_sampling(temperature) if is_ddp else model.softmax_sampling(temperature)
        
    imgs = input.to(device, non_blocking=True).float() / 255.0
    pred = model.module(imgs, gumbel_prob) if is_ddp else model(imgs, gumbel_prob)
    
    det_loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
    det_loss = det_loss[0]
    
    architecture_info = {
        'arch_type': 'continuous',
        'arch': gumbel_prob
    }
    flops = model.module.calculate_flops_new(architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
    # params = model.module.calculate_params_new(architecture_info, est.params_dict)
    # layers = model.module.calculate_layers_new(architecture_info)
    
    output_flops = flops.mean() / 1e3
    # output_params = params.mean() 
    # output_layers = layers.mean() 
    
    squared_error_flops = (output_flops - task_flops) ** 2
    # squared_error_params = (output_params - task_params) ** 2
    
    flops_loss = squared_error_flops * alpha
    # params_loss = squared_error_params * beta
    # layers_loss = output_layers
    train_loss = det_loss + flops_loss

    hardware_losses = {
        'squared_error_flops': squared_error_flops
    }
    train_info = {
        'output_flops' : output_flops,
        'n_targets' : targets.shape[0],
        'n_imgs'    : imgs.shape[-1],
        'architecture' : gumbel_prob
    }

    return train_loss, loss_items, hardware_losses, train_info

def train_epoch_dnas_V2(model, dataloader, theta_dataloader, optimizer, theta_optimizer, cfg, device, task_flops, task_params, 
                     est=None, logger=None, local_rank=0, world_size=0, is_gumbel=False,
                     prefix='', epoch=None, total_epoch=None, logdir='./', ema=None, warmup=True):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    training_losses_m = AverageMeter()
    flops_losses_m = AverageMeter()
    det_losses_m = AverageMeter()
    
    cache_hits = 0
    iterations = len(dataloader)

    end = time.time()
    last_idx = len(dataloader) - 1
    
    batch_size = cfg.DATASET.BATCH_SIZE
    alpha = 0.1        # for flops_loss
    beta = 0.01         # for params_loss
    gamma = 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    eta = 0.01          # for regularization loss, default 0.01
    nw = max(3 * batch_size, 1e3)
    is_ddp = is_parallel(model)
    
    # temperature = model.module.temperature if is_ddp else model.temperature
    mloss = torch.zeros(4, device=device)  # mean losses
    
    pbar = enumerate(dataloader)
    if local_rank in [-1, 0]:
        logger.info(('%10s' * 14) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'lr', 'moment', 'decay', 'temp', 'GFLOPS', 'f_loss'))
        pbar = tqdm(pbar, total=iterations, bar_format='{l_bar}{bar:5}{r_bar}')  # progress bar
    
    def valid_generator():
        while True:
          for x, t, path, shape in theta_dataloader:
            yield x, t, path, shape
    valid_gen = valid_generator()
    
    temperature = model.module.temperature if is_ddp else model.temperature
    arch_prob = model.module.softmax_sampling(temperature=temperature) if is_ddp else model.softmax_sampling(temperature=temperature)
    logger.info(f'Begin Architecture : {str(arch_prob)}')

    for iter_idx, (imgs, targets, paths, _) in pbar:
        ##################################################################
        # 1st. WarmUp Learning Rate
        ##################################################################
        ni = iter_idx + iterations * (epoch- 1)
        if ni <= nw and warmup:
            import math
            lf = lambda x: (((1 + math.cos(x * math.pi / total_epoch)) / 2) ** 1.0) * 0.8 + 0.2
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                if 'initial_lr' not in x:
                    continue
                w_lr = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                
                x['lr'] = w_lr
                if 'momentum' in x:
                    w_momentum = np.interp(ni, xi, [0.9, cfg.momentum])
                    x['momentum'] = w_momentum



        ##################################################################
        ### 2nd. Update SuperNet Architecture
        ##################################################################
        if epoch > cfg.FREEZE_EPOCH:
            # Prepare Data
            input_valid, target_valid, _, _ = next(valid_gen)
            # Prepare Architecture Parameter
            train_loss, loss_items, hardware_losses, train_info = train_step_dnas_V2(model, input_valid, target_valid, False, est, task_flops, task_params, device)
            # squared_error_flops = hardware_losses['squared_error_flops']
            # output_flops        = train_info['output_flops']
            # n_targets           = train_info['n_targets']
            # n_imgs              = train_info['n_imgs']

            
            time_grad= time.time()
            theta_optimizer.zero_grad()
            train_loss.backward()
            theta_optimizer.step()
            if ema is not None:
                ema.update(model, update_arch=True)
        
        ##################################################################
        ### 3rd. Train SuperNet Parameter
        ##################################################################
        train_loss, loss_items, hardware_losses, train_info = train_step_dnas_V2(model, imgs, targets, True, est, task_flops, task_params, device)
        squared_error_flops = hardware_losses['squared_error_flops']
        output_flops        = train_info['output_flops']
        n_targets           = train_info['n_targets']
        n_imgs              = train_info['n_imgs']
        if (iter_idx + 1) % 100 == 0:
            train_prob          = stringify_theta(train_info['architecture'], normalize=True, temperature=temperature)
            logger.info(f'Train Temp : {temperature} Architecture : {train_prob}')
        
        time_grad= time.time()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model, update_arch=False)

        # Basic Info
        for j, x in enumerate(optimizer.param_groups):
            if 'momentum' in x:
                break
        print_lr = x['lr'] if 'lr' in x else 0
        print_m = x['momentum'] if 'momentum' in x else 0
        print_wdecay = x['weight_decay'] if 'weight_decay' in x else 0
        
        # Print
        if local_rank in [-1, 0]:
            ni = iter_idx
            mloss = (mloss * iter_idx + loss_items) / (iter_idx + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 12) % (
                '%g/%g' % (epoch, total_epoch), mem, *mloss, n_targets, n_imgs, print_lr, print_m, print_wdecay, temperature, output_flops, squared_error_flops.detach().cpu())

            date_time = datetime.now().strftime('%m/%d %I:%M %p') + ' | '
            pbar.set_description(date_time + s)
            
            # Plot
            if ni < 3:
                f = str(Path(logdir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)

    arch_prob = model.module.softmax_sampling(temperature=temperature) if is_ddp else model.softmax_sampling(temperature=temperature)
    logger.info(f'End Architecture : {str(arch_prob)}', )

    logger.info(s)        
    torch.cuda.synchronize()


##############################
# Train IZDNAS
##############################
def train_epoch_dnas(model, dataloader, optimizer, cfg, device, task_flops, task_params, zero_cost_data_pair=None,
                     est=None, logger=None, local_rank=0, world_size=0, is_gumbel=False, use_amp=False, zc_func=None,
                     prefix='', epoch=None, total_epoch=None, logdir='./', ema=None, warmup=True, description=""):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    training_losses_m = AverageMeter()
    flops_losses_m = AverageMeter()
    det_losses_m = AverageMeter()
    model_w_dir      = os.path.join(logdir, 'model_weights')
    
    cache_hits = 0
    iterations = len(dataloader)

    end = time.time()
    last_idx = len(dataloader) - 1
    
    batch_size = cfg.DATASET.BATCH_SIZE
    alpha = 0.1        # for flops_loss
    beta = 0.01         # for params_loss
    gamma = 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    eta = 0.01          # for regularization loss, default 0.01
    nw = max(3 * batch_size, 1e3)
    is_ddp = is_parallel(model)
    
    temperature = model.module.temperature if is_ddp else model.temperature
    mloss = torch.zeros(4, device=device)  # mean losses
    

    pbar = enumerate(dataloader)
    if local_rank in [-1, 0]:
        print(('%10s' * 14) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'lr', 'moment', 'decay', 'temp', 'GFLOPS', 'f_loss'))
        pbar = tqdm(pbar, total=iterations, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    
    
    for iter_idx, (imgs, targets, paths, _) in pbar:
        ##################################################################
        ### 1st. Train SuperNet Parameter
        ##################################################################
        # imgs = (batch=2, 3, height, width)
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

        ni = iter_idx + iterations * (epoch- 1)
        if ni <= nw and warmup:
            import math
            lf = lambda x: (((1 + math.cos(x * math.pi / total_epoch)) / 2) ** 1.0) * 0.8 + 0.2
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                if 'initial_lr' not in x:
                    continue
                w_lr = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                
                x['lr'] = w_lr
                if 'momentum' in x:
                    w_momentum = np.interp(ni, xi, [0.9, cfg.momentum])
                    x['momentum'] = w_momentum

        t_infer=time.time()
        if use_amp: 
            precision_determineter = amp.autocast(enabled=True)
            scaler = amp.GradScaler(enabled=True)
        else:
            precision_determineter = DefaultEnter()
            
        with precision_determineter:
            if is_gumbel:
                gumbel_prob = model.module.gumbel_sampling(temperature) if is_ddp else model.gumbel_sampling(temperature)
            else:
                gumbel_prob = model.module.softmax_sampling(temperature) if is_ddp else model.softmax_sampling(temperature)
            if ni % 500 == 0: print(f'tmp={temperature:.4f} prob={gumbel_prob}')
            # print('[Random] sample prob', gumbel_prob)
            pred = model.module(imgs, gumbel_prob) if is_ddp else model(imgs, gumbel_prob)

            det_loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
            det_loss = det_loss[0]
                
            architecture_info = {
                'arch_type': 'continuous',
                'arch': gumbel_prob
            }
            flops = model.module.calculate_flops_new(architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
            # params = model.module.calculate_params_new(architecture_info, est.params_dict)
            # layers = model.module.calculate_layers_new(architecture_info)
            
            output_flops = flops.mean() / 1e3
            # output_params = params.mean() 
            # output_layers = layers.mean() 
            
            squared_error_flops = (output_flops - task_flops) ** 2
            # squared_error_params = (output_params - task_params) ** 2
            # flops_loss = (output_flops - task_flops) ** 2
            
            flops_loss = squared_error_flops * alpha
            # params_loss = squared_error_params * beta
            # layers_loss = output_layers
                
            train_loss = det_loss + flops_loss

            # if iter_idx == 0 and description == "architecture":
            #     logger.info(f'[Roger] Temp {temperature:5.2f} FLOPS : {output_flops:5.2f} Gumbel  {stringify_theta(gumbel_prob)}')
            t_infer=time.time()-t_infer
            
            time_grad= time.time()
            optimizer.zero_grad()
            
            if use_amp:
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                train_loss.backward()
                optimizer.step()

        if ema is not None:
            # description = "architecture", description = "weights"
            if description == "weights":
                ema.update(model, update_arch=False)
            elif description == "architecture":
                ema.update(model, update_arch=True)

        # Basic Info
        for j, x in enumerate(optimizer.param_groups):
            if 'momentum' in x:
                break
        print_lr = x['lr'] if 'lr' in x else 0
        print_m = x['momentum'] if 'momentum' in x else 0
        print_wdecay = x['weight_decay'] if 'weight_decay' in x else 0
        
        # Print
        if local_rank in [-1, 0]:
            mloss = (mloss * iter_idx + loss_items) / (iter_idx + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 12) % (
                '%g/%g' % (epoch, total_epoch), mem, *mloss, targets.shape[0], imgs.shape[-1], print_lr, print_m, print_wdecay, temperature, output_flops, squared_error_flops.detach().cpu())
            pbar.set_description(s)
            
            # Plot
            if ni < 3:
                f = str(Path(logdir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                # if tb_writer and result is not None:
                #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard
        
        # Calculate Zero-Cost
        if ni %  50 == 0:
            f = open(Path(logdir) / 'zcmap.txt', 'a')
            arch_prob = model.module.softmax_sampling(temperature, detach=True) if is_ddp else model.softmax_sampling(temperature, detach=True)
            zc_map = zc_func(model, arch_prob, *zero_cost_data_pair)
            f.write(f'[{ni}-{epoch}-{iter_idx:04d}] {str(zc_map)}\n')  
            f.close() 
            ########################################################################
            ########################################################################
            
            f = open(Path(logdir) / 'zcmap_ema.txt', 'a')
            ############################################
            ema.ema.train()
            for p in ema.ema.parameters():
                p.requires_grad_(True)
            ############################################
            
            arch_prob = ema.ema.module.softmax_sampling(temperature, detach=True) if is_ddp else ema.ema.softmax_sampling(temperature, detach=True)
            zc_map = zc_func(ema.ema, arch_prob, *zero_cost_data_pair)
            f.write(f'[{ni}-{epoch}-{iter_idx:04d}] {str(zc_map)}\n')  
            f.close() 
            
            ############################################
            ema.ema.zero_grad()
            for p in ema.ema.parameters():
                p.requires_grad_(False)
            ema.ema.eval()
            ############################################
            
            model.train()
            ema.ema.train()

        # If iteration is too large, store checkpoint every 2500 iteration
        if iterations > 5000:
            if iter_idx % 2500 == 0:
                torch.save(ema.ema.state_dict(),   os.path.join(model_w_dir, f'ema_pretrained_{epoch}_{iter_idx}.pt'))

    torch.cuda.synchronize()

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
    
    
    alpha = cfg.STAGE2.ALPHA #0.03 # 0.005 # 0.01 # 0.03    # for flops_loss
    beta  = 0.010                                           # for params_loss
    gamma = cfg.STAGE2.GAMMA # 0.01                         # for zero cost loss
    omega = 0.010                                           # for depth loss
    
    alpha = 0.0 if epoch < cfg.STAGE2.HARDWARE_FREEZE_EPOCHS else alpha
    print(f'Gamma(ZC loss weight)={gamma:.3f} Alpha(FLOP loss weight)={alpha:.3f}')
    
    num_iter = cfg.STAGE2.ITERATIONS #2000 # 2880
    if local_rank in [-1, 0]:
        logger.info(('%10s' * 8) % ('Epoch', 'gpu_num', 'Param', 'FLOPS', 'f_loss', 'zc_loss', 'total', 'temp'))
        pbar = tqdm(range(num_iter), total=num_iter, bar_format='{l_bar}{bar:5}{r_bar}')  # progress bar
    
    f=open(os.path.join(logdir, 'train.txt'), 'a')
    for iter_idx in pbar:
        if iter_idx % cfg.STAGE2.ZCMAP_UPDATE_ITER == 0:
        # if True:
        # if iter_idx % 50 == 0: 
        # if iter_idx % 250 == 0: 
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
         
        output_flops, output_params, zc_score = nn_model.calculate_utility(architecture_info, est.flops_dict, est.params_dict, zc_map)
        output_flops /= 1e3
        # zc_score = nn_model.calculate_zc(architecture_info, zc_map)
        # output_flops  = nn_model.calculate_flops_new (architecture_info, est.flops_dict) / 1e3        
        # output_params = nn_model.calculate_params_new(architecture_info, est.params_dict)
        
        #########################################
        # Calculate Loss
        #########################################
        squared_error_flops = (output_flops - task_flops) ** 2
        # squared_error_params = (output_params - task_params) ** 2
        # output_flops,  squared_error_flops  =  flop_mean_square_error(nn_model, est, architecture_info, task_flops)
        

        
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
                '%3d/%3d' % (epoch,cfg.STAGE2.EPOCHS), mem, avg_params.avg, avg_flops.avg, avg_floss.avg, \
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


##############################
# Arch Prob utils
##############################
def theta_zero_grad(arch_prob):
    for stage in arch_prob:
        for key, prob in stage.items():
            if key == 'block_name': continue
            print(key, prob)
            prob.zero_grad()

def theta_grad_norm(arch_prob):
    print(f'{"key":20s}|', end='')
    for i in range(len(arch_prob)):
        stage_str = f'stage{i+1}'
        print(f"{stage_str:8s} ", end='')
    print(f"{'average':8s}")
        
    keys = arch_prob[0].keys()        
    for key in keys:
        if key == 'block_name': continue
        print(f'{key:20s}|', end='')
        norm_list = []
        for stage in arch_prob:
            norm = torch.norm(stage[key].grad, p=2).detach().cpu().numpy()
            norm_list.append(norm)
            print(f'{norm:8.4f} ',end='')
        print(f'{np.mean(norm_list):8.4f} ')
        
        
def theta_grad_norm_v2(gradients):
    n_search = 2
    print(f'{"key":20s}|', end='')
    for i in range(len(gradients)//n_search):
        stage_str = f'stage{i}'
        print(f"{stage_str:>10s} ", end='')
    print(f"{'average':>8s}")
        
    for s_idx in range(n_search):
        head_str = f'search{s_idx}_{len(gradients[s_idx])}'
        print(f'{head_str:20s}|', end='')
        norm_list = []
        for i in range(len(gradients)//n_search):
            norm = torch.norm(gradients[i*n_search+s_idx], p=2).detach().cpu().numpy()
            norm_list.append(norm)
            print(f'{norm:10.4f} ',end='')
        print(f'{np.mean(norm_list):8.4f} ')
