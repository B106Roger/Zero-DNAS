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



PROXY_DICT = {
    'snip'  :  snip.calculate_snip,
    'synflow': synflow.calculate_synflow,
    'naswot' : naswot.calculate_wot,
    'grasp': grasp.calculate_grasp
}

def sample_arch(search_space):
    arch = []
    num_stages = 8
    keys = sorted(search_space.keys())
    for i in range(num_stages):
        tmp=[]
        for key in keys:
            saerch_space_size = len(search_space[key])
            rand_idx = torch.randint(0, saerch_space_size, (1,))
            val = torch.zeros((saerch_space_size,))
            val[rand_idx] = 1.0
                
            tmp.append(val)
        arch.append(tmp)
    return arch    

def largest_arch(search_space):
    arch = []
    num_stages = 8
    keys = sorted(search_space.keys())
    for i in range(num_stages):
        tmp=[]
        for key in keys:
            saerch_space_size = len(search_space[key])
            rand_idx = saerch_space_size - 1
            val = torch.zeros((saerch_space_size,))
            val[rand_idx] = 1.0
                
            tmp.append(val)
        arch.append(tmp)
    return arch    

def smallest_arch(search_space):
    arch = []
    num_stages = 8
    keys = sorted(search_space.keys())
    for i in range(num_stages):
        tmp=[]
        for key in keys:
            saerch_space_size = len(search_space[key])
            rand_idx = 0
            val = torch.zeros((saerch_space_size,))
            val[rand_idx] = 1.0
                
            tmp.append(val)
        arch.append(tmp)
    return arch    


def fullfill_constraints(sample, constraints):
    """
    sample : dict. contain the information about a architecture 
        possible key [arch, flops, params, .....]
    constraints : list. each element is a dict(contain the information about constraints).
        possible  key [constraint_name, operation, value]
    """
    for constraint in constraints:
        constraint_name = constraint['constraint_name']
        operation       = constraint['operation'] # operator.lt
        value           = constraint['value']
        
        if not operation(sample[constraint_name], value) : return False
    return True


# EA Utils       
def crossover(cand1, cand2):
    sample = {
        'arch' : copy.deepcopy(cand1['arch']),
        'arch_type': 'continuous'
    }
    
    rand_stage      = np.random.randint(0, len(sample['arch']))
    search_keys     = []
    for k in sample['arch'][rand_stage].keys():
        if k != 'operator' and k != 'block_name':
            search_keys.append(k)
    rand_component  = np.random.choice(search_keys)
    
    sample['arch'][rand_stage][rand_component] = cand2['arch'][rand_stage][rand_component].clone()
    return sample

def mutation(canddidate):
        sample = {
            'arch' : copy.deepcopy(canddidate['arch']),
            'arch_type': 'continuous'
        }
        rand_stage      = np.random.randint(0, len(sample['arch']))
        search_keys     = []
        for k in sample['arch'][rand_stage].keys():
            if k != 'operator' and k != 'block_name':
                search_keys.append(k)
        rand_component  = np.random.choice(search_keys)
        
        ori_op_idx = np.argmax(sample['arch'][rand_stage][rand_component])
        new_op_idx = ori_op_idx
        while new_op_idx == ori_op_idx and len(sample['arch'][rand_stage][rand_component]) > 1:
            new_op_idx = rd.randint(0, len(sample['arch'][rand_stage][rand_component]) - 1)
        
        sample['arch'][rand_stage][rand_component][ori_op_idx] = 0.
        sample['arch'][rand_stage][rand_component][new_op_idx] = 1.
        return sample

def get_model_info(candidates, info_funcs):
    if isinstance(candidates, list):
        for i in range(len(candidates)):
            arch_info = candidates[i]
            for info_func in info_funcs: 
                name = list(info_func.keys())[0]
                func = info_func[name]
                arch_info[name] = func(arch_info)
    elif isinstance(candidates, dict):
        arch_info = candidates
        for info_func in info_funcs: 
            name = list(info_func.keys())[0]
            func = info_func[name]
            arch_info[name] = func(arch_info)

# EA Batch Utils
def _EA_mutation(candidates, num, info_funcs, constraints=None):
    patience = 0
    indices = list(range(len(candidates)))
    rd.shuffle(indices)
    
    valid_candidates = []
    while num:
        idx = indices[num]
        new_candidate = mutation(candidates[idx])
        get_model_info(new_candidate, info_funcs)
        
        if constraints is not None:
            if not fullfill_constraints(new_candidate, constraints):
                patience+=1
                continue
        
        patience = 0
        valid_candidates.append(new_candidate)
        num-=1    
        
    return valid_candidates

def _EA_crossover(parents, num, info_funcs, constraints=None):
    patience = 0
    indices = list(range(len(parents)))
    
    valid_candidates = []
    while num:
        if patience > 100: 
            print('Some problems while sampling new arch')
            for p in parents:
                print(p)
            print('constraints', constraints)
            raise ValueError("Crossover Error")
            
        rd.shuffle(indices)
        p1_idx, p2_idx = indices[:2]
        p1 = parents[p1_idx]
        p2 = parents[p2_idx]
        
        new_candidate = crossover(p1, p2)
        get_model_info(new_candidate, info_funcs)
        if constraints is not None:
            if not fullfill_constraints(new_candidate, constraints):
                patience+=1
                continue
        
        valid_candidates.append(new_candidate)
        num-=1
        patience = 0
    return valid_candidates

def _EA_sample(sample_function, num, info_funcs, constraints=None):
    """
    Sample #num candidate.
    search_space : dict.
    num : int.
    info_funcs : list. each element is a dict
        dict containt the information name and function to evaluate the information
    """
    patience = 0
    arches = []
    while num:
        arch_info = {'arch': sample_function(), 'arch_type': 'continuous'}
        get_model_info(arch_info, info_funcs)
        # print(arch_info)
        if constraints is not None:
            if not fullfill_constraints(arch_info, constraints):
                # print(f"{num}, {patience}")
                patience+=1
                # if patience > 10: print(f'EA Sampling : {num:3d}/{patience:3d} {arch_info["flops"]:5.2f}\r', end='')
                continue
        # print(f"num = {num}")
        
        arches.append(arch_info)
        num-=1
        patience = 0
    
    return arches


#######################################
# Search Zero-Cost Random
#######################################
def train_epoch_zero_cost_rand(model, dataloader, optimizer, cfg, device, task_flops, task_params, num_arch, patience=None,
                     est=None, logger=None, local_rank=0, prefix='', logdir='./'):
    if patience is None: patience = num_arch * 3
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
    is_ddp = is_parallel(model)
    
    temperature = model.module.temperature if is_ddp else model.temperature
    mloss = torch.zeros(3, device=device)  # mean losses
    
    loader = enumerate(dataloader)
    
    ##################################################################
    ### 0st. Select Dataset
    ##################################################################    
    for iter_idx, (uimgs, targets, paths, _) in loader:
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets  = targets.to(device)
        break
        
    num_valid_arch = 0
    ########################################################
    # Sample Architecture
    ########################################################
    num_stages  = 8
    search_space = model.module.search_space if is_ddp else model.search_space
    keys = sorted(search_space.keys())
    def sample_arch():
        arch = []
        for i in range(num_stages):
            tmp=[]
            for key in keys:
                saerch_space_size = len(search_space[key])
                rand_idx = torch.randint(0, saerch_space_size, (1,))
                val = torch.zeros((saerch_space_size,))
                val[rand_idx] = 1.0
                
                tmp.append(val)
            arch.append(tmp)
        return arch    
    
    ########################################################
    # Random Search Algorithm
    ########################################################
    top10_arch = [] # (arch, score, FLOPS, Params)
    best_snip_value = -99999999.0
    pbar = tqdm(range(patience), total=patience, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        arch_prob = sample_arch()
        architecture_info = {
            'arch_type': 'continuous',
            'arch': arch_prob
        }
        flops = model.module.calculate_flops_new (architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
        param = model.module.calculate_params_new(architecture_info, est.params_dict) if is_ddp else model.calculate_params_new(architecture_info, est.params_dict)
        
        output_flops = flops.mean() / 1e3
        if output_flops > task_flops:
            if abs(output_flops - task_flops) > task_flops * 0.1: continue
        
        num_valid_arch+=1
        
        optimizer.zero_grad()
        model.module.zero_grad() if is_ddp else model.zero_grad()
        snip_value = snip.calculate_snip(model, arch_prob, imgs, targets)
        snip_value = snip_value.detach().cpu().numpy()
        # params = model.module.calculate_params_new(architecture_info, est.params_dict)
        # layers = model.module.calculate_layers_new(architecture_info)
        if best_snip_value < snip_value:
            top10_arch.append((arch_prob, snip_value, output_flops, param))
            top10_arch = sorted(top10_arch, key=lambda item: item[1])
            best_snip_value = snip_value
            logger.info(str(top10_arch[-1]))
        if len(top10_arch) > 10:
            top10_arch.pop(0)
        
        best_arch = top10_arch[-1]
        s=f'valid_arch: {num_valid_arch}/{num_arch}   best_snip: {str(best_arch[1]):8s} flops: {best_arch[2]:5.2f} params: {best_arch[3]:5.2f}'
        pbar.set_description(s)
        if num_valid_arch == num_arch: break
    logger.info(str(top10_arch[-1]))


#######################################
# Search Zero-Cost Aging Evolution
#######################################
def train_epoch_zero_cost_EA(proxy_name, model, dataloader, optimizer, cfg, device, task_flops, task_params, cycles,
                     est=None, logger=None, local_rank=0, prefix='', logdir='./'):
    batch_size = cfg.DATASET.BATCH_SIZE
    is_ddp = is_parallel(model)
    nn_model = model.module if is_ddp else model
    
    naive_model  = model.module if is_ddp else model
    search_space = naive_model.search_space
    model.eval()
    ##################################################################
    ### 0st. Select Dataset
    ##################################################################    
    loader = enumerate(dataloader)
    for iter_idx, (uimgs, targets, paths, _) in loader:
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets  = targets.to(device)
        if iter_idx == 2: break

    
    ##############################################
    # Zero Cost Name (snip, synflow, grasp)
    ##############################################
    # proxy_name = 'synflow'
    info_funcs = [
        {'flops' : (lambda x: nn_model.calculate_flops_new (x, est.flops_dict) / 1e3)},
        {'params': (lambda x: nn_model.calculate_params_new(x, est.params_dict))},
        {proxy_name : (lambda x: PROXY_DICT[proxy_name](model, x['arch'], imgs, targets, optimizer))}
    ]
    constraints = [
        {'constraint_name' : 'flops', 'operation' : operator.lt, 'value' : task_flops * 1.05},
        # {'constraint_name' : 'params', 'operation' : operator.lt, 'value' : task_params},
    ]
    
    if proxy_name == 'synflow' : signs = synflow.linearize(model)
    
    ##############################################
    # Show Largest Network for Debug
    ##############################################
    large_arch    = {'arch' : naive_model.largest_sampling(), 'arch_type': 'continuous'}
    get_model_info(large_arch, info_funcs)
    s = f"large_arch => FLOPS: {large_arch['flops']:.2f} Params: {large_arch['params']:.2f} {proxy_name}: {large_arch[proxy_name]:e}"
    logger.info(s)
    logger.info(f"large_arch => Architecture : {str(large_arch['arch'])}")

    ##############################################
    # Show Smallest Network for Debug
    ##############################################
    small_arch = {'arch' : naive_model.smallest_sampling(), 'arch_type': 'continuous'}
    get_model_info(small_arch, info_funcs)
    s = f"small_arch => FLOPS: {small_arch['flops']:.2f} Params: {small_arch['params']:.2f} {proxy_name}: {small_arch[proxy_name]:e}"
    logger.info(s)
    logger.info(f"small_arch => Architecture : {str(small_arch['arch'])}")

    # res = export_thetas(small_arch['arch'], model, model.model_args, './test123.yaml')
    # print(res)
    # exit()
    # manually_arch = [
    #     {'block_name': 'BottleneckCSP_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0])}, 
    #     {'block_name': 'BottleneckCSP_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0])}, 
    #     {'block_name': 'BottleneckCSP_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0])}, 
    #     {'block_name': 'BottleneckCSP_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0])}, 
    #     {'block_name': 'BottleneckCSP2_Search_num1_gamma0.75', 'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0])},
    #     {'block_name': 'BottleneckCSP2_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0])},
    #     {'block_name': 'BottleneckCSP2_Search_num1_gamma0.75', 'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0])},
    #     {'block_name': 'BottleneckCSP2_Search_num1_gamma0.75', 'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0])} 
    # ]
    # man_arch = {'arch' : manually_arch, 'arch_type': 'continuous'}
    # get_model_info(man_arch, info_funcs)
    # s = f"man_arch => FLOPS: {man_arch['flops']:.2f} Params: {man_arch['params']:.2f} {proxy_name}: {man_arch[proxy_name]:e}"
    # logger.info(s)
    # logger.info(f"man_arch => Architecture : {str(man_arch['arch'])}")
    
    
    # manually_arch = [
    #     {'block_name': 'BottleneckCSP_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0])}, 
    #     {'block_name': 'BottleneckCSP_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0])}, 
    #     {'block_name': 'BottleneckCSP_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0])}, 
    #     {'block_name': 'BottleneckCSP_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0])}, 
    #     {'block_name': 'BottleneckCSP2_Search_num1_gamma0.75', 'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0])},
    #     {'block_name': 'BottleneckCSP2_Search_num1_gamma0.75', 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0])},
    #     {'block_name': 'BottleneckCSP2_Search_num1_gamma0.75', 'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0])},
    #     {'block_name': 'BottleneckCSP2_Search_num1_gamma0.75', 'gamma': torch.tensor([1., 0., 1.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0])} 
    # ]
    # man_arch = {'arch' : manually_arch, 'arch_type': 'continuous'}
    # get_model_info(man_arch, info_funcs)
    # s = f"man_arch22 => FLOPS: {man_arch['flops']:.2f} Params: {man_arch['params']:.2f} {proxy_name}: {man_arch[proxy_name]:e}"
    # logger.info(s)
    # logger.info(f"man_arch22 => Architecture : {str(man_arch['arch'])}")
    # exit()
    ########################################################
    # Init Pool
    ########################################################
    POPULATION_COUNT = 64
    PARENT_COUNT     = 8
    MUTATION_COUNT   = 4
    CROSSOVER_COUNT  = 4
    RANDOM_COUNT     = 2
    DISCARD_COUNT    = MUTATION_COUNT + CROSSOVER_COUNT + RANDOM_COUNT
    TOPK             = 5
    cycles           = 1000
    
    sample_func = lambda : naive_model.random_sampling()
    pools = _EA_sample(sample_func, POPULATION_COUNT, info_funcs, constraints)
    
    best_arch = (sorted(pools, key = lambda x: x[proxy_name]))[-1]
    s = f"Best Arch In Initial Pool => FLOPS: {best_arch['flops']:.2f} Params: {best_arch['params']:.2f} {proxy_name}: {best_arch[proxy_name]:e}"
    logger.info(s)
    logger.info(f"Best Arch In Initial Pool => Historical Best => Architecture : {str(best_arch['arch'])}")
    
    bar = tqdm(range(cycles))
    for iter_idx in bar:
        ########################################################
        # Random Select Candidiate in Pool
        ########################################################
        candidate_idx = sorted(range(len(pools)), key=lambda i: pools[i][proxy_name])
        candidate_idx = candidate_idx[:PARENT_COUNT]
        candidates = [pools[idx] for idx in candidate_idx]
        
        ########################################################
        # Do Mutation
        ########################################################
        new_candidates1 = _EA_mutation(candidates, MUTATION_COUNT, info_funcs, constraints=constraints)
        
        ########################################################
        # Do Crossover
        ########################################################
        new_candidates2 = _EA_crossover(candidates, CROSSOVER_COUNT, info_funcs, constraints=constraints)
        
        ########################################################
        # Random Sample
        ########################################################
        new_candidates3 = _EA_sample(sample_func, RANDOM_COUNT, info_funcs, constraints=constraints)
        
        ########################################################
        # Update Pool and Discard Old Item
        ########################################################
        new_candidates_all = new_candidates1 + new_candidates2 + new_candidates3
        new_candidates_all = sorted(new_candidates_all, key=lambda x: x[proxy_name])
        pools.extend(new_candidates_all)
        for i in range(DISCARD_COUNT): pools.pop(0)

        #############################################
        # Keep Track of Historical Best Architecture
        #############################################
        if best_arch[proxy_name] < new_candidates_all[-1][proxy_name]:
            best_arch = new_candidates_all[-1]
            s = f"Historical Best => FLOPS: {best_arch['flops']:.2f} Params: {best_arch['params']:.2f} {proxy_name}: {best_arch[proxy_name]:e}"
            logger.info(s)
            logger.info(f"Historical Best => Architecture : {str(best_arch['arch'])}")

        sorted_pool = sorted(pools, key=lambda x: x[proxy_name])
        pool_best = sorted_pool[-1]
        s = f"Pool Best => FLOPS: {pool_best['flops']:.2f} Params: {pool_best['params']:.2f} {proxy_name}: {pool_best[proxy_name]:e}"
        bar.set_description(s)

        if iter_idx % 100 == 0:
            s1 = '\nCurrent-Pool-FLOPS '
            s2 = 'Current-Pool-Param '
            s3 = f'Current-Pool-{proxy_name}  '
            for arch in sorted_pool[-10:]:
                s1 += f"{arch['flops']:8.4f} "
                s2 += f"{arch['params']:8.4f} "
                s3 += f"{arch[proxy_name]:8.4f} "
            logger.info(s1)
            logger.info(s2)
            logger.info(s3)
        
    
    logger.info('End Of Algorithm')
    pools = sorted(pools, key=lambda x: x[proxy_name], reverse=True)
    for idx, arch_info in enumerate(pools[:TOPK]):
        s = f"Pool Top{idx+1} => FLOPS: {arch_info['flops']:.2f} Params: {arch_info['params']:.2f} {proxy_name}: {arch_info[proxy_name]:e}"
        logger.info(s)
        logger.info(f"Pool Top{idx+1} => Architecture : {str(arch_info['arch'])}")

    if proxy_name == 'synflow' : synflow.nonlinearize(model, signs)
    return pools[:TOPK], best_arch

##############################
# Train Sensitive
##############################
def train_epoch_sensitive(model, dataloader, optimizer, cfg, device, task_flops, task_params, 
                     est=None, logger=None, local_rank=0, world_size=0, is_gumbel=False,
                     prefix='', epoch=None, total_epoch=None, logdir='./', ema=None):
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
    
    temperature = model.module.temperature if is_ddp else model.temperature
    mloss = torch.zeros(3, device=device)  # mean losses
    
    pbar = enumerate(dataloader)
    if local_rank in [-1, 0]:
        print(('%10s' * 13) % ('Epoch', 'gpu_mem', 'fore', 'back', 'total', 'targets', 'img_size', 'lr', 'moment', 'decay', 'temp', 'GFLOPS', 'f_loss'))
        pbar = tqdm(pbar, total=iterations, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    
    t_data = time.time()
    # for iteration, (input, target) in enumerate(loader):
    for iter_idx, (uimgs, targets, paths, _) in pbar:
        ########################################################
        # Learning WarmUp
        ########################################################
        ni = iter_idx + iterations * (epoch- 1)
        if ni <= nw:
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
        
        ########################################################
        # Sample Architecture
        ########################################################
        arch_theta = torch.cat([theta().reshape(1, -1) for theta in (model.module.thetas if is_ddp else model.thetas )], dim=0)
        if is_gumbel:
            gumbel_prob = nn.functional.gumbel_softmax(arch_theta, temperature, dim=-1)
        else:
            gumbel_prob = nn.functional.softmax(arch_theta, dim=-1)
            
        t_data = time.time() - t_data 
        ##################################################################
        ### 1st. Train SuperNet Parameter
        ##################################################################
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        
        t_infer=time.time()
        pred     = model.module(imgs, gumbel_prob)      if is_ddp else model(imgs, gumbel_prob)
        
        
        drop_mask = torch.ones(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]).cuda()
        drop_mask = torch.nn.functional.dropout(drop_mask, p=0.75, training=True)  # uint8 to float32, 0-255 to 0.0-1.0
        imgs_aug = imgs * drop_mask
        
        
        mask_sizes = [p.shape[2:4] for p in pred[0][1]]
        masks = build_foreground_mask(mask_sizes, targets.to(device), model)  # scaled by batch_size

        pred_aug = model.module(imgs_aug, gumbel_prob)  if is_ddp else model(imgs_aug, gumbel_prob)
                
        ################################################################################
        # if iter_idx < 2: 
        #     import cv2
        #     f = str(Path(logdir) / ('label_batch%g.jpg' % ni))  # filename
        #     result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
            
        #     m1 = str(Path(logdir) / ('masks_batch%g_f1.jpg' % ni))  # filename
        #     m2 = str(Path(logdir) / ('masks_batch%g_f2.jpg' % ni))  # filename
        #     m3 = str(Path(logdir) / ('masks_batch%g_f3.jpg' % ni))  # filename
        #     f_names = [m1,m2,m3]
        #     for i in range(3):
        #         masks[i] = masks[i].cpu().float().numpy().transpose(0,2,3,1).repeat(3, axis=-1)
        #         print('masks[i]', masks[i].shape)
        #         top_img = np.concatenate(masks[i][0:2], axis=0)
        #         # bot_img = np.concatenate([masks[i][2:4]], axis=2)
        #         # ful_img = np.concatenate([top_img, bot_img])
        #         ful_img = top_img
                
        #         ful_img = (ful_img * 255.0).astype(np.uint8)
        #         print('ful_img', ful_img.shape, ful_img.dtype)
        #         ful_img = cv2.resize(ful_img, (208,416), interpolation=cv2.INTER_AREA)
        #         cv2.imwrite(f_names[i], ful_img)
        
        ################################################################################
        sen_loss, loss_items = compute_sensitive_loss(pred[0][1], pred_aug[0][1], targets, masks)  # scaled by batch_size

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
        train_loss = sen_loss + flops_loss
        t_infer=time.time()-t_infer
    
        time_grad= time.time()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # if ema is not None:
        #     ema.update(model)
        
        # Basic Info
        for j, x in enumerate(optimizer.param_groups):
            if 'momentum' in x:
                break
        print_lr = x['lr'] if 'lr' in x else 0
        print_m = x['momentum'] if 'momentum' in x else 0
        print_wdecay = x['weight_decay'] if 'weight_decay' in x else 0
        
        continue
        # Print
        if local_rank in [-1, 0]:
            ni = iter_idx
            mloss = (mloss * iter_idx + loss_items) / (iter_idx + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 11) % (
                '%g/%g' % (epoch, total_epoch), mem, *mloss, targets.shape[0], imgs.shape[-1], print_lr, print_m, print_wdecay, temperature, output_flops, squared_error_flops.detach().cpu())
            pbar.set_description(s)
            
            # Plot
            if ni < 3:
                f = str(Path(logdir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                # if tb_writer and result is not None:
                #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard
    
    
    
        t_data = time.time()
    torch.cuda.synchronize()

    # if iteration % (iterations // 2 -1) == 0:
    #     distributions = softmax(model.module.thetas[0]().detach().cpu().numpy())
    #     print('Distributions in 1 stage:', distributions)
             
    # if iteration % cfg.LOG_INTERVAL == 0:
    #     lrl = [param_group['lr'] for param_group in optimizer.param_groups]
    #     lr = sum(lrl) / len(lrl)
        
    #         # if cfg.SAVE_IMAGES and output_dir:
    #         #     torchvision.utils.save_image(
    #         #         input, os.path.join(
    #         #             output_dir, 'train-batch-%d.jpg' %
    #         #             iteration), padding=0, normalize=True)




def arch_2_subnet(model, architecture):
    subnet = ''
    for layer, layer_arch in zip(model.blocks, architecture):
            for blocks, arch in zip(layer, layer_arch):
                subnet += blocks[arch].get_block_name() + '->'
    return subnet[:-2]

def get_batch_jacobian(model, input, target, random_cand):
    model.zero_grad()
    input.requires_grad_(True)
    (inference_output, output), feature_s, chosen_subnet = model(input, random_cand, calc_metric=True)
    inference_output = sum_arr_tensor(inference_output)
    inference_output.backward(torch.ones_like(inference_output))
    jacob = input.grad.detach()
    return jacob

def validate(model, loader, loss_fn, prioritized_board, cfg, device, log_suffix='', local_rank=0, logger=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        candidates_map = [[], [], [], [], [], [], [], [], [], []]
        for batch_idx, (input, target, paths, _) in enumerate(loader):
            target = target.to(device)
            last_batch = batch_idx == last_idx
            input = input.to(device, non_blocking=True).float() / 255.0
            nb, _, height, width = input.shape
            for idx, candidate in enumerate(prioritized_board.prioritized_board):
                (inference_output, output), _, _ = model(input, candidate[3])
                candidates_map[idx].append(compute_map(inference_output.detach(), target, height, width, nc=model.nc))
        
        average_maps = []
        for candidate_map in candidates_map:
            average_maps.append(sum(candidate_map) / len(candidate_map))
        
        best_candidate_idx, _ = max(enumerate(average_maps), key=operator.itemgetter(1))
        print(best_candidate_idx, average_maps)
        return best_candidate_idx, average_maps
        
    
