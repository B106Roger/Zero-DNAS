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
import imageio, cv2

import  matplotlib.pyplot as plt

PROXY_DICT = {
    'snip'  :  snip.calculate_snip,
    'synflow': synflow.calculate_synflow,
    'naswot' : naswot.calculate_wot,
    'grasp': grasp.calculate_grasp
}

PROXY_MAP_DICT = {
    'naswot': naswot.calculate_zero_cost_map2,
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

def mutation(canddidate, stage=None, num_component=1):
        sample = {
            'arch' : copy.deepcopy(canddidate['arch']),
            'arch_type': 'continuous'
        }
        rand_stage      = np.random.randint(0, len(sample['arch'])) if stage is None else stage
        search_keys     = []
        for k in sample['arch'][rand_stage].keys():
            if k != 'operator' and k != 'block_name':
                search_keys.append(k)
                
        np.random.shuffle(search_keys)
        for rand_component in search_keys[:num_component]:
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

def print_map_func(zc_maps):
    for blk_id, zc_map in enumerate(zc_maps):
        keys = zc_map.keys()
        print(f'{blk_id:5d}', end='')
        for key in keys:
            print(f'{key:>20s}', end='')
        print()
        
        print(f'{blk_id:5d}', end='')
        for key in keys:
            print(f'{zc_map[key]:20.6f}', end='')
        print()

def analyze_map_func(arch_info1, arch_info2, title, img_filename, text_filename):
    zc_maps1 = arch_info1['naswot_map']
    zc_maps2 = arch_info2['naswot_map']
    arch1    = arch_info1['arch']
    arch2    = arch_info2['arch']
    
    fig, axes = plt.subplots(len(zc_maps1))
    fig.suptitle(title)
    f = open(text_filename, 'w')
    for stage_id, (zc_map1, zc_map2) in enumerate(zip(zc_maps1, zc_maps2)):
        keys = zc_map1.keys()
        f.write(f'{"":22s}')
        for key in keys: f.write(f'{key:>15s}')
        f.write('\n')
        
        f.write(f'{"arch1 wot score":>20s}{stage_id:2d}')
        for key in keys: f.write(f'{zc_map1[key]:15.10f}')
        f.write('\n')

        f.write(f'{"arch2 wot score":>20s}{stage_id:2d}')
        for key in keys: f.write(f'{zc_map2[key]:15.10f}')
        f.write('\n')
        
        f.write(f'{"arch1 wot rank":>20s}{stage_id:2d}')
        score1=np.array([zc_map1[key] for key in keys])
        rank1 = (-score1).argsort()[::-1]
        for val in rank1: f.write(f'{val:15d}')
        f.write('\n')
        
        f.write(f'{"arch2 wot rank":>20s}{stage_id:2d}')
        score2=np.array([zc_map2[key] for key in keys])
        rank2 = (-score2).argsort()[::-1]
        for val in rank2: f.write(f'{val:15d}')
        f.write('\n')
        
        rank_diff  = np.abs(rank1-rank2).sum()
        score_diff = np.sum([v for v in zc_map1.values()]) - np.sum([v for v in zc_map2.values()])
        f.write(f'Rank Difference {rank_diff}     Score Difference {score_diff}\n')
        f.write(f"Arch1 FLOPS={arch_info1['flops']:5.2f}G Param={arch_info1['params']:5.2f}M  wot={arch_info1['naswot']}\n")
        f.write(f"Arch2 FLOPS={arch_info2['flops']:5.2f}G Param={arch_info2['params']:5.2f}M  wot={arch_info2['naswot']}\n")
        f.write(f'Arch1 {str(arch_info1["arch"])}\n')
        f.write(f'Arch2 {str(arch_info2["arch"])}\n')
        

        axes[stage_id]
        blk_count  = len(rank1)
        comp_list  = [score1, score2]
        color_list = ['r', 'g']
        x = np.arange(blk_count) * 0.8
        
        for i, score_arr in enumerate(comp_list):
            axes[stage_id].bar(x - 0.2*i, height=score_arr, width=0.2, color=[color_list[i]]*blk_count, align='edge')
        #######################################
        # Basic Math Information
        #######################################
        margin = 0.2
        all_scores = np.concatenate(comp_list)
        center = all_scores.mean()
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
        arr_size  = (max_val-min_val)*0.2
        
        # Arrow Plot
        for ii, (arch, color) in enumerate(zip([arch1, arch2],color_list)):
            loc_idx = arch[stage_id]['gamma'].argmax().numpy() * 4 + arch[stage_id]['n_bottlenecks'].argmax().numpy()
            loc = x[loc_idx] - 0.2 * (ii - 1.0 / len(comp_list))
            axes[stage_id].arrow(loc, min_val+arr_size, 0, -arr_size*0.6666, 
                                 head_width=arr_size*0.8, head_length=arr_size*0.3333, color=color, edgecolor='black')

        # Rank Plot
        for ii, (rank, score, color) in enumerate(zip([rank1, rank2],[score1, score2],color_list)):
            for iii in range(4):
                loc = x[rank[iii]] - 0.2 * (ii - 1.0 / len(comp_list)) + 0.024
                axes[stage_id].text(loc, score[rank[iii]]-arr_size/2, str(iii+1), color=color)

    fig.set_size_inches(15.5, 15.5)
    fig.tight_layout()
    fig.savefig(img_filename)
    f.close()
    return fig

def analyze_map_func2(arch_info_list, title, img_filename, text_filename):
    # zc_maps1 = arch_info1['naswot_map']
    # zc_maps2 = arch_info2['naswot_map']
    # arch1    = arch_info1['arch']
    # arch2    = arch_info2['arch']
    
    fig, axes = plt.subplots(8)
    fig.suptitle(title)
    f = open(text_filename, 'w')
    for stage_id in range(8):
        score_list = []
        rank_list = []
        
        keys = arch_info_list[0]['naswot_map'][stage_id].keys()
        f.write(f'{"":22s}')
        for key in keys: f.write(f'{key:>15s}')
        f.write('\n')
        
        for arch_id, arch_info in enumerate(arch_info_list):
            zc_map = arch_info['naswot_map'][stage_id]

            tmp_str = f"arch{arch_id:02d} wot score"
            f.write(f'{tmp_str:>20s}{stage_id:2d}')
            for key in keys: f.write(f'{zc_map[key]:15.10f}')
            f.write('\n')

        for arch_id, arch_info in enumerate(arch_info_list):
            zc_map = arch_info['naswot_map'][stage_id]
            score  = np.array([zc_map[key] for key in keys])
            rank   = (-score ).argsort()[::-1]
        
            tmp_str = f"arch{arch_id:02d} wot rank"
            f.write(f'{tmp_str:>20s}{stage_id:2d}')
            for val in rank: f.write(f'{val:15d}')
            f.write('\n')
            
            score_list.append(score)
            rank_list.append(rank)
    
        for arch_id, arch_info in enumerate(arch_info_list):
            # rank_diff  = np.abs(rank1-rank2).sum()
            # score_diff = np.sum([v for v in zc_map1.values()]) - np.sum([v for v in zc_map2.values()])
            # f.write(f'Rank Difference {rank_diff}     Score Difference {score_diff}\n')
            f.write(f"Arch{arch_id:02d} FLOPS={arch_info['flops']:5.2f}G Param={arch_info['params']:5.2f}M  wot={arch_info['naswot']}\n")
        
        for arch_id, arch_info in enumerate(arch_info_list):
            f.write(f'Arch{arch_id:02d} {str(arch_info["arch"])}\n')
        

        candidiate_num  = len(rank)
        comp_list  = score_list
        color_list = ['r', 'g', 'b', 'c', 'k', 'm']
        arch_list  = [arch_info['arch'] for arch_info in arch_info_list]
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
        for ii, (arch, color) in enumerate(zip(arch_list,color_list)):
            loc_idx = arch[stage_id]['gamma'].argmax().numpy() * 4 + arch[stage_id]['n_bottlenecks'].argmax().numpy()
            loc = x[loc_idx] - with_val * (ii-len(comp_list)/2) + with_val
            axes[stage_id].arrow(loc, min_val+arr_size, 0, -arr_size*0.6666, 
                                 head_width=arr_size*0.8, head_length=arr_size*0.3333, color=color, edgecolor='black')

        # Rank Plot
        for ii, (rank, score, color) in enumerate(zip(rank_list,comp_list,color_list)):
            for iii in range(4):
                loc = x[rank[iii]] - with_val * (ii-len(comp_list)/2) #+ 0.024
                axes[stage_id].text(loc, score[rank[iii]]-arr_size*0.8, str(iii+1), color=color)

    fig.set_size_inches(15.5, 15.5)
    fig.tight_layout()
    fig.savefig(img_filename)
    f.close()
    return fig

def interpolate_arch(arch1, arch2, alpha):
    """
    arch = arch1 * alpha + arch2 * (1 - alpha)
    """
    arch = []
    for depth in range(len(arch1)):
        tmp_arch = {}
        for key in arch1[depth].keys():
            if key == 'block_name': 
                tmp_arch[key] = arch1[depth][key]
                continue
            tmp_arch[key] = arch1[depth][key] * alpha + arch2[depth][key] * (1. - alpha)
        arch.append(tmp_arch)
    return arch


#######################################
# Test Function
#######################################
def test_zc_map(proxy_name, model, dataloader, optimizer, cfg, device, task_flops, task_params, cycles,
                     est=None, logger=None, local_rank=0, prefix='', logdir='./', output_dir=''):
    batch_size = cfg.DATASET.BATCH_SIZE
    is_ddp = is_parallel(model)
    nn_model = model.module if is_ddp else model
    
    naive_model  = model.module if is_ddp else model
    search_space = naive_model.search_space
    model.eval()
    
    analyze_dir = os.path.join(output_dir, 'analyze_results')
    os.makedirs(analyze_dir, exist_ok=True)
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
    info_funcs = [
        {'flops' : (lambda x: nn_model.calculate_flops_new (x, est.flops_dict) / 1e3)},
        {'params': (lambda x: nn_model.calculate_params_new(x, est.params_dict))},
        {proxy_name : (lambda x: PROXY_DICT[proxy_name](model, x['arch'], imgs, targets, optimizer))},
        {proxy_name+'_map' : (lambda x: PROXY_MAP_DICT[proxy_name](model, x['arch'], imgs, targets, None))}
    ]
    ##################################
    # Sample Function
    ##################################
    sample_func = lambda : naive_model.random_sampling()
    
    
    num_of_sample = 10
    for depth in range(8):
        current_dir = os.path.join(analyze_dir, str(depth))
        os.makedirs(current_dir, exist_ok=True)
        for idx in range(num_of_sample):
            arch1 = sample_func()
            print(arch1)
            arch1_info = {'arch' : arch1, 'arch_type': 'continuous'}
            get_model_info(arch1_info, info_funcs)
            
            arch2_info = mutation(arch1_info, int(depth))
            get_model_info(arch2_info, info_funcs)
            
            image_dir = os.path.join(current_dir, f'{idx}.jpg')
            text_dir  = os.path.join(current_dir, f'{idx}.txt')
            analyze_map_func(arch1_info, arch2_info, image_dir, text_dir)
            
            
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
    
    # manually_arch2 = copy.deepcopy(manually_arch)
    # manually_arch2[3]['gamma']=torch.tensor([1.,0.,0.])
    # man_arch2 = {'arch' : manually_arch2, 'arch_type': 'continuous'}
    # get_model_info(man_arch2, info_funcs)
    # s = f"man_arch22 => FLOPS: {man_arch2['flops']:.2f} Params: {man_arch2['params']:.2f} {proxy_name}: {man_arch2[proxy_name]:e}"
    # logger.info(s)
    # analyze_map_func(man_arch, man_arch2)
    
    # if False:
    #     sample_func = lambda : naive_model.random_sampling()
    #     pools = _EA_sample(sample_func, POPULATION_COUNT, info_funcs, constraints)

def test_zc_map_evolve(proxy_name, model, dataloader, optimizer, cfg, device, task_flops, task_params, cycles,
                     est=None, logger=None, local_rank=0, prefix='', logdir='./', output_dir=''):
    batch_size = cfg.DATASET.BATCH_SIZE
    is_ddp = is_parallel(model)
    nn_model = model.module if is_ddp else model
    
    naive_model  = model.module if is_ddp else model
    search_space = naive_model.search_space
    model.eval()
    
    analyze_dir = os.path.join(output_dir, 'analyze_results')
    os.makedirs(analyze_dir, exist_ok=True)
    ##################################################################
    ### 0st. Select Dataset
    ##################################################################    
    loader = enumerate(dataloader)
    for iter_idx, (uimgs, targets, paths, _) in loader:
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets  = targets.to(device)
        if iter_idx == 2: break

    save_img = uimgs.permute((0,2,3,1)).numpy()
    cv2.imwrite(os.path.join(analyze_dir, 'wot_img0.jpg'), save_img[0])
    cv2.imwrite(os.path.join(analyze_dir, 'wot_img1.jpg'), save_img[1])
    
    ##############################################
    # Zero Cost Name (snip, synflow, grasp)
    ##############################################
    info_funcs = [
        {'flops' : (lambda x: nn_model.calculate_flops_new (x, est.flops_dict) / 1e3)},
        {'params': (lambda x: nn_model.calculate_params_new(x, est.params_dict))},
        {proxy_name : (lambda x: PROXY_DICT[proxy_name](model, x['arch'], imgs, targets, optimizer))},
        {proxy_name+'_map' : (lambda x: PROXY_MAP_DICT[proxy_name](model, x['arch'], imgs, targets, True))}
    ]
    
    # manually_arch = [
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75'}, 
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75'}, 
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75'}, 
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75'}, 
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75'},
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75'},
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75'},
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75'} 
    # ]
    
    manually_arch1 = [
        {'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75',}, 
        {'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75',}, 
        { 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75',},#
        {'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75',}, 
        {'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75',},
        {'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75',},
        {'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75',},
        {'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75',} 
    ]
    man_arch1 = {'arch' : manually_arch1, 'arch_type': 'continuous'}
    get_model_info(man_arch1, info_funcs)
    
    
    stage = 2
    manually_arch2 = copy.deepcopy(manually_arch1)
    manually_arch2[stage]['gamma']         = torch.tensor([0., 1., 0.]) # torch.tensor([0., 0., 1.])
    manually_arch2[stage]['n_bottlenecks'] = torch.tensor([0., 1., 0, 0]) # torch.tensor([0., 1., 0, 0])
    
    man_arch2 = {'arch' : manually_arch2, 'arch_type': 'continuous'}
    get_model_info(man_arch2, info_funcs)
    
    ##################################
    # Sample Function
    ##################################
    steps = 10
    gif_dir = os.path.join(analyze_dir, f'data.mp4')
    with imageio.get_writer(gif_dir, mode='I', fps=2) as writer:
        for step in reversed(range(steps+1)):
            arch = interpolate_arch(manually_arch1, manually_arch2, step/steps)
            arch_info = {'arch' : arch, 'arch_type': 'continuous'}
            get_model_info(arch_info, info_funcs)
            
            image_dir = os.path.join(analyze_dir, f'{step}.jpg')
            text_dir  = os.path.join(analyze_dir, f'{step}.txt')
            fig = analyze_map_func(man_arch1, arch_info, f"WOT Score Analyze {step/steps}", image_dir, text_dir)
            
            fig.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(data)
            
def test_zc_map_param (proxy_name, model, dataloader, optimizer, cfg, device, task_flops, task_params, cycles,
                     est=None, logger=None, local_rank=0, prefix='', logdir='./', output_dir=''):
    batch_size = cfg.DATASET.BATCH_SIZE
    is_ddp = is_parallel(model)
    nn_model = model.module if is_ddp else model
    
    naive_model  = model.module if is_ddp else model
    search_space = naive_model.search_space
    model.eval()
    
    analyze_dir = os.path.join(output_dir, 'analyze_results')
    os.makedirs(analyze_dir, exist_ok=True)
    ##################################################################
    ### 0st. Select Dataset
    ##################################################################    
    loader = enumerate(dataloader)
    for iter_idx, (uimgs, targets, paths, _) in loader:
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets  = targets.to(device)
        if iter_idx == 2: break

    save_img = uimgs.permute((0,2,3,1)).numpy()
    cv2.imwrite(os.path.join(analyze_dir, 'wot_img0.jpg'), save_img[0])
    cv2.imwrite(os.path.join(analyze_dir, 'wot_img1.jpg'), save_img[1])
    
    ##############################################
    # Zero Cost Name (snip, synflow, grasp)
    ##############################################
    info_funcs = [
        {'flops' : (lambda x: nn_model.calculate_flops_new (x, est.flops_dict) / 1e3)},
        {'params': (lambda x: nn_model.calculate_params_new(x, est.params_dict))},
        {proxy_name : (lambda x: PROXY_DICT[proxy_name](model, x['arch'], imgs, targets, optimizer))},
        {proxy_name+'_map' : (lambda x: PROXY_MAP_DICT[proxy_name](model, x['arch'], imgs, targets, True))}
    ]
    
    # manually_arch = [
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75'}, 
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75'}, 
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75'}, 
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75'}, 
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75'},
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75'},
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75'},
    #     {'gamma': torch.tensor([0.3333, 0.3333, 0.3333]), 'n_bottlenecks': torch.tensor([0.25, 0.25, 0.25, 0.25]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75'} 
    # ]
    
    manually_arch1 = [
        {'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75',}, 
        {'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75',}, 
        { 'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75',},#
        {'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0]), 'block_name': 'BottleneckCSP_Search_num1_gamma0.75',}, 
        {'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75',},
        {'gamma': torch.tensor([0., 0., 1.]), 'n_bottlenecks': torch.tensor([0., 1., 0, 0]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75',},
        {'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75',},
        {'gamma': torch.tensor([1., 0., 0.]), 'n_bottlenecks': torch.tensor([1., 0., 0, 0]), 'block_name': 'BottleneckCSP2_Search_num1_gamma0.75',} 
    ]
    man_arch1 = {'arch' : manually_arch1, 'arch_type': 'continuous'}
    get_model_info(man_arch1, info_funcs)
    
    
    # stage = 2
    # manually_arch2 = copy.deepcopy(manually_arch1)
    # manually_arch2[stage]['gamma']         = torch.tensor([0., 1., 0.]) # torch.tensor([0., 0., 1.])
    # manually_arch2[stage]['n_bottlenecks'] = torch.tensor([0., 1., 0, 0]) # torch.tensor([0., 1., 0, 0])
    
    # man_arch2 = {'arch' : manually_arch2, 'arch_type': 'continuous'}
    # get_model_info(man_arch2, info_funcs)
    
    ##################################
    # Sample Function
    ##################################
    samples = 6
    score_list = []
    wot_score_list = []
    for step in range(samples):
        score_list.append(copy.deepcopy(man_arch1['naswot_map']))
        wot_score_list.append(man_arch1['naswot'])
        model._initialize_weights()
        get_model_info(man_arch1, info_funcs)

    image_dir = os.path.join(analyze_dir, f'result.jpg')
    text_dir  = os.path.join(analyze_dir, f'result.txt')
    arch_info_list = [
        {
            'arch' : man_arch1['arch'],
            'arch_type' : 'continuous',
            'naswot_map' : score,
            'flops' : man_arch1['flops'],
            'params' : man_arch1['params'],
            'naswot' : wot_score,
            
            
        } for score, wot_score in zip(score_list, wot_score_list) 
    ]
    fig = analyze_map_func2(arch_info_list, f"WOT Score Analyze", image_dir, text_dir)
        