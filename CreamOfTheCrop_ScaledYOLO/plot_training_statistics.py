from pathlib import Path
import json
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import imageio
import argparse
from itertools import combinations
from tqdm import tqdm
matplotlib.rc('font', size=15)

parser = argparse.ArgumentParser(description='Plot the theta distribution according to the serach result')
parser.add_argument('--type',     type=str, default='zcmap_ema', help='zcmap_ema or zcmap or train')
args = parser.parse_args()


EXP_NAME_LIST = [
    '0818_izdnas-every_act3-S42',
    '0818_izdnas-every_act3-S43',
    '0818_izdnas-every_act3-S44',
    '0818_izdnas-every_act3-S45',
    
]

experiment_name = [Path(f'experiments/workspace/train/{exp_name}/') / (args.type + '.txt') for exp_name in EXP_NAME_LIST]


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

# zcmap.txt
def arch_generator(filename_list):
    f_list = []
    for filename in filename_list:
        f= open(filename, 'r')
        f_list.append(f)

    zc_map_iter_list = []
    for idx, items in enumerate(zip(*f_list)):
        # if idx % 2 == 1:
        zc_map_list = []
        if True:
            idx_string, *content = items[0].split(' ')
            
            # Parse Epoch and Iteration
            idx_string = idx_string[1:-1]
            total_idx, epoch_idx, iter_idx = idx_string.split('-')
            total_idx, epoch_idx, iter_idx = int(total_idx), int(epoch_idx), int(iter_idx)
            
            
            for item in items:
                idx_string, *content = item.split(' ')
                # Parse Architecture Prob
                zc_map_string = ''.join(content)
                zc_map_string = zc_map_string.replace("gamma", "g").replace("n_bottlenecks", "n")
                zc_map_string = zc_map_string.replace("inf", "0")
                zc_map = eval(zc_map_string)
                zc_map_list.append(zc_map)
            
            if total_idx % 1000 == 0:
                zc_map_iter_list.append((total_idx, zc_map_list))
            
            
    return zc_map_iter_list

# train.txt
def arch_generator2(filename):
    f= open(filename, 'r')
    for idx, item in enumerate(f):
        if idx % 2 == 1:
            idx_string, *content = item.split(' ')
            
            # Parse Epoch and Iteration
            idx_string = idx_string[1:-1]
            epoch_idx, iter_idx = idx_string.split('-')
            epoch_idx, iter_idx = int(epoch_idx), int(iter_idx)
            
            # Parse Architecture Prob
            arch_prob_raw = ''.join(content)
            arch_prob_raw = arch_prob_raw.replace("gamma", "g").replace("n_bottlenecks", "n")
            arch_prob_raw = arch_prob_raw.replace("inf", "0")
            

            print(epoch_idx, iter_idx)
            if (epoch_idx==1 and iter_idx <= 50): continue
            
            arch = eval(arch_prob_raw)
            if iter_idx % 200 == 0:
            # if True:
                yield (epoch_idx, iter_idx), arch


if __name__ == '__main__':
    if 'zcmap' in args.type:   zc_map_iter_list = arch_generator(experiment_name)
    # elif 'train' in args.type: gen = arch_generator2(thetas_path)
    else: raise ValueError(f'invalid args.type={args.type}')
        
    for (iter_idx, zc_maps) in zc_map_iter_list:
        num_zc_map = len(zc_maps)
        
        num_stages = len(zc_maps[0])
        
        iter_rank_diff  = []
        iter_score_diff = []
        for s in range(num_stages):
            keys  = zc_maps[0][s].keys()
            
            single_depth_zcmaps = np.array([
                np.array([
                    zc_map[s][key] for key in keys
                ]) for zc_map in zc_maps
            ])
            
            comb_list = list(combinations(range(num_zc_map),2))
            single_depth_zcmaps_rank = np.argsort(single_depth_zcmaps, axis=1)
            tmp_stage_rank_diff = []
            tmp_stage_score_diff= []
            for comb in comb_list:
                rank0 = single_depth_zcmaps_rank[comb[0]]
                rank1 = single_depth_zcmaps_rank[comb[1]]
                tmp_stage_rank_diff.append(np.abs(rank0-rank1).sum())
                
                score0 = single_depth_zcmaps[comb[0]]
                score1 = single_depth_zcmaps[comb[1]]
                tmp_stage_score_diff.append(np.abs(score0-score1).sum())
                
            iter_rank_diff.append(np.mean(tmp_stage_rank_diff))
            iter_score_diff.append(np.mean(tmp_stage_score_diff))
        iter_rank_diff =  np.mean(iter_rank_diff)
        iter_score_diff = np.mean(iter_score_diff)
        print(f'Iteartion: {iter_idx:06d} rank_diff={iter_rank_diff:8.4f} score_diff={iter_score_diff:8.4f}')
        