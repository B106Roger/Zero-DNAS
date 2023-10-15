from pathlib import Path
import json
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import imageio
import argparse
import json
import os
from tqdm import tqdm


def one_plot():
    ###############################
    # Input Area
    ###############################
    # label_list = ['t5','t3','t1']
    label_list = ['t3']
    
    data_list = [
        # '0818_izdnas-every_act3',
        '0821_izdnas-t3-every_act3',
        # '0823_izdnas-t1-every_act3'
    ]
    # label_list = ['t5','t1']
    # data_list = [
    #     '0916_v7_init2_t5',
    #     '0914_v7_init2_t1'
    # ]
    ###############################
    ###############################
    # Custom Data Preparation Area
    ###############################
    parent_folder = os.path.join('experiments', 'workspace', 'statistics')
    filename_list = [os.path.join(parent_folder, data_foldname, 'ranking_analyze_overall.json') for data_foldname in data_list]
    ###############################


    # fig, axes = plt.subplots(1, 1, figsize=(8,10))
    fig, axes = plt.subplots(1, 1, figsize=(4,5))

    axes2 = axes.twinx() #
    for filename, f_label in zip(filename_list, label_list):
        with open(filename, 'r') as f:
            data = json.load(f)

        axes.plot( data['tau']['x'], data['tau']['y'], label=f'tau-{f_label}')
        axes2.plot(data['var']['x'], data['var']['y'], label=f'var-{f_label}', linestyle='--') #

    step = int((data['tau']['x'][-1] - data['tau']['x'][0]) / (len(data['tau']['x'])-1))
    color = 'tab:red' #
    # color = 'black'
    
    axes.set_ylabel('Kendals Tau', color = color)
    axes.tick_params(axis='y', labelcolor=color)
    axes.set_xticks(np.arange(data['tau']['x'][0], data['tau']['x'][-1]+step, step))
    
    color = 'tab:blue'   #
    axes2.set_ylabel('Zero Cost Score', color = color) #
    axes2.tick_params(axis='y', labelcolor=color) #

    axes.set_xlabel('Epoch')
    axes.set_title(f"Model Statistics")
    axes.legend(loc='center right')
    line1, label1 = axes.get_legend_handles_labels()#
    line2, label2 = axes2.get_legend_handles_labels()#
    axes.legend(line1+line2, label1+label2, loc='center right')#

    plt.tight_layout()
    plt.savefig(f'ranking_analyze_overall_V2.jpg', dpi=300)
    plt.close(fig)

def multi_plot_rank():
    ###############################
    # Input Area
    ###############################
    label_list = ['t5','t3','t1']
    data_list = [
        '0818_izdnas-every_act3',
        '0821_izdnas-t3-every_act3',
        '0823_izdnas-t1-every_act3'
    ]
    # label_list = ['t5','t1']
    # data_list = [
    #     '0916_v7_init2_t5',
    #     '0914_v7_init2_t1'
    # ]
    ###############################
    ###############################
    # Custom Data Preparation Area
    ###############################
    parent_folder = os.path.join('experiments', 'workspace', 'statistics')
    filename_list = [os.path.join(parent_folder, data_foldname, 'ranking_analyze_stage.json') for data_foldname in data_list]
    ###############################
    
    fig, axes = plt.subplots(4, 2, figsize=(12,15))
    for exp_idx, (filename, f_label) in enumerate(zip(filename_list, label_list)):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        for stage_idx in range(len(data)):
            row_idx = stage_idx // 2
            col_idx = stage_idx %  2
            axes1   = axes[row_idx,  col_idx]
            # axes2   = axes[row_idx,  col_idx].twinx()
            
            axes1.plot(data[stage_idx]['tau']['x'], data[stage_idx]['tau']['y'], label=f'tau-{f_label}')
            # axes2.plot(data[stage_idx]['var']['x'], data[stage_idx]['var']['y'], label=f'var-{f_label}', linestyle='--')
            
            if exp_idx == len(filename_list) - 1:
                print(f'execution {exp_idx} {stage_idx}')
                color = 'tab:red'
                axes1.set_ylabel('Kendals Tau', color = color)
                axes1.tick_params(axis='y', labelcolor=color)
                
                # color = 'tab:blue'   
                # axes2.set_ylabel('Zero Cost Score', color = color)
                # axes2.tick_params(axis='y', labelcolor=color)
                
                axes1.set_xlabel('Epoch')
                axes1.set_title(f"Stage {stage_idx} Statistics")
                # line1, label1 = axes1.get_legend_handles_labels()
                # line2, label2 = axes2.get_legend_handles_labels()
                # axes1.legend(line1+line2, label1+label2, loc='center right')
                axes1.legend()

    plt.tight_layout()
    plt.savefig(f'ranking_analyze_stage.jpg', dpi=300)
    plt.close(fig)


def multi_plot_param():
    ###############################
    # Input Area
    ###############################
    label_list = ['t3']
    data_list = [
        # '0818_izdnas-every_act3',
        '0821_izdnas-t3-every_act3',
        # '0823_izdnas-t1-every_act3'
    ]
    # label_list = ['t5','t1']
    # data_list = [
    #     '0916_v7_init2_t5',
    #     '0914_v7_init2_t1'
    # ]
    ###############################
    ###############################
    # Custom Data Preparation Area
    ###############################
    parent_folder = os.path.join('experiments', 'workspace', 'statistics')
    filename_list = [os.path.join(parent_folder, data_foldname, 'param_analyze.json') for data_foldname in data_list]
    ###############################
    
    fig, axes = plt.subplots(4, 2, figsize=(8,10))
    for exp_idx, (filename, f_label) in enumerate(zip(filename_list, label_list)):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        
        for stage_idx in range(len(data[0])):
            row_idx = stage_idx // 2
            col_idx = stage_idx %  2
            axes1   = axes[row_idx,  col_idx]
            # axes2   = axes[row_idx,  col_idx].twinx()
            
            for seed_idx, seed in enumerate(range(42,46)):
                # axes1.plot(data[seed_idx][stage_idx]['mean']['x'], data[seed_idx][stage_idx]['mean']['y'], label=f'mean-S{seed}')
                axes1.plot(data[seed_idx][stage_idx]['var']['x'], data[seed_idx][stage_idx]['var']['y'], label=f'var-S{seed}')
                
            # axes2.plot(data[stage_idx]['var']['x'], data[stage_idx]['var']['y'], label=f'var-{f_label}', linestyle='--')
            
            if exp_idx == len(filename_list) - 1:
                print(f'execution {exp_idx} {stage_idx}')
                # color = 'tab:red'
                color = 'black'
                
                axes1.set_ylabel('Variance', color = color)
                axes1.tick_params(axis='y', labelcolor=color)
                
                # color = 'tab:blue'   
                # axes2.set_ylabel('Zero Cost Score', color = color)
                # axes2.tick_params(axis='y', labelcolor=color)
                
                axes1.set_xlabel('Epoch')
                axes1.set_title(f"Stage {stage_idx} Params")
                # line1, label1 = axes1.get_legend_handles_labels()
                # line2, label2 = axes2.get_legend_handles_labels()
                # axes1.legend(line1+line2, label1+label2, loc='center right')
                
                step = int((data[seed_idx][stage_idx]['var']['x'][-1] - data[seed_idx][stage_idx]['var']['x'][0]) / (len(data[seed_idx][stage_idx]['var']['x'])-1)) * 2
                axes1.set_xticks(np.arange(data[seed_idx][stage_idx]['var']['x'][0], data[seed_idx][stage_idx]['var']['x'][-1]+step, step))
                
                axes1.legend()

    plt.tight_layout()
    plt.savefig(f'param_analyze_stage.jpg', dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    one_plot()
    # multi_plot()
    # multi_plot_param()