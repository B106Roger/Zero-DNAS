# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import torch
from pathlib import Path
import json

from lib.utils.flops_counter import get_model_complexity_info
from lib.models.blocks.yolo_blocks import get_algorithm_type
from copy import deepcopy
import numpy as np


class FlopsEst(object):
    def __init__(self, model, input_shape=(2, 3, 416, 416), search_space=None):
        self.search_space = deepcopy(search_space)
        self.block_num = len(model.blocks)
        self.choice_num = len(model.blocks[0])
        self.flops_dict = {}
        self.params_dict = {}

       
        model = model.cpu()

        self.params_fixed = 0
        self.flops_fixed = 0
        
        # if Path('flops_dict.json').exists() and Path('params_dict.json').exists():
        if False:
            self.load_flops_dict()
            self.load_params_dict()
        else:

        # Compute Stem FLOPs
            print('Computing stem FLOPs. Resolution ', input_shape)
            flops_dynamic = 0
            params_dynamic = 0
            
            # the largest params, its corresponding flops and params
            max_param_arch_flops, max_param_arch_params = 0, 0
            # the smallest params, its corresponding flops and params
            min_param_arch_flops, min_param_arch_params = 0, 0
            
            
            flops, params, spatial_dimension = get_model_complexity_info(
                model.conv_stem, (input_shape[1], input_shape[2], input_shape[3]), as_strings=False, print_per_layer_stat=False)
            self.params_fixed += params / 1e6
            self.flops_fixed += flops / 1e6
            detect_input_dims = []

            # Compute Blocks FLOPs
            print('Computing blocks FLOPs...')
            for block_id, block in enumerate(model.blocks):
                block_id = str(block_id)
                self.flops_dict[block_id] = {}
                self.params_dict[block_id] = {}
                for module_id, module in enumerate(block):
                    module_id = str(module_id)
                    self.flops_dict[block_id][module_id] = {}
                    self.params_dict[block_id][module_id] = {}
                    flops_list = []
                    param_list = []
                    for choice_id, choice in enumerate(module):
                        print(f'block {block_id} | module_id {module_id} | choice_id {choice_id} | {choice.get_block_name()}', end='  ')
                        choice_id = str(choice_id)
                        if choice.get_block_name() == 'concat':
                            out_chs = choice.block_arguments['out_chs']
                            detect_input_dims.append((1, out_chs, spatial_dimension[0], spatial_dimension[1]))

                            self.flops_dict[block_id][module_id][choice_id] = 0                 # FLOPS(M)
                            self.params_dict[block_id][module_id][choice_id] = 0                # Params(M)
                            # Store the flop and param for each choices
                            flops_list.append(0)
                            param_list.append(0)
                            print()
                            
                            
                        elif 'up' in choice.get_block_name():
                            spatial_dimension = (
                                spatial_dimension[0] * choice.block_arguments['scale_factor'],
                                spatial_dimension[1] * choice.block_arguments['scale_factor']
                            )
                            in_chs = choice.block_arguments['out_chs']
                            input_shape = (in_chs, spatial_dimension[0], spatial_dimension[1])
                            macs, params, next_spatial_dimension = get_model_complexity_info(
                                choice, 
                                input_res=input_shape, 
                                as_strings=False, 
                                print_per_layer_stat=False
                            )
                            flops = macs * 2
                            params = 0
                            print(f'[Up] flops {spatial_dimension}', flops / 1e6, 'M', '-'*40)
                            # flops = 0                            
                            #######################################
                            self.flops_dict[block_id][module_id][choice_id] =0 # flops / 1e6    # FLOPS(M)
                            self.params_dict[block_id][module_id][choice_id] = 0                # Params(M)
                            # Store the flop and param for each choices

                            flops_dynamic += flops / 1e6
                            params_dynamic += params / 1e6
                            # Store the flop and param for each choices
                            flops_list.append(flops / 1e6)
                            param_list.append(params / 1e6)
                        elif 'bottlecsp' not in choice.get_block_name():
                            in_chs = choice.block_arguments['in_chs']
                            input_shape = (in_chs, spatial_dimension[0], spatial_dimension[1])
                            macs, params, next_spatial_dimension = get_model_complexity_info(
                                choice, 
                                input_res=input_shape, 
                                as_strings=False, 
                                print_per_layer_stat=False
                            )
                            algorithm_type = get_algorithm_type()
                            if algorithm_type == 'ZeroDNAS_Egor':
                                flops = macs
                            elif algorithm_type == 'DNAS' or algorithm_type == 'ZeroCost':
                                flops = macs * 2
                            else:
                                raise ValueError(f"Invalid algorithm type {algorithm_type}")
                            spatial_dimension = next_spatial_dimension

                            self.flops_dict[block_id][module_id][choice_id] = flops / 1e6       # FLOPS(M)
                            self.params_dict[block_id][module_id][choice_id] = params / 1e6     # Params(M)

                            flops_dynamic += flops / 1e6
                            params_dynamic += params / 1e6
                            # Store the flop and param for each choices
                            flops_list.append(flops / 1e6)
                            param_list.append(params / 1e6)
                        else:
                            in_chs = choice.block_arguments['in_chs']
                            input_shape = (in_chs, spatial_dimension[0], spatial_dimension[1])
                            macs, params, next_spatial_dimension = get_model_complexity_info(
                                choice, 
                                input_res=input_shape, 
                                as_strings=False, 
                                print_per_layer_stat=False
                            )
                            algorithm_type = get_algorithm_type()
                            if algorithm_type == 'ZeroDNAS_Egor':
                                flops = macs
                            elif algorithm_type == 'DNAS' or algorithm_type == 'ZeroCost':
                                flops = macs * 2
                            else:
                                raise ValueError(f"Invalid algorithm type {algorithm_type}")
                            flops_dynamic += flops / 1e6
                            params_dynamic += params / 1e6
                            ##############################################
                            gamma_list = self.search_space['gamma']
                            depth_list = self.search_space['n_bottlenecks']
                            idx = -1
                            print()
                            if choice_id == str(len(module) - 1):
                                for gamma in gamma_list:
                                    for n_bottleneck in depth_list:
                                        idx+=1
                                        ba=choice.block_arguments
                                        choice_instance = type(choice)(c1=ba['in_chs'], c2=ba['out_chs'], n=n_bottleneck, gamma_space=[gamma])
                                        macs, params, next_spatial_dimension = get_model_complexity_info(
                                            choice_instance, 
                                            input_res=input_shape, 
                                            as_strings=False, 
                                            print_per_layer_stat=False
                                        )
                                        algorithm_type = get_algorithm_type()
                                        if algorithm_type == 'ZeroDNAS_Egor':
                                            flops = macs
                                        elif algorithm_type == 'DNAS' or algorithm_type == 'ZeroCost':
                                            flops = macs * 2
                                        else:
                                            raise ValueError(f"Invalid algorithm type {algorithm_type}")
                                        spatial_dimension = next_spatial_dimension

                                        self.flops_dict[block_id][module_id][str(idx)] = flops / 1e6       # FLOPS(M)
                                        self.params_dict[block_id][module_id][str(idx)] = params / 1e6     # Params(M)

                                        # flops_dynamic += flops / 1e6
                                        # params_dynamic += params / 1e6
                                        # Store the flop and param for each choices
                                        flops_list.append(flops / 1e6)
                                        param_list.append(params / 1e6)
                                

                    max_param_arch_params += param_list[np.argmax(param_list)] # / 1e6
                    max_param_arch_flops  += flops_list[np.argmax(param_list)] # / 1e6
                    min_param_arch_params += param_list[np.argmin(param_list)] # / 1e6
                    min_param_arch_flops  += flops_list[np.argmin(param_list)] # / 1e6
                    
            # Compute Head FLOPs
            print('Computing head FLOPs...')
            detect_input_dims = [detect_input_dims[1], detect_input_dims[0], detect_input_dims[-1]] # 256, 512, 1024
            print('detect_input_dims', detect_input_dims)
            flops, params, next_spatial_dimension = get_model_complexity_info(
                model.yolo_detector, 
                detect_input_dims, 
                as_strings=False, 
                print_per_layer_stat=False
            )
            self.params_fixed += params / 1e6
            self.flops_fixed += flops / 1e6
            self.flops_dict['flops_fixed'] = self.flops_fixed
            self.params_dict['params_fixed'] = self.params_fixed
            self.save_flops_dict()
            self.save_params_dict()
            
            max_param_arch_flops  += self.flops_fixed
            max_param_arch_params += self.params_fixed
            min_param_arch_flops  += self.flops_fixed
            min_param_arch_params += self.params_fixed
            print(f'Largest  SubNet  Params: {max_param_arch_params:8.4f}  |  FLOPS: {max_param_arch_flops/1e3:8.4f}G ')
            print(f'Smallest SubNet  Params: {min_param_arch_params:8.4f}  |  FLOPS: {min_param_arch_flops/1e3:8.4f}G ')
            print(f'SuperNet GFLOPS = Fixed + Choices = {self.flops_fixed/1e3:8.4f} + {flops_dynamic/1e3:8.4f} = {(self.flops_fixed+flops_dynamic)/1e3:8.4f}')
            print(f'SuperNet MParam = Fixed + Choices = {self.params_fixed:8.4f} + {params_dynamic:8.4f} = {self.params_fixed+params_dynamic:8.4f}')
            
            # conv_last
            # flops, params, next_spatial_dimension = get_model_complexity_info(model.global_pool, tuple(
            #     input.shape[1:]), as_strings=False, print_per_layer_stat=False)
            # self.params_fixed += params / 1e6
            # self.flops_fixed += flops / 1e6

            # input = model.global_pool(input)

            # globalpool
            # flops, params = get_model_complexity_info(model.conv_head, tuple(
            #     input.shape[1:]), as_strings=False, print_per_layer_stat=False)
            # self.params_fixed += params / 1e6
            # self.flops_fixed += flops / 1e6

    # return params (M)
    def get_params(self, arch):
        params = 0
        for block_id, block in enumerate(arch):
            for module_id, choice in enumerate(block):
                if choice == -1:
                    continue
                params += self.params_dict[block_id][module_id][choice]
        return params + self.params_fixed

    # return flops (M)
    def get_flops(self, arch):
        # print('flops_table:', self.flops_dict)
        flops = 0
        for block_id, block in enumerate(arch):
            for module_id, choice in enumerate(block):
                # print('block:', block_id)
                # print('module:', module_id)
                # print('choice:', choice)
                flops += self.flops_dict[block_id][module_id][choice]
        return flops + self.flops_fixed

    def save_flops_dict(self):
        with open('flops_dict.json', 'w') as f:
            json.dump(self.flops_dict, f)

    def load_flops_dict(self):
        with open('flops_dict.json') as f:
            self.flops_dict = json.load(f)

    def save_params_dict(self):
        with open('params_dict.json', 'w') as f:
            json.dump(self.params_dict, f)

    def load_params_dict(self):
        with open('params_dict.json') as f:
            self.params_dict = json.load(f)