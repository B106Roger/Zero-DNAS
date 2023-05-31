# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import torch
from pathlib import Path
import json

from lib.utils.flops_counter import get_model_complexity_info
from lib.models.blocks.yolo_blocks import get_algorithm_type
from lib.models.blocks.yolo_blocks import BottleneckCSP, BottleneckCSP2
from copy import deepcopy
import numpy as np


class FlopsEst(object):
    def __init__(self, model, input_shape=(2, 3, 416, 416), search_space=None):
        self.search_space = deepcopy(search_space)
        self.block_num = len(model.blocks)
        # self.choice_num = len(model.blocks[0])
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
            spatial_dimension = input_shape[2:]
            # the largest params, its corresponding flops and params
            max_param_arch_flops, max_param_arch_params = 0, 0
            # the smallest params, its corresponding flops and params
            min_param_arch_flops, min_param_arch_params = 0, 0

            detect_input_idx  = model.blocks[-1].block_arguments['f']
            detect_input_dims = []

            # Compute Blocks FLOPs
            print('Computing blocks FLOPs...')
            for block_id, block in enumerate(model.blocks):
                block_id = str(block_id)
                
                in_chs  = block.block_arguments['in_chs']
                out_chs = block.block_arguments['out_chs']
                
                self.flops_dict[block_id] = {}
                self.params_dict[block_id] = {}
                
                flops_list = []
                param_list = []
                if 'Search' in block.__class__.__name__:
                    if 'Composite_Search' in block.__class__.__name__:
                        pass
                    else:
                        input_shape = (in_chs, spatial_dimension[0], spatial_dimension[1])
                        ###########################################################################
                        # Calculate SuperNet Real GFLOPS and Params
                        ###########################################################################
                        # Params raw value
                        macs, params, next_spatial_dimension = get_model_complexity_info(
                            block, 
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
                        ###########################################################################
                        
                        
                        ###########################################################################
                        # Calculate SuperNet Real GFLOPS and Params
                        ###########################################################################
                        # options, key2idx, search_keys = block.generate_options()
                        options, search_keys = block.generate_options()
                        
                        print(options, search_keys)
                        for option in options:
                            args = dict(
                                c1=in_chs,
                                c2=out_chs,
                                n=block.search_space['n_bottlenecks'][option[search_keys.index('n_bottlenecks')]],
                                e=block.search_space['gamma'     ][option[search_keys.index('gamma')]],
                                # n=option[key2idx['bottleneck']],
                                # e=option[key2idx['gamma']],
                            )
                            classname = block.__class__.__name__.replace('_Search', '')
                            ChoiceClass = eval(classname)
                            choice_block = ChoiceClass(**args)
                            
                            
                            macs, params, next_spatial_dimension = get_model_complexity_info(
                                choice_block, 
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
                            
                            query_key = '-'.join([f'{key}{block.search_space[key][option[key_idx]]}' for key_idx, key in enumerate(search_keys)])
                            self.flops_dict[block_id][query_key]  = flops / 1e6       # FLOPS(M)
                            self.params_dict[block_id][query_key] = params / 1e6      # Params(M)
                            
                            # Store the flop and param for each choices
                            flops_list.append(flops / 1e6)
                            param_list.append(params / 1e6)
                        ###########################################################################
                else:
                    if block.get_block_name() == 'concat':
                        out_chs = block.block_arguments['out_chs']
                        flops, params = 0, 0
                    elif 'up' in block.get_block_name():
                        # spatial_dimension = (
                        #     spatial_dimension[0] * block.block_arguments['scale_factor'],
                        #     spatial_dimension[1] * block.block_arguments['scale_factor']
                        # )
                        in_chs = block.block_arguments['in_chs']
                        input_shape = (in_chs, spatial_dimension[0], spatial_dimension[1])
                        
                        # Params raw value
                        macs, params, next_spatial_dimension = get_model_complexity_info(
                            block, 
                            input_res=input_shape, 
                            as_strings=False, 
                            print_per_layer_stat=False
                        )
                        params = 0
                        # FLOPS raw value
                        algorithm_type = get_algorithm_type()
                        if algorithm_type == 'ZeroDNAS_Egor':
                            flops = macs
                        # This is the correct flop calculation
                        elif algorithm_type == 'DNAS' or algorithm_type == 'ZeroCost':
                            flops = macs * 2
                        else:
                            raise ValueError(f"Invalid algorithm type {algorithm_type}")
                        # print(f'[Up] Original FLOPS {flops} MFLOPS => Approx FLOPS={0}')
                        # flops = 0.0
                        print(f'[Up] flops {spatial_dimension}', flops / 1e6, 'M', '-'*40)
                    else:
                        in_chs  = block.block_arguments['in_chs']
                        out_chs  = block.block_arguments['out_chs']
                        
                        if 'Detect' in block.__class__.__name__:
                            input_shape = detect_input_dims
                            print(detect_input_dims)
                        else:
                            input_shape = (in_chs, spatial_dimension[0], spatial_dimension[1])
                        
                        ###########################################################################
                        # Calculate SuperNet Real GFLOPS and Params
                        ###########################################################################
                        # Params raw value
                        macs, params, next_spatial_dimension = get_model_complexity_info(
                            block, 
                            input_res=input_shape, 
                            as_strings=False, 
                            print_per_layer_stat=False
                        )
                        # FLOPS raw value
                        algorithm_type = get_algorithm_type()
                        if algorithm_type == 'ZeroDNAS_Egor':
                            flops = macs
                        # This is the correct flop calculation
                        elif algorithm_type == 'DNAS' or algorithm_type == 'ZeroCost':
                            flops = macs * 2
                        else:
                            raise ValueError(f"Invalid algorithm type {algorithm_type}")
                    
                                        
                    # Store the flop and param for each choices
                    self.flops_fixed  += flops / 1e6
                    self.params_fixed += params / 1e6
                    
                    self.flops_dict[block_id]['0']  = flops  / 1e6  # FLOPS(M)
                    self.params_dict[block_id]['0'] = params / 1e6  # Params(M)
                    
                    # Store the flop and param for each choices
                    flops_list.append(flops / 1e6)
                    param_list.append(params / 1e6)
                    
                    
                if int(block_id) in detect_input_idx:
                    detect_input_dims.append((1, out_chs, next_spatial_dimension[0], next_spatial_dimension[1]))
                
                
                print(f'{block_id} {block.get_block_name()} in : {spatial_dimension} out : {next_spatial_dimension}')
                spatial_dimension = next_spatial_dimension    
                max_param_arch_params += param_list[np.argmax(param_list)] # / 1e6
                max_param_arch_flops  += flops_list[np.argmax(flops_list)] # / 1e6
                min_param_arch_params += param_list[np.argmin(param_list)] # / 1e6
                min_param_arch_flops  += flops_list[np.argmin(flops_list)] # / 1e6
                

            self.flops_dict['flops_fixed'] = self.flops_fixed
            self.params_dict['params_fixed'] = self.params_fixed
            self.save_flops_dict()
            self.save_params_dict()
            
            # max_param_arch_flops  += self.flops_fixed
            # max_param_arch_params += self.params_fixed
            # min_param_arch_flops  += self.flops_fixed
            # min_param_arch_params += self.params_fixed
            print(f'Largest  SubNet  Params: {max_param_arch_params:8.4f}  |  FLOPS: {max_param_arch_flops/1e3:8.4f}G ')
            print(f'Smallest SubNet  Params: {min_param_arch_params:8.4f}  |  FLOPS: {min_param_arch_flops/1e3:8.4f}G ')
            print(f'SuperNet GFLOPS = Fixed + Choices = {self.flops_fixed/1e3:8.4f} + {flops_dynamic/1e3:8.4f} = {(self.flops_fixed+flops_dynamic)/1e3:8.4f}')
            print(f'SuperNet MParam = Fixed + Choices = {   self.params_fixed:8.4f} + {   params_dynamic:8.4f} = {    self.params_fixed+params_dynamic:8.4f}')
            
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