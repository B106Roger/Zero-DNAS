# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import torch
from pathlib import Path
import os
import json

from lib.utils.flops_counter import get_model_complexity_info
from lib.models.blocks.yolo_blocks import *
from lib.models.blocks.yolo_blocks_search import BottleneckCSP_Search, BottleneckCSP2_Search,\
    ELAN_Search, ELAN2_Search
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import numpy as np

def mac_flop_converter(macs, algorithm_type):
    if algorithm_type == 'ZeroDNAS_Egor':
        # legacy code use wrong flops 
        flops = macs
    elif algorithm_type == 'DNAS' or algorithm_type == 'ZeroCost':
        # This is the correct case.
        flops = macs * 2
    else:
        raise ValueError(f"Invalid algorithm type {algorithm_type}")
    return flops

class FlopsEst(object):
    def __init__(self, model, input_shape=(2, 3, 416, 416), search_space=None, signature=None):
        self.search_space = deepcopy(search_space)
        self.block_num = len(model.blocks)
        # self.choice_num = len(model.blocks[0])
        self.flops_dict = {}
        self.params_dict = {}
        
        flops_filename  = os.path.basename(signature) +'_flops.json'
        params_filename = os.path.basename(signature) +'_params.json'

        model = model.cpu()

        self.params_fixed = 0
        self.flops_fixed = 0
        
        if Path(flops_filename).exists() and Path(params_filename).exists():
            self.load_flops_dict(flops_filename)
            self.load_params_dict(params_filename)
        else:
            flops_dynamic = 0
            params_dynamic = 0
            spatial_dimension = input_shape[2:]
            # the largest params, its corresponding flops and params
            largest_arch_flops, largest_arch_params = 0, 0
            # the smallest params, its corresponding flops and params
            smallest_arch_flops, smallest_arch_params = 0, 0

            dimension_list    = [spatial_dimension]
            channel_list      = [3]
            detect_input_idx  = model.blocks[-1].block_arguments['f']
            detect_input_dims = []

            # Compute Blocks FLOPs
            print('Computing blocks FLOPs...')
            for block_id, block in enumerate(model.blocks):
                block_id = str(block_id)
                
                in_chs  = block.block_arguments['in_chs']
                out_chs = block.block_arguments['out_chs']
                f       = block.block_arguments['f']
                
                if isinstance(f, list): # Case of Detect, IDetect
                    dimension_factor = 1
                    s_factor = dimension_factor**2
                    
                    input_shape = [
                        (1,
                         channel_list[f_i], 
                         dimension_list[f_i][0] // dimension_factor, 
                         dimension_list[f_i][1] // dimension_factor
                        ) for f_i in f]
                else:
                    dimension_factor = 1
                    tmp_dim = dimension_list[f][0] 
                    while tmp_dim % 4 == 0 and dimension_factor < 16:
                        tmp_dim = tmp_dim // 2
                        dimension_factor *= 2
                    s_factor = dimension_factor**2
                    
                    input_shape = (
                        in_chs, 
                        dimension_list[f][0] // dimension_factor, 
                        dimension_list[f][1] // dimension_factor)
                
                self.flops_dict[block_id] = {}
                self.params_dict[block_id] = {}
                
                flops_list = []
                param_list = []
                if 'Search' in block.__class__.__name__:
                    if 'Composite_Search' in block.__class__.__name__:
                        pass
                    else:
                        ###########################################################################
                        # Calculate SuperNet Real GFLOPS and Params
                        ###########################################################################
                        # Params raw value
                        macs, params, next_spatial_dimension = get_model_complexity_info(
                            block, 
                            input_res=input_shape, 
                            as_strings=False, 
                            print_per_layer_stat=False,
                        )
                        algorithm_type = get_algorithm_type()
                        flops  = mac_flop_converter(macs, algorithm_type) / 1e6 * s_factor    # (M)
                        params = params / 1e6                                                 # (M)
                        next_spatial_dimension = (next_spatial_dimension[0] * dimension_factor, next_spatial_dimension[1] * dimension_factor)
                        
                        dimension_list.append(next_spatial_dimension)
                        channel_list.append(out_chs)
                        
                        flops_dynamic += flops
                        params_dynamic += params
                        ###########################################################################
                        
                        
                        ###########################################################################
                        # Calculate SuperNet Real GFLOPS and Params
                        ###########################################################################
                        options, search_keys = block.generate_options()
                        
                        if block.__class__ in [BottleneckCSP_Search, BottleneckCSP2_Search]:
                            for option in options:
                                args = dict(
                                    c1=in_chs,
                                    c2=out_chs,
                                    n=block.search_space['n_bottlenecks'][option[search_keys.index('n_bottlenecks')]],
                                    e=block.search_space['gamma'     ][option[search_keys.index('gamma')]],
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
                                flops  = mac_flop_converter(macs, algorithm_type) / 1e6  * s_factor # (M)
                                params = params / 1e6                                    # (M)
                                next_spatial_dimension = (next_spatial_dimension[0] * dimension_factor, next_spatial_dimension[1] * dimension_factor)
                                
                                query_key = '-'.join([f'{key}{block.search_space[key][option[key_idx]]}' for key_idx, key in enumerate(search_keys)])
                                self.flops_dict[block_id][query_key]  = flops
                                self.params_dict[block_id][query_key] = params
                                
                                # Store the flop and param for each choices
                                flops_list.append(flops)
                                param_list.append(params)
                        
                        elif block.__class__ in [ELAN_Search, ELAN2_Search]:
                            comb_list = block._connection_combination(block.search_space['connection'])
                            pbar = tqdm(enumerate(options))
                            for op_idx, option in pbar:
                                args_cn         = int(block.search_space['gamma'     ][option[search_keys.index('gamma')]] * block.base_cn)
                                args_connection = comb_list[option[search_keys.index('connection')]]
                                args = dict(
                                    c1=in_chs,
                                    c2=out_chs,
                                    cn=args_cn,
                                    connection=args_connection,
                                    n= -np.min(args_connection)+2 if len(args_connection) > 0 else 2,
                                )
                                # print('[Roger] args', args)
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
                                flops  = mac_flop_converter(macs, algorithm_type) / 1e6  * s_factor # (M)
                                params = params / 1e6                                   # (M)
                                next_spatial_dimension = (next_spatial_dimension[0] * dimension_factor, next_spatial_dimension[1] * dimension_factor)

                                query_key = f'cn{args_cn}-con{str(args_connection)}'
                                self.flops_dict[block_id][query_key]  = flops
                                self.params_dict[block_id][query_key] = params
                                
                                # Store the flop and param for each choices
                                flops_list.append(flops)
                                param_list.append(params)
                                
                                pbar.set_description(f'largest block {option} FLOPS: {max(flops_list)/1e3}G Params: {max(param_list)}M')
                        ###########################################################################
                else:
                    if block.__class__ is Concat:
                        dimension_list.append(dimension_list[block.block_arguments['f'][0]])
                        channel_list.append(out_chs)
                        flops, params = 0, 0
                    elif block.__class__ in [nn.Upsample, Upsample]:
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
                        flops  = mac_flop_converter(macs, algorithm_type) / 1e6  * s_factor# (M)
                        params = params / 1e6                                   # (M)
                        next_spatial_dimension = (next_spatial_dimension[0] * dimension_factor, next_spatial_dimension[1] * dimension_factor)
                        
                        dimension_list.append(next_spatial_dimension)
                        channel_list.append(out_chs)
                    else:
                        # Params raw value
                        macs, params, next_spatial_dimension = get_model_complexity_info(
                            block, 
                            input_res=input_shape, 
                            as_strings=False, 
                            print_per_layer_stat=False
                        )
                        # FLOPS raw value
                        algorithm_type = get_algorithm_type()
                        flops  = mac_flop_converter(macs, algorithm_type) / 1e6  * s_factor# (M)
                        params = params / 1e6                                   # (M)
                        if next_spatial_dimension is not None:
                            next_spatial_dimension = (next_spatial_dimension[0] * dimension_factor, next_spatial_dimension[1] * dimension_factor)
                        dimension_list.append(next_spatial_dimension)
                        channel_list.append(out_chs)
                                        
                    # Store the flop and param for each choices
                    self.flops_fixed  += flops
                    self.params_fixed += params
                    
                    self.flops_dict[block_id]['0']  = flops
                    self.params_dict[block_id]['0'] = params
                    
                    # Store the flop and param for each choices
                    flops_list.append(flops)
                    param_list.append(params)
                    
                    
                if int(block_id) in detect_input_idx:
                    detect_input_dims.append((1, out_chs, next_spatial_dimension[0], next_spatial_dimension[1]))
                    
                if block_id == '0':
                    dimension_list = dimension_list[1:]
                    channel_list   = channel_list[1:]
                
                if next_spatial_dimension is not None:
                    in_dim  = (in_chs, )  + tuple(spatial_dimension)
                    out_dim = (out_chs, ) + tuple(next_spatial_dimension)
                
                if 'Search' not in block.__class__.__name__:
                    print(f'{block_id:2s} {block.__class__.__name__:20s} in : {str(in_dim):20s} out : {str(out_dim):20s}     FLOPS: {flops/1e3:5.2f}G  PARAMS: {params:.2f}M')
                else:
                    print(f'{block_id:2s} {block.__class__.__name__:20s} in : {str(in_dim):20s} out : {str(out_dim):20s} MAX FLOPS: {max(flops_list)/1e3:5.2f}G  PARAMS: {max(param_list):.2f}M')
                    print(f'{block_id:2s} {block.__class__.__name__:20s} in : {str(in_dim):20s} out : {str(out_dim):20s} MIN FLOPS: {min(flops_list)/1e3:5.2f}G  PARAMS: {min(param_list):.2f}M')
                    
                spatial_dimension = next_spatial_dimension    
                largest_arch_params += param_list[np.argmax(flops_list)]
                largest_arch_flops  += flops_list[np.argmax(flops_list)]
                
                smallest_arch_params += param_list[np.argmin(flops_list)]
                smallest_arch_flops  += flops_list[np.argmin(flops_list)]
                

            self.flops_dict['flops_fixed']      = self.flops_fixed
            self.flops_dict['flops_dynamic']    = flops_dynamic
            self.flops_dict['flops_supernet']   = self.flops_fixed + flops_dynamic
            self.flops_dict['largest_arch']     = largest_arch_flops
            self.flops_dict['smallest_arch']    = smallest_arch_flops
            self.flops_dict['model_config']     = signature
            
            self.params_dict['params_fixed']    = self.params_fixed
            self.params_dict['params_dynamic']  = params_dynamic
            self.params_dict['params_supernet'] = self.params_fixed + params_dynamic
            self.params_dict['largest_arch']     = largest_arch_params
            self.params_dict['smallest_arch']    = smallest_arch_params
            self.params_dict['model_config']    = signature
            
            self.save_flops_dict(flops_filename)
            self.save_params_dict(params_filename)
            
        print(f"Largest  SubNet  Params: { self.params_dict['largest_arch']:8.4f}  |  FLOPS: { self.flops_dict['largest_arch']/1e3:8.4f}G ")
        print(f"Smallest SubNet  Params: {self.params_dict['smallest_arch']:8.4f}  |  FLOPS: {self.flops_dict['smallest_arch']/1e3:8.4f}G ")
        print(f"SuperNet GFLOPS = Fixed + Choices = {self.flops_dict['flops_fixed']/1e3:8.4f} + {self.flops_dict['flops_dynamic']/1e3:8.4f} = {self.flops_dict['flops_supernet']/1e3:8.4f}")
        print(f"SuperNet MParam = Fixed + Choices = {  self.params_dict['params_fixed']:8.4f} + {  self.params_dict['params_dynamic']:8.4f} = {self.params_dict['params_supernet']:8.4f}")


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

    def save_flops_dict(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.flops_dict, f)

    def load_flops_dict(self, filename):
        with open(filename) as f:
            self.flops_dict = json.load(f)

    def save_params_dict(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.params_dict, f)

    def load_params_dict(self, filename):
        with open(filename) as f:
            self.params_dict = json.load(f)