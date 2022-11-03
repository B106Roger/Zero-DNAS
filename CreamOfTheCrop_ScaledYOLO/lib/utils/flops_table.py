# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import torch
from pathlib import Path
import json

from lib.utils.flops_counter import get_model_complexity_info


class FlopsEst(object):
    def __init__(self, model, input_shape=(2, 3, 416, 416)):
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
            print('Computing stem FLOPs...')
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
                    for choice_id, choice in enumerate(module):
                        choice_id = str(choice_id)
                        if choice.get_block_name() == 'concat':
                            out_chs = choice.block_arguments['out_chs']
                            detect_input_dims.append((1, out_chs, spatial_dimension[0], spatial_dimension[1]))
                            self.flops_dict[block_id][module_id][choice_id] = 0
                            # Params(M)
                            self.params_dict[block_id][module_id][choice_id] = 0
                            
                        elif 'up' in choice.get_block_name():
                            spatial_dimension = (
                                spatial_dimension[0] * choice.block_arguments['scale_factor'],
                                spatial_dimension[1] * choice.block_arguments['scale_factor']
                            )
                            self.flops_dict[block_id][module_id][choice_id] = 0
                            # Params(M)
                            self.params_dict[block_id][module_id][choice_id] = 0
                            
                        else:
                            in_chs = choice.block_arguments['in_chs']
                            input_shape = (in_chs, spatial_dimension[0], spatial_dimension[1])
                            flops, params, next_spatial_dimension = get_model_complexity_info(
                                choice, 
                                input_res=input_shape, 
                                as_strings=False, 
                                print_per_layer_stat=False
                            )
                            spatial_dimension = next_spatial_dimension
                            # Flops(M)
                            self.flops_dict[block_id][module_id][choice_id] = flops / 1e6
                            # Params(M)
                            self.params_dict[block_id][module_id][choice_id] = params / 1e6


            # Compute Head FLOPs
            print('Computing head FLOPs...')
            detect_input_dims = [detect_input_dims[1], detect_input_dims[0], detect_input_dims[-1]] # 256, 512, 1024
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