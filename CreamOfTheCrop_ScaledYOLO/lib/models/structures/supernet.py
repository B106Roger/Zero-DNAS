# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

from numpy.core.shape_base import block
from lib.utils.builder_util import *
from lib.utils.search_structure_supernet import *
from lib.models.builders.build_supernet import *
import numpy as np
from lib.utils.op_by_layer_dict import flops_op_dict
from lib.models.blocks.yolo_blocks import Detect, BottleneckCSP, BottleneckCSP2, C3, get_algorithm_type
# from mish_cuda import MishCuda as Mish
from torch import optim
import torch
from torch.nn import ModuleList
from timm.models.efficientnet_builder import resolve_bn_args, round_channels
from timm.models.efficientnet_blocks import create_conv2d
from lib.utils.wot_hooks import counting_forward_hook, counting_backward_hook


from timm.models.layers import SelectAdaptivePool2d
from timm.models.layers.activations import hard_sigmoid
from lib.utils.kd_utils import FeatureAdaptation
from lib.utils.synflow import synflow, sum_arr, sum_arr_tensor

class Theta(nn.Module):
    def __init__(self, parameters):
        super(Theta, self).__init__()
        self.thetas = nn.Parameter(parameters)

    def forward(self):
        return self.thetas
# Supernet Structures
class SuperNet(nn.Module):

    def __init__(
            self,
            block_args,
            choices,
            num_classes=1000,
            in_chans=3,
            stem_size=32,
            num_features=1280,
            head_bias=True,
            channel_multiplier=1.0,
            pad_type='',
            act_layer=nn.ReLU,
            drop_rate=0.,
            drop_path_rate=0.,
            slice=4,
            se_kwargs=None,
            norm_layer=nn.BatchNorm2d,
            logger=None,
            norm_kwargs=None,
            global_pool='avg',
            resunit=False,
            dil_conv=False,
            verbose=False,
            init_temp=3):
        super(SuperNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        self.logger = logger
        # self.hetero_choices = [8, 4, 2] #from n_bottlenecks choices
        self.hetero_choices = choices['n_bottlenecks'] #[8, 6, 4, 2]
        # self.choices = calculate_choices(choices)
        self.choices = 1
        for choice_key, choice_list in choices.items():
            self.choices *= len(choice_list)
        self.block_to_choice_map = []
        self.ignore_stages = [0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 15, 17, 18, 19, 21, 22, 23, 25]
        self.temperature = init_temp
        # self._initialize_weights()
        self.map_choices_to_blocks(choices)
        self.thetas = self.initialize_thetas(block_args) # current thetas
        self.thetas_main = self.initialize_thetas(block_args) # for outer meta update
        self.thetas_pi = self.initialize_thetas(block_args) # for inner meta update
        self.thetas_pi_optimizer = optim.Adam(params=self.thetas_pi.parameters(),
                                       lr=0.1,
                                       weight_decay=5 * 1e-4)
        # self.thetas = self.initialize_biased_thetas(block_args)
        
            
        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(
            self._in_chs, stem_size, 3, stride=1, padding=pad_type)
        self.bn1 = nn.GroupNorm(1, stem_size)
        self.act1 = act_layer(inplace=True) if isinstance(act_layer, nn.ReLU) else act_layer()
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = SuperNetBuilder(
            choices,
            channel_multiplier,
            8,
            None,
            32,
            pad_type,
            act_layer,
            se_kwargs,
            norm_layer,
            norm_kwargs,
            drop_path_rate,
            verbose=verbose,
            resunit=resunit,
            dil_conv=dil_conv,
            logger=self.logger)
        self.blocks, self.save = builder(self._in_chs, block_args) # build middle stages
        self._in_chs = builder.in_chs

        self.K = 0
        # Head + Pooling
        # self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        # self.conv_head = create_conv2d(
        #     self._in_chs,
        #     self.num_features,
        #     1,
        #     padding=pad_type,
        #     bias=head_bias)
        # self.act2 = act_layer(inplace=True) if isinstance(act_layer, nn.ReLU) else act_layer()

        # Classifier
        # self.classifier = nn.Linear(
        #     self.num_features *
        #     self.global_pool.feat_mult(),
        #     self.num_classes)
        
        self.yolo_detector = Detect(
            nc=self.num_classes,
            anchors=(
                [12,16, 19,36, 40,28],  # P3/8
                [36,75, 76,55, 72,146],  # P4/16
                [142,110, 192,243, 459,401]),
            ch=(256, 512, 1024)
        )

        # Build strides, anchors
        random_cand = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0, 0, 0], [0], [0], [0, 0, 0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        # s = 256  # 2x min stride
        s = 256  # 2x min stride
        
        forward_once, _, _ = self.forward(torch.zeros(1, 3, s, s), random_cand, first_run=True)
        self.yolo_detector.stride = torch.tensor([s / x.shape[-2] for x in forward_once])  # forward
        self.yolo_detector.anchors /= self.yolo_detector.stride.view(-1, 1, 1)
        # check_anchor_order(m)
        self.stride = self.yolo_detector.stride
        
        # [Roger]
        algorithm_type = get_algorithm_type()
        if algorithm_type == 'ZeroCost':
            efficientnet_init_weights(self)
        elif algorithm_type == 'DNAS':
            self._initialize_weights()
        else:
            raise ValueError(f"Invalid algorithm type {algorithm_type}")

    def get_classifier(self):
        return self.classifier

    def initialize_thetas(self, block_args):
        thetas = nn.ModuleList()
        print('len(block_args) - len(self.ignore_stages)', len(block_args) - len(self.ignore_stages))
        for i in range(len(block_args) - len(self.ignore_stages)):
            thetas.extend([Theta(torch.Tensor([1.0 / self.choices for i in range(self.choices)]))])
        return thetas

    def update_pi(self):
        for m_from, m_to in zip(self.thetas.modules(), self.thetas_pi.modules()):
            if not isinstance(m_to, ModuleList) and not isinstance(m_from, ModuleList):
                m_to.thetas.data = m_from.thetas.data.clone()

        self.thetas = self.thetas_pi

    def point_grad_to(self, device):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.thetas_main.parameters(), self.thetas_pi.parameters()):
            if p.grad is None:
                p.grad = torch.Tensor(torch.zeros(p.size())).to(device)
                
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def update_main(self):
        self.thetas = self.thetas_main

    def initialize_biased_thetas(self, block_args):
        thetas = nn.ModuleList()
        for i in range(len(block_args) - len(self.ignore_stages)):
            thetas.extend([Theta(torch.Tensor([0.0 if i != 3 else 100.0 for i in range(self.choices)]))])
 
        return thetas

    def map_choices_to_blocks(self, choices):
        for gamma in choices['gamma']:
            for n_bottlenecks in choices['n_bottlenecks']:
                self.block_to_choice_map.append({'n_bottlenecks': n_bottlenecks, 'gamma': gamma})


    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(),
            num_classes) if self.num_classes else None

    def forward_features(self, x, architecture, first_run=False, calc_metric=False):
        # Pass data through stem
        # print('Initial shape:', x.shape)
        y = []
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        chosen_subnet = ''
        current_theta = 0
        # print('Shape after stem:', x.shape)
        # Pass data through chosen subnet
        block_id = 0
        for layer, layer_arch in zip(self.blocks, architecture):            
            # layer => <class 'torch.nn.modules.container.ModuleList'> 
            for blocks, arch in zip(layer, layer_arch):
                # blocks => <class 'torch.nn.modules.container.ModuleList'>
                if arch == -1:
                    continue

                if (arch <= 3):
                    chosen_block = blocks[0]
                elif (arch > 3 and arch <= 7):
                    chosen_block = blocks[1]
                    arch -= len(self.hetero_choices)
                elif (arch > 7):
                    chosen_block = blocks[2]
                    arch -= len(self.hetero_choices) * 2
                # if (arch <= 2):
                #     chosen_block = blocks[0]
                # elif (arch > 2):
                #     chosen_block = blocks[1]
                #     arch -= len(self.hetero_choices)

                chosen_block = blocks[0]
                # print(chosen_subnet)
                if chosen_block.block_args.get('from_concat') != None:
                    f = chosen_block.block_args.get('from_concat')
                    x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  
                                  
                if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                    if not first_run:
                        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                        current_operation = 0
                        operators_outputs = []
                        for op in blocks:
                            operators_outputs.append(op(x) * soft_mask_variables[current_operation])
                            current_operation += 1
  
                        x = sum(operators_outputs)
                        current_theta += 1
                    else:
                        n_bottlenecks = self.hetero_choices[arch]
                        x = chosen_block(x, n_bottlenecks)
                        
                        

                else:
                    x = chosen_block(x)
                   

                # print(f'Shape after {chosen_block.get_block_name()}:', x.shape)
                y.append(x if chosen_block.i in self.save else None)
                
                if chosen_block.block_args.get('last') != None:
                    x = [y[j] for j in [21, 25, 29]]
            
            block_id += 1
                # if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                #     block_name = chosen_block.get_block_name().split('_')[0]
                #     gamma = chosen_block.get_block_name().split('_')[-1]
                #     chosen_subnet += block_name + '_num' + str(n_bottlenecks) + '->' + gamma 
                # else:
                #     chosen_subnet += chosen_block.get_block_name() + '->'
        # Final processing

        # print('Shape before detection:', [output.shape for output in x])
        feature_out = []
        
        x = self.yolo_detector(x, first_run, calc_metric)
        # print('Final shape:', [out.shape for out in x])
        chosen_subnet = [None]
        return x, feature_out, chosen_subnet

    def forward_features_confinuous(self, x, distributions, first_run=False, calc_metric=False):
        """
        x: input image. torch.float32(b, c, h, w)
        distributions: architecture distributions parameter. torch.float32()
        """
        # Pass data through stem
        # print('Initial shape:', x.shape)
        if distributions is None:
            distributions = torch.cat([theta().reshape(1, -1) for theta in self.thetas], dim=0)
            distributions = distributions.detach()
            distributions = nn.functional.softmax(distributions, dim=-1)
        
        y = []
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        # print('[Roger] SuperNet.forward_features_confinuous self.act1', x.shape)
        # print('[Roger] SuperNet.forward_features_confinuous architecture', architecture)
        
        chosen_subnet = ''
        current_theta = 0
        # print('Shape after stem:', x.shape)    
        # Pass data through chosen subnet
        block_id = 0
        for layer in self.blocks:           
            num_of_choice_blocks = len(layer)
            # layer => <class 'torch.nn.modules.container. ModuleList'> 
            # print(f'[Roger] SuperNet.forward_features_confinuous block_id: {block_id:02d} choices: {num_of_choice_blocks}')
            # print(f'[Roger] \tlayer_arch: {[item for item in layer_arch]}')
            # print(f'[Roger] \tlayer type: {type(layer)}')
            
            for blocks in layer:
                # blocks => <class 'torch.nn.modules.container.ModuleList'>
                # print(f'[Roger] \t\tblocks type: {type(blocks)}')

                chosen_block = blocks[0]
                # print(chosen_subnet)
                if chosen_block.block_args.get('from_concat') != None:
                    f = chosen_block.block_args.get('from_concat')
                    x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  
                                  
                if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                    if not first_run:
                        # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                        soft_mask_variables = distributions[current_theta]
                        # print(f'{current_theta:2d} candidate layer: {soft_mask_variables.detach().cpu().numpy()}')
                        current_operation = 0
                        operators_outputs = []
                        for op in blocks:
                            operators_outputs.append(op(x) * soft_mask_variables[current_operation])
                            current_operation += 1
                        x = sum(operators_outputs)
                    else:
                        n_bottlenecks = self.hetero_choices[-1]
                        x = chosen_block(x, n_bottlenecks)
                    
                    current_theta += 1
                else:
                    x = chosen_block(x)
                   
                # print(f'Shape after {chosen_block.get_block_name()}:', x.shape)
                y.append(x if chosen_block.i in self.save else None)
                
                if chosen_block.block_args.get('last') != None:
                    x = [y[j] for j in [21, 25, 29]]
                    # print('trigger last')
                    # for xi, xx in enumerate(x):
                    #     print(f'{xi}-th feature map : {xx.shape}')
            block_id += 1

        feature_out = []
        # print('pass to yolo-detector')
        x = self.yolo_detector(x, first_run, calc_metric)
        # print('Final shape:', [out.shape for out in x])
        chosen_subnet = [None]
        return x, feature_out, chosen_subnet


    def forward(self, x, architecture=None, first_run=False, calc_metric=False):
        x, feature_out, chosen_subnet = self.forward_features_confinuous(x, architecture, first_run=first_run, calc_metric=calc_metric)
        # x, feature_out, chosen_subnet = self.forward_features_confinuous(x, architecture, first_run=first_run, calc_metric=calc_metric)
        # x = x.flatten(1)
        # if self.drop_rate > 0.:
        #     x = F.dropout(x, p=self.drop_rate, training=self.training)
        # return self.classifier(x), chosen_subnet[:-2]
        return x, feature_out, chosen_subnet[:-2]

    def calculate_flops(self, architecture, flops_dict, overall_flops, first_run=False, calc_metric=False):
        current_theta = 0
        block_id = 0
        for layer, layer_arch in zip(self.blocks, architecture):            
            for blocks, arch in zip(layer, layer_arch):
                if arch == -1:
                    continue

                if (arch <= 3):
                    chosen_block = blocks[0]
                elif (arch > 3 and arch <= 7):
                    chosen_block = blocks[1]
                    arch -= len(self.hetero_choices)
                elif (arch > 7):
                    chosen_block = blocks[2]
                    arch -= len(self.hetero_choices) * 2

                chosen_block = blocks[0]
                                  
                if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                    soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)

                    operators_flops = 0
                    current_operation = 0
                    for operation in blocks:
                        operator_flops = flops_dict[str(block_id)]['0'][str(current_operation)]
                        operators_flops = operators_flops + (operator_flops * soft_mask_variables[current_operation])
                        current_operation += 1

                    
                    overall_flops = overall_flops + (operators_flops)
                    current_theta += 1
                else:
                    overall_flops = overall_flops + flops_dict[str(block_id)]['0']['0']
            block_id += 1
        overall_flops += flops_dict['flops_fixed']
        return overall_flops
    
    def calculate_parameters(self, architecture, params_dict, overall_params, first_run=False, calc_metric=False):
        current_theta = 0
        block_id = 0
        for layer, layer_arch in zip(self.blocks, architecture):            
            for blocks, arch in zip(layer, layer_arch):
                if arch == -1:
                    continue

                if (arch <= 3):
                    chosen_block = blocks[0]
                elif (arch > 3 and arch <= 7):
                    chosen_block = blocks[1]
                    arch -= len(self.hetero_choices)
                elif (arch > 7):
                    chosen_block = blocks[2]
                    arch -= len(self.hetero_choices) * 2

                chosen_block = blocks[0]
                                  
                if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                    soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)

                    operators_params = 0
                    current_operation = 0
                    for operation in blocks:
                        operator_params = params_dict[str(block_id)]['0'][str(current_operation)]
                        operators_params = operators_params + (operator_params * soft_mask_variables[current_operation])
                        current_operation += 1

                    
                    overall_params = overall_params + (operators_params)
                    current_theta += 1
                else:
                    overall_params = overall_params + params_dict[str(block_id)]['0']['0']
            
            block_id += 1
        overall_params = overall_params + params_dict['params_fixed']
        return overall_params

    def calculate_layers(self, architecture, overall_layers, first_run=False, calc_metric=False):
        current_theta = 0
        block_id = 0
        for layer, layer_arch in zip(self.blocks, architecture):            
            for blocks, arch in zip(layer, layer_arch):
                if arch == -1:
                    continue

                if (arch <= 3):
                    chosen_block = blocks[0]
                elif (arch > 3 and arch <= 7):
                    chosen_block = blocks[1]
                    arch -= len(self.hetero_choices)
                elif (arch > 7):
                    chosen_block = blocks[2]
                    arch -= len(self.hetero_choices) * 2

                chosen_block = blocks[0]
                                  
                if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                    soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)

                    operators_layers = 0
                    current_operation = 0
                    for operation in blocks:
                        operator_layers = operation.n
                        operators_layers = operators_layers + (operator_layers * soft_mask_variables[current_operation])
                        current_operation += 1

                    
                    overall_layers = overall_layers + (operators_layers)
                    current_theta += 1
            
            block_id += 1
        return overall_layers

    def _initialize_weights(self):
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.yolo_detector  # Detect() module
        cf=None
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            with torch.no_grad():
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # Initialize Backbone
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)

    def calculate_synflow(self, x, architecture, overall_synflow, first_run=False, calc_metric=False):
        y = []
        x = self.conv_stem(x)
        #Layers with CSP
        synflow_map = {
            0: {}, 
            1: {}, 
            2: {}, 
            3: {}, 
            4: {}, 
            5: {}, 
            6: {},
            7: {},
        }
        x = self.bn1(x)
        x = self.act1(x)
        chosen_subnet = ''
        current_theta = 0
        block_id = 0
        for layer, layer_arch in zip(self.blocks, architecture):            
            for blocks, arch in zip(layer, layer_arch):
                if arch == -1:
                    continue

                if (arch <= 3):
                    chosen_block = blocks[0]
                elif (arch > 3 and arch <= 7):
                    chosen_block = blocks[1]
                    arch -= len(self.hetero_choices)
                elif (arch > 7):
                    chosen_block = blocks[2]
                    arch -= len(self.hetero_choices) * 2

                chosen_block = blocks[0]
                if chosen_block.block_args.get('from_concat') != None:
                    f = chosen_block.block_args.get('from_concat')
                    x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  
                                  
                if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                    soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)

                    operators_overall_synflow = 0
                    current_operation = 0
                    number_of_layers = 0
                    for op in blocks:
                        operators_synflow = []
                        for layer in op.modules():
                            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                                operators_synflow.append(synflow(layer))
                                number_of_layers += 1
                        # operators_overall_synflow = operators_overall_synflow + sum_arr(operators_synflow) * soft_mask_variables[current_operation]
                        synflow_map[current_theta][current_operation] = sum_arr(operators_synflow)
                        current_operation += 1
                    
                    if operators_overall_synflow > 1e9:
                        operators_overall_synflow = operators_overall_synflow / 1e5

                    
                    overall_synflow = overall_synflow + (operators_overall_synflow)
                    x = chosen_block(x)
                    current_theta += 1
                else:
                    x = chosen_block(x)
                    

                y.append(x if chosen_block.i in self.save else None)
                
                if chosen_block.block_args.get('last') != None:
                    x = [y[j] for j in [21, 25, 29]]
            
            block_id += 1

        
        x = self.yolo_detector(x, first_run, calc_metric)
        print(synflow_map)
        exit()
        return overall_synflow

    def calculate_wot_train(self, overall_wot, architecture, wot_map, first_run=False, calc_metric=False):
        chosen_subnet = ''
        current_theta = 0
        block_id = 0
        for layer, layer_arch in zip(self.blocks, architecture):            
            for blocks, arch in zip(layer, layer_arch):
                if arch == -1:
                    continue

                if (arch <= 3):
                    chosen_block = blocks[0]
                elif (arch > 3 and arch <= 7):
                    chosen_block = blocks[1]
                    arch -= len(self.hetero_choices)
                elif (arch > 7):
                    chosen_block = blocks[2]
                    arch -= len(self.hetero_choices) * 2

                chosen_block = blocks[0] 
                                  
                if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                    soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    current_operation = 0
                    number_of_layers = 0
                    operators_wot = []
                    for op in blocks:
                        current_operator_wot = wot_map[current_theta][current_operation]
                        operators_wot.append(current_operator_wot * soft_mask_variables[current_operation])
                        current_operation += 1
                    
                    
                    overall_wot = overall_wot + sum(operators_wot)
                    current_theta += 1
               
            
            block_id += 1

        return overall_wot


    def calculate_wot(self, x, architecture, overall_wot=None, first_run=False, calc_metric=False):
        if overall_wot == None:
            overall_wot = 0
        y = []
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        wot_map = {
            0: {}, 
            1: {}, 
            2: {}, 
            3: {}, 
            4: {}, 
            5: {}, 
            6: {},
            7: {},
        }
        chosen_subnet = ''
        current_theta = 0
        block_id = 0
        for layer_id, (layer, layer_arch) in enumerate(zip(self.blocks, architecture)):            
            for blocks, arch in zip(layer, layer_arch):
                if arch == -1:
                    continue

                if (arch <= 3):
                    chosen_block = blocks[0]
                elif (arch > 3 and arch <= 7):
                    chosen_block = blocks[1]
                    arch -= len(self.hetero_choices)
                elif (arch > 7):
                    chosen_block = blocks[2]
                    arch -= len(self.hetero_choices) * 2

                chosen_block = blocks[0]
                if chosen_block.block_args.get('from_concat') != None:
                    f = chosen_block.block_args.get('from_concat')
                    x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  
                
                # is_choices = isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)
                # from_concat = str(chosen_block.block_args.get("from_concat"))
                # strides = str(chosen_block.block_args.get("stride"))
                # last = str(chosen_block.block_args.get("last"))
                # print(f'[calculate_wot] layer_id: {layer_id} layer_arch: {str(layer_arch):11s} arch: {arch} Type: {chosen_block.__class__.__name__:18s} from_concat: {from_concat:<10s} strides: {strides:<10s} last: {last:<10s}')
                # if chosen_block.__class__.__name__ == 'Concat':
                #     for a in x:
                #         print(a.shape)
                
                # if layer_id in [18, 22]:
                #     print(x.shape)
                
                if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                    soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)

                    operators_wot = []
                    operators_outputs = []
                    current_operation = 0
                    number_of_layers = 0
                    for op in blocks:
                        operators_outputs.append(op(x) * soft_mask_variables[current_operation])
                        s, ld = np.linalg.slogdet(self.K)
                        operators_wot.append(ld * soft_mask_variables[current_operation])
                        wot_map[current_theta][current_operation] = ld
                        current_operation += 1

                        self.K = np.full_like(self.K, 0)
                    
                    
                    overall_wot = overall_wot + sum(operators_wot)
                    x = sum(operators_outputs)
                    current_theta += 1
                else:
                    x = chosen_block(x)
                    

                y.append(x if chosen_block.i in self.save else None)
                
                if chosen_block.block_args.get('last') != None:
                    x = [y[j] for j in [21, 25, 29]]
            
            block_id += 1


        x = self.yolo_detector(x, first_run, calc_metric)
        return wot_map
    
    def calculate_beta_reg(self):
        losses = 0
        total_stages = len(self.thetas)
        for i in range(total_stages):
            losses += torch.logsumexp(torch.as_tensor(self.thetas[i]()), 0)
        losses /= total_stages
        return losses
    
    def forward_meta(self, features):
        return self.meta_layer(features.view(1, -1))

    def rand_parameters(self, architecture, meta=False):
        for name, param in self.named_parameters(recurse=True):
            if 'meta' in name and meta:
                yield param
            elif 'blocks' not in name and 'meta' not in name and (not meta):
                yield param

        if not meta:
            for layer, layer_arch in zip(self.blocks, architecture):
                for blocks, arch in zip(layer, layer_arch):
                    if arch == -1:
                        continue
                    for name, param in blocks[arch].named_parameters(
                            recurse=True):
                        yield param

    @torch.no_grad()
    def linearize(self):
        signs = {}
        for name, param in self.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])


    def calculate_synflow_metric(self, architecture):
        metric_array = []
        for layer, layer_arch in zip(self.blocks, architecture):
            for blocks, arch in zip(layer, layer_arch):
                if arch == -1:
                    continue

                # if (arch <= 2):
                #     chosen_block = blocks[0]
                # elif (arch > 2):
                #     chosen_block = blocks[1]
                #     arch -= len(self.hetero_choices)
                
                if (arch <= 3):
                    chosen_block = blocks[0]
                elif (arch > 3 and arch <= 7):
                    chosen_block = blocks[1]
                    arch -= len(self.hetero_choices)
                elif (arch > 7):
                    chosen_block = blocks[2]
                    arch -= len(self.hetero_choices) * 2
                

            
                for layer in chosen_block.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        metric_array.append(synflow(layer))
                # print(sum_arr(metric_array))
        return sum_arr(metric_array)

    def calculate_full_synflow(self):
        metric_array = []
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                metric_array.append(synflow(layer))
        
        return sum_arr_tensor(metric_array)


    #########################################################################
    ######################## WeiJie Implementation ##########################
    def calculate_flops_new(self, architecture_info, flops_dict):
        architecture = architecture_info['arch']
        architecture_type = architecture_info['arch_type']
        
        current_theta = 0
        block_id = 0
        overall_flops = 0
        for layer_idx, layer in enumerate(self.blocks):            
            for block_idx, blocks in enumerate(layer):
                
                # [Discrete Mode] use architecture to sample subnetwork to do inference
                if architecture_type == 'discrete':
                    arch=architecture[layer_idx][block_idx]
                    if arch == -1: 
                        continue
                    else:
                        chosen_block_idx = arch // 4
                        n_bottlenecks = arch % 4
                    
                    # Determine which block to use in nn.MoudleList
                    chosen_block = blocks[0]
                    # [Block Inference]
                    if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                        overall_flops = overall_flops + flops_dict[str(block_id)]['0'][str(arch)]
                    else:
                        overall_flops = overall_flops + flops_dict[str(block_id)]['0']['0']
                        
                # [Continuous Mode] use architecture distribtuion and weighted-sum their output to do inference
                elif architecture_type == 'continuous':
                    chosen_block = blocks[0]
                    if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                        # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                        soft_mask_variables = architecture[current_theta]

                        operators_flops = 0
                        current_operation = 0
                        for operation in blocks:
                            operator_flops = flops_dict[str(block_id)]['0'][str(current_operation)]
                            operators_flops = operators_flops + (operator_flops * soft_mask_variables[current_operation])
                            current_operation += 1

                        overall_flops = overall_flops + (operators_flops)
                        current_theta += 1
                    else:
                        overall_flops = overall_flops + flops_dict[str(block_id)]['0']['0']
            
            block_id += 1
        overall_flops += flops_dict['flops_fixed']
        return overall_flops
    
    def calculate_params_new(self, architecture, params_dict):
        architecture = architecture_info['arch']
        architecture_type = architecture_info['arch_type']
        
        current_theta = 0
        block_id = 0
        overall_params = 0
        for layer_idx, layer in enumerate(self.blocks):            
            for block_idx, blocks in enumerate(layer):
                
                # [Discrete Mode] use architecture to sample subnetwork to do inference
                if architecture_type == 'discrete':
                    arch=architecture[layer_idx][block_idx]
                    if arch == -1: 
                        continue
                    else:
                        chosen_block_idx = arch // 4
                        n_bottlenecks = arch % 4
                    
                    # Determine which block to use in nn.MoudleList
                    chosen_block = blocks[0]
                    # [Block Inference]
                    if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                        overall_params = overall_params + params_dict[str(block_id)]['0'][str(arch)]
                    else:
                        overall_params = overall_params + params_dict[str(block_id)]['0']['0']
                        
                # [Continuous Mode] use architecture distribtuion and weighted-sum their output to do inference
                elif architecture_type == 'continuous':
                    chosen_block = blocks[0]
                    if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                        # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                        soft_mask_variables = architecture[current_theta]

                        operators_params = 0
                        current_operation = 0
                        for operation in blocks:
                            operator_params = params_dict[str(block_id)]['0'][str(current_operation)]
                            operators_params = operators_params + (operator_params * soft_mask_variables[current_operation])
                            current_operation += 1

                        overall_params = overall_params + (operators_params)
                        current_theta += 1
                    else:
                        overall_params = overall_params + params_dict[str(block_id)]['0']['0']
            
            block_id += 1
        overall_params += params_dict['params_fixed']
        return overall_params

    def calculate_layers_new(self, architecture):
        architecture = architecture_info['arch']
        architecture_type = architecture_info['arch_type']
                
        current_theta = 0
        block_id = 0
        overall_layers = 0
        for layer, layer_arch in zip(self.blocks, architecture):            
            for blocks, arch in zip(layer, layer_arch):
                # [Discrete Mode] use architecture to sample subnetwork to do inference
                if architecture_type == 'discrete':
                    arch=architecture[layer_idx][block_idx]
                    if arch == -1: 
                        continue
                    else:
                        chosen_block_idx = arch // 4
                        n_bottlenecks = arch % 4
                    
                    # Determine which block to use in nn.MoudleList
                    chosen_block = blocks[0]
                    operator_layers = chosen_block.n
                    # [Block Inference]
                    if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                        overall_layers = overall_layers + operator_layers
                
                # [Continuous Mode] use architecture distribtuion and weighted-sum their output to do inference
                elif architecture_type == 'continuous':
                    chosen_block = blocks[0]
                    if (isinstance(chosen_block, BottleneckCSP) or isinstance(chosen_block, BottleneckCSP2)):
                        # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                        soft_mask_variables = architecture[current_theta]

                        operators_layers = 0
                        current_operation = 0
                        for operation in blocks:
                            operator_layers = operation.n
                            operators_layers = operators_layers + (operator_layers * soft_mask_variables[current_operation])
                            current_operation += 1

                        overall_layers = overall_layers + (operators_layers)
                        current_theta += 1
                        
            block_id += 1
        return overall_layers


class Classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        return self.classifier(x)

def calculate_choices(choices):
    n_choices = 0
    for key in choices:
        for choice in choices[key]:
            print(choice)
            n_choices += 1
    return n_choices

def gen_supernet(flops_minimum=0, flops_maximum=600, choices=None, **kwargs):
    print('choices: ', choices)
    if choices is None:
        print('choices is none !! please use correct choices')
    # choices = {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]} # big
    # choices = {'n_bottlenecks': [0, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]} # tiny


    #1 4 0 0
    # act_layer = HardSwish
    act_layer = nn.ReLU

    arch_def = [
        # 
        # Backbone
        # stage 0
        ['cn_r1_k3_s2_cin32_cout64'],
        # stage 1
        ['bottle_r1_k1_s1_cin64_cout64'], # BottleNeck [64]
        # stage 2
        ['cn_r1_k3_s2_cin64_cout128'],
        # stage 3
        ['bottlecsp_r1_k1_s1_num2_gamma0.5_cin128_cout128'],  # BottleNeck [128]
        # stage 4
        ['cn_r1_k3_s2_cin128_cout256'],
        # stage 5
        ['bottlecsp_r1_k1_s1_num8_gamma0.5_cin256_cout256'], #BottleNeck [256]
        # stage 6
        ['cn_r1_k3_s2_cin256_cout512'],
        # stage 7
        ['bottlecsp_r1_k1_s1_num8_gamma0.5_cin512_cout512'], # BottleNeckCSP [512]
        # stage 8
        ['cn_r1_k3_s2_cin512_cout1024'],
        # stage 9
        ['bottlecsp_r1_k1_s1_num8_gamma0.5_cin1024_cout1024'], # BottleneckCSP [1024]
        # stage 10
        ['sppcsp_r1_k1_s1_cin1024_cout512'],
        # stage 11
        ['cn_np_r1_k1_s1_cin512_cout256', 'up_r1_mnearest_sf2_c256', 'cn_r1_k1_s1_cin512_cout256_f8'],
        # stage 12
        ['concat_r1_c512_f-2'],
        # stage 13
        ['bottlecsp2_r1_k1_s1_num2_gamma0.5_cin512_cout256'], # should be csp2
        # stage 14
        ['cn_np_r1_k1_s1_cin256_cout128', 'up_r1_mnearest_sf2_c128', 'cn_r1_k1_s1_cin256_cout128_f6'],
        # stage 15
        ['concat_r1_c256_f-2'],
        # stage 16
        ['bottlecsp2_r1_k1_s1_num2_gamma0.5_cin256_cout128'],
        # stage 17
        ['cn_r1_k3_s1_cin128_cout256'],
        # stage 18
        ['cn_r1_k3_s2_cin128_cout256_f-2'],
        # stage 19
        ['concat_r1_c512_f16'],
        # stage 20
        ['bottlecsp2_r1_k1_s1_num2_gamma0.5_cin512_cout256'], #should be csp2
        # stage 21
        ['cn_r1_k3_s1_cin256_cout512'],
        # stage 22
        ['cn_r1_k3_s2_cin256_cout512_f-2'],
        # stage 23
        ['concat_r1_c1024_f11'],
        # stage 24
        ['bottlecsp2_r1_k1_s1_num2_gamma0.5_cin1024_cout512'], #should be csp2
        # stage 25
        ['cn_r1_k3_s1_cin512_cout1024_last'],
    ]


    sta_num, arch_def, resolution = search_for_layer(flops_op_dict, arch_def, flops_minimum, flops_maximum)
    if sta_num is None or arch_def is None or resolution is None:
        raise ValueError('Invalid FLOPs Settings')
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        choices=choices,
        stem_size=32,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(
            act_layer=nn.ReLU,
            gate_fn=hard_sigmoid,
            reduce_mid=True,
            divisor=8),
        **kwargs,
    )
    model = SuperNet(**model_kwargs)
    print('save:', model.save)
    return model, sta_num, resolution
