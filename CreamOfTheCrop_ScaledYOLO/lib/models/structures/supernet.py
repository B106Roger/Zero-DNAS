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
from lib.utils.general import check_anchor_order, random_testing
import math
import yaml
import random
class Theta(nn.Module):
    def __init__(self, parameters):
        super(Theta, self).__init__()
        self.thetas = nn.Parameter(parameters)

    def forward(self):
        return self.thetas

# Reference: https://github.com/yandexdataschool/gumbel_lstm/blob/master/gumbel_sigmoid.py
def gumbel_sigmoid(logits, temperature = 1.0):
    eps=1e-20
    
    #sample from Gumbel(0, 1)
    uniform1 = torch.rand(logits.shape)
    uniform2 = torch.rand(logits.shape)
    
    noise = -torch.log(torch.log(uniform2 + eps)/torch.log(uniform1 + eps) + eps)
    
    return torch.nn.functional.sigmoid((logits+noise) / temperature)

# Supernet Structures
class SuperNet(nn.Module):

    def __init__(
            self,
            model_args,
            choices,
            num_classes=1000,
            in_chans=3,
            stem_size=32,
            num_features=1280,
            head_bias=True,
            channel_multiplier=1.0,
            pad_type='',
            act_layer=nn.ReLU,
            # drop_rate=0.,
            # drop_path_rate=0.,
            # slice=4,
            se_kwargs=None,
            norm_layer=nn.BatchNorm2d,
            logger=None,
            norm_kwargs=None,
            device='cpu',
            # global_pool='avg',
            # resunit=False,
            # dil_conv=False,
            verbose=False,
            init_temp=3):
        super(SuperNet, self).__init__()
        block_args = model_args['backbone'] + model_args['head']
        self.searchable_block_idx = self.identify_searchable(block_args)
        self.num_classes = num_classes
        self.num_features = num_features
        # self.drop_rate = drop_rate
        self._in_chs = in_chans
        self.logger = logger
        # self.hetero_choices = [8, 4, 2] #from n_bottlenecks choices
        # self.hetero_choices = choices['n_bottlenecks'] #[8, 6, 4, 2]
        self.search_space = deepcopy(choices)
        # self.choices = calculate_choices(choices)
        # self.choices = 1
        # for choice_key, choice_list in choices.items():
        #     self.choices *= len(choice_list)
        # self.block_to_choice_map = []
        # self.ignore_stages = [0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 15, 17, 18, 19, 21, 22, 23, 25]
        self.temperature = init_temp
        self.model_args = model_args
        # self._initialize_weights()
        # self.map_choices_to_blocks(choices)
        # self.thetas = self.initialize_thetas(block_args) # current thetas
        # self.thetas_main = self.initialize_thetas(block_args) # for outer meta update
        # self.thetas_pi = self.initialize_thetas(block_args) # for inner meta update
        # self.thetas_pi_optimizer = optim.Adam(params=self.thetas_pi.parameters(),
        #                                lr=0.1,
        #                                weight_decay=5 * 1e-4)
        # self.thetas = self.initialize_biased_thetas(block_args)

        self.blocks, self.save = parse_model(model_args, ch=[in_chans])  # model, savelist, ch_out
        
        self.thetas_main = self.init_arch_parameter(device, self.temperature, 'uniform')
        print(self.thetas_main)
        
        # Build strides, anchors
        m = self.blocks[-1]  # Detect()
        if isinstance(m, Detect) or isinstance(m, IDetect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, in_chans, s, s))  ])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            # self._initialize_detector_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        
        # [Roger]
        DEBUG=False
        if not DEBUG:
            algorithm_type = get_algorithm_type()
            if algorithm_type == 'ZeroDNAS_Egor':
                efficientnet_init_weights(self)
            elif algorithm_type == 'DNAS' or algorithm_type =='ZeroCost':
                self._initialize_weights(True)
            else:
                raise ValueError(f"Invalid algorithm type {algorithm_type}")
        else:
            # self._initialize_efficientnet() # efficientnet_init_weights(self) &  self._initialize_detector_biases()
            self._initialize_weights(True)
            self._initialize_detector_biases()
            pass
            
    def initialize_thetas(self, block_args):
        thetas = nn.ModuleList()
        keys = sorted(self.search_space.keys())
        for i in range(len(block_args) - len(self.ignore_stages)):
            vals = nn.ModuleList()
            for key in keys:
                search_space_len = len(self.search_space[key])
                vals.extend([Theta(torch.Tensor([
                    1.0 / search_space_len for i in range(search_space_len)
                ]))])
            thetas.append(vals)
        return thetas

    def update_main(self):
        self.thetas = self.thetas_main

    def identify_searchable(self, block_args):
        searchable_block_id = []
        for block_id, block_arg in enumerate(block_args):
            if 'Search' in block_arg[2]:
                searchable_block_id.append(block_id) 
        return searchable_block_id
    
    def initialize_biased_thetas(self, block_args):
        thetas = nn.ModuleList()
        for i in range(len(block_args) - len(self.ignore_stages)):
            thetas.extend([Theta(torch.Tensor([0.0 if i != 3 else 100.0 for i in range(self.choices)]))])
 
        return thetas

    def forward_features_confinuous(self, x, distributions, first_run=False, calc_metric=False):
        """
        x: input image. torch.float32(b, c, h, w)
        distributions: architecture distributions parameter. torch.float32()
        """
        # Pass data through stem
        # print('Initial shape:', x.shape)
        SHOW_FEATURE_STATS = False
        if distributions is None:
            distributions = self.softmax_sampling(temperature=self.temperature, detach=True)
        
        stage_idx = 0
        y, dt = [], []  # outputs
        profile = False
        for idx, m in enumerate(self.blocks):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))


            if 'Search' in m.__class__.__name__:
                stage_args = distributions[stage_idx]
                x = m(x, stage_args)
                stage_idx+=1
                if SHOW_FEATURE_STATS:
                    tmp_feat = x.detach().cpu()
                    a,b,c = tmp_feat.min().numpy(), tmp_feat.max().numpy(), tmp_feat.mean().numpy()
                    d= tmp_feat.data[0,4,4,4].numpy()
                    print(f'{idx:02d} {m.__class__.__name__:30s} min {a:10.8e} max {b:10.8e} mean {c:10.8e} first ele: {d:10.8e}')
            else:
                x = m(x)  # run
                if SHOW_FEATURE_STATS:
                    if 'Detect' not in m.__class__.__name__:
                        tmp_feat = x.detach().cpu()
                        a,b,c = tmp_feat.min().numpy(), tmp_feat.max().numpy(), tmp_feat.mean().numpy()
                        d= tmp_feat.data[0,4,4,4].numpy()
                        print(f'{idx:02d} {m.__class__.__name__:30s} min {a:10.8e} max {b:10.8e} mean {c:10.8e} first ele: {d:10.8e}')          
                    else:
                        if len(x) == 2: res = x[1]
                        else: res=x
                        
                        for feat_idx, feat in enumerate(res):
                            tmp_feat = feat.detach().cpu()
                            a,b,c = tmp_feat.min().numpy(), tmp_feat.max().numpy(), tmp_feat.mean().numpy()
                            if len(feat.shape) == 5:
                                d= tmp_feat.sum()
                            else:
                                d= tmp_feat.sum()
                            print(f'{stage_idx+2:02d} {m.__class__.__name__:30s}[{feat_idx}] min {a:10.8e} max {b:10.8e} mean {c:10.8e} first ele: {d:10.8e}')                  
                            
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))

        chosen_subnet = [None]
        return x, y, chosen_subnet

    def forward(self, x, architecture=None, first_run=False, calc_metric=False):
        x, feature_out, chosen_subnet = self.forward_features_confinuous(x, architecture, first_run=first_run, calc_metric=calc_metric)

        
        return x #, feature_out, chosen_subnet[:-2]


    ################################################
    # Initialize Network Parameter
    ################################################
    def _initialize_detector_biases(self):
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.blocks[-1]  # Detect() module
        cf=None
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            with torch.no_grad():
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_weights(self, first=False):
        # # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.yolo_detector  # Detect() module
        # cf=None
        # for mi, s in zip(m.m, m.stride):  # from
        #     b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        #     with torch.no_grad():
        #         b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        #         b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
        #     mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # self._initialize_detector_biases()
        # Initialize Backbone
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                if first: continue
                # pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        torch.nn.init.uniform_(m.bias, -bound, bound)
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

    def _initialize_efficientnet(self):
        efficientnet_init_weights(self)
        self._initialize_detector_biases()

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

    ################################################
    # Sampling Method
    ################################################
    def init_arch_parameter(self, device, temperature, init_type):
        arch = []
        for block_id, block in enumerate(self.blocks):
            if 'Search' in block.__class__.__name__:
                arch.append(block.init_arch_parameter(device, temperature, init_type))
        return arch

    def get_optimizer_parameter(self):
        result = []
        for arch in self.thetas_main:
            if 'Composite' in arch['block_name']:
                result.append(arch['operators_choice'])
                
                for choice_arch in arch['operators']:
                    block_choice_arch = {}
                    for key, value in choice_arch.items():
                        if key == 'block_name': 
                            continue
                        else:
                            result.append(choice_arch[key])
                        
            elif 'Search' in arch['block_name']:
                for key, value in arch.items():
                    if key == 'block_name': 
                        continue
                    else:
                        result.append(arch[key])
            
        return result

    def random_sampling(self, device='cpu'):
        res = []
        for arch in self.thetas_main:
            block_arch={}
            if 'Composite' in arch['block_name']:
                idx = random.randint(0, len(sample['arch']) - 1)
                choice_prob = torch.zeros_like(arch['operators_choice'])
                choice_prob[idx] = 1.0
                block_arch['operators_choice'] = choice_prob.to(device)
                
                block_arch['operators'] = []
                for choice_arch in arch['operators']:
                    block_choice_arch = {}
                    for key, value in choice_arch.items():
                        if key == 'block_name': 
                            block_choice_arch[key] = value
                        else:
                            idx = np.random.randint(0, len(choice_arch[key]))
                            arch_prob = torch.zeros_like(choice_arch[key])
                            arch_prob[idx] = 1.0
                            block_choice_arch[key] = arch_prob.to(device)
                    block_arch['operators'].append(block_choice_arch)
                        
            elif 'Search' in arch['block_name']:
                if 'BottleneckCSP' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        else:
                            idx = np.random.randint(0, len(arch[key]))
                            arch_prob = torch.zeros_like(arch[key])
                            arch_prob[idx] = 1.0
                            block_arch[key] = arch_prob.to(device)
                elif 'ELAN' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        elif key == 'connection':
                            arch_prob = torch.zeros_like(arch[key])
                            for i in range(len(arch_prob)):
                                arch_prob[i] = np.random.choice([0.0, 1.0], 1)
                            block_arch[key] = arch_prob.to(device)
                        else:
                            idx = np.random.randint(0, len(arch[key]))
                            arch_prob = torch.zeros_like(arch[key])
                            arch_prob[idx] = 1.0
                            block_arch[key] = arch_prob.to(device)
                else:
                    raise ValueError(f"Unrecognize Block Type : {arch['block_name']}")
            
            res.append(block_arch)
        return res
    
    def gumbel_sampling(self, temperature=None, device='cpu', detach=False):
        """
        this.tehtas_main = {
            {
                'block_name': 'Composite_op2',
                'operator_choice' : [0.5, 0.5],
                'operator' : [
                    {
                        'block_name' : 'BottleneckCSP_Search_num2_gamma1.0',
                        'gamma' :         [0.33, 0.33, 0.33],
                        'n_bottlenecks' : [0.33, 0.33, 0.33],
                        
                    },
                    {
                        'block_name' : 'BottleneckCSP2_Search_num2_gamma1.0',
                        'gamma' :         [0.33, 0.33, 0.33],
                        'n_bottlenecks' : [0.33, 0.33, 0.33],
                        
                    }
                ]
            },
            {
                'block_name' : 'BottleneckCSP_Search_num2_gamma1.0',
                'gamma' :         [0.33, 0.33, 0.33],
                'n_bottlenecks' : [0.33, 0.33, 0.33],
            },
            {
                'block_name' : 'BottleneckCSP2_Search_num2_gamma1.0',
                'gamma' :         [0.33, 0.33, 0.33],
                'n_bottlenecks' : [0.33, 0.33, 0.33],
            }
        }
        """
        if temperature is None:
            temperature = self.temperature
        DEBUG = False
        res = []
        for stage_idx, arch in enumerate(self.thetas_main):
            block_arch={}
            if 'Composite' in arch['block_name']:
                # raise ValueError(f'{stage_idx} is composite')
                choice_prob = arch['operators_choice'].clone()
                block_arch['operators_choice'] = torch.nn.functional.gumbel_softmax(choice_prob.to(device), temperature)
                
                block_arch['operators'] = []
                for choice_arch in arch['operators']:
                    block_choice_arch = {}
                    for key, value in choice_arch.items():
                        if key == 'block_name': 
                            block_choice_arch[key] = value
                        else:
                            arch_prob = choice_arch[key].clone()
                            gumbel_value = torch.nn.functional.gumbel_softmax(arch_prob.to(device), temperature)
                            block_choice_arch[key] = gumbel_value if not detach else gumbel_value.detach()
                    block_arch['operators'].append(block_choice_arch)
                        
            elif 'Search' in arch['block_name']:
                if 'BottleneckCSP' in arch['block_name']:
                    search_idx = 0
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        else:
                            arch_prob = arch[key].clone()
                            gumbel_value = torch.nn.functional.gumbel_softmax(arch_prob.to(device), temperature, dim=-1)
                            block_arch[key] = gumbel_value if not detach else gumbel_value.detach()
                            search_idx += 1
                elif 'ELAN' in arch['block_name']:
                    search_idx = 0
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        elif key == 'connection':
                            arch_prob = arch[key].clone()
                            gumbel_value = gumbel_sigmoid(arch_prob.to(device), temperature)
                            block_arch[key] = gumbel_value if not detach else gumbel_value.detach()
                            search_idx += 1
                        else:
                            arch_prob = arch[key].clone()
                            gumbel_value = torch.nn.functional.gumbel_softmax(arch_prob.to(device), temperature, dim=-1)
                            block_arch[key] = gumbel_value if not detach else gumbel_value.detach()
                            search_idx += 1
                else:
                    raise ValueError(f"Unrecognize Block Type : {arch['block_name']}")       
            else:
                raise ValueError(f"Invalid Block Name {arch['block_name']}")
            res.append(block_arch)
        return res
    
    def softmax_sampling(self, temperature=None, detach=False, device='cpu'):
        """
        this.tehtas_main = {
            
        }
        """
        if temperature is None:
            temperature = self.temperature
            print(f'temperature is none, so using default temp {temperature}')
            
        res = []
        for arch in self.thetas_main:
            block_arch={}
            if 'Composite' in arch['block_name']:
                choice_prob = arch['operators_choice'].clone()
                block_arch['operators_choice'] = torch.nn.functional.softmax(choice_prob.to(device) / temperature, dim=-1)
                if detach: block_arch['operators_choice'] = block_arch['operators_choice'].detach()
                
                block_arch['operators'] = []
                for choice_arch in arch['operators']:
                    block_choice_arch = {}
                    for key, value in choice_arch.items():
                        if key == 'block_name': 
                            block_choice_arch[key] = value
                        else:
                            arch_prob = choice_arch[key].clone()
                            block_choice_arch[key] = torch.nn.functional.softmax(arch_prob.to(device) / temperature, dim=-1)
                            if detach: block_choice_arch[key] = block_choice_arch[key].detach()
                    block_arch['operators'].append(block_choice_arch)
                        
            elif 'Search' in arch['block_name']:
                if 'BottleneckCSP' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        else:
                            arch_prob = arch[key].clone()
                            block_arch[key] = torch.nn.functional.softmax(arch_prob.to(device) / temperature, dim=-1)
                            if detach: block_arch[key] = block_arch[key].detach()
                elif 'ELAN' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        elif key == 'connection':
                            arch_prob = arch[key].clone()
                            block_arch[key] = torch.nn.functional.sigmoid(arch_prob.to(device) / temperature)                            
                            if detach: block_arch[key] = block_arch[key].detach()
                        else:
                            arch_prob = arch[key].clone()
                            block_arch[key] = torch.nn.functional.softmax(arch_prob.to(device) / temperature, dim=-1)
                            if detach: block_arch[key] = block_arch[key].detach()
                else:
                    raise ValueError(f"Unrecognize Block Type : {arch['block_name']}")
            else:
                raise ValueError(f"Invalid Block Name {arch['block_name']}")
            res.append(block_arch)
        return res
    
    def largest_sampling(self):
        res = []
        for arch in self.thetas_main:
            block_arch={}
            if 'Composite' in arch['block_name']:
                choice_prob = torch.zeros_like(arch['operators_choice'])
                choice_prob[0] = 1.0
                block_arch['operators_choice'] = choice_prob
                
                block_arch['operators'] = []
                for choice_arch in arch['operators']:
                    block_choice_arch = {}
                    for key, value in choice_arch.items():
                        if key == 'block_name': 
                            block_choice_arch[key] = value
                        else:
                            arch_prob = torch.zeros_like(choice_arch[key])
                            arch_prob[-1] = 1.0
                            block_choice_arch[key] = arch_prob
                    block_arch['operators'].append(block_choice_arch)
                        
            elif 'Search' in arch['block_name']:
                if 'BottleneckCSP' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        else:
                            arch_prob = torch.zeros_like(arch[key])
                            arch_prob[-1] = 1.0
                            block_arch[key] = arch_prob
                elif 'ELAN' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        elif key == 'connection':
                            arch_prob = torch.ones_like(arch[key])
                            block_arch[key] = arch_prob
                        else:
                            arch_prob = torch.zeros_like(arch[key])
                            arch_prob[-1] = 1.0
                            block_arch[key] = arch_prob
            else:
                raise ValueError(f"Invalid Block Name {arch['block_name']}")
            res.append(block_arch)
        return res
    
    def smallest_sampling(self):
        res = []
        for arch in self.thetas_main:
            block_arch={}
            if 'Composite' in arch['block_name']:
                choice_prob = torch.zeros_like(arch['operators_choice'])
                choice_prob[0] = 1.0
                block_arch['operators_choice'] = choice_prob
                
                block_arch['operators'] = []
                for choice_arch in arch['operators']:
                    block_choice_arch = {}
                    for key, value in choice_arch.items():
                        if key == 'block_name': 
                            block_choice_arch[key] = value
                        else:
                            arch_prob = torch.zeros_like(choice_arch[key])
                            arch_prob[0] = 1.0
                            block_choice_arch[key] = arch_prob
                    block_arch['operators'].append(block_choice_arch)
                        
            elif 'Search' in arch['block_name']:
                if 'BottleneckCSP' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        else:
                            arch_prob = torch.zeros_like(arch[key])
                            arch_prob[0] = 1.0
                            block_arch[key] = arch_prob
                elif 'ELAN' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        elif key == 'connection':
                            arch_prob = torch.zeros_like(arch[key])
                            block_arch[key] = arch_prob
                        else:
                            arch_prob = torch.zeros_like(arch[key])
                            arch_prob[0] = 1.0
                            block_arch[key] = arch_prob
            
            else:
                raise ValueError(f"Invalid Block Name {arch['block_name']}")
            res.append(block_arch)
        return res
    
    def discretize_sampling(self, arches=None):
        if arches is None:
            arches = self.thetas_main
            
        res = []
        for arch in arches:
            block_arch={}
            if 'Composite' in arch['block_name']:
                # choice_prob = torch.zeros_like(arch['operators_choice'])
                # choice_prob[0] = 1.0
                # block_arch['operators_choice'] = choice_prob
                
                # block_arch['operators'] = []
                # for choice_arch in arch['operators']:
                #     block_choice_arch = {}
                #     for key, value in choice_arch.items():
                #         if key == 'block_name': 
                #             block_choice_arch[key] = value
                #         else:
                #             arch_prob = torch.zeros_like(choice_arch[key])
                #             arch_prob[0] = 1.0
                #             block_choice_arch[key] = arch_prob
                #     block_arch['operators'].append(block_choice_arch)
                pass

            elif 'Search' in arch['block_name']:
                if 'BottleneckCSP' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        else:
                            max_idx   = arch[key].argmax().detach().cpu().numpy()
                            arch_prob = torch.zeros_like(arch[key])
                            arch_prob[max_idx] = 1.0
                            block_arch[key] = arch_prob
                elif 'ELAN' in arch['block_name']:
                    for key, value in arch.items():
                        if key == 'block_name': 
                            block_arch[key] = value
                        elif key == 'connection':
                            arch_prob = torch.zeros_like(arch[key])
                            for i in range(len(arch[key])):
                                arch_prob[i] = 1.0 if arch[key][i] > 0.0 else 0.0
                            block_arch[key] = arch_prob
                        else:
                            max_idx   = arch[key].argmax().detach().cpu().numpy()
                            arch_prob = torch.zeros_like(arch[key])
                            arch_prob[max_idx] = 1.0
                            block_arch[key] = arch_prob
                     
            else:
                raise ValueError(f"Invalid Block Name {arch['block_name']}")
            
            res.append(block_arch)
        return res
    
    ################################################
    # Utility Function
    # WeiJie Implementation
    ################################################ 
    def calculate_flops_new(self, architecture_info, flops_dict):
        """
        Params
        ------
        architecutre_info : 
        flops_dict : type,
        Returns
        -------
        overall_flops: torch.tensor(1,), M-FLOPS
        """
        SHOW_FLOP_STAT = False
        keys = sorted(self.search_space.keys())
        architecture = architecture_info['arch']
        architecture_type = architecture_info['arch_type']
        
        current_theta = 0
        overall_flops = 0
        for block_idx, block in enumerate(self.blocks):         
            block_idx = str(block_idx)
            # [Discrete Mode] use architecture to sample subnetwork to do inference
            if architecture_type == 'discrete':
                raise ValueError("Not Implement")

            # [Continuous Mode] use architecture distribtuion and weighted-sum their output to do inference
            elif architecture_type == 'continuous':
                layer_flops = 0.0
                if block.__class__ in [BottleneckCSP_Search, BottleneckCSP2_Search]:
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    options, search_keys = block.generate_options()
                    
                    for option in options:
                        prob = 1.0
                        query_keys = []
                        for key_idx, key in enumerate(search_keys):
                            option_index = option[key_idx]
                            option_value = block.search_space[key][option_index]
                            option_prob  = soft_mask_variables[key][option_index]
                            query_keys.append(f'{key}{option_value}')
                            prob *= option_prob
                            
                        
                        query_key      = '-'.join(query_keys)
                        choice_flops = flops_dict[block_idx][query_key]
                        if SHOW_FLOP_STAT: print(f'[FLOPS opt={query_keys}] flops={choice_flops} prob={prob} mut={choice_flops * prob}')
                        layer_flops += choice_flops * prob

                    current_theta += 1
                elif block.__class__ in [ELAN_Search, ELAN2_Search]:
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    options, search_keys = block.generate_options()
                    comb_list       = block._connection_combination(block.search_space['connection'])
                    comb_list_index = block._connection_combination(block.search_space['connection'], index=True)

                    for option in options:
                        args_cn             = int(block.search_space['gamma'     ][option[search_keys.index('gamma')]] * block.base_cn)
                        args_connection     = comb_list[option[search_keys.index('connection')]]
                        args_connection_idx = comb_list_index[option[search_keys.index('connection')]]
                        prob = 1.0
                        
                        selected_con   = args_connection_idx
                        unselected_con = list(set(comb_list_index[-1]) - set(args_connection_idx))
                        
                        prob *= soft_mask_variables['gamma'][option[search_keys.index('gamma')]]
                        prob *= torch.prod(      soft_mask_variables['connection'][selected_con])
                        prob *= torch.prod(1.0 - soft_mask_variables['connection'][unselected_con])
                        # for connection_idx in args_connection_idx:
                        #     prob *= soft_mask_variables['connection'][connection_idx]
                        query_key = f'cn{args_cn}-con{str(args_connection)}'

                        choice_flops = flops_dict[block_idx][query_key]
                        if SHOW_FLOP_STAT: print(f'[FLOPS opt={query_key}] flops={choice_flops} prob={prob} mut={choice_flops * prob}')
                        layer_flops += choice_flops * prob

                    current_theta += 1
                else:
                    layer_flops = flops_dict[block_idx]['0']
                
                if SHOW_FLOP_STAT: print(f'[FLOPS {block_idx}]={layer_flops}')
                overall_flops += layer_flops
        
        return overall_flops
    
    def calculate_params_new(self, architecture_info, params_dict):
        """
        Params
        ------
        Returns
        -------
        overall_params: torch.tensor(1,), M-PARAMS
        """
        SHOW_FLOP_STAT=False
        keys = sorted(self.search_space.keys())
        architecture = architecture_info['arch']
        architecture_type = architecture_info['arch_type']
        
        current_theta = 0
        overall_params = 0
        for block_idx, block in enumerate(self.blocks):            
            block_idx = str(block_idx)
            # [Discrete Mode] use architecture to sample subnetwork to do inference
            if architecture_type == 'discrete':
                raise ValueError("Not Implement")

            # [Continuous Mode] use architecture distribtuion and weighted-sum their output to do inference
            elif architecture_type == 'continuous':
                layer_params = 0.
                if block.__class__ in [BottleneckCSP_Search, BottleneckCSP2_Search]:
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    options, search_keys = block.generate_options()
                    
                    for option in options:
                        prob = 1.0
                        query_keys = []
                        for key_idx, key in enumerate(search_keys):
                            option_index = option[key_idx]
                            option_value = block.search_space[key][option_index]
                            option_prob  = soft_mask_variables[key][option_index]
                            query_keys.append(f'{key}{option_value}')
                            prob *= option_prob
                            
                        
                        query_key      = '-'.join(query_keys)
                        choice_params = params_dict[block_idx][query_key]

                        layer_params += choice_params * prob
                    current_theta += 1
                elif block.__class__ in [ELAN_Search, ELAN2_Search]:
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    options, search_keys = block.generate_options()
                    comb_list       = block._connection_combination(block.search_space['connection'])
                    comb_list_index = block._connection_combination(block.search_space['connection'], index=True)

                    for option in options:
                        args_cn             = int(block.search_space['gamma'     ][option[search_keys.index('gamma')]] * block.base_cn)
                        args_connection     = comb_list[option[search_keys.index('connection')]]
                        args_connection_idx = comb_list_index[option[search_keys.index('connection')]]
                        prob = 1.0
                        
                        selected_con   = args_connection_idx
                        unselected_con = list(set(comb_list_index[-1]) - set(args_connection_idx))
                        
                        prob *= soft_mask_variables['gamma'][option[search_keys.index('gamma')]]
                        prob *= torch.prod(      soft_mask_variables['connection'][selected_con])
                        prob *= torch.prod(1.0 - soft_mask_variables['connection'][unselected_con])
                        # for connection_idx in args_connection_idx:
                        #     prob *= soft_mask_variables['connection'][connection_idx]
                        query_key = f'cn{args_cn}-con{str(args_connection)}'

                        choice_params = params_dict[block_idx][query_key]
                        if SHOW_FLOP_STAT: print(f'[PARAM opt={query_key}] parms={choice_params} prob={prob} mut={choice_params * prob}')
                        layer_params += choice_params * prob
                    current_theta += 1
                else:
                    layer_params += params_dict[block_idx]['0']

                overall_params += layer_params

        return overall_params

    def calculate_layers_new(self, architecture):
        architecture = architecture_info['arch']
        architecture_type = architecture_info['arch_type']
                
        current_theta = 0
        block_id = 0
        overall_layers = 0
        for layer, layer_arch in zip(self.blocks, architecture):
            block_idx = str(block_idx)
            # [Discrete Mode] use architecture to sample subnetwork to do inference
            if architecture_type == 'discrete':
                raise ValueError("Not Implement")

            # [Continuous Mode] use architecture distribtuion and weighted-sum their output to do inference
            elif architecture_type == 'continuous':
                if 'Search' in block.__class__.__name__:
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    
                    if block.__class__.__name__ == ' Composite_Search':
                        pass
                    else:
                        expected_layer = 0
                        for i in range(len(block.search_space['n_bottlenecks'])):
                            depth_value = block.search_space['n_bottlenecks'][i]
                            depth_prob  = soft_mask_variables['n_bottlenecks'][i]
                            expected_layer += depth_value * depth_prob
                            
                        overall_layers += expected_layer
                        current_theta += 1
                        
            block_id += 1
        return overall_layers

    def calculate_zc(self, architecture_info, zc_map):
        """
        Params
        ------
        architecutre_info : architecture distribution
        zc_map : a dict that store the zero cost value of each candidate block
        
        Returns
        -------
        overall_zc: torch.tensor(1,), M-FLOPS
        """
        SHOW_ZC_STAT = False
        keys = sorted(self.search_space.keys())
        architecture = architecture_info['arch']
        architecture_type = architecture_info['arch_type']
        
        current_theta = 0
        overall_zc = 0
        for block_idx, block in enumerate(self.blocks):         
            block_idx = str(block_idx)
            # [Discrete Mode] use architecture to sample subnetwork to do inference
            if architecture_type == 'discrete':
                raise ValueError("Not Implement")

            # [Continuous Mode] use architecture distribtuion and weighted-sum their output to do inference
            elif architecture_type == 'continuous':
                layer_zc = 0.0
                if block.__class__ in [BottleneckCSP_Search, BottleneckCSP2_Search]:
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    options, search_keys = block.generate_options()
                    
                    for option in options:
                        prob = 1.0
                        query_keys = []
                        for key_idx, key in enumerate(search_keys):
                            option_index = option[key_idx]
                            option_value = block.search_space[key][option_index]
                            option_prob  = soft_mask_variables[key][option_index]
                            query_keys.append(f'{key}{option_value}')
                            prob *= option_prob
                            
                        
                        query_key      = '-'.join(query_keys)
                        choice_zc = zc_map[current_theta][query_key]
                        if SHOW_ZC_STAT: print(f'[ZC opt={query_keys}] flops={choice_zc} prob={prob} mut={choice_zc * prob}')
                        layer_zc += choice_zc * prob

                    current_theta += 1
                
                elif block.__class__ in [ELAN_Search, ELAN2_Search]:
                    import time
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    # st = time.time()
                    options, search_keys = block.generate_options()
                    # st1_time = time.time() - st
                    
                    # st = time.time()
                    comb_list       = block._connection_combination(block.search_space['connection'])
                    comb_list_index = block._connection_combination(block.search_space['connection'], index=True)
                    # st2_time = time.time() - st
                    
                    # st = time.time()
                    for option in options:
                        args_cn             = int(block.search_space['gamma'     ][option[search_keys.index('gamma')]] * block.base_cn)
                        args_connection     = comb_list[option[search_keys.index('connection')]]
                        args_connection_idx = comb_list_index[option[search_keys.index('connection')]]
                        prob = 1.0
                        
                        selected_con   = args_connection_idx
                        unselected_con = list(set(comb_list_index[-1]) - set(args_connection_idx))
                        
                        prob *= soft_mask_variables['gamma'][option[search_keys.index('gamma')]]
                        prob *= torch.prod(      soft_mask_variables['connection'][selected_con])
                        prob *= torch.prod(1.0 - soft_mask_variables['connection'][unselected_con])
                        # for connection_idx in args_connection_idx:
                        #     prob *= soft_mask_variables['connection'][connection_idx]
                        query_key = f'cn{args_cn}-con{str(args_connection)}'
                        
                        choice_zc = zc_map[current_theta][query_key]
                        if SHOW_ZC_STAT: print(f'[ZC opt={query_keys}] flops={choice_zc} prob={prob} mut={choice_zc * prob}')
                        layer_zc += choice_zc * prob

                    # st3_time = time.time() - st
                    # print(f'st1_time={st1_time:.8f} st2_time={st2_time:.8f} st3_time={st3_time:.8f} total={st1_time+st2_time+st3_time:.8f}')
                    current_theta += 1
                else:
                    pass
                    # layer_flops = flops_dict[block_idx]['0']
                
                if SHOW_ZC_STAT: print(f'[ZC {block_idx}]={layer_zc}')
                overall_zc += layer_zc
        
        return overall_zc

    def calculate_utility(self, architecture_info, flops_dict, params_dict, zc_map):
        """
        Params
        ------
        architecutre_info : 
        flops_dict : type,
        Returns
        -------
        overall_flops: torch.tensor(1,), M-FLOPS
        """
        SHOW_FLOP_STAT = False
        keys = sorted(self.search_space.keys())
        architecture = architecture_info['arch']
        architecture_type = architecture_info['arch_type']
        
        current_theta = 0
        overall_flops = 0
        overall_params= 0
        overall_zc    = 0
        for block_idx, block in enumerate(self.blocks):         
            block_idx = str(block_idx)
            # [Discrete Mode] use architecture to sample subnetwork to do inference
            if architecture_type == 'discrete':
                raise ValueError("Not Implement")

            # [Continuous Mode] use architecture distribtuion and weighted-sum their output to do inference
            elif architecture_type == 'continuous':
                layer_flops = 0.0
                layer_params= 0.0
                layer_zc    = 0.0
                if block.__class__ in [BottleneckCSP_Search, BottleneckCSP2_Search]:
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    options, search_keys = block.generate_options()
                    
                    for option in options:
                        prob = 1.0
                        query_keys = []
                        for key_idx, key in enumerate(search_keys):
                            option_index = option[key_idx]
                            option_value = block.search_space[key][option_index]
                            option_prob  = soft_mask_variables[key][option_index]
                            query_keys.append(f'{key}{option_value}')
                            prob *= option_prob
                            
                        
                        query_key      = '-'.join(query_keys)
                        choice_flops = flops_dict[block_idx][query_key]
                        choice_params = params_dict[block_idx][query_key]
                        choice_zc = zc_map[current_theta][query_key]
                        
                        layer_flops += choice_flops * prob
                        layer_params+= choice_params* prob
                        layer_zc    += choice_zc    * prob
                        
                        if SHOW_FLOP_STAT: print(f'[FLOPS opt={query_keys}] flops={choice_flops} prob={prob} mut={choice_flops * prob}')
                        

                    current_theta += 1
                elif block.__class__ in [ELAN_Search, ELAN2_Search]:
                    # soft_mask_variables = nn.functional.gumbel_softmax(self.thetas[current_theta](), self.temperature)
                    soft_mask_variables = architecture[current_theta]
                    options, search_keys = block.generate_options()
                    comb_list       = block._connection_combination(block.search_space['connection'])
                    comb_list_index = block._connection_combination(block.search_space['connection'], index=True)

                    for option in options:
                        args_cn             = int(block.search_space['gamma'     ][option[search_keys.index('gamma')]] * block.base_cn)
                        args_connection     = comb_list[option[search_keys.index('connection')]]
                        args_connection_idx = comb_list_index[option[search_keys.index('connection')]]
                        prob = 1.0
                        
                        selected_con   = args_connection_idx
                        unselected_con = list(set(comb_list_index[-1]) - set(args_connection_idx))
                        
                        prob *= soft_mask_variables['gamma'][option[search_keys.index('gamma')]]
                        prob *= torch.prod(      soft_mask_variables['connection'][selected_con])
                        prob *= torch.prod(1.0 - soft_mask_variables['connection'][unselected_con])
                        
                        query_key = f'cn{args_cn}-con{str(args_connection)}'

                        choice_flops = flops_dict[block_idx][query_key]
                        choice_params = params_dict[block_idx][query_key]
                        choice_zc = zc_map[current_theta][query_key]

                        layer_flops += choice_flops * prob
                        layer_params+= choice_params* prob
                        layer_zc    += choice_zc    * prob

                        if SHOW_FLOP_STAT: print(f'[FLOPS opt={query_key}] flops={choice_flops} prob={prob} mut={choice_flops * prob}')

                    current_theta += 1
                else:
                    layer_flops = flops_dict[block_idx]['0']
                    layer_params= params_dict[block_idx]['0']
                
                overall_flops += layer_flops
                overall_params+= layer_params
                overall_zc    += layer_zc
                
                if SHOW_FLOP_STAT: print(f'[FLOPS {block_idx}]={layer_flops}')
                
        
        return overall_flops, overall_params, overall_zc
    

    #########################################################################
    # ZeroDNAS Function
    #########################################################################
    def generate_proxy_map(self, x, distributions, proxy_func):
        """
        x: input image. torch.float32(b, c, h, w)
        distributions: architecture distributions parameter. torch.float32()
        """
        # Pass data through stem
        # print('Initial shape:', x.shape)
        SHOW_FEATURE_STATS = False
        keys = sorted(self.search_space.keys())
        if distributions is None:
            distributions = self.softmax_sampling()
        
        stage_idx = 0
        y, dt = [], []  # outputs
        profile = False
        for idx, m in enumerate(self.blocks):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))


            if 'Search' in m.__class__.__name__:
                stage_args = distributions[stage_idx]
                x = m(x, stage_args)
                if SHOW_FEATURE_STATS:
                    a,b = x.detach().cpu().min().numpy(), x.detach().cpu().max().numpy()
                    print(f'{idx:02d} {m.__class__.__name__:30s} min {a:10.8e} max {b:10.8e}')
                stage_idx+=1
            else:
                x = m(x)  # run
                if SHOW_FEATURE_STATS:
                    if 'Detect' not in m.__class__.__name__:
                        a,b = x.detach().cpu().min().numpy(), x.detach().cpu().max().numpy()
                        print(f'{idx:02d} {m.__class__.__name__:30s} min {a:10.8e} max {b:10.8e}')            
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))

        chosen_subnet = [None]
        return x, y, chosen_subnet
    
    def update_proxy_map(self, architecture_info):
        pass
    
class Classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        return self.classifier(x)

def gen_supernet(model_args, **kwargs):

    # choices = {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]} # big
    # choices = {'n_bottlenecks': [0, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]} # tiny

    choices = model_args['search_space']
    blocks_args     = model_args['backbone'] + model_args['head']
    
    print('choices: ', choices)

    # exit()
    #1 4 0 0
    # act_layer = HardSwish
    act_layer = nn.ReLU

    # arch_def = [
    #     # 
    #     # Backbone
    #     # stage 0
    #     ['cn_r1_k3_s2_cin32_cout64'],
    #     # stage 1
    #     ['bottle_r1_k1_s1_cin64_cout64'], # BottleNeck [64]
    #     # stage 2
    #     ['cn_r1_k3_s2_cin64_cout128'],
    #     # stage 3
    #     ['bottlecsp_r1_k1_s1_num2_gamma0.5_cin128_cout128'],  # BottleNeck [128]
    #     # stage 4
    #     ['cn_r1_k3_s2_cin128_cout256'],
    #     # stage 5
    #     ['bottlecsp_r1_k1_s1_num8_gamma0.5_cin256_cout256'], #BottleNeck [256]
    #     # stage 6
    #     ['cn_r1_k3_s2_cin256_cout512'],
    #     # stage 7
    #     ['bottlecsp_r1_k1_s1_num8_gamma0.5_cin512_cout512'], # BottleNeckCSP [512]
    #     # stage 8
    #     ['cn_r1_k3_s2_cin512_cout1024'],
    #     # stage 9
    #     ['bottlecsp_r1_k1_s1_num8_gamma0.5_cin1024_cout1024'], # BottleneckCSP [1024]
    #     # stage 10
    #     ['sppcsp_r1_k1_s1_cin1024_cout512'],
    #     # stage 11
    #     ['cn_np_r1_k1_s1_cin512_cout256', 'up_r1_mnearest_sf2_c256', 'cn_r1_k1_s1_cin512_cout256_f8'],
    #     # stage 12
    #     ['concat_r1_c512_f-2'],
    #     # stage 13
    #     ['bottlecsp2_r1_k1_s1_num2_gamma0.5_cin512_cout256'], # should be csp2
    #     # stage 14
    #     ['cn_np_r1_k1_s1_cin256_cout128', 'up_r1_mnearest_sf2_c128', 'cn_r1_k1_s1_cin256_cout128_f6'],
    #     # stage 15
    #     ['concat_r1_c256_f-2'],
    #     # stage 16
    #     ['bottlecsp2_r1_k1_s1_num2_gamma0.5_cin256_cout128'],
    #     # stage 17
    #     ['cn_r1_k3_s1_cin128_cout256'],
    #     # stage 18
    #     ['cn_r1_k3_s2_cin128_cout256_f-2'],
    #     # stage 19
    #     ['concat_r1_c512_f16'],
    #     # stage 20
    #     ['bottlecsp2_r1_k1_s1_num2_gamma0.5_cin512_cout256'], #should be csp2
    #     # stage 21
    #     ['cn_r1_k3_s1_cin256_cout512'],
    #     # stage 22
    #     ['cn_r1_k3_s2_cin256_cout512_f-2'],
    #     # stage 23
    #     ['concat_r1_c1024_f11'],
    #     # stage 24
    #     ['bottlecsp2_r1_k1_s1_num2_gamma0.5_cin1024_cout512'], #should be csp2
    #     # stage 25
    #     ['cn_r1_k3_s1_cin512_cout1024_last'],
    # ]


    # sta_num, arch_def, resolution = search_for_layer(flops_op_dict, arch_def, flops_minimum, flops_maximum)
    # if sta_num is None or arch_def is None or resolution is None:
    #     raise ValueError('Invalid FLOPs Settings')
    model_kwargs = dict(
        # block_args=decode_arch_def(arch_def),
        model_args=model_args,
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
    # return model, sta_num, resolution
    return model, None, None
    
