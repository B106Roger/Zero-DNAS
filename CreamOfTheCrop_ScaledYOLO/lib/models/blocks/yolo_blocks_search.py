import math
import itertools
import torch
import copy
import torch.nn as nn
import numpy as np
from mish_cuda import MishCuda as Mish
from torch.nn import ReLU, LeakyReLU, SiLU

from lib.utils.synflow import synflow, sum_arr
from lib.models.blocks.yolo_blocks import Conv, Bottleneck, DEFAULT_ACTIVATION, V4_DEFAULT_ACTIVATION, V7_DEFAULT_ACTIVATION


TYPE = None # ZeroDNAS_Egor or DNAS or ZeroCost
SHOW_FEATURE_STATS = False
############################################
# Important Noticing for Default Value
############################################
# groupnorm : number of group = 1 by default
############################################
NORMALIZATION = {
    'batchnorm': nn.BatchNorm2d,
    'groupnorm': lambda num_feat : nn.GroupNorm(1, num_feat), 
}
ACTIVATION = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'mish': Mish,
}

def feature_inspection(x, prefix=''):
    tmp_feat = x.detach().cpu()
    a,b,c = tmp_feat.min().numpy(), tmp_feat.max().numpy(), tmp_feat.mean().numpy()
    d= tmp_feat.data[0,4,4,4].numpy()
    print(f'[Inspection {prefix}] min {a:10.8e} max {b:10.8e} mean {c:10.8e} first ele: {d:10.8e}')

def set_algorithm_type(type_literal):
    """
    Option
        ZeroDNAS_Egor:
            1. use GroupNormalization to replace all the BatchNormalization
            2. use MACS as the flop constraint [FLOPS = MAC * 2]
            3. use 'efficientnet_init_weights' to initialize yolo-detector
        DNAS:
            1. use BatchNormalization
            2. use 'Real' flop constraint [FLOPS = MAC * 2]
            3. use 'the initialize method from ScaledYOLO source code'
        ZeroCost:
            1. use GroupNormalization to replace all the BatchNormalization
            2. use 'Real' flop constraint [FLOPS = MAC * 2]
            3. use 'the initialize method from ScaledYOLO source code'
    """
    global TYPE
    if type_literal not in ['DNAS', 'ZeroDNAS_Egor', 'ZeroCost']:
        raise ValueError(f"Invalid Search Algorithm {type_literal}")    
    TYPE = type_literal
    
def get_algorithm_type():
    global TYPE
    if TYPE not in ['DNAS', 'ZeroDNAS_Egor', 'ZeroCost']:
        raise ValueError(f"Didn't Initialize the TYPE parameter.")    
    return TYPE    

def init_value(temperature, value_count, init_type, total_step=0.3):
    if init_type == 'uniform':
        return torch.nn.Parameter(torch.zeros((value_count, )))
    elif init_type == 'step':
        base_count = np.float32(value_count)
        step_count = np.arange(value_count).astype(np.float32).sum()
        
        step = total_step / step_count
        base = (1.0 - total_step) / base_count
        prob = np.ones((value_count,)) * base + np.arange(value_count) *step
        
        tmp_log = temperature * np.log(prob)
        res = tmp_log - tmp_log.mean()
        
        return torch.nn.Parameter(torch.tensor(res))

class GeneralOpeartor_Search(nn.Module):
    def __init__(self):
        super(GeneralOpeartor_Search, self).__init__()
        self.search_space = {}
        
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

    def calculate_synflow_metric(self):
        metric_array = []
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                metric_array.append(synflow(layer))

        return sum_arr(metric_array)

    def get_block_name(self):
        return self.block_name

    def generate_options(self):
        """
        for option in results:
            for key in keys:
                index = option[keys.index(key)]
                value = self.search_space[key][index]
        """
        keys = list(sorted(self.search_space.keys()))
        results = list(itertools.product(*[list(range(len(self.search_space[key]))) for key in keys]))
        
        return results, keys
    
    def init_arch_parameter(self, device='cpu'):
        arch = {'block_name' : self.block_name}
        return arch

class BottleneckCSP_Search(GeneralOpeartor_Search):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # Modify from    https://github.com/chiahuilin0531/ScaledYOLOv4
    def __init__(self, c1, c2, gamma_space, bottleneck_space, shortcut=True, g=1, bn='batchnorm', act='mish'): #, bn='groupnorm', act='relu'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP_Search, self).__init__()
        # self.gamma_space      = gamma_space
        # self.bottleneck_space = bottleneck_space
        self.search_space['gamma']         = copy.deepcopy(gamma_space)
        self.search_space['n_bottlenecks'] = copy.deepcopy(bottleneck_space)
        
        e=max(gamma_space)
        channel_values = [int(gamma*c2) for gamma in gamma_space]
        n = max(bottleneck_space)
        
        ########################################################################################
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        # if TYPE=='ZeroDNAS_Egor' or TYPE=='ZeroCost':
        #     self.bn = nn.GroupNorm(1, 2 * c_)  # applied to cat(cv2, cv3)
        #     self.act = nn.ReLU()
        # elif TYPE=='DNAS':
        #     self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        #     self.act = Mish()
        # else:
        #     raise ValueError(f'Invalid Type: {TYPE}')
        self.bn  = NORMALIZATION[bn](2 * c_)
        self.act = ACTIVATION[act]()
        
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.block_name = f'{self.__class__.__name__}_n{n}g{e}'
        ########################################################################################
        
        
        # masks = (num_of_mask, max_channel)
        masks = torch.zeros((len(gamma_space), max(channel_values)))
        # print('[Roger] BottleneckCSP channel_values', channel_values)
        for idx, num_channel in enumerate(channel_values):
            masks[idx, :num_channel] = 1.0
            # print(f'masks[{idx}, :{num_channel}]', masks[idx].mean())
        self.register_buffer('channel_masks', masks.clone())
            
    def forward(self, x, args=None):
        """
        Parameters
        ----------
        x: float32, (b, c, w, h)
        args: dict
            'gamma_dist': float32, (num_of_gamma_choices,)
        """
        mask = 1.0
        if args is not None:
            if 'gamma' in args.keys() and args['gamma'] is not None:
                mask = 0.0
                for i in range(len(self.channel_masks)):
                    # print('[Roger] BottleneckCSP gamma', self.channel_masks[i].mean())
                    mask += self.channel_masks[i] *  args['gamma'][i]
                mask = mask.reshape(1,-1,1,1)
        
            if 'n_bottlenecks' in args.keys() and args['n_bottlenecks'] is not None:
                depth_vals = self.search_space['n_bottlenecks'] # args['n_bottlenecks_val']
                depth_dist = args['n_bottlenecks']
                
                m_out = self.cv1(x, mask)
                if SHOW_FEATURE_STATS: feature_inspection(m_out, 'self.cv1')
                aggregate_feature = 0
                depth_idx = 0
                for depth in range(len(self.m) + 1):
                    if depth == depth_vals[depth_idx]:
                        aggregate_feature += m_out * depth_dist[depth_idx]
                        if SHOW_FEATURE_STATS: feature_inspection(m_out, f'depth={depth} prob={depth_dist[depth_idx]}')
                        depth_idx += 1
                    if depth == depth_vals[-1]: break
                    m_out = self.m[depth](m_out, mask)
                m_out = aggregate_feature
                
            else:
                m_out = self.cv1(x, mask)
                for m in self.m:
                    m_out = m(m_out, mask)
        else:
            m_out = self.cv1(x, mask)
            for m in self.m:
                m_out = m(m_out, mask)
        
        y1 = self.cv3(m_out) * mask
        y2 = self.cv2(x) * mask
        if SHOW_FEATURE_STATS: feature_inspection(m_out, 'self.cv3')
    
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    def init_arch_parameter(self, device='cpu', temperature=5.0, init_type='uniform'):
        arch = super(BottleneckCSP_Search, self).init_arch_parameter(device)
        for key, candidates in self.search_space.items():
            length    = len(candidates)
            # arch[key] = torch.nn.Parameter(torch.ones((length, )) / length).to(device)
            arch[key] = init_value(temperature, length, init_type=init_type).to(device)
            

        return arch
    
class BottleneckCSP2_Search(GeneralOpeartor_Search):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, gamma_space, bottleneck_space, shortcut=False, g=1, bn='batchnorm', act='mish'): #bn='groupnorm', act='relu'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2_Search, self).__init__()
        # self.gamma_space      = gamma_space
        # self.bottleneck_space = bottleneck_space
        self.search_space['gamma']      = gamma_space
        self.search_space['n_bottlenecks'] = bottleneck_space
        
        e = max(gamma_space)
        channel_values = [int((gamma+0.5)*c2) for gamma in gamma_space]
        n = max(bottleneck_space) 
        
        ########################################################################################
        c_ = int(c2 * (e + 0.5))  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn  = NORMALIZATION[bn](2 * c_)
        self.act = ACTIVATION[act]()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.block_name = f'{self.__class__.__name__}_n{n}g{e}'
        ########################################################################################
        
        # masks = (num_of_mask, max_channel)
        masks = torch.zeros((len(gamma_space), max(channel_values)))
        # print('[Roger] BottleneckCSP2 channel_values', channel_values)
        for idx, num_channel in enumerate(channel_values):
            masks[idx, :num_channel] = 1
            # print(f'masks[{idx}, :{num_channel}]', masks[idx].mean())
        masks = torch.nn.parameter.Parameter(masks, requires_grad=False)
        self.register_buffer('channel_masks', masks.clone())

    def forward(self, x, args=None):
        """
        Parameters
        ----------
        x: float32, (b, c, w, h)
        args: dict
            'gamma_dist': float32, (num_of_gamma_choices,)
        """
        mask = 1.0
        if args is not None:
            if 'gamma' in args.keys() and args['gamma'] is not None:
                mask = 0.0
                for i in range(len(self.channel_masks)):
                    # print('[Roger] BottleneckCSP2 gamma', self.channel_masks[i].mean())
                    mask += self.channel_masks[i] * args['gamma'][i]
                mask = mask.reshape(1,-1,1,1)
        
            if 'n_bottlenecks' in args.keys() and args['n_bottlenecks'] is not None:
                depth_vals = self.search_space['n_bottlenecks'] # args['n_bottlenecks_val']
                depth_dist = args['n_bottlenecks']
                
                x1 = m_out = self.cv1(x, mask)
                # if SHOW_FEATURE_STATS:feature_inspection(m_out, 'self.cv1')
                aggregation = 0
                depth_idx = 0
                for depth in range(len(self.m) + 1):
                    if depth == depth_vals[depth_idx]:
                        aggregation += m_out * depth_dist[depth_idx]
                        depth_idx += 1
                    if depth == depth_vals[-1]: break
                    # if SHOW_FEATURE_STATS:feature_inspection(m_out, f'Bef depth={depth} mask={mask.sum()}')
                    m_out = self.m[depth](m_out, mask)
                    # if SHOW_FEATURE_STATS: feature_inspection(m_out, f'Aft depth={depth} c_={self.m[depth].c_} g={self.m[depth].g} add={self.m[depth].add}')
                m_out = aggregation
            else:
                x1 = m_out = self.cv1(x, mask)
                for m in self.m:
                    m_out = m(m_out, mask)
        else:
            x1 = m_out = self.cv1(x, mask)
            for m in self.m:
                m_out = m(m_out, mask)
            
        y1 = m_out
        y2 = self.cv2(x1) * mask
        if SHOW_FEATURE_STATS:feature_inspection(y2, 'self.cv2')
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))
    
        
        # if self.search_type == 0 or self.search_type == 1:
        #     x1 = m_out = self.cv1(x) * mask
            
        #     for m in self.m[:n_bottlenecks]:
        #         m_out = m(m_out) * mask
                
        #     y1 = m_out
        #     y2 = self.cv2(x1) * mask
        #     return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    def init_arch_parameter(self, device='cpu', temperature=5.0, init_type='uniform'):
        arch = super(BottleneckCSP2_Search, self).init_arch_parameter()
        for key, candidates in self.search_space.items():
            length    = len(candidates)
            # arch[key] = torch.nn.Parameter(torch.ones((length, )) / length).to(device)
            arch[key] = init_value(temperature, length, init_type=init_type).to(device)
        return arch

class Composite_Search(GeneralOpeartor_Search):
    def __init__(self, operators):
        super(Composite_Search, self).__init__()
        # self.search_space = {'operator_choice' : list(range(len(operators)))}
        self.search_space = None
        
        self.operators = operators
        self.block_name = f'{self.__class__.__name__}_op{len(operators)}'
        
    def forawrd(self, x, args_list):
        out = 0
        for i, (block, args) in enumerate(zip(self.operators, args_list)):
            out += block(x, args)
            
        return out
    
    def init_arch_parameter(self, device='cpu', temperature=1.0):
        arch = super(Composite_Search).init_arch_parameter(device)
        arch['operators'] = []
        for operator in self.operators:
            if 'Search' in operator.__class__.__name__:
                arch['operators'].append(operator.init_arch_parameter(device))
            else:
                arch['operators'].append(None)

        
        length = len(self.operators)
        arch['operators_choice'] = torch.nn.Parameter(torch.ones((length,)) / length)
        return arch


#############################################################
# YOLOv7 Operator
#############################################################

class ELAN_Search(GeneralOpeartor_Search):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, cn, connection_space, gamma_space):  # ch_in, ch_out, number, shortcut, groups, expansion
        """
        connection_space:
            [-1,-2,-3,-4]
        gamma_space
        """
        super(ELAN_Search, self).__init__()
        self.search_space['gamma']         = copy.deepcopy(gamma_space)
        self.search_space['connection']    = copy.deepcopy(connection_space)
        self.base_cn = cn
        connection    =  -np.min(connection_space)
        cn            =   (np.max(gamma_space) * self.base_cn).astype(np.int32)
        
        c_ = (len(connection_space) + 2) * cn
        n  = len(connection_space)
        # print(c2, cn, connection_space, n)

        self.cv1 = Conv(c1, cn, 1, 1)
        self.cv2 = Conv(c1, cn, 1, 1)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.m = nn.Sequential(*[Conv(cn, cn, 3, 1) for _ in range(n)])
        
        self.block_name = f'{self.__class__.__name__}_n{n}g{cn}'
        
        channel_values = (np.array(gamma_space) * self.base_cn).astype(np.int32)
        # masks = (num_of_mask, max_channel)
        masks = torch.zeros((len(gamma_space), max(channel_values)))
        for idx, num_channel in enumerate(channel_values):
            masks[idx, :num_channel] = 1.0
        self.register_buffer('channel_masks', masks.clone())

    def forward(self, x, args=None):
        """
        gamma: [0.3333, 0.3333, 0.3333]
        connection: [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        """
        mask = 1.0
        n_connection_list = torch.ones((len(self.m), ))
        if args is not None:
            if 'gamma' in args.keys():
                mask = 0.0
                for i in range(len(self.channel_masks)):
                    mask += self.channel_masks[i] *  args['gamma'][i]
                mask = mask.reshape(1,-1,1,1)
            if 'connection' in args.keys():
                n_connection_list = torch.flip(args['connection'], dims=(0,))
        
        y1 = self.cv1(x, mask)
        y2 = self.cv2(x, mask)
        out = y2

        feat_list = [y1, y2]
        for idx, m in enumerate(self.m):
            out = m(out, mask) 
            feat_list.append(out * n_connection_list[idx])

        return self.cv3(torch.cat(feat_list, dim=1))

    def _connection_combination(self, connection, index=False):
        """
        generate the Combination of connection by index
        connection=[-1,-2,-3]
        self._connection_combination() -> [
            (), 
            (0), (1), (2), 
            (0,1), (1,2), (0,2),
            (0,1,2)
        ]
        """
        results = []
        for L in range(len(connection) + 1):
            target = list(range(len(connection))) if index else connection 
            for subset in itertools.combinations(target, L):
                results.append(list(subset))
        return results

    def generate_options(self):
        """
        for option in results:
            for key in keys:
                index = option[keys.index(key)]
                value = self.search_space[key][index]
        """
        keys = list(sorted(self.search_space.keys()))
        iter_object = []
        for key in keys:
            if key == 'gamma':
                search_space_idx = list(range(len(self.search_space[key])))
                iter_object.append(search_space_idx)
            elif key == 'connection':
                comb_space_idx = list(range(len(self._connection_combination(self.search_space[key]))))
                iter_object.append(comb_space_idx)
        results = list(itertools.product(*iter_object))
        
        return results, keys

    def init_arch_parameter(self, device='cpu', temperature=5.0, init_type='uniform'):
        arch = super(ELAN_Search, self).init_arch_parameter()
        for key, candidates in self.search_space.items():
            length    = len(candidates)
            if key == 'connection':
                arch[key] = torch.nn.Parameter(torch.ones((length, ))).to(device)
            else:
                arch[key] = init_value(temperature, length, init_type=init_type).to(device)
        return arch


class ELAN2_Search(GeneralOpeartor_Search):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, cn, connection_space, gamma_space):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ELAN2_Search, self).__init__()
        self.search_space['gamma']         = copy.deepcopy(gamma_space)
        self.search_space['connection']    = copy.deepcopy(connection_space)
        self.base_cn = cn        
        connection    =  -np.min(connection_space)
        cn            =   (np.max(gamma_space) * self.base_cn).astype(np.int32)
        
        c_ = (len(connection_space) + 4) * cn
        n  = len(connection_space)
        # print(c2, cn, connection_space, n)

        self.cv1 = Conv(c1, cn*2, 1, 1)
        self.cv2 = Conv(c1, cn*2, 1, 1)
        self.m = nn.Sequential(*[Conv(cn, cn, 3, 1) if _!=0 else Conv(cn*2, cn, 3, 1)for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1)
        self.block_name = f'{self.__class__.__name__}_n{n}g{cn}'
        
        channel_values = (np.array(gamma_space) * self.base_cn).astype(np.int32)
        # masks = (num_of_mask, max_channel)
        masks1 = torch.zeros((len(gamma_space), max(channel_values)))
        masks2 = torch.zeros((len(gamma_space), max(channel_values) * 2))
        for idx, num_channel in enumerate(channel_values):
            masks1[idx, :num_channel]     = 1.0
            masks2[idx, :(num_channel*2)] = 1.0
        self.register_buffer('channel_masks1', masks1.clone())
        self.register_buffer('channel_masks2', masks2.clone())

    def forward(self, x, args=None):
        mask1 = 1.0
        mask2 = 1.0
        n_connection_list = torch.ones((len(self.m),))
        if args is not None:
            if 'gamma' in args.keys():
                mask1 = 0.0
                mask2 = 0.0
                for i in range(len(self.channel_masks1)):
                    mask1 += self.channel_masks1[i] *  args['gamma'][i]
                    mask2 += self.channel_masks2[i] *  args['gamma'][i]

                mask1 = mask1.reshape(1,-1,1,1)
                mask2 = mask2.reshape(1,-1,1,1)
                
            if 'connection' in args.keys():
                n_connection_list = torch.flip(args['connection'], dims=(0,))
            
        y1 = self.cv1(x, mask2)
        y2 = self.cv2(x, mask2)
        out = y2

        feat_list = [y1, y2]
        for idx, m in enumerate(self.m):
            out = m(out, mask1) 
            feat_list.append(out * n_connection_list[idx])

        return self.cv3(torch.cat(feat_list, dim=1))

    def _connection_combination(self, connection, index=False):
        """
        generate the Combination of connection by index
        connection=[-1,-2,-3]
        self._connection_combination() -> [
            (), 
            (0), (1), (2), 
            (0,1), (1,2), (0,2),
            (0,1,2)
        ]
        """
        results = []
        for L in range(len(connection) + 1):
            target = list(range(len(connection))) if index else connection 
            for subset in itertools.combinations(target, L):
                results.append(list(subset))
        return results

    def generate_options(self):
        """
        for option in results:
            for key in keys:
                index = option[keys.index(key)]
                value = self.search_space[key][index]
        """
        keys = list(sorted(self.search_space.keys()))
        iter_object = []
        for key in keys:
            if key == 'gamma':
                search_space_idx = list(range(len(self.search_space[key])))
                iter_object.append(search_space_idx)
            elif key == 'connection':
                comb_space_idx = list(range(len(self._connection_combination(self.search_space[key]))))
                iter_object.append(comb_space_idx)
        results = list(itertools.product(*iter_object))
        
        return results, keys

    def init_arch_parameter(self, device='cpu', temperature=5.0, init_type='uniform'):
        arch = super(ELAN2_Search, self).init_arch_parameter()
        for key, candidates in self.search_space.items():
            length    = len(candidates)
            if key == 'connection':
                arch[key] = torch.nn.Parameter(torch.ones((length, ))).to(device)
            else:
                arch[key] = init_value(temperature, length, init_type=init_type).to(device)
        return arch