import math
import itertools
import torch
import copy
import torch.nn as nn
from mish_cuda import MishCuda as Mish
# from torch.nn import ReLU as Mish
from lib.utils.synflow import synflow, sum_arr
from lib.models.blocks.yolo_blocks import Conv, Bottleneck


TYPE = None # ZeroDNAS_Egor or DNAS or ZeroCost

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
    
    def init_arch_parameter(self):
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
        self.block_name = f'{self.__class__.__name__}_num{n}_gamma{e}'
        ########################################################################################
        
        
        # self.masks = (num_of_mask, max_channel)
        self.masks = torch.zeros((len(gamma_space), max(channel_values)))
        for idx, num_channel in enumerate(channel_values):
            self.masks[idx, :num_channel] = 1
        self.masks = torch.nn.parameter.Parameter(self.masks, requires_grad=False)
            
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
                for i in range(len(self.masks)):
                    mask += self.masks[i] *  args['gamma'][i]
                mask = mask.reshape(1,-1,1,1)
        
            if 'n_bottlenecks' in args.keys() and args['n_bottlenecks'] is not None:
                depth_vals = self.search_space['n_bottlenecks'] # args['n_bottlenecks_val']
                depth_dist = args['n_bottlenecks']
                
                m_out = self.cv1(x) * mask
                aggregate_feature = 0
                depth_idx = 0
                for depth in range(len(self.m) + 1):
                    if depth == depth_vals[depth_idx]:
                        aggregate_feature += m_out * depth_dist[depth_idx]
                        depth_idx += 1
                    if depth == depth_vals[-1]: break
                    m_out = self.m[depth](m_out) * mask
                m_out = aggregate_feature
            else:
                m_out = self.cv1(x)   * mask
                for m in self.m:
                    m_out = m(m_out) * mask
        else:
            m_out = self.cv1(x)   * mask
            m_out = self.m(m_out) * mask
        
        y1 = self.cv3(m_out) * mask
        y2 = self.cv2(x) * mask
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    def init_arch_parameter(self):
        arch = super(BottleneckCSP_Search, self).init_arch_parameter()
        for key, candidates in self.search_space.items():
            length    = len(candidates)
            arch[key] = torch.nn.Parameter(torch.ones((length, )) / length)
        return arch
    
class BottleneckCSP2_Search(GeneralOpeartor_Search):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, gamma_space, bottleneck_space, shortcut=True, g=1, bn='batchnorm', act='mish'): #bn='groupnorm', act='relu'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2_Search, self).__init__()
        # self.gamma_space      = gamma_space
        # self.bottleneck_space = bottleneck_space
        self.search_space['gamma']      = gamma_space
        self.search_space['n_bottlenecks'] = bottleneck_space
        
        e=max(gamma_space)
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
        self.block_name = f'{self.__class__.__name__}_num{n}_gamma{e}'
        ########################################################################################
        
        # self.masks = (num_of_mask, max_channel)
        self.masks = torch.zeros((len(gamma_space), max(channel_values)))
        for idx, num_channel in enumerate(channel_values):
            self.masks[idx, :num_channel] = 1
        self.masks = torch.nn.parameter.Parameter(self.masks, requires_grad=False)

    def forward(self, x, n_bottlenecks=None, args=None):
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
                for i in range(len(self.masks)):
                    mask += self.masks[i] * args['gamma'][i]
                mask = mask.reshape(1,-1,1,1)
        
            if 'n_bottlenecks' in args.keys() and args['n_bottlenecks'] is not None:
                depth_vals = self.search_space['n_bottlenecks'] # args['n_bottlenecks_val']
                depth_dist = args['n_bottlenecks']
                
                x1 = m_out = self.cv1(x) * mask
                aggregation = 0
                depth_idx = 0
                for depth in range(len(self.m) + 1):
                    if depth == depth_vals[depth_idx]:
                        aggregation += m_out * depth_dist[depth_idx]
                        depth_idx += 1
                    if depth == depth_vals[-1]: break
                    m_out = self.m[depth](m_out) * mask
                m_out = aggregation
            else:
                x1 = m_out = self.cv1(x) * mask
                for m in self.m:
                    m_out = m(m_out) * mask
        else:
            x1 = m_out = self.cv1(x) * mask
            m_out = self.m(m_out) * mask
            
        y1 = m_out
        y2 = self.cv2(x1) * mask
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))
    
        
        # if self.search_type == 0 or self.search_type == 1:
        #     x1 = m_out = self.cv1(x) * mask
            
        #     for m in self.m[:n_bottlenecks]:
        #         m_out = m(m_out) * mask
                
        #     y1 = m_out
        #     y2 = self.cv2(x1) * mask
        #     return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    def init_arch_parameter(self):
        arch = super(BottleneckCSP2_Search, self).init_arch_parameter()
        for key, candidates in self.search_space.items():
            length    = len(candidates)
            arch[key] = torch.nn.Parameter(torch.ones((length, )) / length)
        return arch

class Composite_Search(GeneralOpeartor_Search):
    def __init__(self, operators):
        super(Composite_Search, self).__init__()
        self.search_space = {'operator_choice' : list(range(len(operators)))}
        self.operators = operators
        self.block_name = f'{self.__class__.__name__}_op{len(operators)}'
        
    def forawrd(self, x, args_list):
        out = 0
        for i, (block, args) in enumerate(zip(self.operators, args_list)):
            out += block(x, args)
            
        return out
    
    def init_arch_parameter(self):
        arch = super(BottleneckCSP2_Search).init_arch_parameter()
        arch['operators'] = []
        for operator in self.operators:
            if 'Search' in operator.__class__.__name__:
                arch['operators'].append(operator.init_arch_parameter())
            else:
                arch['operators'].append(None)

        
        length = len(self.operators)
        arch['operators_choice'] = torch.nn.Parameter(torch.ones((length,)) / length)
        return arch
