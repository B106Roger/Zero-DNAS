import math

import torch
import torch.nn as nn
from mish_cuda import MishCuda as Mish
from lib.utils.synflow import synflow, sum_arr

TYPE = None # ZeroDNAS_Egor or DNAS or ZeroCost

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

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        if TYPE=='ZeroDNAS_Egor' or TYPE=='ZeroCost':
            self.bn = nn.GroupNorm(1, c2)
            self.act = nn.ReLU() if act else nn.Identity()
        elif TYPE=='DNAS':
            self.bn = nn.BatchNorm2d(c2)
            self.act = Mish() if act else nn.Identity()
        else:
            raise ValueError(f'Invalid Type: {TYPE}')
        self.block_name = f'cn_k{k}_s{s}'


    def get_block_name(self):
        return self.block_name

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class ConvNP(nn.Module):
    # Not Prunable
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvNP, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        if TYPE=='ZeroDNAS_Egor' or TYPE=='ZeroCost':
            self.bn = nn.GroupNorm(1, c2)
            self.act = nn.ReLU() if act else nn.Identity()
        elif TYPE=='DNAS':
            self.bn = nn.BatchNorm2d(c2)
            self.act = Mish() if act else nn.Identity()
        else:
            raise ValueError(f'Invalid Type: {TYPE}')
        self.block_name = f'cn_k{k}_s{s}'


    def get_block_name(self):
        return self.block_name

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.block_name = f'bottle'


    def get_block_name(self):
        return self.block_name

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension
        self.block_name = f'concat'

    def get_block_name(self):
        return self.block_name

    def forward(self, x):
        return torch.cat(x, self.d)

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # Modify from    https://github.com/chiahuilin0531/ScaledYOLOv4
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, gamma_space=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        e=max(gamma_space)
        channel_values = [int(gamma*c2) for gamma in gamma_space]
        self.n = n
        self.search_type = self.search_space_id
        if self.search_type == 0:
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
            self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
            self.cv4 = Conv(2 * c_, c2, 1, 1)
            if TYPE=='ZeroDNAS_Egor' or TYPE=='ZeroCost':
                self.bn = nn.GroupNorm(1, 2 * c_)  # applied to cat(cv2, cv3)
                self.act = nn.ReLU()
            elif TYPE=='DNAS':
                self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
                self.act = Mish()
            else:
                raise ValueError(f'Invalid Type: {TYPE}')
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        elif self.search_type == 1:
            c_ = int(c2)             # hidden channels
            c_h = int(c2 * e)        # hidden channels. extract deep feature.
            c_s = int(c2 * (1 - e))  # shown  channels.
            self.cv1 = Conv(c1, c_h, 1, 1)
            self.cv2 = nn.Conv2d(c1, c_s, 1, 1, bias=False)
            self.cv3 = nn.Conv2d(c_h, c_h, 1, 1, bias=False)
            self.cv4 = Conv(c2, c2, 1, 1)
            if TYPE=='ZeroDNAS_Egor' or TYPE=='ZeroCost':
                self.bn = nn.GroupNorm(1, c2)  # applied to cat(cv2, cv3)
                self.act = nn.ReLU()
            elif TYPE=='DNAS':
                self.bn = nn.BatchNorm2d(c2)  # applied to cat(cv2, cv3)
                self.act = Mish()
            else:
                raise ValueError(f'Invalid Type: {TYPE}')
        
            self.m = nn.Sequential(*[Bottleneck(c_h, c_h, shortcut, g, e=1.0) for _ in range(n)])
        self.block_name = f'bottlecsp_num{n}_gamma{e}'
        
        # self.masks = (num_of_mask, max_channel)
        self.masks = torch.zeros((len(gamma_space), max(channel_values)))
        for idx, num_channel in enumerate(channel_values):
            self.masks[idx, :num_channel] = 1
        self.masks = torch.nn.parameter.Parameter(self.masks, requires_grad=False)
            

    @classmethod
    def set_search_space(cls, search_space_id):
        cls.search_space_id = search_space_id

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

    def forward(self, x, n_bottlenecks=None, args=None):
        """
        Parameters
        ----------
        x: float32, (b, c, w, h)
        n_bottlenecks: integer
        args: dict
            'gamma_dist': float32, (num_of_gamma_choices,)
        """
        mask = 1.0
        if args is not None:
            if 'gamma_dist' in args.keys() and args['gamma_dist'] is not None:
                # mask = self.masks * args['gamma_dist']
                mask = 0.0
                for i in range(len(self.masks)):
                    mask += self.masks[i] *  args['gamma_dist'][i]
                mask = mask.reshape(1,-1,1,1)
        if (n_bottlenecks == None): n_bottlenecks = len(self.m)

        m_out = self.cv1(x) * mask
        for m in self.m[:n_bottlenecks]:
            m_out = m(m_out) * mask
        y1 = self.cv3(m_out) * mask
        y2 = self.cv2(x) * mask
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    # def forward(self, x):
    #     y1 = self.cv3(self.m(self.cv1(x)))
    #     y2 = self.cv2(x)
    #     return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        self.block_name = f'c3_num{n}_gamma{e}'


    def get_block_name(self):
        return self.block_name
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Upsample(nn.Module):
    def __init__(self, size, scale_factor, mode):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode)
        self.block_name = f'up_{size}_{mode}'

    def get_block_name(self):
        return self.block_name

    def forward(self, x):
        return self.upsample(x)


class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, gamma_space=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        e=max(gamma_space)
        channel_values = [int((gamma+0.5)*c2) for gamma in gamma_space]
        self.n = n
        self.search_type = self.search_space_id
        if self.search_type == 0:
            c_ = int(c2)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            if TYPE=='ZeroDNAS_Egor' or TYPE=='ZeroCost':
                self.bn = nn.GroupNorm(1, 2 * c_) 
                self.act = nn.ReLU()
            elif TYPE=='DNAS':
                self.bn = nn.BatchNorm2d(2 * c_) 
                self.act = Mish()
            else:
                raise ValueError(f'Invalid Type: {TYPE}')
        
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        elif self.search_type == 1:
            c_ = int(c2 * (e + 0.5))  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            if TYPE=='ZeroDNAS_Egor' or TYPE=='ZeroCost':
                self.bn = nn.GroupNorm(1, 2 * c_) 
                self.act = nn.ReLU()
            elif TYPE=='DNAS':
                self.bn = nn.BatchNorm2d(2 * c_) 
                self.act = Mish()
            else:
                raise ValueError(f'Invalid Type: {TYPE}')

            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        elif self.search_type == 2:
            c_  = int(c2)
            c_h = int(c_ * (0.5 + e))  # hidden channels
            c_s = int(c_ * (1.5 - e))  # shown  channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = nn.Conv2d(c_, c_s, 1, 1, bias=False)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            self.bn = nn.GroupNorm(1, 2 * c_) 
            self.act = nn.ReLU()
            self.pre_m = Conv(c_, c_h, 1, 1)
            self.m = nn.Sequential(*[Bottleneck(c_h, c_h, shortcut, g, e=1.0) for _ in range(n)])
        elif self.search_type == 3:
            c_  = int(c2)
            c_h = self.c_h = int(c_ * (0.5 + e))  # hidden channels
            c_s = self.c_s = int(c_ * (1.5 - e))  # shown  channels
            self.cv1 = Conv(c1, c_ * 2, 1, 1)
            self.cv2 = nn.Conv2d(c_s, c_s, 1, 1, bias=False)
            self.cv3 = Conv(2 * c_, c2, 1, 1)
            self.bn = nn.GroupNorm(1, 2 * c_) 
            self.act = nn.ReLU()
            self.m = nn.Sequential(*[Bottleneck(c_h, c_h, shortcut, g, e=1.0) for _ in range(n)])

        self.block_name = f'bottlecsp2_num{n}_gamma{e}'
        
        # self.masks = (num_of_mask, max_channel)
        self.masks = torch.zeros((len(gamma_space), max(channel_values)))
        for idx, num_channel in enumerate(channel_values):
            self.masks[idx, :num_channel] = 1
        self.masks = torch.nn.parameter.Parameter(self.masks, requires_grad=False)

    @classmethod
    def set_search_space(cls, search_space_id):
        cls.search_space_id = search_space_id

    def get_block_name(self):
        return self.block_name

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

    def forward(self, x, n_bottlenecks=None, args=None):
        """
        Parameters
        ----------
        x: float32, (b, c, w, h)
        n_bottlenecks: integer
        args: dict
            'gamma_dist': float32, (num_of_gamma_choices,)
        """
        mask = 1.0
        if args is not None:
            if 'gamma_dist' in args.keys() and args['gamma_dist'] is not None:
                # mask = self.masks * args['gamma_dist']
                mask = 0.0
                for i in range(len(self.masks)):
                    mask += self.masks[i] *  args['gamma_dist'][i]
                # mask = mask.unsqueeze(0)
                mask = mask.reshape(1,-1,1,1)
        if (n_bottlenecks == None): n_bottlenecks = len(self.m)
        
        if self.search_type == 0 or self.search_type == 1:
            x1 = m_out = self.cv1(x) * mask
            
            for m in self.m[:n_bottlenecks]:
                m_out = m(m_out) * mask
                
            y1 = m_out
            y2 = self.cv2(x1) * mask
            return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        elif self.search_type == 2:
            x1 = self.cv1(x)
            if (n_bottlenecks == None):
                y1 = self.m(self.pre_m(x1))
            else:
                y1 = self.m[:n_bottlenecks](self.pre_m(x1))
            y2 = self.cv2(x1)
            return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))    
        elif self.search_type == 3:
            x1 = self.cv1(x)
            x11 = x1[:, :self.c_h]
            x12 = x1[:, self.c_h:]
            
            if (n_bottlenecks == None):
                y1 = self.m(x11)
            else:
                y1 = self.m[:n_bottlenecks](x11)
                
            y2 = self.cv2(x12)
            return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        
        
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.block_name = f'spp'


    def get_block_name(self):
        return self.block_name

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))



class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        if TYPE=='ZeroDNAS_Egor' or TYPE=='ZeroCost':
            self.bn = nn.GroupNorm(1, 2 * c_) 
            self.act = nn.ReLU()
        elif TYPE=='DNAS':
            self.bn = nn.BatchNorm2d(2 * c_) 
            self.act = Mish()
        else:
            raise ValueError(f'Invalid Type: {TYPE}')

        self.cv7 = Conv(2 * c_, c2, 1, 1)
        self.block_name = f'sppcsp'


    def get_block_name(self):
        return self.block_name

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.export = False  # onnx export
        self.block_name = f'detect'


    def get_block_name(self):
        return self.block_name

    def forward(self, x, first_run=False, calc_metric=False):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not first_run:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not calc_metric:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        
        return x if first_run else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()