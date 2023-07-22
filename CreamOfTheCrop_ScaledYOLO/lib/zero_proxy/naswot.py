import torch
import torchvision
import torch.nn.functional as F
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.models.blocks.yolo_blocks import *
from lib.models.blocks.yolo_blocks_search import *

import numpy as np

PREPROCESSED=False
BATCH_SIZE=2
def preprocess_model(model):
    def init_forward_hook(module, inp, out):
        # print('counting hook', str(type(module)))
        # try:
        # if not module.visited_backwards:
        #     return
        model.K = 0.0
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        inp = inp[:BATCH_SIZE]
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
    
    def counting_forward_hook(module, inp, out):
        # print('counting hook', str(type(module)))
        # try:
        # if not module.visited_backwards:
        #     return
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        inp = inp[:BATCH_SIZE]
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()


    block_num = len(model.blocks)
    block_id  = 0
    first_hook = True
    white_list = [BottleneckCSP_Search, BottleneckCSP2_Search, Conv, Bottleneck, SPPCSP]
    for module_idx, (name, module) in enumerate(model.named_modules()):
        # print(str(type(module)), name)
        # if 'Mish' in str(type(module)):
        # if 'Search' in str(type(module)):
        if name == f'blocks.{block_id}':
            if module.__class__ in white_list:
                print('register hook', name, str(type(module)))
                module.visited_backwards = False
                if first_hook:
                    module.register_forward_hook(init_forward_hook)
                    # module.register_forward_hook(create_init_hook(name))
                    first_hook = False
                else:
                    module.register_forward_hook(counting_forward_hook)
                    # module.register_forward_hook(create_hook(name))
            block_id += 1

def calculate_wot(model, arch_prob, inputs, targets, opt=None):
    global PREPROCESSED
    if not PREPROCESSED: 
        preprocess_model(model)
        PREPROCESSED = True
    
    is_ddp = is_parallel(model)
    model = model.module if is_ddp else model
    
    pred     = model(inputs, arch_prob)
    
    s, ld = np.linalg.slogdet(model.K)
    wot_value = ld
    return wot_value


PREPROCESSED_BLOCK=False
BATCH_SIZE=2
def preprocess_block(model):
    def forward_hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        out = out.view(out.size(0), -1)
        out = out[:BATCH_SIZE]
        x = (out > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        model.K += K.cpu().numpy() + K2.cpu().numpy()
        # print('[Forward Hook]', type(module), model.K.flatten())
        
    def search_forward_hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        out = out.view(out.size(0), -1)
        out = out[:BATCH_SIZE]
        x = (out > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        module.tmp_K = K.cpu().numpy() + K2.cpu().numpy()
        # print('[Forward Hook]',type(module), module.tmp_K.flatten() + model.K.flatten(), out.mean())
        # print('[Forward Hook]',type(module), end=' ')
        # for val in (module.tmp_K.flatten() + model.K.flatten()):
        #     print(val, end=' ')
        # print(out.mean())

    block_num = len(model.blocks)
    block_id  = 0
    first_hook = True
    white_list = [BottleneckCSP_Search, BottleneckCSP2_Search, Conv, Bottleneck, SPPCSP, ELAN_Search,ELAN2_Search]
    for module_idx, (name, module) in enumerate(model.named_modules()):
        if name == f'blocks.{block_id}':
            if module.__class__ in white_list:
                print('register hook', name, str(type(module)))
                module.visited_backwards = False
                if 'Search' in module.__class__.__name__:
                    module.register_forward_hook(search_forward_hook)
                else:
                    module.register_forward_hook(forward_hook)
                    
            block_id += 1

def calculate_zero_cost_map(model, arch_prob, inputs, targets, opt=None):
    global PREPROCESSED_BLOCK
    if not PREPROCESSED_BLOCK: 
        preprocess_block(model)
        PREPROCESSED_BLOCK = True
    model.eval()
    model.K = 0.0
    zc_maps = []
    """
    x: input image. torch.float32(b, c, h, w)
    distributions: architecture distributions parameter. torch.float32()
    """
    distributions = arch_prob
    
    stage_idx = 0
    x = inputs
    y, dt = [], []  # outputs
    for idx, m in enumerate(model.blocks):
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        if   m.__class__ in [BottleneckCSP_Search, BottleneckCSP2_Search]:
            stage_args = distributions[stage_idx]
            #####################################################
            zc_map = {}
            options, search_keys = m.generate_options()
            for option in options:
                block_args = {}
                query_keys = []
                for key_idx, key in enumerate(search_keys):
                    option_index = option[key_idx]
                    option_value = m.search_space[key][option_index]
                    
                    # Block Args for search block
                    block_args[key] = torch.zeros((len(m.search_space[key]),))
                    block_args[key][option_index] = 1.0
                    
                    if opt == True:
                        query_keys.append(f'{key[0]}{option_value}')
                    elif opt is None:
                        query_keys.append(f'{key}{option_value}')
                query_key      = '-'.join(query_keys)
                
                # Calculate Each Zero-Cost FLOPS
                _ = m(x, args=block_args)
                s, ld = np.linalg.slogdet(m.tmp_K)
                # Note that the negative symbol is to make the wot score larger in negative side
                zc_map[query_key] = -ld
            #####################################################
            
            zc_maps.append(zc_map)
            x = m(x, args=stage_args)
            model.K += m.tmp_K
            stage_idx+=1
        elif m.__class__ in [ELAN_Search, ELAN2_Search]:
            stage_args = distributions[stage_idx]
            #####################################################
            zc_map = {}
            options, search_keys = m.generate_options()
            
            comb_list       = m._connection_combination(m.search_space['connection'])
            comb_list_index = m._connection_combination(m.search_space['connection'], index=True)
            for option in options:
                block_args = {}
                query_keys = []
                
                args_cn             = int(m.search_space['gamma'     ][option[search_keys.index('gamma')]] * m.base_cn)
                args_connection     = comb_list[option[search_keys.index('connection')]]
                args_connection_idx = comb_list_index[option[search_keys.index('connection')]]
                
                block_args['connection'] = torch.zeros((len(m.search_space['connection']),))
                
                for connection_idx in args_connection_idx:
                    block_args['connection'][connection_idx] = 1.
                
                block_args['gamma'] = torch.zeros((len(m.search_space['gamma']),))
                block_args['gamma'][option[search_keys.index('gamma')]] = 1.0

                query_key = f'cn{args_cn}-con{str(args_connection)}'
                # Calculate Each Zero-Cost FLOPS
                _ = m(x, args=block_args)
                s, ld = np.linalg.slogdet(m.tmp_K)
                # Note that the negative symbol is to make the wot score larger in negative side
                zc_map[query_key] = -ld
            #####################################################
            
            zc_maps.append(zc_map)
            x = m(x, args=stage_args)
            model.K += m.tmp_K
            stage_idx+=1
        
        else:
            x = m(x)  # run
                    
        y.append(x if m.i in model.save else None)  # save output
    return zc_maps