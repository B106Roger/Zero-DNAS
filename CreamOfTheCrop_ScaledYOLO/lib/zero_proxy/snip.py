import torch
import torchvision
import torch.nn.functional as F
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.models.blocks.yolo_blocks import *

#############################################################################################
# Reference                                                                                 #
# https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/snip.py #
#############################################################################################
# Note                                                                                      #
# Only Calculate the SNIP value for convolution weights(not inlcude bias)                   #
#############################################################################################

def calculate_snip(model, arch_prob, inputs, targets, opt=None):
    is_ddp = is_parallel(model)
    model = model.module if is_ddp else model
    DEBUG = False
    
    pred     = model(inputs, arch_prob)
    
    det_loss, loss_items = compute_loss(pred[1], targets, model)  # scaled by batch_size
    det_loss = det_loss[0]

    ##################################
    # Discard Gradient 
    ##################################
    model.zero_grad()
    if opt is not None: opt.zero_grad()
    
    det_loss.backward()

    keys = sorted(model.search_space.keys())
     
    snip_value = 0
    for block_idx, layer in enumerate(model.blocks):
            
        if ('Search' in layer.__class__.__name__):
            for n, p in layer.named_parameters():
                if 'mask' in n:
                    assert(p.grad is None)
                else:
                    tmp_value = (p * p.grad).abs().sum()
                    snip_value += tmp_value
                    if DEBUG:
                        if 'bn' not in n:
                            print(f'{n} : ', p.shape, p.sum(), tmp_value.detach().cpu(), )
                        

    ##################################
    # Discard Gradient 
    ##################################
    model.zero_grad()
    if opt is not None: opt.zero_grad()
    
    return snip_value.detach().cpu().numpy()

def calculate_zero_cost_map(model, arch_prob, inputs, targets=None, short_name=None):
    # print('snip','short_name', short_name)
    model.eval()
    zc_maps = []
    distributions = arch_prob

    ##################################
    # Discard Gradient 
    ##################################
    model.zero_grad()
    
    ##################################
    # Calculate Gradient
    ##################################
    pred     = model(inputs, arch_prob)
    det_loss, loss_items = compute_loss(pred[1], targets, model)  # scaled by batch_size
    print('loss_items', loss_items)
    det_loss = det_loss[0]
    det_loss.backward()

    stage_idx = 0
    x = inputs
    y, dt = [], []  # outputs
    for idx, m in enumerate(model.blocks):
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
                    
                    if short_name == True:
                        query_keys.append(f'{key[0]}{option_value}')
                    elif short_name is None:
                        query_keys.append(f'{key}{option_value}')
                query_key      = '-'.join(query_keys)
                
                # Calculate Each Zero-Cost FLOPS
                used_params, mask_funcs = m.get_masked_params(block_args)
                snip_value = []
                for p_idx, (p, m_func) in enumerate(zip(used_params, mask_funcs)):
                    mask_p = m_func(p.grad * p).detach()
                    snip_value.append(mask_p.abs().sum())
                            
                zc_value = sum(snip_value)
                # Note that the negative symbol is to make the snip score larger in negative side
                zc_map[query_key] = -zc_value.cpu()
            #####################################################
            
            zc_maps.append(zc_map)
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
                block_args['connection'][args_connection_idx] = 1.0
                # for connection_idx in args_connection_idx:
                #     block_args['connection'][connection_idx] = 1.
                
                block_args['gamma'] = torch.zeros((len(m.search_space['gamma']),))
                block_args['gamma'][option[search_keys.index('gamma')]] = 1.0

                query_key = f'cn{args_cn}-con{str(args_connection)}'
                zc_value = 0.0
                
                # Note that the negative symbol is to make the wot score larger in negative side
                zc_map[query_key] = zc_value
            #####################################################
            stage_idx+=1

    ##################################
    # Discard Gradient 
    ##################################
    model.zero_grad()
    
    return zc_maps