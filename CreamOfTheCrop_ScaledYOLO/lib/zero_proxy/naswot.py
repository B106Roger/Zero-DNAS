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

def calculate_zero_cost_map(model, arch_prob, inputs, targets, opt=None):
    zc_map = {}
    return zc_map