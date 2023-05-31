import torch
import torchvision
import torch.nn.functional as F
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.models.blocks.yolo_blocks import *

PREPROCESSED=False
BATCH_SIZE=2
def preprocess_model(model):
    
    def init_forward_hook(module, inp, out):
        # print('init hook', str(type(module)))
        model.K = 0.0
        # exit()
    
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
        
    for module_idx, (name, module) in enumerate(model.named_modules()):
        # print(f'{module_idx:02d} {name:25s} {str(type(module))}')
        if 'Mish' in str(type(module)):
            print('register hook', name)
            module.visited_backwards = False
            module.register_forward_hook(counting_forward_hook)
        elif module_idx == 3:
            module.visited_backwards = False
            module.register_forward_hook(init_forward_hook)
            
def calculate_wot(model, arch_prob, inputs, targets, opt=None):
    global PREPROCESSED
    if not PREPROCESSED: 
        preprocess_model(model)
        PREPROCESSED = True
    
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
        chosen_block = layer
            
        if ('Search' in layer.__class__.__name__):
            for n, p in chosen_block.named_parameters():
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

