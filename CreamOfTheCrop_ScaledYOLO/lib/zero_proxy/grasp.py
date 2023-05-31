import torch
import torchvision
import torch.nn.functional as F
import torch.autograd as autograd
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.models.blocks.yolo_blocks import *
import time


# TODO
"""
GraSP: Gradient Signal Preservation
    grasp = param * -(H*param_grad)
    Hessian-gradient Product: H*param_grad
"""
def calculate_grasp(model, arch_prob, inputs, targets, opt=None):
    # start = time.time()
    is_ddp = is_parallel(model)
    model = model.module if is_ddp else model
    DEBUG = False

    # Get all applicable param
    """ Time = 0.002 """
    param = []
    # loop1 = time.time()
    for block_idx, layer in enumerate(model.blocks):
        chosen_block = layer    
        if ('Search' in layer.__class__.__name__):
            for n, p in chosen_block.named_parameters(): # n -> layer names, p -> parameters (theta)
                if 'mask' in n:
                    assert(p.grad is None)
                else:
                    param.append(p)
                    p.requires_grad_(True)
    # print(f"idx={block_idx}")
    # print(f"loop1 = {time.time() - loop1}")

    """ Calculate model detection loss """
    """ Time = 0.68 """
    # loop2 = time.time()
    # Forward/grad pass 1
    pred     = model(inputs, arch_prob)
    det_loss, loss_items = compute_loss(pred[1], targets, model)  # scaled by batch_size
    det_loss = det_loss[0]
    grad_w = autograd.grad(det_loss, param, create_graph=False, allow_unused=True)
    # print(len(grad_w))
    # Forward/grad pass 2
    pred     = model(inputs, arch_prob)
    det_loss, loss_items = compute_loss(pred[1], targets, model)  # scaled by batch_size
    det_loss = det_loss[0]
    grad_f = autograd.grad(det_loss, param, create_graph=True, allow_unused=True)
    # print(len(grad_f))
    ##################################
    # Discard Gradient 
    ##################################
    model.zero_grad()
    if opt is not None: opt.zero_grad()
    # print(f"loop2 = {time.time() - loop2}")

    # Accumulate gradients computed in previous step and call backwards
    """ Time = 0.032 """
    z, count = 0, 0
    # loop3 = time.time()
    for block_idx, layer in enumerate(model.blocks):
        chosen_block = layer    
        if ('Search' in layer.__class__.__name__):
            for n, p in chosen_block.named_parameters(): # n -> layer names, p -> parameters (theta)
                if 'mask' in n:
                    pass
                else:
                    if grad_w[count] is not None:
                        z += (grad_w[count] * grad_f[count]).sum()
                        count += 1
    # print(f"loop3 = {time.time() - loop3}")
    """ Time = 132 """
    z.backward()
    # print(f"z backward = {time.time() - loop3}")
     
    """ Calculate GraSP values """
    """ Time = 0.16 """
    grasp_value = 0
    # loop4 = time.time()
    for block_idx, layer in enumerate(model.blocks):
        chosen_block = layer
            
        if ('Search' in layer.__class__.__name__):
            for n, p in chosen_block.named_parameters(): # n -> layer names, p -> parameters (theta)
                # Masked layer -> don't need to calculate zero-cost
                if 'mask' in n:
                    assert(p.grad is None)
                else:
                    # print(f"p={p}, grad={p.grad}")
                    if p.grad is not None:
                        tmp_value = (-p.data * p.grad).sum() # -theta*Hg
                        grasp_value += tmp_value
                    if DEBUG:
                        if 'bn' not in n:
                            print(f'{n} : ', p.shape, p.sum(), tmp_value.detach().cpu(), )
    # print(f"loop4 = {time.time() - loop4}")
    ##################################
    # Discard Gradient 
    ##################################
    model.zero_grad()
    if opt is not None: opt.zero_grad()
    # print(f"time = {time.time()-start}")
    return grasp_value.detach().cpu().numpy()
