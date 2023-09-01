import numpy as np
import torch
import torchvision
from torch import nn
from lib.utils.util import *
from lib.utils.general import compute_loss, is_parallel
from lib.models.blocks.yolo_blocks import *
from lib.models.blocks.yolo_blocks_search import *


"""
Get the average gradient of every N data batches
"""
def get_mean_grad(grad_dict, grad, step_iter=1):
    if step_iter==1:
        for i, modname in enumerate(grad.keys()):
            # grad_dict[modname] = np.array(grad[modname]) / 8
            grad_dict[modname] = [np.mean(np.abs(np.array(grad[modname])), axis=0)]
    else:
        for i, modname in enumerate(grad.keys()):
            # grad_dict[modname].append(np.array(grad[modname]) / 8)
            grad_dict[modname].append(np.mean(np.abs(np.array(grad[modname])), axis=0))
    return grad_dict

"""
Get the gradient of one data batch
"""
def getgrad(model, grad_dict, step_iter=0):
    if step_iter==0:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                grad_dict[name]=[module.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                grad_dict[name].append(module.weight.grad.data.cpu().reshape(-1).numpy())
    return grad_dict

"""
Calculate zico score
"""
def zico(grad_dict):
    allgrad_array=None
    # Change to numpy array
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0] # return idx of non zero value
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))
    # print(nsr_mean_sum_abs)
    return nsr_mean_sum_abs

def calculate_zico(model, arch_prob, inputs, targets, opt=None):
    # model: model, arch_prob: architecture, inputs: images, targets: labels, opt: optimizer
    is_ddp = is_parallel(model)
    model = model.module if is_ddp else model
    DEBUG = False

    # print(len(inputs))
    # print(inputs[0].size())
    ##################################
    # Calculate different gradient across data batch
    ##################################    
    grad_dict = {}
    grad = {}
    # model.train()
    # model.cuda()
    for i in range(len(inputs)): # for each data batch -> batch=2, size=8 or 128 -> size=16 * 8
        pred     = model(inputs[i], arch_prob)
        det_loss, loss_items = compute_loss(pred[1], targets[i], model)  # scaled by batch_size
        det_loss = det_loss[0]
        model.zero_grad()
        det_loss.backward()
        grad = getgrad(model, grad, i)
        
        # Average each N batch to match size=128 (256)
        # if (i+1)%16 == 0:
        #     grad_dict= get_mean_grad(grad_dict, grad, (i+1)//16) # Start from 1
        
    cost = zico(grad)
    # cost = zico(grad_dict)
    # print(f"zico score: {cost}")
    return cost
