import torch
from tqdm import tqdm
import json

def synflow(layer):
    if layer.weight.grad is not None:
        return torch.abs(layer.weight * layer.weight.grad)
    else:
        return torch.zeros_like(layer.weight)

def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()

def sum_arr_tensor(arr):
    sum = 0.
    for i in range(len(arr)):
        sum = sum + torch.sum(arr[i])
    return sum

def compute_block_synflow(block, input, n_bottlenecks=None):
    
    signs = block.linearize()
    if n_bottlenecks is None:
        output = block.forward(input)
    else:
        output = block.forward(input, n_bottlenecks)
    block.last = torch.nn.Conv2d(output.shape[1], 32, 1, 1)
    torch.sum(block.last(output)).backward()
    metric = block.calculate_synflow_metric()
    block.zero_grad()
    block.last = None
    block.nonlinearize(signs)
    return metric/1e16
    

