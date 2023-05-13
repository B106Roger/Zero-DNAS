import torch
import torchvision
import torch.nn.functional as F
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.models.blocks.yolo_blocks import *


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


# def snip_forward_conv2d(self, x):
#         return F.conv2d(x, self.weight * self.weight_mask, self.bias,
#                         self.stride, self.padding, self.dilation, self.groups)

# def snip_forward_linear(self, x):
#         return F.linear(x, self.weight * self.weight_mask, self.bias)

# @measure('snip', bn=True, mode='param')
# def compute_snip_per_weight(net, inputs, targets, mode, loss_fn, split_data=1):
    # for layer in net.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
    #         layer.weight.requires_grad = False

    #     # Override the forward methods:
    #     if isinstance(layer, nn.Conv2d):
    #         layer.forward = types.MethodType(snip_forward_conv2d, layer)

    #     if isinstance(layer, nn.Linear):
    #         layer.forward = types.MethodType(snip_forward_linear, layer)

    # # Compute gradients (but don't apply them)
    # net.zero_grad()
    # N = inputs.shape[0]
    # for sp in range(split_data):
    #     st=sp*N//split_data
    #     en=(sp+1)*N//split_data
    
    #     outputs = net.forward(inputs[st:en])
    #     loss = loss_fn(outputs, targets[st:en])
    #     loss.backward()

    # # select the gradients that we want to use for search/prune
    # def snip(layer):
    #     if layer.weight_mask.grad is not None:
    #         return torch.abs(layer.weight_mask.grad)
    #     else:
    #         return torch.zeros_like(layer.weight)
    
    # grads_abs = get_layer_metric_array(net, snip, mode)

    # return grads_abs