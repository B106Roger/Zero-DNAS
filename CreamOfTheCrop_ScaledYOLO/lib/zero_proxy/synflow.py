import torch
from tqdm import tqdm
import json


################################################################################################
# Reference                                                                                    #
# https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/synflow.py #
################################################################################################
# Note                                                                                         #
#                                                                                              #
################################################################################################


#convert params to their abs. Keep sign for converting it back.
@torch.no_grad()
def linearize(model):
    signs = {}
    for name, param in model.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs

#convert to orig values
@torch.no_grad()
def nonlinearize(model, signs):
    for name, param in model.state_dict().items():
        if 'weight_mask' not in name:
            param.mul_(signs[name])
                

def calculate_synflow(model, arch_prob, inputs, targets, _=None):

    device = inputs.device
    # is_linear = True
    DEBUG = False

    # keep signs of all params
    # if is_linear:
    #     signs = linearize(model)
    
    # Compute gradients with input of 1s 
    model.zero_grad()
    model.double()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = model.forward(inputs)
    output = output[1]
    res = 0
    for item in (output):
        if DEBUG:
            print(item.shape, item.detach().cpu().max(), item.detach().cpu().min())
        res += item.sum()
    torch.sum(res).backward() 

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        # if layer.weight.grad is not None:
        #     return torch.abs(layer.weight * layer.weight.grad)
        # else:
        #     return torch.zeros_like(layer.weight)
        synflow_value = 0
        for n, p in layer.named_parameters():
            if 'mask' in n:
                assert(p.grad is None)
            elif 'bn' not in n or DEBUG:
                tmp_value = (p * p.grad).abs()
                tmp_value2 = tmp_value.sum().detach().cpu()
                synflow_value += tmp_value2
                
                if DEBUG:
                    a,b = p.detach().cpu().min(), p.detach().cpu().max()
                    c,d = p.grad.detach().cpu().min(), p.grad.detach().cpu().max()
                    
                    ss = str(p.detach().cpu().numpy().shape)
                    print(f'P {layer.__class__.__name__:20s} {n:20s} : {ss:30s} {a:10.7e} {b:10.7e}', )
                    print(f'G {layer.__class__.__name__:20s} {n:20s} : {ss:30s} {c:10.7e} {d:10.7e}', )
                
        return synflow_value

    grads_abs = get_layer_metric_array(model, synflow)
    # print('model.blocks[-1].training', model.blocks[-1].training)
    # apply signs of all params
    # if is_linear:
    #     nonlinearize(model, signs)

    return torch.tensor(grads_abs).sum()

def get_layer_metric_array(net, metric): 
    metric_array = []

    for mod_id, layer in enumerate(net.blocks):
        # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        # if 'Search' in layer.__class__.__name__:
        metric_array.append(metric(layer))
    
    return metric_array