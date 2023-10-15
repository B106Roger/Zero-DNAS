# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import sys
import argparse
import logging
import torch.nn as nn
import torch
import math
import yaml
import os
import copy
from torch import optim as optim
from thop import profile, clever_format

from timm.utils import *
from lib.models.blocks.yolo_blocks_search import *
from lib.models.blocks.yolo_blocks import *

from lib.config import cfg
from lib.utils.general import non_max_suppression, clip_coords, xywh2xyxy, box_iou, ap_per_class


def get_path_acc(model, path, val_loader, args, val_iters=50):
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            if batch_idx >= val_iters:
                break
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            output = model(input, path)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(
                    0,
                    reduce_factor,
                    reduce_factor).mean(
                    dim=2)
                target = target[0:target.size(0):reduce_factor]

            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            torch.cuda.synchronize()

            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

    return (prec1_m.avg, prec5_m.avg)


def get_logger(file_path):
    """ Make python logger """
    log_format = '%(asctime)s | %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger('')

    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def add_weight_decay_supernet(model, args, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    meta_layer_no_decay = []
    meta_layer_decay = []
    other=[]
    total_param = 0
    for name, param in model.named_parameters():
        
        if not param.requires_grad or 'thetas' in name:
            continue  # frozen weights
        total_param += param.numel()
        if '.bias' in name:
            no_decay.append(param)
        elif '.weight' in name and '.bn' not in name:
            decay.append(param)
        else:
            other.append(param)
        # if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
        #     no_decay.append(param)
        # else:
        #     decay.append(param)
    print(f'decay: {len(decay)}')
    print(f'no_decay: {len(no_decay)}')
    print(f'other: {len(other)}')
    
    
    print(f'args.lr: { args.lr}')
    print(f'total_param: {total_param/1e6:.2f}M      Exact Value: {total_param}')
    
    
    
    return [
        {'params': other,                                       'lr': args.lr},
        {'params': decay,       'weight_decay': weight_decay,   'lr': args.lr},
        {'params': no_decay,    'weight_decay': weight_decay,   'lr': args.lr},
        # {'params': other,                                       'lr': args.lr},
    ]


def compute_map(inf_out, targets, height, width, nc, conf_thres=0.001, iou_thres=0.5):
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    device = targets.device
    seen = 0
    whwh = torch.Tensor([width, height, width, height]).to(device)
    stats, ap, ap_class = [], [], []
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=False)
    for si, pred in enumerate(output):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        seen += 1

        if pred is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Clip boxes to image bounds
        clip_coords(pred, (height, width))
    
        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5]) * whwh

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
   
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    return map50

def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False

def create_optimizer_supernet(args, model, has_apex, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    print(f'create_optimizer_supernet || opt_lower: {opt_lower} || weight_decay: {args.weight_decay} || momentum: {args.momentum}')
    thetas_lr = args.theta_optimizer.LR
    thetas_weight_decay = args.theta_optimizer.WEIGHT_DECAY
    # thetas_params = model.thetas_main.parameters()
    thetas_params = model.get_optimizer_parameter()
    print('thetas_params', thetas_params)
    
    # print(args)

    weight_decay = args.weight_decay
    # print('LR:', args.lr)
    # print('WEIGHT_DECAY:', weight_decay)
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        weight_decay /= args.lr
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay_supernet(model, args, weight_decay)
        weight_decay = 0.
    
    
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(
        ), 'APEX and CUDA required for fused optimizers'

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    theta_optimizer = torch.optim.Adam(params=thetas_params,
                                       lr=thetas_lr,
                                       weight_decay=thetas_weight_decay)
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        # print('Model Optimizer Here !!', parameters)
        optimizer = optim.SGD(
            parameters,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov=True)
        print(f'ck momentum: {args.momentum} weight_decay: {weight_decay} ')
    elif opt_lower == 'momentum':
        optimizer = optim.SGD(
            parameters,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov=False)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, weight_decay=weight_decay, eps=args.opt_eps)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer, theta_optimizer


def convert_lowercase(cfg):
    keys = cfg.keys()
    lowercase_keys = [key.lower() for key in keys]
    values = [cfg.get(key) for key in keys]
    for lowercase_key, value in zip(lowercase_keys, values):
        cfg.setdefault(lowercase_key, value)
    return cfg


def get_model_flops_params(model, input_size=(1, 3, 224, 224)):
    input = torch.randn(input_size)
    macs, params = profile(deepcopy(model), inputs=(input,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def create_supernet_scheduler(cfg, optimizer):
    if cfg.EPOCHS == 0:
        lf = lambda x: x
    else:
        lf = lambda x: (((1 + math.cos(x * math.pi / cfg.EPOCHS)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # ITERS = cfg.EPOCHS * \
    #     (1280000 / (cfg.NUM_GPU * cfg.DATASET.BATCH_SIZE))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
    #     cfg.LR - step / ITERS) if step <= ITERS else 0, last_epoch=-1)
    return scheduler, cfg.EPOCHS


def write_thetas(output_dir, thetas, epoch, temperature):
    alpha_distributions = stringify_theta(thetas, normalize=False)
    beta_distributions  = stringify_theta(thetas, normalize=True, temperature=temperature)
    
    with open(os.path.join(output_dir, 'alpha_distribution.txt'), 'a') as f:
        f.write(f'epoch: {epoch}')
        f.writelines(str(alpha_distributions)+'\n')
    with open(os.path.join(output_dir, 'beta_distribution.txt'), 'a') as f:
        f.write(f'epoch: {epoch}')
        f.writelines(str(beta_distributions)+'\n')


def stringify_theta(thetas, normalize=False, temperature=1.):
    """
    Convert Tensor to Numpy value in Architecture Parameter
    
    Parameter
    ---------
    normalize  : bool,  determine whether to use softmax to normalize or not
    temperature: float, determine the temperature when using softmax normalize the architecture
    
    Return
    ------
    list of archtiecture value
    """
    if normalize:
        softmax_func = lambda x: torch.nn.functional.softmax(x / temperature).detach().cpu().numpy()
        sigmoid_func = lambda x: torch.nn.functional.sigmoid(x / temperature).detach().cpu().numpy()
    else:
        softmax_func = sigmoid_func = lambda x: x.detach().cpu().numpy()
    
    
    write_arch = []
    for arch in thetas:
        tmp_arch = { 'block_name': arch['block_name'] }
        if arch['block_name'] == 'Composite_Search':
            tmp_arch['operators_choice'] = normalize_func(arch['operators_choice'])
            tmp_arch['operators'] = stringify_theta(arch['operators'], normalize=normalize, temperature=temperature)
        else:
            for key, value in arch.items():
                if key == 'block_name': pass
                elif key == 'connection': 
                    tmp_arch[key] = sigmoid_func(arch[key])
                else:                     tmp_arch[key] = softmax_func(arch[key])
                    
        write_arch.append(tmp_arch)
    return write_arch


def thetas_to_archtecture(thetas, model):
    """
    thetas: only accepct normalize thetas
    """
    export_thetas = copy.deepcopy(thetas)
    theta_idx = 0
    for depth, m in enumerate(model.blocks):
        if 'Search' in m.__class__.__name__:
            if m.__class__ in [Composite_Search]:
                best_option_idx = thetas[theta_idx]['operator_choice'].argmax().cpu().numpy()
                export_thetas[theta_idx] = thetas[theta_idx]['operators'][best_option_idx]
                ##############################################
                m = m.operators[best_option_idx]
                if m.__class__ in [BottleneckCSP, BottleneckCSP2, ELAN_Search, ELAN2_Search]:
                    search_space = m.search_space
                    for key in search_space.keys():
                        best_option_idx = thetas[theta_idx][key].argmax().cpu().numpy()
                        export_thetas[theta_idx][key] = search_space[key][best_option_idx]
            elif m.__class__ in [BottleneckCSP_Search, BottleneckCSP2_Search]:
                search_space = m.search_space
                for key in search_space.keys():
                    best_option_idx = thetas[theta_idx][key].argmax().cpu().numpy()
                    export_thetas[theta_idx][key] = search_space[key][best_option_idx]
            elif m.__class__ in [ELAN_Search, ELAN2_Search]:
                search_space = m.search_space
                print(f'thetas[{theta_idx}]={thetas[theta_idx]}')
                for key in search_space.keys():
                    if key == 'connection':
                        con_len = len(search_space['connection'])
                        best_con_idx = torch.arange(con_len)[thetas[theta_idx][key] > 0.5].numpy().tolist()
                        best_con = np.array(search_space['connection'])[best_con_idx].tolist()
                        export_thetas[theta_idx][key] = best_con
                    else:
                        best_option_idx = thetas[theta_idx][key].argmax().cpu().numpy()
                        export_thetas[theta_idx][key] = search_space[key][best_option_idx]
                    
            theta_idx += 1
    return export_thetas

def export_thetas(thetas, model, origin_config, output_file):
    """
    thetas: only accept normalized thetas, or the program would go wrong 
    """
    export_config = copy.deepcopy(origin_config)
    discrete_thetas = thetas_to_archtecture(thetas, model)
    # Config For ScaledYOLOv4
    gamma_list = []
    theta_idx = 0
    for part in ['backbone', 'head']:
        for depth in range(len(export_config[part])):
            f,n,m,arg = export_config[part][depth]
            m = eval(m)
            if m in [BottleneckCSP_Search, BottleneckCSP2_Search]:
                # Process Block Name
                export_config[part][depth][2] = export_config[part][depth][2].replace('_Search','')
                # Process Block Depth
                if 'n_bottlenecks' in discrete_thetas[theta_idx].keys():
                    export_config[part][depth][1] = discrete_thetas[theta_idx]['n_bottlenecks']
                # Process Block Gamma
                if 'gamma' in discrete_thetas[theta_idx].keys():
                    gamma_list.append(discrete_thetas[theta_idx]['gamma'])
                theta_idx += 1
            
            if m in [ELAN_Search, ELAN2_Search]:
                # Process Block Name
                export_config[part][depth][2] = export_config[part][depth][2].replace('_Search','')
                # Process Block Depth
                if 'connection' in discrete_thetas[theta_idx].keys():
                    export_config[part][depth][3].append(discrete_thetas[theta_idx]['connection'])
                # Process Block Gamma
                if 'gamma' in discrete_thetas[theta_idx].keys():
                    gamma_list.append(discrete_thetas[theta_idx]['gamma'])
                    base_cn = export_config[part][depth][3][1]
                    export_config[part][depth][3][1] = int(discrete_thetas[theta_idx]['gamma'] * base_cn)
                theta_idx += 1
                
            if m in [Detect, IDetect]:
                arg.pop()
    
    export_config['csp_gammas'] = gamma_list
    
    with open(output_file, 'w') as f:
        documents = yaml.dump(export_config, default_flow_style=True, sort_keys=False)
        documents = documents.replace('Upsample', 'nn.Upsample')
        f.write(documents)
    return export_config