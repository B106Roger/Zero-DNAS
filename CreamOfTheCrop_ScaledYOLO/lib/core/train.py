# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

from pathlib import Path
import numpy as np
import time, os
import torchvision
import torch.nn.functional as F
import itertools
import random
from datetime import datetime
from tqdm import tqdm
from torch.cuda import amp
from scipy.special import softmax
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.utils.kd_utils import compute_loss_KD
from lib.utils.synflow import sum_arr_tensor
from lib.zero_proxy import snip
from lib.utils.general import random_testing


def get_device_info():
    print('--------------GPU Info -------------------')
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f'total    : {t/1e9:.2f}GB')
    print(f'free     : {f/1e9:.2f}GB')
    print(f'used     : {a/1e9:.2f}GB')

def mlc_loss(arch_param):
    y_pred_neg = arch_param
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    aux_loss = torch.mean(neg_loss)
    return aux_loss

##############################
# Train Zero-Cost Metrics
##############################
def train_epoch(epoch, model, loader, optimizer, loss_fn, prioritized_board, MetaMN, cfg, device, synflow_cache, task_flops, task_params, theta_optimizer=None,
                est=None, logger=None, lr_scheduler=None, saver=None,
                output_dir='', model_ema=None, local_rank=0, world_size=0, test_loader=None, arch_sampler=None, train_theta=False, wot_map=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    training_losses_m = AverageMeter()
    flops_losses_m = AverageMeter()
    overall_losses_m = AverageMeter()
    kd_losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    cache_hits = 0
    iterations = 2880

    end = time.time()
    last_idx = len(loader) - 1
    test_iter = itertools.cycle(test_loader)
    # for batch_idx, (input, target, paths, _) in enumerate(loader):
        # target = target.to(device)
        # last_batch = batch_idx == last_idx
        # data_time_m.update(time.time() - end)
        # input = input.to(device, non_blocking=True).float() / 255.0
        # test_input, test_target, _, _ = next(test_iter)
        # test_input = test_input.to(device, non_blocking=True).float() / 255.0
        # test_target = test_target.to(device)
        # nb, _, height, width = input.shape
        # nb_test, _, test_height, test_width = test_input.shape
        
    for iteration in range(iterations):
        
        prob = prioritized_board.get_prob()

        random_cand = prioritized_board.sample_according_to_thetas( 
            ignore_stages=model.module.ignore_stages,
            thetas=model.module.thetas)

        # cand_flops = est.get_flops(random_cand)

        random_cand = random_cand[1:]
        random_cand.insert(0, [0]) # [0] for stem

        if iteration % 200 == 0: record_theta = True
        else: record_theta = False
        #######################################
        # Finetuning part
        model.module.update_main()
        # task_flops, task_params = (TASK_FLOPS, TASK_PARAMS)
        training_loss, loss_dict = training_step(task_flops, task_params, device, model, random_cand, est, wot_map, cfg, record_theta)
        flops_loss, params_loss, wot_loss, layer_loss = loss_dict['flops_loss'], loss_dict['params_loss'], loss_dict['zero_cost'], loss_dict['layer_loss']
        reg_losss=loss_dict['reg_loss']
        
        theta_optimizer.zero_grad()
        training_loss.backward()
        theta_optimizer.step()
        #######################################
        # print(random_cand)
        # random_cand.append([0])
        #######################################
        # Meta learning part
        # (flops, params)
        # tasks= [(18, 60), (10.3, 50.4), (5.7, 30.1), (4.1, 24.2)]
        
        # num_updates = 5

        # task_flops, task_params = random.choice(tasks)
        # model.module.update_pi()
        # for i in range(num_updates):
        #     training_loss, flops_loss, params_loss, _ = training_step(task_flops, task_params, test_input, model, random_cand, est)
        #     # print(flops_loss)
        #     model.module.thetas_pi_optimizer.zero_grad()
        #     training_loss.backward()
        #     model.module.thetas_pi_optimizer.step()
        
        # theta_optimizer.zero_grad()
        # model.module.point_grad_to(device)
        # theta_optimizer.step()
        # model.module.update_main()
        # training_loss, flops_loss, params_loss, _ = training_step(task_flops, task_params, test_input, model, random_cand, est)
        
        ########################################
            # kd_loss, feature_t, teacher_cand = None, None, None
        # else:
        #     (inference_output, output), feature_s, chosen_subnet = model(input, random_cand, est.flops_dict)
        #     # print(chosen_subnet)
        #     valid_loss, loss_items = compute_loss(output, target.to(device), model)
        #     if local_rank != -1:
        #             valid_loss *= world_size
        #     # get soft label from teacher cand
        #     # with torch.no_grad():
        #     #     (teacher_inference_output, teacher_output), feature_t, chosen_subnet = model(input, teacher_cand)
           
        #     # kd_loss = compute_loss_KD(model.feature_adaptation, model, target, teacher_output, feature_s, feature_t)
        #     loss = valid_loss

        
        reduced_loss_training = training_loss.data
        reduced_loss_flops = flops_loss.data

        # if epoch >= cfg.SUPERNET.META_STA_EPOCH:

        #     with torch.no_grad():
        #         (test_inference_output, test_output), test_feature_s, chosen_subnet = model(test_input, random_cand)
        #         map50 = compute_map(test_inference_output.detach(), test_target, test_height, test_width, nc=model.nc)

        #     prioritized_board.update_prioritized_board(input, epoch, map50, cand_flops, random_cand, chosen_subnet)
        # else:
        #     map50 = 0
        map50 = 0
        torch.cuda.synchronize()

        # if kd_loss is not None:
        #     kd_losses_m.update(kd_loss.item(), input.size(0))
        batch_size = 2
        training_losses_m.update(reduced_loss_training.item(), batch_size)
        flops_losses_m.update(reduced_loss_flops.item(), batch_size)
        overall_losses_m.update(reduced_loss_flops.item() * reduced_loss_training.item(), batch_size)
        prec1_m.update(map50, batch_size)
        batch_time_m.update(time.time() - end)
        # print('theta shape', model.module.thetas_main[0]().shape)

        if iteration % (iterations // 2 -1) == 0:
            distributions = softmax(model.module.thetas[0]().detach().cpu().numpy())
            print('Distributions in 1 stage:', distributions)
            


        if iteration % cfg.LOG_INTERVAL == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Flops_loss: {flops_loss.val:>9.6f} ({flops_loss.avg:>6.4f})  '
                    'Overall_loss: {overall_loss:>9.6f}  '
                    'temperature: {temperature:>7.4f}  '
                    'Params_loss: {params_loss:>7.4f}  '
                    'Wot_loss: {wot_loss:>7.4f}  '
                    'Layer_loss: {layer_loss:>7.4f}  '
                    'Reg_loss: {reg_losss:>7.4} '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        iteration, len(loader),
                        100. * iteration / last_idx,
                        loss=training_losses_m,
                        flops_loss=flops_losses_m,
                        overall_loss=training_loss,
                        params_loss=params_loss,
                        wot_loss=wot_loss,
                        layer_loss=layer_loss,
                        reg_losss=reg_losss,
                        top1=prec1_m,
                        top5=prec5_m,
                        temperature=model.module.temperature,
                        batch_time=batch_time_m,
                        rate=batch_size * cfg.NUM_GPU / batch_time_m.val,
                        rate_avg=batch_size * cfg.NUM_GPU / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if cfg.SAVE_IMAGES and output_dir:
                    torchvision.utils.save_image(
                        input, os.path.join(
                            output_dir, 'train-batch-%d.jpg' %
                            iteration), padding=0, normalize=True)

        #     if saver is not None and cfg.RECOVERY_INTERVAL and (
        #             last_batch or (batch_idx + 1) % cfg.RECOVERY_INTERVAL == 0):
        #         saver.save_recovery(model, optimizer, cfg, epoch,
        #                             model_ema=model_ema, batch_idx=batch_idx)

        #     end = time.time()
        # if lr_scheduler is not None:
        #         lr_scheduler.step()

        if local_rank == 0:
            for idx, i in enumerate(prioritized_board.prioritized_board):
        
                logger.info("No.{} {}".format(idx, i[:4]))
        # return OrderedDict([('loss', losses_m.avg)])
        
        
    return map50, synflow_cache



def training_step(task_flops, task_params, device, model, random_cand, est, wot_map, cfg, record_theta=False):
    output_dir = os.path.join(cfg.SAVE_PATH, cfg.exp_name)
    GPU_NUMBER = 2
    model.module.zero_grad()
    overall_flops = torch.zeros((GPU_NUMBER, 1)).to(device)
    overall_synflow = torch.zeros((GPU_NUMBER, 1)).to(device)
    overall_params = torch.zeros((GPU_NUMBER, 1)).to(device)
    overall_wot = torch.zeros((GPU_NUMBER, 1)).to(device)
    overall_layers = torch.zeros((GPU_NUMBER, 1)).to(device)
    # one_datasample = torch.ones(1, 3, 416, 416).to(device)
    

    #############################################
    # Synflow part
    # signs = model.module.linearize()
    # model.eval()
    # (inference_output, output), feature_s, chosen_subnet = model(x=one_datasample, architecture=random_cand, calc_metric=True)
    # sum_arr_tensor(inference_output).backward()
    # output_synflow = model.module.calculate_synflow(one_datasample, random_cand, overall_synflow)
    # model.module.nonlinearize(signs)
    ##############################################
    output_wot = model.module.calculate_wot_train(overall_wot, random_cand, wot_map)
    output_flops = model.module.calculate_flops(random_cand, est.flops_dict, overall_flops)
    output_params = model.module.calculate_parameters(random_cand, est.params_dict, overall_params)
    output_layers = model.module.calculate_layers(random_cand, overall_layers)
    reg_loss = model.module.calculate_beta_reg()    
    
    output_flops = output_flops.mean()
    output_wot = output_wot.mean()
    output_layers = output_layers.mean()
    # output_synflow = output_synflow.mean()
    output_params = output_params.mean()
    with open(os.path.join(output_dir, f'flops-{task_flops}-wot-precalculated-it2880.txt'), 'a') as flops_file:
        flops_file.write(str(output_flops.item()))
        flops_file.write('\n')
    with open(os.path.join(output_dir, f'params-{task_params}-wot-precalculatedit2880.txt'), 'a') as params_file:
        params_file.write(str(output_params.item()))
        params_file.write('\n')
    if record_theta:
        with open(os.path.join(output_dir, f'history_thetas.txt'), 'a') as thetas_file:
            beta_distributions=[]
            for theta in model.module.thetas:
                alpha=theta().detach().cpu().numpy()
                beta_distributions.append(softmax(alpha))
            thetas_file.write(str(beta_distributions))
            thetas_file.write('\n')
    # for params_loss
    alpha = 0.03
    # for flops_loss
    beta = 0.01
    # for zero cost loss
    gamma = 0.01
    # for depth loss
    omega = 0.01
    # for regularization loss, default 0.01
    eta = 0.01
    # training_loss, loss_items = compute_loss(output, target, model)
    output_flops = output_flops / 1000
    flops_loss = torch.log(output_flops ** beta)
    params_loss = torch.log(output_params ** alpha)
    
    squared_error_flops = (output_flops - task_flops) ** 2
    squared_error_params = (output_params - task_params) ** 2

    flops_loss = squared_error_flops * alpha
    params_loss = squared_error_params * beta
    layers_loss = output_layers * omega
    # wot_loss = 1/torch.log(output_wot ** gamma)
    wot_loss = -(output_wot * gamma)
    # synflow_loss = 1/torch.log(output_synflow ** gamma)
    
    if cfg.BETA_REG: 
        reg_loss = reg_loss * eta
    else:
        reg_loss = reg_loss * 0.
        
    
    loss = wot_loss + flops_loss + params_loss + layers_loss + reg_loss
    loss_dict = {
        'flops_loss': flops_loss,
        'params_loss': params_loss,
        'layer_loss': layers_loss,
        'zero_cost': wot_loss,
        'reg_loss': reg_loss
    }

    return loss, loss_dict
    # return loss, overall_flops.mean(), overall_params.mean(), wot_loss



##############################
# Train DNAS
##############################
def train_step_dnas_V2(model, input, targets, is_gumbel, est, task_flops, task_params, device):
    alpha = 0.1        # for flops_loss
    beta = 0.01         # for params_loss
    gamma = 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    eta = 0.01          # for regularization loss, default 0.01
    is_ddp = is_parallel(model)
    temperature = model.module.temperature if is_ddp else model.temperature

    if is_gumbel:
        gumbel_prob = model.module.gumbel_sampling(temperature) if is_ddp else model.gumbel_sampling(temperature)
    else:
        gumbel_prob = model.module.softmax_sampling(temperature) if is_ddp else model.softmax_sampling(temperature)
        
    imgs = input.to(device, non_blocking=True).float() / 255.0
    pred = model.module(imgs, gumbel_prob) if is_ddp else model(imgs, gumbel_prob)
    
    det_loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
    det_loss = det_loss[0]
    
    architecture_info = {
        'arch_type': 'continuous',
        'arch': gumbel_prob
    }
    flops = model.module.calculate_flops_new(architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
    # params = model.module.calculate_params_new(architecture_info, est.params_dict)
    # layers = model.module.calculate_layers_new(architecture_info)
    
    output_flops = flops.mean() / 1e3
    # output_params = params.mean() 
    # output_layers = layers.mean() 
    
    squared_error_flops = (output_flops - task_flops) ** 2
    # squared_error_params = (output_params - task_params) ** 2
    
    flops_loss = squared_error_flops * alpha
    # params_loss = squared_error_params * beta
    # layers_loss = output_layers
    train_loss = det_loss + flops_loss

    hardware_losses = {
        'squared_error_flops': squared_error_flops
    }
    train_info = {
        'output_flops' : output_flops,
        'n_targets' : targets.shape[0],
        'n_imgs'    : imgs.shape[-1],
        'architecture' : gumbel_prob
    }

    return train_loss, loss_items, hardware_losses, train_info


def train_epoch_dnas(model, dataloader, optimizer, cfg, device, task_flops, task_params, 
                     est=None, logger=None, local_rank=0, world_size=0, is_gumbel=False, use_amp=False,
                     prefix='', epoch=None, total_epoch=None, logdir='./', ema=None, warmup=True, description=""):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    training_losses_m = AverageMeter()
    flops_losses_m = AverageMeter()
    det_losses_m = AverageMeter()
    
    cache_hits = 0
    iterations = len(dataloader)

    end = time.time()
    last_idx = len(dataloader) - 1
    
    batch_size = cfg.DATASET.BATCH_SIZE
    alpha = 0.1        # for flops_loss
    beta = 0.01         # for params_loss
    gamma = 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    eta = 0.01          # for regularization loss, default 0.01
    nw = max(3 * batch_size, 1e3)
    is_ddp = is_parallel(model)
    
    temperature = model.module.temperature if is_ddp else model.temperature
    mloss = torch.zeros(4, device=device)  # mean losses
    

    pbar = enumerate(dataloader)
    if local_rank in [-1, 0]:
        print(('%10s' * 14) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'lr', 'moment', 'decay', 'temp', 'GFLOPS', 'f_loss'))
        pbar = tqdm(pbar, total=iterations, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    
    
    # t_data = time.time()
    DBG_ITER=10
    DEBUG = False
    for iter_idx, (imgs, targets, paths, _) in pbar:

        # t_data = time.time() - t_data 
        ##################################################################
        ### 1st. Train SuperNet Parameter
        ##################################################################
        # imgs = (batch=2, 3, height, width)
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        if iter_idx <=DBG_ITER and DEBUG:
            random_testing(f'train iter {iter_idx}')
            print(imgs.sum(), imgs.mean(), imgs.min(), imgs.max())
        ni = iter_idx + iterations * (epoch- 1)
        if ni <= nw and warmup:
            import math
            lf = lambda x: (((1 + math.cos(x * math.pi / total_epoch)) / 2) ** 1.0) * 0.8 + 0.2
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                if 'initial_lr' not in x:
                    continue
                w_lr = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                
                x['lr'] = w_lr
                if 'momentum' in x:
                    w_momentum = np.interp(ni, xi, [0.9, cfg.momentum])
                    x['momentum'] = w_momentum

        t_infer=time.time()
        if False and use_amp: 
            scaler = amp.GradScaler(enabled=True)
            with amp.autocast(enabled=True):
                if is_gumbel:
                    gumbel_prob = model.module.gumbel_sampling(temperature) if is_ddp else model.gumbel_sampling(temperature)
                else:
                    gumbel_prob = model.module.softmax_sampling(temperature) if is_ddp else model.softmax_sampling(temperature)
                
                pred = model.module(imgs, gumbel_prob) if is_ddp else model(imgs, gumbel_prob)
                
                # det_loss, loss_items = compute_loss(pred[0][1], targets.to(device), model)  # scaled by batch_size
                det_loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                det_loss = det_loss[0]
                
                architecture_info = {
                    'arch_type': 'continuous',
                    'arch': gumbel_prob
                }
                flops = model.module.calculate_flops_new(architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
                # params = model.module.calculate_params_new(architecture_info, est.params_dict)
                # layers = model.module.calculate_layers_new(architecture_info)
                
                output_flops = flops.mean() / 1e3
                # output_params = params.mean() 
                # output_layers = layers.mean() 
                
                squared_error_flops = (output_flops - task_flops) ** 2
                # squared_error_params = (output_params - task_params) ** 2
                # flops_loss = (output_flops - task_flops) ** 2
                
                flops_loss = squared_error_flops * alpha
                # params_loss = squared_error_params * beta
                # layers_loss = output_layers

                train_loss = det_loss + flops_loss
            t_infer=time.time()-t_infer
            
            time_grad= time.time()
            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if is_gumbel:
                gumbel_prob = model.module.gumbel_sampling(temperature=temperature, device=device) if is_ddp else model.gumbel_sampling(temperature=temperature, device=device)
            else:
                gumbel_prob = model.module.softmax_sampling(temperature=temperature) if is_ddp else model.softmax_sampling(temperature=temperature)
            
            # print('[Random] sample prob', gumbel_prob)
            pred = model.module(imgs, gumbel_prob) if is_ddp else model(imgs, gumbel_prob)
            if DEBUG: print('[Test] feature ', pred[0].mean())

            # det_loss, loss_items = compute_loss(pred[0][1], targets.to(device), model)  # scaled by batch_size
            # DEBUG = False
            if True:
                det_loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                det_loss = det_loss[0]
                # logger.info(f'iter {iter_idx} {str(loss_items)}')
                
            else:
                det_loss = 0
                loss_items = torch.tensor([0.0, 0.0, 0.0, 0.0]).cuda()
            architecture_info = {
                'arch_type': 'continuous',
                'arch': gumbel_prob
            }
            flops = model.module.calculate_flops_new(architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
            # params = model.module.calculate_params_new(architecture_info, est.params_dict)
            # layers = model.module.calculate_layers_new(architecture_info)
            
            output_flops = flops.mean() / 1e3
            # output_params = params.mean() 
            # output_layers = layers.mean() 
            
            squared_error_flops = (output_flops - task_flops) ** 2
            # squared_error_params = (output_params - task_params) ** 2
            # flops_loss = (output_flops - task_flops) ** 2
            
            flops_loss = squared_error_flops * alpha
            # params_loss = squared_error_params * beta
            # layers_loss = output_layers
                
            train_loss = det_loss + flops_loss
            t_infer=time.time()-t_infer

            # if iter_idx == 0 and description == "architecture":
            #     logger.info(f'[Roger] Temp {temperature:5.2f} FLOPS : {output_flops:5.2f} Gumbel  {stringify_theta(gumbel_prob)}')
            
            time_grad= time.time()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
        if ema is not None:
            # description = "architecture", description = "weights"
            if description == "weights":
                ema.update(model, update_arch=False)
            elif description == "architecture":
                ema.update(model, update_arch=True)
                
        
        # Basic Info
        for j, x in enumerate(optimizer.param_groups):
            if 'momentum' in x:
                break
        print_lr = x['lr'] if 'lr' in x else 0
        print_m = x['momentum'] if 'momentum' in x else 0
        print_wdecay = x['weight_decay'] if 'weight_decay' in x else 0
        
        # Print
        if local_rank in [-1, 0]:
            ni = iter_idx
            mloss = (mloss * iter_idx + loss_items) / (iter_idx + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 12) % (
                '%g/%g' % (epoch, total_epoch), mem, *mloss, targets.shape[0], imgs.shape[-1], print_lr, print_m, print_wdecay, temperature, output_flops, squared_error_flops.detach().cpu())
            pbar.set_description(s)
            
            # Plot
            if ni < 3:
                f = str(Path(logdir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                # if tb_writer and result is not None:
                #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard
    
        if iter_idx <= DBG_ITER and DEBUG:
            print('flops', flops)
            print(f'task_flops={task_flops} output_flops={output_flops.detach().cpu().numpy()} squared_error_flops={squared_error_flops.detach().cpu().numpy()} flops_loss={flops_loss.detach().cpu().numpy()}')
            random_testing(f'end of iter {iter_idx} output_flops: {output_flops.detach().cpu().numpy()} {flops_loss.detach().cpu().numpy()} {det_loss.detach().cpu().numpy()}')
        if iter_idx == DBG_ITER and DEBUG: exit()
    
        # t_data = time.time()
    torch.cuda.synchronize()


def train_epoch_dnas_V2(model, dataloader, theta_dataloader, optimizer, theta_optimizer, cfg, device, task_flops, task_params, 
                     est=None, logger=None, local_rank=0, world_size=0, is_gumbel=False,
                     prefix='', epoch=None, total_epoch=None, logdir='./', ema=None, warmup=True):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    training_losses_m = AverageMeter()
    flops_losses_m = AverageMeter()
    det_losses_m = AverageMeter()
    
    cache_hits = 0
    iterations = len(dataloader)

    end = time.time()
    last_idx = len(dataloader) - 1
    
    batch_size = cfg.DATASET.BATCH_SIZE
    alpha = 0.1        # for flops_loss
    beta = 0.01         # for params_loss
    gamma = 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    eta = 0.01          # for regularization loss, default 0.01
    nw = max(3 * batch_size, 1e3)
    is_ddp = is_parallel(model)
    
    # temperature = model.module.temperature if is_ddp else model.temperature
    mloss = torch.zeros(4, device=device)  # mean losses
    
    pbar = enumerate(dataloader)
    if local_rank in [-1, 0]:
        logger.info(('%10s' * 14) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'lr', 'moment', 'decay', 'temp', 'GFLOPS', 'f_loss'))
        pbar = tqdm(pbar, total=iterations, bar_format='{l_bar}{bar:5}{r_bar}')  # progress bar
    
    def valid_generator():
        while True:
          for x, t, path, shape in theta_dataloader:
            yield x, t, path, shape
    valid_gen = valid_generator()
    
    temperature = model.module.temperature if is_ddp else model.temperature
    arch_prob = model.module.softmax_sampling(temperature=temperature) if is_ddp else model.softmax_sampling(temperature=temperature)
    logger.info(f'Begin Architecture : {str(arch_prob)}')

    for iter_idx, (imgs, targets, paths, _) in pbar:
        ##################################################################
        # 1st. WarmUp Learning Rate
        ##################################################################
        ni = iter_idx + iterations * (epoch- 1)
        if ni <= nw and warmup:
            import math
            lf = lambda x: (((1 + math.cos(x * math.pi / total_epoch)) / 2) ** 1.0) * 0.8 + 0.2
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                if 'initial_lr' not in x:
                    continue
                w_lr = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                
                x['lr'] = w_lr
                if 'momentum' in x:
                    w_momentum = np.interp(ni, xi, [0.9, cfg.momentum])
                    x['momentum'] = w_momentum



        ##################################################################
        ### 2nd. Update SuperNet Architecture
        ##################################################################
        if epoch > cfg.FREEZE_EPOCH:
            # Prepare Data
            input_valid, target_valid, _, _ = next(valid_gen)
            # Prepare Architecture Parameter
            train_loss, loss_items, hardware_losses, train_info = train_step_dnas_V2(model, input_valid, target_valid, False, est, task_flops, task_params, device)
            # squared_error_flops = hardware_losses['squared_error_flops']
            # output_flops        = train_info['output_flops']
            # n_targets           = train_info['n_targets']
            # n_imgs              = train_info['n_imgs']

            
            time_grad= time.time()
            theta_optimizer.zero_grad()
            train_loss.backward()
            theta_optimizer.step()
            if ema is not None:
                ema.update(model, update_arch=True)
        
        ##################################################################
        ### 3rd. Train SuperNet Parameter
        ##################################################################
        train_loss, loss_items, hardware_losses, train_info = train_step_dnas_V2(model, imgs, targets, True, est, task_flops, task_params, device)
        squared_error_flops = hardware_losses['squared_error_flops']
        output_flops        = train_info['output_flops']
        n_targets           = train_info['n_targets']
        n_imgs              = train_info['n_imgs']
        if (iter_idx + 1) % 100 == 0:
            train_prob          = stringify_theta(train_info['architecture'], normalize=True, temperature=temperature)
            logger.info(f'Train Temp : {temperature} Architecture : {train_prob}')
        
        time_grad= time.time()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model, update_arch=False)

        # Basic Info
        for j, x in enumerate(optimizer.param_groups):
            if 'momentum' in x:
                break
        print_lr = x['lr'] if 'lr' in x else 0
        print_m = x['momentum'] if 'momentum' in x else 0
        print_wdecay = x['weight_decay'] if 'weight_decay' in x else 0
        
        # Print
        if local_rank in [-1, 0]:
            ni = iter_idx
            mloss = (mloss * iter_idx + loss_items) / (iter_idx + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 12) % (
                '%g/%g' % (epoch, total_epoch), mem, *mloss, n_targets, n_imgs, print_lr, print_m, print_wdecay, temperature, output_flops, squared_error_flops.detach().cpu())

            date_time = datetime.now().strftime('%m/%d %I:%M %p') + ' | '
            pbar.set_description(date_time + s)
            
            # Plot
            if ni < 3:
                f = str(Path(logdir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)

    arch_prob = model.module.softmax_sampling(temperature=temperature) if is_ddp else model.softmax_sampling(temperature=temperature)
    logger.info(f'End Architecture : {str(arch_prob)}', )

    logger.info(s)        
    torch.cuda.synchronize()


def train_epoch_flop(model, dataloader, optimizer, cfg, device, task_flops, task_params, 
                     est=None, logger=None, local_rank=0, world_size=0, is_gumbel=False, use_amp=False,
                     prefix='', epoch=None, total_epoch=None, logdir='./', ema=None, warmup=True, description=""):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    training_losses_m = AverageMeter()
    flops_losses_m = AverageMeter()
    det_losses_m = AverageMeter()
    
    cache_hits = 0
    iterations = len(dataloader)

    end = time.time()
    last_idx = len(dataloader) - 1
    
    batch_size = cfg.DATASET.BATCH_SIZE
    alpha = 0.1        # for flops_loss
    beta = 0.01         # for params_loss
    gamma = 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    eta = 0.01          # for regularization loss, default 0.01
    nw = max(3 * batch_size, 1e3)
    is_ddp = is_parallel(model)
    
    temperature = model.module.temperature if is_ddp else model.temperature
    mloss = torch.zeros(4, device=device)  # mean losses
    

    pbar = enumerate(dataloader)
    if local_rank in [-1, 0]:
        print(('%10s' * 14) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'lr', 'moment', 'decay', 'temp', 'GFLOPS', 'f_loss'))
        pbar = tqdm(pbar, total=iterations, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    
    
    # t_data = time.time()
    DBG_ITER=10
    DEBUG = False
    for iter_idx, (imgs, targets, paths, _) in pbar:

        # t_data = time.time() - t_data 
        ##################################################################
        ### 1st. Train SuperNet Parameter
        ##################################################################
        # imgs = (batch=2, 3, height, width)
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        if iter_idx <=DBG_ITER and DEBUG:
            random_testing(f'train iter {iter_idx}')
            print(imgs.sum(), imgs.mean(), imgs.min(), imgs.max())
        ni = iter_idx + iterations * (epoch- 1)
        if ni <= nw and warmup:
            import math
            lf = lambda x: (((1 + math.cos(x * math.pi / total_epoch)) / 2) ** 1.0) * 0.8 + 0.2
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                if 'initial_lr' not in x:
                    continue
                w_lr = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                
                x['lr'] = w_lr
                if 'momentum' in x:
                    w_momentum = np.interp(ni, xi, [0.9, cfg.momentum])
                    x['momentum'] = w_momentum

        t_infer=time.time()
        if False and use_amp: 
            scaler = amp.GradScaler(enabled=True)
            with amp.autocast(enabled=True):
                if is_gumbel:
                    gumbel_prob = model.module.gumbel_sampling(temperature=temperature) if is_ddp else model.gumbel_sampling(temperature=temperature)
                else:
                    gumbel_prob = model.module.softmax_sampling(temperature=temperature) if is_ddp else model.softmax_sampling(temperature=temperature)
                
                pred = model.module(imgs, gumbel_prob) if is_ddp else model(imgs, gumbel_prob)
                
                # det_loss, loss_items = compute_loss(pred[0][1], targets.to(device), model)  # scaled by batch_size
                det_loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                det_loss = det_loss[0]
                
                architecture_info = {
                    'arch_type': 'continuous',
                    'arch': gumbel_prob
                }
                flops = model.module.calculate_flops_new(architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
                # params = model.module.calculate_params_new(architecture_info, est.params_dict)
                # layers = model.module.calculate_layers_new(architecture_info)
                
                output_flops = flops.mean() / 1e3
                # output_params = params.mean() 
                # output_layers = layers.mean() 
                
                squared_error_flops = (output_flops - task_flops) ** 2
                # squared_error_params = (output_params - task_params) ** 2
                # flops_loss = (output_flops - task_flops) ** 2
                
                flops_loss = squared_error_flops * alpha
                # params_loss = squared_error_params * beta
                # layers_loss = output_layers

                train_loss = det_loss + flops_loss
            t_infer=time.time()-t_infer
            
            time_grad= time.time()
            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if is_gumbel:
                gumbel_prob = model.module.gumbel_sampling(temperature, device=device) if is_ddp else model.gumbel_sampling(temperature, device=device)
            else:
                gumbel_prob = model.module.softmax_sampling(temperature, device=device) if is_ddp else model.softmax_sampling(temperature, device=device)
            
            # print('[Random] sample prob', gumbel_prob)
            pred = model.module(imgs, gumbel_prob) if is_ddp else model(imgs, gumbel_prob)
            if DEBUG: print('[Test] feature ', pred[0].mean())

            # det_loss, loss_items = compute_loss(pred[0][1], targets.to(device), model)  # scaled by batch_size
            # DEBUG = False
            if False:
                det_loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                det_loss = det_loss[0]
                logger.info(f'iter {iter_idx} {str(loss_items)}')
                
            else:
                det_loss = 0
                loss_items = torch.tensor([0.0, 0.0, 0.0, 0.0]).cuda()
            architecture_info = {
                'arch_type': 'continuous',
                'arch': gumbel_prob
            }
            flops = model.module.calculate_flops_new(architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
            # params = model.module.calculate_params_new(architecture_info, est.params_dict)
            # layers = model.module.calculate_layers_new(architecture_info)
            
            output_flops = flops.mean() / 1e3
            # output_params = params.mean() 
            # output_layers = layers.mean() 
            
            squared_error_flops = (output_flops - task_flops) ** 2
            # squared_error_params = (output_params - task_params) ** 2
            # flops_loss = (output_flops - task_flops) ** 2
            
            flops_loss = squared_error_flops * alpha
            # params_loss = squared_error_params * beta
            # layers_loss = output_layers
                
            train_loss = det_loss + flops_loss
            t_infer=time.time()-t_infer

            # if iter_idx == 0 and description == "architecture":
            #     logger.info(f'[Roger] Temp {temperature:5.2f} FLOPS : {output_flops:5.2f} Gumbel  {stringify_theta(gumbel_prob)}')
            
            time_grad= time.time()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
        if ema is not None:
            # description = "architecture", description = "weights"
            if description == "weights":
                ema.update(model, update_arch=False)
            elif description == "architecture":
                ema.update(model, update_arch=True)
                
        
        # Basic Info
        for j, x in enumerate(optimizer.param_groups):
            if 'momentum' in x:
                break
        print_lr = x['lr'] if 'lr' in x else 0
        print_m = x['momentum'] if 'momentum' in x else 0
        print_wdecay = x['weight_decay'] if 'weight_decay' in x else 0
        
        # Print
        if local_rank in [-1, 0]:
            ni = iter_idx
            mloss = (mloss * iter_idx + loss_items) / (iter_idx + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 12) % (
                '%g/%g' % (epoch, total_epoch), mem, *mloss, targets.shape[0], imgs.shape[-1], print_lr, print_m, print_wdecay, temperature, output_flops, squared_error_flops.detach().cpu())
            pbar.set_description(s)
            
            # Plot
            if ni < 3:
                f = str(Path(logdir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                # if tb_writer and result is not None:
                #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard
    
        if iter_idx <= DBG_ITER and DEBUG:
            print('flops', flops)
            print(f'task_flops={task_flops} output_flops={output_flops.detach().cpu().numpy()} squared_error_flops={squared_error_flops.detach().cpu().numpy()} flops_loss={flops_loss.detach().cpu().numpy()}')
            random_testing(f'end of iter {iter_idx} output_flops: {output_flops.detach().cpu().numpy()} {flops_loss.detach().cpu().numpy()} {det_loss.detach().cpu().numpy()}')
        if iter_idx == DBG_ITER and DEBUG: exit()
    
        # t_data = time.time()
    torch.cuda.synchronize()



##############################
# Train Sensitive
##############################
def train_epoch_sensitive(model, dataloader, optimizer, cfg, device, task_flops, task_params, 
                     est=None, logger=None, local_rank=0, world_size=0, is_gumbel=False,
                     prefix='', epoch=None, total_epoch=None, logdir='./', ema=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    training_losses_m = AverageMeter()
    flops_losses_m = AverageMeter()
    det_losses_m = AverageMeter()
    
    cache_hits = 0
    iterations = len(dataloader)

    end = time.time()
    last_idx = len(dataloader) - 1
    
    batch_size = cfg.DATASET.BATCH_SIZE
    alpha = 0.1        # for flops_loss
    beta = 0.01         # for params_loss
    gamma = 0.01        # for zero cost loss
    omega = 0.01        # for depth loss
    eta = 0.01          # for regularization loss, default 0.01
    nw = max(3 * batch_size, 1e3)
    is_ddp = is_parallel(model)
    
    temperature = model.module.temperature if is_ddp else model.temperature
    mloss = torch.zeros(3, device=device)  # mean losses
    
    pbar = enumerate(dataloader)
    if local_rank in [-1, 0]:
        print(('%10s' * 13) % ('Epoch', 'gpu_mem', 'fore', 'back', 'total', 'targets', 'img_size', 'lr', 'moment', 'decay', 'temp', 'GFLOPS', 'f_loss'))
        pbar = tqdm(pbar, total=iterations, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    
    t_data = time.time()
    # for iteration, (input, target) in enumerate(loader):
    for iter_idx, (uimgs, targets, paths, _) in pbar:
        ########################################################
        # Learning WarmUp
        ########################################################
        ni = iter_idx + iterations * (epoch- 1)
        if ni <= nw:
            import math
            lf = lambda x: (((1 + math.cos(x * math.pi / total_epoch)) / 2) ** 1.0) * 0.8 + 0.2
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                if 'initial_lr' not in x:
                    continue
                w_lr = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                
                x['lr'] = w_lr
                if 'momentum' in x:
                    w_momentum = np.interp(ni, xi, [0.9, cfg.momentum])
                    x['momentum'] = w_momentum
        
        ########################################################
        # Sample Architecture
        ########################################################
        arch_theta = torch.cat([theta().reshape(1, -1) for theta in (model.module.thetas if is_ddp else model.thetas )], dim=0)
        if is_gumbel:
            gumbel_prob = nn.functional.gumbel_softmax(arch_theta, temperature, dim=-1)
        else:
            gumbel_prob = nn.functional.softmax(arch_theta, dim=-1)
            
        t_data = time.time() - t_data 
        ##################################################################
        ### 1st. Train SuperNet Parameter
        ##################################################################
        # imgs = (batch=2, 3, height, width)
        imgs     = uimgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        
        t_infer=time.time()
        pred     = model.module(imgs, gumbel_prob)      if is_ddp else model(imgs, gumbel_prob)
        
        
        drop_mask = torch.ones(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]).cuda()
        drop_mask = torch.nn.functional.dropout(drop_mask, p=0.75, training=True)  # uint8 to float32, 0-255 to 0.0-1.0
        imgs_aug = imgs * drop_mask
        
        
        mask_sizes = [p.shape[2:4] for p in pred[0][1]]
        masks = build_foreground_mask(mask_sizes, targets.to(device), model)  # scaled by batch_size

        pred_aug = model.module(imgs_aug, gumbel_prob)  if is_ddp else model(imgs_aug, gumbel_prob)
                
        ################################################################################
        # if iter_idx < 2: 
        #     import cv2
        #     f = str(Path(logdir) / ('label_batch%g.jpg' % ni))  # filename
        #     result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
            
        #     m1 = str(Path(logdir) / ('masks_batch%g_f1.jpg' % ni))  # filename
        #     m2 = str(Path(logdir) / ('masks_batch%g_f2.jpg' % ni))  # filename
        #     m3 = str(Path(logdir) / ('masks_batch%g_f3.jpg' % ni))  # filename
        #     f_names = [m1,m2,m3]
        #     for i in range(3):
        #         masks[i] = masks[i].cpu().float().numpy().transpose(0,2,3,1).repeat(3, axis=-1)
        #         print('masks[i]', masks[i].shape)
        #         top_img = np.concatenate(masks[i][0:2], axis=0)
        #         # bot_img = np.concatenate([masks[i][2:4]], axis=2)
        #         # ful_img = np.concatenate([top_img, bot_img])
        #         ful_img = top_img
                
        #         ful_img = (ful_img * 255.0).astype(np.uint8)
        #         print('ful_img', ful_img.shape, ful_img.dtype)
        #         ful_img = cv2.resize(ful_img, (208,416), interpolation=cv2.INTER_AREA)
        #         cv2.imwrite(f_names[i], ful_img)
        
        ################################################################################
        sen_loss, loss_items = compute_sensitive_loss(pred[0][1], pred_aug[0][1], targets, masks)  # scaled by batch_size

        architecture_info = {
            'arch_type': 'continuous',
            'arch': gumbel_prob
        }
        flops = model.module.calculate_flops_new(architecture_info, est.flops_dict) if is_ddp else model.calculate_flops_new(architecture_info, est.flops_dict)
        # params = model.module.calculate_params_new(architecture_info, est.params_dict)
        # layers = model.module.calculate_layers_new(architecture_info)
        
        output_flops = flops.mean() / 1e3
        # output_params = params.mean() 
        # output_layers = layers.mean() 
        squared_error_flops = (output_flops - task_flops) ** 2
        # squared_error_params = (output_params - task_params) ** 2
        # flops_loss = (output_flops - task_flops) ** 2
        flops_loss = squared_error_flops * alpha
        # params_loss = squared_error_params * beta
        # layers_loss = output_layers
        train_loss = sen_loss + flops_loss
        t_infer=time.time()-t_infer
    
        time_grad= time.time()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Basic Info
        for j, x in enumerate(optimizer.param_groups):
            if 'momentum' in x:
                break
        print_lr = x['lr'] if 'lr' in x else 0
        print_m = x['momentum'] if 'momentum' in x else 0
        print_wdecay = x['weight_decay'] if 'weight_decay' in x else 0
        
        continue
        # Print
        if local_rank in [-1, 0]:
            ni = iter_idx
            mloss = (mloss * iter_idx + loss_items) / (iter_idx + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 11) % (
                '%g/%g' % (epoch, total_epoch), mem, *mloss, targets.shape[0], imgs.shape[-1], print_lr, print_m, print_wdecay, temperature, output_flops, squared_error_flops.detach().cpu())
            pbar.set_description(s)
            
            # Plot
            if ni < 3:
                f = str(Path(logdir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                # if tb_writer and result is not None:
                #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard
    
    
    
        t_data = time.time()
    torch.cuda.synchronize()

    # if iteration % (iterations // 2 -1) == 0:
    #     distributions = softmax(model.module.thetas[0]().detach().cpu().numpy())
    #     print('Distributions in 1 stage:', distributions)
             
    # if iteration % cfg.LOG_INTERVAL == 0:
    #     lrl = [param_group['lr'] for param_group in optimizer.param_groups]
    #     lr = sum(lrl) / len(lrl)
        
    #         # if cfg.SAVE_IMAGES and output_dir:
    #         #     torchvision.utils.save_image(
    #         #         input, os.path.join(
    #         #             output_dir, 'train-batch-%d.jpg' %
    #         #             iteration), padding=0, normalize=True)




def arch_2_subnet(model, architecture):
    subnet = ''
    for layer, layer_arch in zip(model.blocks, architecture):
            for blocks, arch in zip(layer, layer_arch):
                subnet += blocks[arch].get_block_name() + '->'
    return subnet[:-2]

def get_batch_jacobian(model, input, target, random_cand):
    model.zero_grad()
    input.requires_grad_(True)
    (inference_output, output), feature_s, chosen_subnet = model(input, random_cand, calc_metric=True)
    inference_output = sum_arr_tensor(inference_output)
    inference_output.backward(torch.ones_like(inference_output))
    jacob = input.grad.detach()
    return jacob


def validate(model, loader, loss_fn, prioritized_board, cfg, device, log_suffix='', local_rank=0, logger=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        candidates_map = [[], [], [], [], [], [], [], [], [], []]
        for batch_idx, (input, target, paths, _) in enumerate(loader):
            target = target.to(device)
            last_batch = batch_idx == last_idx
            input = input.to(device, non_blocking=True).float() / 255.0
            nb, _, height, width = input.shape
            for idx, candidate in enumerate(prioritized_board.prioritized_board):
                (inference_output, output), _, _ = model(input, candidate[3])
                candidates_map[idx].append(compute_map(inference_output.detach(), target, height, width, nc=model.nc))
        
        average_maps = []
        for candidate_map in candidates_map:
            average_maps.append(sum(candidate_map) / len(candidate_map))
        
        best_candidate_idx, _ = max(enumerate(average_maps), key=operator.itemgetter(1))
        print(best_candidate_idx, average_maps)
        return best_candidate_idx, average_maps
        
    
