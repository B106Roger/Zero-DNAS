# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import time, os
import torchvision
import torch.nn.functional as F
import itertools
import random
from scipy.special import softmax
from lib.utils.util import *
from lib.utils.general import compute_loss
from lib.utils.kd_utils import compute_loss_KD
from lib.utils.synflow import sum_arr_tensor

TASK_FLOPS = 11.9 #11.9          # e.g TASK_FLOPS = 5   means 50 GFLOPs
TASK_PARAMS = 52.5  # 52.5        # e.g TASK_PARAMS = 32 means 32 million parameters.
# supernet train function
def train_epoch(epoch, model, loader, optimizer, loss_fn, prioritized_board, MetaMN, cfg, device, synflow_cache, theta_optimizer=None,
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

        #######################################
        # Finetuning part
        model.module.update_main()
        task_flops, task_params = (TASK_FLOPS, TASK_PARAMS)
        training_loss, flops_loss, params_loss, wot_loss, layer_loss = training_step(task_flops, task_params, device, model, random_cand, est, wot_map, cfg)
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



def training_step(task_flops, task_params, device, model, random_cand, est, wot_map, cfg):
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
    output_flops = output_flops.mean()
    output_wot = output_wot.mean()
    output_layers = output_layers.mean()
    # output_synflow = output_synflow.mean()
    output_params = output_params.mean()
    with open(os.path.join(output_dir, f'flops-{TASK_FLOPS}-wot-precalculated-it2880.txt'), 'a') as flops_file:
        flops_file.write(str(output_flops.item()))
        flops_file.write('\n')
    with open(os.path.join(output_dir, f'params-{TASK_PARAMS}-wot-precalculatedit2880.txt'), 'a') as params_file:
        params_file.write(str(output_params.item()))
        params_file.write('\n')
    
    alpha = 0.03
    beta = 0.01
    gamma = 0.01
    # gamma = 0.001
    omega = 0.01
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
    # print('flops_loss', flops_loss)
    # print('params_loss', params_loss)
    # print('synflow loss', synflow_loss)
    # exit()
    # loss = synflow_loss * flops_loss
    # loss = synflow_loss + flops_loss + params_loss
    # loss = flops_loss
    # loss = wot_loss + flops_loss + params_loss
    loss = wot_loss + flops_loss + params_loss + layers_loss
    # print(loss)

    return loss, flops_loss, params_loss, wot_loss, layers_loss
    # return loss, overall_flops.mean(), overall_params.mean(), wot_loss

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
        
    
