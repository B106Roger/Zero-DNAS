# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import time
import torchvision
import torch.nn.functional as F
import itertools

from lib.utils.util import *
from lib.utils.general import compute_loss
from lib.utils.kd_utils import compute_loss_KD

# supernet train function
def train_epoch(epoch, model, loader, optimizer, loss_fn, prioritized_board, MetaMN, cfg, device, synflow_cache,
                est=None, logger=None, lr_scheduler=None, saver=None,
                output_dir='', model_ema=None, local_rank=0, world_size=0, test_loader=None, arch_sampler=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    kd_losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    cache_hits = 0
    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    test_iter = itertools.cycle(test_loader)
    for batch_idx, (input, target, paths, _) in enumerate(loader):
        target = target.to(device)
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input = input.to(device, non_blocking=True).float() / 255.0
        test_input, test_target, _, _ = next(test_iter)
        test_input = test_input.to(device, non_blocking=True).float() / 255.0
        test_target = test_target.to(device)
        nb, _, height, width = input.shape
        nb_test, _, test_height, test_width = test_input.shape
        

        # if epoch <= cfg.SUPERNET.META_STA_EPOCH:
        #     prob = prioritized_board.get_prob()

        #     random_cand = prioritized_board.get_cand_with_prob(
        #         prob=prob, 
        #         ignore_stages=model.ignore_stages)
        #     cand_flops = est.get_flops(random_cand)
        # else:
            # Uniform
            #######################################################
            # prob = prioritized_board.get_prob()

            # random_cand = prioritized_board.get_cand_with_prob(
            #     prob=prob, 
            #     ignore_stages=model.ignore_stages)
            # cand_flops = est.get_flops(random_cand)
            #######################################################

            # Attentive min loss
            #######################################################
            # target_flops = arch_sampler.sample_one_target_flops()
            # candidates = arch_sampler.sample_archs_according_to_flops(
            #     target_flops, 
            #     ignore_stages=model.ignore_stages, 
            #     block_to_choice_map=model.block_to_choice_map
            # )            
            # candidate_losses = []
            # for candidate in candidates:
                # with torch.no_grad():
                #     (_, output), _, _ = model(test_input, candidate['arch'])
                #     candidate_loss, _ = compute_loss(output, test_target, model)
                #     candidate_losses.append(candidate_loss)
            # best_candidate_idx, _ = min(enumerate(candidate_losses), key=operator.itemgetter(1))
            # best_candidate = candidates[best_candidate_idx]
            # random_cand = best_candidate['arch']
            # cand_flops = best_candidate['flops']
            ########################################################

            # Attentive synflow
            ########################################################
        target_flops = arch_sampler.sample_one_target_flops()
        candidates = arch_sampler.sample_archs_according_to_flops(
            target_flops, 
            ignore_stages=model.ignore_stages, 
            block_to_choice_map=model.block_to_choice_map
        )  
        candidate_synflows = []
        # signs = model.linearize()
        
        for candidate in candidates:
            # metric = synflow_cache.get(str(candidate['arch']))
            metric = est.get_synflow(candidate['arch'])
            # if metric is None:
            #     (inference_output, output), _, _ = model.forward(torch.ones(1, 3, 416, 416).to(device), candidate['arch'], calc_metric=True)
            #     torch.sum(inference_output[0]).backward()
            #     metric = model.calculate_synflow_metric(candidate['arch'])/1e10
            #     synflow_cache[str(candidate['arch'])] = metric
            #     model.zero_grad()
                
            # else:
            #     cache_hits += 1

            candidate_synflows.append(metric)
        # model.nonlinearize(signs)
        best_candidate_idx, _ = max(enumerate(candidate_synflows), key=operator.itemgetter(1))
        best_candidate = candidates[best_candidate_idx]
        
        random_cand = best_candidate['arch']
        cand_flops = best_candidate['flops']

        random_cand = random_cand[1:]
        random_cand.insert(0, [0]) # [0] for stem
        # print(random_cand)
        # random_cand.append([0])

        if prioritized_board.board_size() == 0 or epoch <= cfg.SUPERNET.META_STA_EPOCH:
            (inference_output, output), feature_s, chosen_subnet = model(input, random_cand)

            # print(chosen_subnet)

            loss, loss_items = compute_loss(output, target.to(device), model)
            if local_rank != -1:
                    loss *= world_size
            kd_loss, feature_t, teacher_cand = None, None, None
        else:
            (inference_output, output), feature_s, chosen_subnet = model(input, random_cand)
            # print(chosen_subnet)
            valid_loss, loss_items = compute_loss(output, target.to(device), model)
            if local_rank != -1:
                    valid_loss *= world_size
            # get soft label from teacher cand
            # with torch.no_grad():
            #     (teacher_inference_output, teacher_output), feature_t, chosen_subnet = model(input, teacher_cand)
           
            # kd_loss = compute_loss_KD(model.feature_adaptation, model, target, teacher_output, feature_s, feature_t)
            loss = valid_loss

        if batch_idx % 30 == 0:
            print('loss:', loss.item())
            print('cache_hits:', cache_hits)
            print('synflow_cache_length:', len(synflow_cache))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reduced_loss = loss.data
        with torch.no_grad():
            (test_inference_output, test_output), test_feature_s, chosen_subnet = model(test_input, random_cand)

        # reduced_loss = reduce_tensor(loss.data, cfg.NUM_GPU)
        

        map50 = compute_map(test_inference_output.detach(), test_target, test_height, test_width, nc=model.nc)
       
        prioritized_board.update_prioritized_board(input, epoch, map50, cand_flops, random_cand, chosen_subnet)

        torch.cuda.synchronize()

        # if kd_loss is not None:
        #     kd_losses_m.update(kd_loss.item(), input.size(0))
        losses_m.update(reduced_loss.item(), input.size(0))
        prec1_m.update(map50, input.size(0))
        batch_time_m.update(time.time() - end)


        if last_batch or batch_idx % cfg.LOG_INTERVAL == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if local_rank == 0:
                print(test_height, test_width)
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'KD-Loss: {kd_loss.val:>9.6f} ({kd_loss.avg:>6.4f})  '
                    'map: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        kd_loss=kd_losses_m,
                        top1=prec1_m,
                        top5=prec5_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * cfg.NUM_GPU / batch_time_m.val,
                        rate_avg=input.size(0) * cfg.NUM_GPU / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if cfg.SAVE_IMAGES and output_dir:
                    torchvision.utils.save_image(
                        input, os.path.join(
                            output_dir, 'train-batch-%d.jpg' %
                            batch_idx), padding=0, normalize=True)

    #     if saver is not None and cfg.RECOVERY_INTERVAL and (
    #             last_batch or (batch_idx + 1) % cfg.RECOVERY_INTERVAL == 0):
    #         saver.save_recovery(model, optimizer, cfg, epoch,
    #                             model_ema=model_ema, batch_idx=batch_idx)

    #     end = time.time()
    if lr_scheduler is not None:
            lr_scheduler.step()

    if local_rank == 0:
        for idx, i in enumerate(prioritized_board.prioritized_board):
    
           logger.info("No.{} {}".format(idx, i[:4]))
    # return OrderedDict([('loss', losses_m.avg)])
    with open('cache_hits.txt', 'w+') as out_file:
        out_file.write(str(cache_hits))
        out_file.write('\n')
    
    print('cache_hits', str(cache_hits))
    print(len(synflow_cache))
    return map50, synflow_cache


def arch_2_subnet(model, architecture):
    subnet = ''
    for layer, layer_arch in zip(model.module.blocks, architecture):
            for blocks, arch in zip(layer, layer_arch):
                subnet += blocks[arch].get_block_name() + '->'
    return subnet[:-2]



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
        
    
