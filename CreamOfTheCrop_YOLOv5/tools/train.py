# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import os
import sys
from datetime import datetime
import yaml
import torch
import numpy as np
import torch.nn as nn
import tqdm

import _init_paths

# import timm packages
from timm.utils import CheckpointSaver, update_summary
# from timm.loss import LabelSmoothingCrossEntropy
# from timm.data import Dataset, create_loader
# from timm.models import resume_checkpoint

# import apex as distributed package otherwise we use torch.nn.parallel.distributed as distributed package
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    USE_APEX = True
except ImportError:
    from torch.nn.parallel import DataParallel as DDP
    USE_APEX = False

# import models and training functions
from lib.utils.flops_table import FlopsEst
from lib.core.train import train_epoch, validate
from lib.models.structures.supernet import gen_supernet
from lib.models.PrioritizedBoard import PrioritizedBoard
from lib.models.MetaMatchingNetwork import MetaMatchingNetwork
from lib.config import DEFAULT_CROP_PCT, IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from lib.utils.util import parse_config_args, get_logger, \
    create_optimizer_supernet, create_supernet_scheduler
from lib.utils.datasets import create_dataloader
from lib.utils.kd_utils import FeatureAdaptation
from lib.models.blocks.yolo_blocks import Conv, ConvNP
from lib.utils.general import check_img_size, labels_to_class_weights
from lib.utils.torch_utils import select_device
from lib.utils.attentive_sampling import collect_samples
from scipy.special import softmax
from lib.models.AttentiveNasSampler import ArchSampler



def main():
    args, cfg = parse_config_args('super net training')
    # resolve logging
    output_dir = os.path.join(cfg.SAVE_PATH,
                              "{}-{}".format(datetime.now().strftime('%m%d-%H:%M:%S'),
                                             cfg.MODEL))

    if args.local_rank == 0:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        logger_path = os.path.join(output_dir, "train.log")
        with open(logger_path, 'w') as file:
            pass
        logger = get_logger(logger_path)
    else:
        logger = None

    # initialize distributed parameters
    # torch.cuda.set_device(args.local_rank)
    device = select_device(args.device, batch_size=cfg.DATASET.BATCH_SIZE)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = 1
    args.global_rank = -1
    if args.local_rank == 0:
        logger.info(
            'Training on Process %d with %d GPUs.',
                args.local_rank, cfg.NUM_GPU)

    # fix random seeds
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # generate supernet
    model, sta_num, resolution = gen_supernet(
        flops_minimum=cfg.SUPERNET.FLOPS_MINIMUM,
        flops_maximum=cfg.SUPERNET.FLOPS_MAXIMUM,
        num_classes=cfg.DATASET.NUM_CLASSES,
        drop_rate=cfg.NET.DROPOUT_RATE,
        global_pool=cfg.NET.GP,
        resunit=cfg.SUPERNET.RESUNIT,
        dil_conv=cfg.SUPERNET.DIL_CONV,
        slice=cfg.SUPERNET.SLICE,
        verbose=cfg.VERBOSE,
        logger=logger)
    
    # print(model)
    # initialize meta matching networks
    MetaMN = MetaMatchingNetwork(cfg)
    
    # number of choice blocks in supernet
    choice_num = model.choices # First bottlecsp
    if args.local_rank == 0:
        logger.info('Supernet created, param count: %d', (
            sum([m.numel() for m in model.parameters()])))
        logger.info('resolution: %d', (resolution))
        logger.info('choice number: %d', (choice_num))

    #initialize prioritized board
    prioritized_board = PrioritizedBoard(cfg, CHOICE_NUM=choice_num, sta_num=sta_num, acc_gap=0.06)
    # print(model.blocks[1])
    prunable_module_type = (nn.BatchNorm2d, )
    prunable_modules = []
    CBL_idx = []
    for idx, module in enumerate(model.modules()):
        if isinstance(module, ConvNP):
            CBL_idx.append(idx)    
    # initialize flops look-up table
    model_est = FlopsEst(model, input_shape=(2, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE))
    if args.collect_samples > 0:
        collect_samples(args.collect_samples, model, prioritized_board, model_est)
        exit()

    def counting_forward_hook(module, inp, out):
        # try:
        # if not module.visited_backwards:
        #     return
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        model.module.K = model.module.K + K.cpu().numpy() + K2.cpu().numpy()
       
    for name, module in model.named_modules():
        if 'ReLU' in str(type(module)):
            module.visited_backwards = False
            #hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)
            # module.register_backward_hook(counting_backward_hook)
    # optionally resume from a checkpoint
    optimizer_state = None
    resume_epoch = None
    if cfg.AUTO_RESUME:
        optimizer_state, resume_epoch = resume_checkpoint(
            model, cfg.RESUME_PATH)

    # create optimizer and resume from checkpoint
    if args.resume_theta_training:
        print(f'Resuming training from: {args.resume_theta_training}')
        model.thetas_main = torch.load(args.resume_theta_training)
    optimizer, theta_optimizer = create_optimizer_supernet(cfg, model, USE_APEX)
    
    
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state['optimizer'])

    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [cfg.DATASET.IMAGE_SIZE] * 2]
    
    # convert model to distributed mode
    if cfg.BATCHNORM.SYNC_BN:
        try:
            if USE_APEX:
                model = convert_syncbn_model(model)
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if args.local_rank == 0:
                logger.info('Converted model to use Synchronized BatchNorm.')
        except Exception as exception:
            logger.info(
                'Failed to enable Synchronized BatchNorm. '
                'Install Apex or Torch >= 1.1 with Exception %s', exception)
    
    cuda = device.type != 'cpu'
    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs')

    model = model.to(device)
    
    # create learning rate scheduler
    lr_scheduler, num_epochs = create_supernet_scheduler(cfg, optimizer)

    start_epoch = resume_epoch if resume_epoch is not None else 0
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: %d', num_epochs)

    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    

    # Trainloader
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if args.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, args.data)  # check
    
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    
    dataloader, dataset = create_dataloader(train_path, imgsz, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect,
                                            world_size=args.world_size)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, args.data, nc - 1)


    # Testloader
    if args.local_rank in [-1, 0]:
        # ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader = create_dataloader(test_path, imgsz_test, cfg.DATASET.BATCH_SIZE, gs, args, hyp=hyp, augment=False,
                                       cache=args.cache_images, rect=True, local_rank=-1, world_size=args.world_size)[0]

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    arch_sampler = ArchSampler('candidate_samples_8_6_4_2_025_05_075.txt', 1250, model_est, prioritized_board=prioritized_board)
    
    if args.collect_synflows > 0:
        collect_synflows(args.collect_synflows, model, arch_sampler, device)
        exit()
    
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = train_loss_fn
    # for i in range(100):
    #     print(arch_sampler.sample_one_target_flops())
    # exit()
    synflow_cache = {}
    # initialize training parameters
    eval_metric = cfg.EVAL_METRICS
    best_metric, best_epoch, saver, best_children_pool = None, None, None, []
    if args.local_rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        # saver = CheckpointSaver(
        #     checkpoint_dir=output_dir,
        #     decreasing=decreasing)

    # training scheme
    for batch_idx, (input, target, paths, _) in enumerate(dataloader):
        target = target.to(device)
        input = input.to(device, non_blocking=True).float() / 255.0

        wot_map = model.module.calculate_wot(input, model.module.random_cand)
        break
    try:
        write_thetas(model.module.thetas_main, -1)
        for epoch in range(num_epochs):
            print(f'current epoch: {epoch}')
            train_metrics = train_epoch(epoch, model, dataloader, optimizer,
                                        train_loss_fn, prioritized_board, MetaMN, cfg,
                                        theta_optimizer=theta_optimizer,
                                        synflow_cache=synflow_cache,
                                        lr_scheduler=lr_scheduler, saver=saver,
                                        output_dir=output_dir, logger=logger,
                                        est=model_est, 
                                        local_rank=args.local_rank, 
                                        device=device, 
                                        world_size=args.world_size, 
                                        test_loader=testloader,
                                        arch_sampler=arch_sampler,
                                        train_theta=True, wot_map=wot_map)
            print('Writing thetas!')                            
            write_thetas(model.module.thetas_main, epoch)
            torch.save(model.module.thetas_main, 'thetas_weights/thetas.pt')
            print('Decreasing temperature!')
            model.module.temperature = model.module.temperature * np.exp(-0.065)
            # evaluate one epoch
        best_candidate_idx, best_candidate_map = validate(model, testloader, validate_loss_fn,
                                prioritized_board, cfg, device=device,
                                local_rank=args.local_rank, logger=logger)

            # update_summary(epoch, train_metrics, eval_metrics, os.path.join(
            #     output_dir, 'summary.csv'), write_header=best_metric is None)

            # if saver is not None:
            #     # save proper checkpoint with eval metric
            #     save_metric = eval_metrics[eval_metric]
            #     best_metric, best_epoch = saver.save_checkpoint(
            #         model, optimizer, cfg,
            #         epoch=epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass


def write_thetas(thetas, epoch):
    distributions = []
    for theta in thetas:
        distributions.append(softmax(theta().detach().cpu().numpy()))

    with open('thetas-v5-wot-precalculated-flops11.4-params46.6-layers-it2880-temp3-dec0.065-voc-second.txt', 'a') as f:
        f.write(f'epoch: {epoch}\n')
        f.writelines(str(distributions))

if __name__ == '__main__':
    main()
