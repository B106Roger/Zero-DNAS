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
import shutil

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
from lib.models.blocks.yolo_blocks import Conv, ConvNP, BottleneckCSP, BottleneckCSP2, set_algorithm_type
from lib.utils.general import check_img_size, labels_to_class_weights
from lib.utils.torch_utils import select_device
from lib.utils.attentive_sampling import collect_samples
from scipy.special import softmax
from lib.models.AttentiveNasSampler import ArchSampler

def config_backup(backup_path, args):
    os.makedirs(backup_path, exist_ok=True)
    shutil.copy(args.cfg,  os.path.join(backup_path, os.path.basename(args.cfg)))
    shutil.copy(args.hyp,  os.path.join(backup_path, os.path.basename(args.hyp)))
    shutil.copy(args.data, os.path.join(backup_path, os.path.basename(args.data)))
    shutil.copy('lib/core/train.py', os.path.join(backup_path, 'core_train.py'))
    shutil.copy('tools/train.py', os.path.join(backup_path, 'tools_train.py'))
    shutil.copy('lib/models/structures/supernet.py', os.path.join(backup_path, 'supernet.py'))
    shutil.copy('lib/models/blocks/yolo_blocks.py', os.path.join(backup_path, 'yolo_blocks.py'))
    shutil.copy('lib/models/builders/build_supernet.py', os.path.join(backup_path, 'build_supernet.py'))
    
    
    
    with open(os.path.join(backup_path, 'commandline.txt'), 'w') as f:
        f.writelines(' '.join(sys.argv))
    
task_dict = {
    'NAS-SS': { 'GFLOPS': 5.7,  'PARAMS': 32.0, 'CHOICES': {'n_bottlenecks': [0, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-S':  { 'GFLOPS': 7.0,  'PARAMS': 36.0, 'CHOICES': {'n_bottlenecks': [0, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-M':  { 'GFLOPS': 9.0,  'PARAMS': 40.0, 'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS':    { 'GFLOPS': 11.9, 'PARAMS': 52.5, 'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
    'NAS-L':  { 'GFLOPS': 16.5, 'PARAMS': 70.2, 'CHOICES': {'n_bottlenecks': [8, 6, 4, 2], 'gamma': [0.25, 0.5, 0.75]}},
}
task_name = 'NAS-L'
TASK_FLOPS      = task_dict[task_name]['GFLOPS']     # e.g TASK_FLOPS  = 5  means 50 GFLOPs
TASK_PARAMS     = task_dict[task_name]['PARAMS']     # e.g TASK_PARAMS = 32 means 32 million parameters.
SEARCH_SPACES   = task_dict[task_name]['CHOICES']
def main():
    args, cfg = parse_config_args('super net training')
    # resolve logging
    # output_dir = os.path.join(cfg.SAVE_PATH,
    #                           "{}-{}".format(datetime.now().strftime('%m%d-%H:%M:%S'),
    #                                          cfg.MODEL))
    output_dir = os.path.join(cfg.SAVE_PATH, cfg.exp_name)
    output_bakup_dir = os.path.join(output_dir, 'config')
    config_backup(output_bakup_dir, args)

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
    # set block static argument
    set_algorithm_type('ZeroCost')
    BottleneckCSP.set_search_space(cfg.search_space.BOTTLENECK_CSP)
    BottleneckCSP2.set_search_space(cfg.search_space.BOTTLENECK_CSP2)
    
    # generate supernet
    print('SEARCH_SPACES', SEARCH_SPACES)
    model, sta_num, resolution = gen_supernet(
        flops_minimum=cfg.SUPERNET.FLOPS_MINIMUM,
        flops_maximum=cfg.SUPERNET.FLOPS_MAXIMUM,
        choices=SEARCH_SPACES,
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
    # initialize flops look-up table. 
    # It's well-known that "MACS * 2 = FLOPs"
    # Note that the batch should be 2, because the origin code is used to calculate MACS 
    model_est = FlopsEst(model, input_shape=(None, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE))
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
        print(f'Using {torch.cuda.device_count()} GPUs. device: {device}')

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

    # arch_sampler = ArchSampler('candidate_samples_8_6_4_2_025_05_075.txt', 1250, model_est, prioritized_board=prioritized_board)
    arch_sampler = None
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
        random_cand = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0, 0, 0], [0], [0], [0, 0, 0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        wot_map = model.module.calculate_wot(input, random_cand)
        break
    print('#'*15 + '   forward_variance   ' + '#'*15 )
    model.module.forward_variance()
    print('#'*15 + '   forward_stage_variance   ' + '#'*15 )
    model.module.forward_stage_variance()
    print('#'*15 + '   forward_whole_variance   ' + '#'*15 )
    model.module.forward_whole_variance()


def write_thetas(output_dir, thetas, epoch):
    alpha_distributions = []
    beta_distributions = []
    for theta in thetas:
        alpha=theta().detach().cpu().numpy()
        alpha_distributions.append(alpha)
        beta_distributions.append(softmax(alpha))

    with open(os.path.join(output_dir, 'alpha_distribution.txt'), 'a') as f:
        f.write(f'epoch: {epoch}')
        f.writelines(str(alpha_distributions)+'\n')
    with open(os.path.join(output_dir, 'beta_distribution.txt'), 'a') as f:
        f.write(f'epoch: {epoch}')
        f.writelines(str(beta_distributions)+'\n')

def write_final_thetas(output_dir, thetas):
    beta_distributions = []
    for theta in thetas:
        alpha=theta().detach().cpu().numpy()
        beta_distributions.append(softmax(alpha))

    with open(os.path.join(output_dir, 'thetas.txt'), 'w') as f:
        f.writelines(str(beta_distributions)+'\n')

if __name__ == '__main__':
    main()
