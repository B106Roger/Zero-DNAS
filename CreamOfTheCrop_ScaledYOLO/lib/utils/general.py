import glob
import math
import os
import random
import shutil
import subprocess
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from sys import platform
from copy import deepcopy

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from lib.utils.torch_utils import init_seeds, is_parallel, select_device, time_synchronized

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_seeds(seed=seed)


def get_latest_run(search_dir='./runs'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime)


def check_git_status():
    # Suggest 'git pull' if repo is out of date
    if platform in ['linux', 'darwin'] and not os.path.isfile('/.dockerenv'):
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    print('\nAnalyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to generate improved anchors, please wait...' % bpr)
        na = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print('New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int32)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int32), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

def compute_loss(p, targets, model, logger=None):  # predictions, targets, model
    """
    ****************************************************************************************************
    p[0] torch.Size([8, 3, 52, 52, 25])
    p[1] torch.Size([8, 3, 26, 26, 25])
    p[2] torch.Size([8, 3, 13, 13, 25])
    targets torch.Size([36, 6])
    s 1.0
    (1.4 if np >= 4 else 1.)= 1.0
    bs 8
    ****************************************************************************************************
    Return
    ------
    loss : torch.tensor([loss_value])
    
    """
    device = targets.device
    # print('targets:', targets.shape)
    # print([pr.shape for pr in p]) 
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model, logger)  # targets
    # tcls = [torch.tensor(num_of_all_obj_across_batch,) ... number of fpn]
    # tbox = [torch.tensor(num_of_all_obj_across_batch, 4) ... number of fpn]
    # indices = [element ... number of fpn]
    #   element => tuple (batch_idx, anchor_idx, grid_y_idx, grid_x_idx)
    # anchors = [torch.tensor(num_of_all_obj_across_batch, 2) ... number of fpn]
        
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if np == 5 else balance
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets


            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=False)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss
            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio
            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE
          
            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling
    lbox *= h['giou'] * s
    lobj *= h['obj'] * s * (1.4 if np >= 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size
    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

def compute_sensitive_loss(p, p_aug, targets, masks):
    """
    Parameters
    ----------
    p: model prediction from normal image
        [(batch, num_of_anchor, fpn_h, fpn_w, 4+1+cls) ......]
    p_aug: model prediction from normal image
        [(batch, num_of_anchor, fpn_h, fpn_w, 4+1+cls) ......]
    targets:
        not in used parameter
    masks: foreground mask
        [(batch, 1, fpn_h, fpn_w, 1) ......]

    Returns
    -------
    sensitive : torch.tensor, (1,)
    loss_item : torch.tensor, (3,) => [fore_sensitive, back_sensitive, sensitive]

    """
    fore_sensitive = 0
    back_sensitive = 0
    for (pi, pi_aug, mask) in zip(p, p_aug, masks):
        mask =  mask.unsqueeze(-1)
        pos_mask = mask
        neg_mask = 1-mask
        pix_count = mask.shape[2] * mask.shape[3]
        # [Roger]
        # pos_ratio = neg_ratio = (2,)
        pos_cnt   = pos_mask.sum((1,2,3,4))
        neg_cnt   = neg_mask.sum((1,2,3,4))
        pos_ratio = torch.nan_to_num(1.0 / pos_cnt)
        neg_ratio = torch.nan_to_num(1.0 / neg_cnt)
        # [Roger]
        # mse_diff = (batch, num_of_anchor, fpn_h, fpn_w, 4+1+num_of_cls)
        mse_diff = torch.abs(pi - pi_aug)
        
        fore_sensitive += - (mse_diff * pos_mask).sum((1,2,3,4)) * pos_ratio
        back_sensitive +=   (mse_diff * neg_mask).sum((1,2,3,4)) * neg_ratio

    fore_sensitive = fore_sensitive.mean()
    back_sensitive = back_sensitive.mean()
    sensitive = fore_sensitive + back_sensitive
    
    return sensitive, torch.tensor((
        fore_sensitive.clone(), 
        back_sensitive.clone(), 
        sensitive.clone(), 
    )).detach()

def build_targets(p, targets, model, logger=None):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.blocks[-1] if is_parallel(model) else model.blocks[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    
    # ai=(num_of_anchor, num_of_bbox)
    # [Roger]
    # input
    #   ai[:, :, None]=(num_of_anchor, num_of_bbox, 1)
    #   targets.repeat(na, 1, 1)=(num_of_anchor, num_of_bbox, 6)
    # output
    #   targets=(num_of_anchor, num_of_bbox, 7)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets
    # [Roger]
    # na, number of anchor, should always be 3
    # nt, number of target,
    # ai=(num_of_anchor, num_of_target), float32
    # off=(5,2) 

    for i in range(det.nl):
        # [Roger]
        # p[i] = (batch, num_of_anchor, fpn_h, fpn_w, num_cls+5)
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors      # [Roger]
        t = targets * gain              # targets=(num_of_anchor, num_of_bbox, 7) 7=>[batch,cls,xywh,anchor_idx]
                                        # gain   =[1,1,fpn_w, fpn_h, fpn_w, fpn_h, 1]
        
        # [Roger]
        # anchor=(3,2), float32, 3 anchor per fpn, xy
        # gain=(7,), float32, [1, 1, fpn_w, fpn_h, fpn_w, fpn_h, 1]
        # t=(3, num_of_targets, 7), float32 => the intemediate 4 value is at the grid coordinate [0, fpn_size]
        
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter
        
            # [Roger]
            # r=(num_of_anchor, num_of_bbox, 2), float32, the scale ratio for anchor to reach the wh of bbox
            # j=(num_of_anchor, num_of_bbox),    float32, choose which anchor is most suitable for the target
            # model.hyp["anchor_t"]=4.0
            # t=(num_of_bbox, 7) 7=>(batch, cls, xywh, anchor_idx)
            
            # [Roger]
            # gxy=(num_of_bbox, 2) coordinate for grid in xy
            # gxi=(num_of_bbox, 2) coordinate for grid in xy but in reverse order.
            # j=(5, num_of_bbox)
            # t=(increase_num_of_bbox, 2)
            # offsets=(increase_num_of_bbox, 2)
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            # [Roger]
            # j = k = l = m = (1, increase_num_of_bbox)
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T # g=0.5, j=>index for y,         k=>index for x
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T # g=0.5, l=>index for inverse y, m=>index for inverse x
            
            # [Roger]
            # j=(5,increase_num_of_bbox)
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            # [Roger]
            # simply increase the number of trainig sample, but have no idea why it do that @@?
            # increase the left-right-top-bot pixel for training, at most 5 pixel for training single bbox
            # t=(increase_num_of_bbox, 7)
            # t.repeat((5, 1, 1))=(5, increase_num_of_bbox, 7)
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            # offsets => (num_of_targets, 2) => contain one of coordinate
            # (0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.0, -0.5), (-0.5, 0.0)
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        # [Roger]
        # gain[3],gain[2] => fpn size, fpn size
        # gigj => the coordiante in fpn_size coordinate
        # gxy  => gxy 
        indices.append((b, a, gj.clamp_(0, gain[3]), gi.clamp_(0, gain[2])))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
    return tcls, tbox, indices, anch

def build_foreground_mask(mask_sizes, batch_size, targets, model):
    """
    mask_sizes:
    targets: (num_of_bbox, 6)
    """
    # mask_sizes = [torch.tensor(fpn.shape[2:4]) for fpn in p]
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.yolo_detector if is_parallel(model) else model.yolo_detector  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    
    

    
    # [Roger]
    # input
    #   ai            =(num_of_anchor, num_of_bbox)
    #   ai[:, :, None]=(num_of_anchor, num_of_bbox, 1)
    #   targets.repeat(na, 1, 1)=(num_of_anchor, num_of_bbox, 6)
    # output
    #   targets=(num_of_anchor, num_of_bbox, 7)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets
    # [Roger]
    # na, number of anchor, should always be 3
    # nt, number of target,
    # ai=(num_of_anchor, num_of_target), float32
    # off=(5,2) 
    masks = []
    for i, mask_size in enumerate(mask_sizes):
        # [Roger]
        # p[i] = (batch, num_of_anchor, fpn_h, fpn_w, num_cls+5)
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(mask_size.clone().detach().requires_grad_(False))[[1,0,1,0]]  # xyxy gain
        grid_y, grid_x = torch.meshgrid(torch.arange(mask_size[0]), torch.arange(mask_size[1]))
        mask = torch.zeros(batch_size, 1, mask_size[0], mask_size[1]).cuda()

        # Match targets to anchors      # [Roger]
        t = targets * gain              # targets=(num_of_anchor, num_of_bbox, 7) 7=>[batch,cls,xywh,anchor_idx]
                                        # gain   =[1,1,fpn_w, fpn_h, fpn_w, fpn_h, 1]
        
        
        # [Roger]
        # anchor=(3,2), float32, 3 anchor per fpn, xy
        # gain=(7,), float32, [1, 1, fpn_w, fpn_h, fpn_w, fpn_h, 1]
        # t=(3, num_of_targets, 7), float32 => the intemediate 4 value is at the grid coordinate [0, fpn_size]
        
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter
            
            xy_size = torch.tensor([[mask_size[1], mask_size[0]]]).cuda()
            # t_start = t_end = (t, 2)
            t_start = (t[:, 2:4] - t[:, 4:6] / 2)
            t_end   = (t[:, 2:4] + t[:, 4:6] / 2)
            
            # t_start = t_end = (t, 5) => [batch, x1, y1, x2, y2]
            t_infos = torch.cat([t[:, 0:1], t_start, t_end], -1).int()

            for t_info in t_infos:
                mask[t_info[0], 0, t_info[2]:t_info[4], t_info[1]:t_info[3]] = 1
                
        masks.append(mask)
    return masks

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        st= time.time()
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            # print(f'no bbox in image {xi}')
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        # print(f'process image{xi} time {time.time() - st}')
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def intersect(box_a, box_b):

    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]

def jaccard(box_a, box_b, iscrowd:bool=False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / (union + 0.0000001)

    return out if use_batch else out.squeeze(0)

def jaccard_diou(box_a, box_b, iscrowd:bool=False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    x1 = ((box_a[:, :, 2]+box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3]+box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2]+box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3]+box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)

    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    D = (((x2 - x1)**2 + (y2 - y1)**2) / ((cr-cl)**2 + (cb-ct)**2 + 1e-7))
    out = inter / area_a if iscrowd else inter / (union + 1e-7) - D ** 0.7
    return out if use_batch else out.squeeze(0)

def box_diou(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    clt=torch.min(boxes1[:, None, :2], boxes2[:, :2])
    crb=torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    x1=(boxes1[:, None, 0] + boxes1[:, None, 2])/2
    y1=(boxes1[:, None, 1] + boxes1[:, None, 3])/2
    x2=(boxes2[:, None, 0] + boxes2[:, None, 2])/2
    y2=(boxes2[:, None, 1] + boxes2[:, None, 3])/2
    d=(x1-x2.t())**2 + (y1-y2.t())**2
    c=((crb-clt)**2).sum(dim=2)

    inter = (rb - lt).clamp(min=0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter) - (d / c)**0.6  # iou = inter / (area1 + area2 - inter)

# For batch mode Cluster-Weighted NMS
def non_max_suppression2(prediction, conf_thres=0.1, iou_thres=0.6, max_box=1500, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]   
    pred1 = (prediction < -1).float()[:,:max_box,:6]    # pred1.size()=[batch, max_box, max_box] denotes boxes without offset by class
    pred2 = pred1[:,:,:4]+0   # pred2 denotes boxes with offset by class
    batch_size = prediction.shape[0]   
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        x = x[x[:, 4].argsort(descending=True)]
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes

        boxes = (x[:, :4].clone() + c.view(-1, 1) * max_wh)[:max_box]  # boxes (offset by class), scores
        pred2[xi,:] = torch.cat((boxes, pred2[xi,:]), 0)[:max_box]        # If less than max_box, padding 0.
        pred1[xi,:] = torch.cat((x[:max_box], pred1[xi,:]), 0)[:max_box]

    # Batch mode Cluster-Weighted NMS

    iou = jaccard_diou(pred2, pred2).triu_(diagonal=1)    # switch to 'jaccard_diou' function for using Cluster-DIoU-NMS
    B = iou
    for i in range(200):
        A=B
        maxA=A.max(dim=1)[0]
        E = (maxA<iou_thres).float().unsqueeze(2).expand_as(A)
        B=iou.mul(E)
        if A.equal(B)==True:
            break
    keep = (maxA <= iou_thres) 
    weights = (B*(B>0.8) + torch.eye(max_box).cuda().expand(batch_size,max_box,max_box)) * (pred1[:,:,4].reshape((batch_size,1,max_box)))
    pred1[:,:, :4]=torch.matmul(weights,pred1[:,:,:4]) / weights.sum(2, keepdim=True)   # weighted coordinates

    for jj in range(batch_size):
        output[jj] = pred1[jj][keep[jj]]

    return output


def strip_optimizer(f='weights/best.pt', s=''):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/train2017/'):  # from utils.utils import *; coco_only_people()
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def crop_images_random(path='../images/', scale=0.50):  # from utils.utils import *; crop_images_random()
    # crops images into random squares up to scale fraction
    # WARNING: overwrites images!
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        img = cv2.imread(file)  # BGR
        if img is not None:
            h, w = img.shape[:2]

            # create random mask
            a = 30  # minimum size (pixels)
            mask_h = random.randint(a, int(max(a, h * scale)))  # mask height
            mask_w = mask_h  # mask width

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.utils import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def output_to_target(output, width, height):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    elif isinstance(output, list):
        output = [item.cpu() if item is not None else None for item in output]

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)


def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')


# Plotting functions ---------------------------------------------------------------------------------------------------
def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='YOLO')
    plt.plot(x, yb ** 2, '.-', label='YOLO ^2')
    plt.plot(x, yb ** 1.6, '.-', label='YOLO ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    # if os.path.isfile(fname):  # do not overwrite
    #     print('[Alert] To prevent overwrite file, lib.utils.general.plot_images do nothing and return immediately.')
    #     return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)


def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_study_txt(f='study.txt', x=None):  # from utils.utils import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in ['']:
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        for i in range(7):
            ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
            ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, :j], y[3, :j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=Path(f).stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [33.8, 39.6, 43.0, 47.5, 49.4, 50.7],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid()
    ax2.set_xlim(0, 30)
    ax2.set_ylim(28, 50)
    ax2.set_yticks(np.arange(30, 55, 5))
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig('study_mAP_latency.png', dpi=300)
    plt.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_labels(labels, save_dir=''):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig(Path(save_dir) / 'labels.png', dpi=200)
    plt.close()


def plot_evolution(yaml_file='runs/evolve/hyp_evolved.yaml'):  # from utils.utils import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 10), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(5, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
                # y_smooth = butter_lowpass_filtfilt(y)
                # ax[i].plot(x, np.gradient(y_smooth), marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=(), labels=(),
                 save_dir=''):  # from utils.utils import *; plot_results()
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov3
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        files = glob.glob(str(Path(save_dir) / 'results*.txt')) + glob.glob('../../Downloads/results*.txt')
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else Path(f).stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except:
            print('Warning: Plotting error for %s, skipping file' % f)

    fig.tight_layout()
    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)

# Testing Data. Display Loss and mAP
def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         save_txt=False,
         logger=None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(opt.device, batch_size=batch_size)
        merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
        if save_txt:
            out = Path('inference/output')
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    if logger is not None:
        s = ' ' * 18 + '| ' + s
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    #model = model.to(memory_format=torch.channels_last)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
        # if batch_i == 10: break
        
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            [inf_out, train_out] = model(img)  # inference and training outputs
            #inf_out, train_out = model(img.to(memory_format=torch.channels_last), augment=augment)  # inference and training outputs
            t0_current = time_synchronized() - t
            t0 += t0_current

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
            t1_current = time_synchronized() - t
            t1 += t1_current
            
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                txt_path = str(out / Path(paths[si]).stem)
                pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in pred:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = Path(paths[si]).stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

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



        # Plot images
        if batch_i < 1:
            f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            # print(f'[Test] targets {targets[targets[:,0]==0].shape}', targets[targets[:,0]==0])
            plot_images(img, targets, paths, str(f), names)  # ground truth
            f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
            preds = output_to_target(output, width, height)
            # print(f'[Test] preds {preds[preds[:,0]==0].shape}', preds[preds[:,0]==0])
            plot_images(img, preds, paths, str(f), names)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    if logger is not None:
        logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    else:
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

_RNAD_=0
def random_testing(tag='', out=[True, False, False]):
    global _RNAD_
    _RNAD_+=1
    if tag == '': tag = 'START'
    print(f'[Random Test{_RNAD_}] {tag}')
    a=torch.rand((4,), dtype=torch.float32).numpy()
    b=np.random.rand(4)
    c=random.random()
    if out[0]: print(f'[Random Test{_RNAD_}] pytorch {a}')
    if out[1]: print(f'[Random Test{_RNAD_}] numpy   {b}')
    if out[2]: print(f'[Random Test{_RNAD_}] python  {c:.6f}')
    

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates      = updates  # number of EMA updates
        self.updates_arch = updates  # number of EMA updates
        self.decay        = lambda x: decay * (1 - math.exp(-x / 2000))  #2000 decay exponential ramp (to help early epochs)
        self.decay_arch   = lambda x: decay * (1 - math.exp(-x / 100))  #25 decay exponential ramp (to help early epochs)
        
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model, update_arch):
        # Update EMA parameters
        # nn.module parameters only !!
        if not update_arch:
            with torch.no_grad():
                self.updates += 1
                d = self.decay(self.updates)

                msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
                for k, v in self.ema.state_dict().items():
                    if v.dtype.is_floating_point:
                        v *= d
                        v += (1. - d) * msd[k].detach()

        # Update theta parameters
        if update_arch:
            with torch.no_grad():
                self.updates_arch += 1
                d = self.decay_arch(self.updates_arch)
                
                model_theta = model.module.thetas_main if is_parallel(model) else model.thetas_main
                for stage_idx, arch in enumerate(self.ema.thetas_main):
                    if 'Composite' in arch['block_name']:
                        # ENA Update
                        arch['operators_choice'] *= d
                        arch['operators_choice'] += (1. - d) * model_theta[stage_idx]['operators_choice'].detach()

                        for choice_arch in arch['operators']:
                            for op_idx, (key, value) in enumerate(choice_arch.items()):
                                if key != 'block_name':
                                    # ENA Update
                                    choice_arch[key] *= d
                                    choice_arch[key] += (1. - d) * model_theta[stage_idx]['operators'][op_idx][key].detach()
                        
                    elif 'Search' in arch['block_name']:
                        for key, value in arch.items():
                            if key != 'block_name':
                                ###########################
                                # ENA Update
                                ###########################
                                # arch[key] *= d
                                # arch[key] += (1. - d) * model_theta[stage_idx][key].detach()
                                
                                ###########################
                                # Directly Update
                                ###########################
                                arch[key] = model_theta[stage_idx][key].detach()
        
        # Update temperature
        self.ema.temperature = model.module.temperature if is_parallel(model) else model.temperature
        
        
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
