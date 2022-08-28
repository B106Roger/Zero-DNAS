import torch
from torch import nn

def compute_loss_KD(feature_adaptation_layer, model, targets, output_t, feature_s, feature_t):
    lambda_feature = 0.0001

    lfeature = 0
    tcls, tbox, indices, anchor_vec, kd_masks = build_targets_kd(
        output_t, targets, model)
    norms1 = kd_masks[0].sum() + 10e-7
    norms2 = kd_masks[1].sum() + 10e-7
    norms3 = kd_masks[2].sum() + 10e-7

    kd_mask_1 = kd_masks[0].unsqueeze(1)
    kd_mask_2 = kd_masks[1].unsqueeze(1)
    kd_mask_3 = kd_masks[2].unsqueeze(1)

    lfeature += (torch.pow(feature_t[0] - feature_adaptation_layer(feature_s[0], 1), 2)
                 * kd_mask_1).sum() / norms1
    lfeature += (torch.pow(feature_t[1] - feature_adaptation_layer(feature_s[1], 2), 2)
                 * kd_mask_2).sum() / norms2
    lfeature += (torch.pow(feature_t[2] - feature_adaptation_layer(feature_s[2], 3), 2)
                 * kd_mask_3).sum() / norms3

    return lfeature * lambda_feature

def build_kd_mask(p, b, a, gj, gi, tbox):
    box_predictions = []
    box_predictions.append(p[..., :4])
    nB = box_predictions[0].size(0)
    nA = box_predictions[0].size(1)
    nG = box_predictions[0].size(2)
    
    
    iou_scores = torch.zeros(
                nB, nA, nG, nG, dtype=torch.float32).to(tbox.device)
    iou_scores[b, a, gj, gi] = (bbox_iou_another(box_predictions[0][b, a, gj,
                                                                            gi], tbox, x1y1x2y2=False))
    threshold = 0.5  # Threshold for IOU mask
    kd_mask = torch.zeros(
        nB, nG, nG, dtype=torch.float32).to(tbox.device)
    for i in range(nB):
        max_iou = iou_scores[i].max()
        max_iou_thresh = max_iou * threshold
        mask_per_img, _ = torch.max(iou_scores[i], dim=0)
        mask_per_img = (mask_per_img > max_iou_thresh).float()
        kd_mask[i, :, :] = mask_per_img
    return kd_mask


def build_targets_kd(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    all_kd_masks = []
    det = model.module.yolo_detector if is_parallel(model) else model.yolo_detector  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        a = t[:, 6].long()  # anchor indices

        kd_mask = build_kd_mask(p[i], b, a, gj, gi, tbox[i])

        # Append
        indices.append((b, a, gj.clamp_(0, gain[3]), gi.clamp_(0, gain[2])))  # image, anchor, grid indices
          # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        all_kd_masks.append(kd_mask)

    return tcls, tbox, indices, anch, all_kd_masks


def bbox_iou_another(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """

    box1 = box1.float()
    box2 = box2.float()

    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


class FeatureAdaptation(nn.Module):
    def __init__(self):
        super(FeatureAdaptation, self).__init__()
        self.adaptation_layer1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU())
        self.adaptation_layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU())
        self.adaptation_layer3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU())

    def forward(self, features, layer):
        if layer == 1:
            return self.adaptation_layer1(features)
        elif layer == 2:
            return self.adaptation_layer2(features)
        elif layer == 3:
            return self.adaptation_layer3(features)

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


