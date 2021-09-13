import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np

def ciou_yolov4(bboxes_a, bboxes_b, xyxy=True, CIOU=False):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if CIOU:
        c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v)
        return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def build_targets_yolov4(pred, labels, n_classes, anchors, anch_masks, num_anchors, nH, nW, n_ch, ignore_thre):
    # target assignment
    pred = pred.data
    batchsize = pred.size(0)  # batch=4
    tgt_mask = torch.zeros(batchsize, num_anchors, nH, nW, 4 + n_classes).cuda()
    obj_mask = torch.ones(batchsize, num_anchors, nH, nW).cuda()
    tgt_scale = torch.zeros(batchsize, num_anchors, nH, nW, 2).cuda()
    target = torch.zeros(batchsize, num_anchors, nH, nW, n_ch).cuda()

    # labels batch*50*5
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

    truth_x_all = labels[:, :, 1] * nW
    truth_y_all = labels[:, :, 2] * nH
    truth_w_all = labels[:, :, 3] * nW
    truth_h_all = labels[:, :, 4] * nH
    truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
    truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

    for b in range(batchsize):
        n = int(nlabel[b])
        if n == 0:
            continue
        truth_box = torch.zeros(n, 4).cuda()
        truth_box[:n, 2] = truth_w_all[b, :n]
        truth_box[:n, 3] = truth_h_all[b, :n]
        truth_i = truth_i_all[b, :n]
        truth_j = truth_j_all[b, :n]

        # calculate iou between truth and reference anchors
        ref_anchors = np.zeros((num_anchors, 4), dtype=np.float32)
        ref_anchors[:, 2] = np.array(anchors[0:-1:2], dtype=np.float32)
        ref_anchors[:, 3] = np.array(anchors[1::2], dtype=np.float32)
        ref_anchors = torch.from_numpy(ref_anchors)
        anchor_ious_all = ciou_yolov4(truth_box.cpu(), ref_anchors, CIOU=True)

        best_n_all = anchor_ious_all.argmax(dim=1)
        best_n = best_n_all % 3
        best_n_mask = ((best_n_all == anch_masks[0]) | (best_n_all == anch_masks[1]) | (best_n_all == anch_masks[2]))

        if sum(best_n_mask) == 0:
            continue

        truth_box[:n, 0] = truth_x_all[b, :n]
        truth_box[:n, 1] = truth_y_all[b, :n]

        pred_ious = ciou_yolov4(pred[b].view(-1, 4), truth_box, xyxy=False)
        pred_best_iou, _ = pred_ious.max(dim=1)
        pred_best_iou = (pred_best_iou > ignore_thre)
        pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
        # set mask to zero (ignore) if pred matches truth
        obj_mask[b] = ~ pred_best_iou

        for ti in range(best_n.shape[0]):
            if best_n_mask[ti] == 1:
                i, j = truth_i[ti], truth_j[ti]
                a = best_n[ti]
                obj_mask[b, a, j, i] = 1
                tgt_mask[b, a, j, i, :] = 1
                target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / anchors[2 * a] + 1e-16)
                target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / anchors[2 * a + 1] + 1e-16)
                target[b, a, j, i, 4] = 1
                target[b, a, j, i, 5 + labels[b, ti, 0].to(torch.int16).cpu().numpy()] = 1
                tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / nW / nH)
    return obj_mask, tgt_mask, tgt_scale, target

class Yolov4Loss(nn.Module):
    def __init__(self, num_classes=2, num_anchors=3, anchors=[0.68,1.38,0.97,1.87,1.03,2.65], anchor_mask=[0, 1, 2]):
        super(Yolov4Loss, self).__init__()
        self.num_classes = num_classes # 20,80
        self.anchors = anchors #40,15, 90,45, 120,55, 190,65, 220,88
        self.num_anchors = num_anchors #5
        self.anchor_mask = anchor_mask
        self.anchor_step = len(anchors) / num_anchors #2
        self.scale_x_y = 1.0
        self.ignore_thre = 0.5

    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        batchsize = output.data.size(0) #batch=4
        nA = self.num_anchors #3
        nC = self.num_classes #80
        nH = output.data.size(2) #13 26 52
        nW = output.data.size(3) #13
        n_ch = 5 + nC

        output = output.view(batchsize, self.n_anchors, n_ch, nH, nW)
        output = output.permute(0, 1, 3, 4, 2)  # b 3 26 26 85

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

        grid_x = torch.arange(nW, dtype=torch.float).repeat(batchsize, 3, nW, 1).cuda()
        grid_y = torch.arange(nH, dtype=torch.float).repeat(batchsize, 3, nH, 1).permute(0, 1, 3, 2).cuda()
        anchor_w = torch.from_numpy(np.array(self.anchors[0:-1:2], dtype=np.float32)).repeat(batchsize, nH, nW, 1).permute(0, 3, 1, 2).cuda()
        anchor_h = torch.from_numpy(np.array(self.anchors[1::2], dtype=np.float32)).repeat(batchsize, nH, nW, 1).permute(0, 3, 1, 2).cuda()
        pred = output[..., :4].clone()
        pred[..., 0] += grid_x
        pred[..., 1] += grid_y
        pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w
        pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h

        obj_mask, tgt_mask, tgt_scale, obj_target = \
            build_targets_yolov4(pred, target, nC, self.anchors, self.anchor_mask, nA, nH, nW, n_ch, self.ignore_thre)

        # loss calculation
        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        obj_target[..., 4] *= obj_mask
        obj_target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        obj_target[..., 2:4] *= tgt_scale

        loss_xy = F.binary_cross_entropy(input=output[..., :2], target=obj_target[..., :2], weight=tgt_scale * tgt_scale, reduction='sum')
        loss_wh = F.mse_loss(input=output[..., 2:4], target=obj_target[..., 2:4], reduction='sum') / 2
        loss_obj = F.binary_cross_entropy(input=output[..., 4], target=obj_target[..., 4], reduction='sum')
        loss_cls = F.binary_cross_entropy(input=output[..., 5:], target=obj_target[..., 5:], reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss / batchsize








