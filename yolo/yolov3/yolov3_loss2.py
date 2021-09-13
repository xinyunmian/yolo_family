import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
	return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

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

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def to_cpu(tensor):
    return tensor.detach().cpu()

def build_targets_yolov3New(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor
    FloatTensor = torch.cuda.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nH = pred_boxes.size(2)
    nW = pred_boxes.size(3)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nH, nW).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nH, nW).fill_(1)
    class_mask = FloatTensor(nB, nA, nH, nW).fill_(0)
    iou_scores = FloatTensor(nB, nA, nH, nW).fill_(0)
    tx = FloatTensor(nB, nA, nH, nW).fill_(0)
    ty = FloatTensor(nB, nA, nH, nW).fill_(0)
    tw = FloatTensor(nB, nA, nH, nW).fill_(0)
    th = FloatTensor(nB, nA, nH, nW).fill_(0)
    tcls = FloatTensor(nB, nA, nH, nW, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = torch.cuda.FloatTensor(target[:, 2:6].shape)
    target_boxes[:, 0] = target[:, 2] * nW
    target_boxes[:, 1] = target[:, 3] * nH
    target_boxes[:, 2] = target[:, 4] * nW
    target_boxes[:, 3] = target[:, 5] * nH
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

class Yolov3LossNew(nn.Module):
    def __init__(self, num_classes=2, anchors=None, num_anchors=3, stride=32):
        super(Yolov3LossNew, self).__init__()
        self.num_classes = num_classes # 20,80
        self.num_anchors = num_anchors #3
        self.anchor_step = int(len(anchors) / num_anchors) #2
        self.stride = stride
        self.anchors = anchors #40,15, 90,45, 120,55, 190,65, 220,88
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.metrics = {}
        self.thresh = 0.5

    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        nB = output.data.size(0) #batch=4
        nA = self.num_anchors #3
        nC = self.num_classes #2
        nH = output.data.size(2) #13 26 52
        nW = output.data.size(3) #13

        output = output.view(nB, nA, (5 + nC), nH, nW)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(output[..., 0])  # Center x
        y = torch.sigmoid(output[..., 1])  # Center y
        w = output[..., 2]  # Width
        h = output[..., 3]  # Height
        pred_conf = torch.sigmoid(output[..., 4])  # Conf
        pred_cls = torch.sigmoid(output[..., 5:])  # Cls pred.

        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(torch.cuda.FloatTensor)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(torch.cuda.FloatTensor)
        scaled_anchors = torch.FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]).cuda()
        anchor_w = scaled_anchors[:, 0].view(1, nA, 1, 1)
        anchor_h = scaled_anchors[:, 1].view(1, nA, 1, 1)

        pred_boxes = torch.cuda.FloatTensor(nB, nA, nH, nW, 4)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets_yolov3New(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=target,
            anchors=scaled_anchors,
            ignore_thres=self.thresh,
        )

        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()
        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            "loss": to_cpu(total_loss).item(),
            "x": to_cpu(loss_x).item(),
            "y": to_cpu(loss_y).item(),
            "w": to_cpu(loss_w).item(),
            "h": to_cpu(loss_h).item(),
            "conf": to_cpu(loss_conf).item(),
            "cls": to_cpu(loss_cls).item(),
            "cls_acc": to_cpu(cls_acc).item(),
            "recall50": to_cpu(recall50).item(),
            "recall75": to_cpu(recall75).item(),
            "precision": to_cpu(precision).item(),
            "conf_obj": to_cpu(conf_obj).item(),
            "conf_noobj": to_cpu(conf_noobj).item(),
            "grid_size": nH,
        }
        return total_loss / nB


class log_Loss(nn.Module):
    def __init__(self):
        super(log_Loss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(3), requires_grad=True).cuda()

    def forward(self, loss1, loss2, loss3,):
        var = torch.exp(-self.log_vars)
        loss_sum = (loss1 * var[0] + self.log_vars[0]) + (loss2 * var[1] + self.log_vars[1]) + (loss3 * var[2] + self.log_vars[2])
        return loss_sum