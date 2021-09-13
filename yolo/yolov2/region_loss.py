import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def bbox_iou(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea

def build_label(pred_boxes, target, anchors, num_anchors, nH, nW, noobject_scale, object_scale, sil_thresh, seen=15000):
    nB = target.size(0) #batch=4
    nA = num_anchors #5
    anchor_step = int(len(anchors) / num_anchors) #2
    conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask = torch.zeros(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW)

    nAnchors = nA * nH * nW
    nPixels = nH * nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
            cur_ious = torch.max(cur_ious, bbox_iou(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        cur_ious = cur_ious.view(nA, nH, nW)
        conf_mask[b][cur_ious > sil_thresh] = 0
    if seen < 12800:
        tx.fill_(0.5)
        ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            gt_box = torch.FloatTensor([0, 0, gw, gh])
            for n in range(nA):
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                anchor_box = torch.FloatTensor([0, 0, aw, ah])
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n

            gt_box = torch.FloatTensor([gx, gy, gw, gh])
            pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t * 5 + 1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t * 5 + 2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw / anchors[anchor_step * best_n])
            th[b][best_n][gj][gi] = math.log(gh / anchors[anchor_step * best_n + 1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t * 5]
            if iou > 0.5:
                nCorrect = nCorrect + 1
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class loss_v2(nn.Module):
    def __init__(self, num_classes=2, anchors=[], num_anchors=5):
        super(loss_v2, self).__init__()
        self.num_classes = num_classes # 20,80
        self.anchors = anchors #40,15, 90,45, 120,55, 190,65, 220,88
        self.num_anchors = num_anchors #5
        self.anchor_step = int(len(anchors) / num_anchors) #2
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6

    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        nB = output.data.size(0) #batch=4
        nA = self.num_anchors #5
        nC = self.num_classes #2
        nH = output.data.size(2) #13
        nW = output.data.size(3) #13

        output = output.view(nB, nA, (5 + nC), nH, nW)
        x = torch.sigmoid(output.index_select(2, torch.tensor([0]).cuda()).view(nB, nA, nH, nW))
        y = torch.sigmoid(output.index_select(2, torch.tensor([1]).cuda()).view(nB, nA, nH, nW))
        w = output.index_select(2, torch.tensor([2]).cuda()).view(nB, nA, nH, nW)
        h = output.index_select(2, torch.tensor([3]).cuda()).view(nB, nA, nH, nW)
        conf = torch.sigmoid(output.index_select(2, torch.tensor([4]).cuda()).view(nB, nA, nH, nW))
        cls = output.index_select(2, torch.linspace(5, 5 + nC - 1, nC).long().cuda())
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)

        pred_boxes = torch.cuda.FloatTensor(4, nB * nA * nH * nW)
        # pred_boxes = torch.zeros(size = (4, nB * nA * nH * nW)).cuda()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        xv = x.view(nB * nA * nH * nW)
        yv = y.view(nB * nA * nH * nW)
        wv = w.view(nB * nA * nH * nW)
        hv = h.view(nB * nA * nH * nW)
        pred_boxes[0] = xv.data + grid_x
        pred_boxes[1] = yv.data + grid_y
        pred_boxes[2] = torch.exp(wv.data) * anchor_w
        pred_boxes[3] = torch.exp(hv.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls \
            = build_label(pred_boxes, target.data, self.anchors, nA, nH, nW, self.noobject_scale, self.object_scale, self.thresh)

        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum())

        tx = tx.cuda()
        ty = ty.cuda()
        tw = tw.cuda()
        th = th.cuda()
        tconf = tconf.cuda()
        tcls = tcls[cls_mask].view(-1).long().cuda()

        coord_mask = coord_mask.cuda()
        conf_mask = conf_mask.cuda().sqrt()
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC).cuda()
        cls = cls[cls_mask].view(-1, nC)

        loss_x = self.coord_scale * nn.MSELoss(reduction='sum')(x * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.MSELoss(reduction='sum')(y * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * nn.MSELoss(reduction='sum')(w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss(reduction='sum')(h * coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(reduction='sum')(conf * conf_mask, tconf * conf_mask) / 2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction='sum')(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # print('nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
        # nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
        # loss_conf.item(), loss_cls.item(), loss.item()))
        return loss / nB

def bbox_iou_new(box1, box2, x1y1x2y2=True):
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
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def build_targets_yolov2(target, anchors, num_anchors, inh, inw, nclass, sil_thresh):
    target = target.data
    nB = target.size(0) #batch=4
    nA = num_anchors #5
    nC = nclass
    nH = inh
    nW = inw
    ignore_thresh = sil_thresh

    obj_mask = torch.zeros(nB, nA, nH, nW)
    noobj_mask = torch.ones(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW, nC)

    for b in range(nB):
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            # gt_box = [0, 0, gw, gh]
            gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0)
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((nA, 2)), np.array(anchors)), 1))
            anch_ious = bbox_iou_new(gt_box, anchor_shapes)
            noobj_mask[b, anch_ious > ignore_thresh, gj, gi] = 0
            best_n = np.argmax(anch_ious)

            obj_mask[b][best_n][gj][gi] = 1
            noobj_mask[b][best_n][gj][gi] = 0
            tx[b][best_n][gj][gi] = gx - gi
            ty[b][best_n][gj][gi] = gy - gj
            tw[b][best_n][gj][gi] = math.log(gw / anchors[best_n, 0])
            th[b][best_n][gj][gi] = math.log(gh / anchors[best_n, 1])
            tconf[b][best_n][gj][gi] = 1
            class_index = int(target[b][t * 5])
            tcls[b][best_n][gj][gi][class_index] = 1
    return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls

class yolov2_loss(nn.Module):
    def __init__(self, num_classes=2, anchors=[], num_anchors=5, stride=32):
        super(yolov2_loss, self).__init__()
        self.num_classes = num_classes # 20,80
        self.num_anchors = num_anchors #3
        self.anchor_step = 2 #2
        self.stride = stride
        self.anchors = anchors #40,15, 90,45, 120,55, 190,65, 220,88
        # self.anchors = [anchor / self.stride for anchor in anchors]
        self.thresh = 0.6

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

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

        scaled_anchors = [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]

        obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls = build_targets_yolov2(target, scaled_anchors, nA, nH, nW, nC, self.thresh)

        obj_mask, noobj_mask = obj_mask.cuda(), noobj_mask.cuda()
        tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
        tconf, tcls = tconf.cuda(), tcls.cuda()

        loss_x = self.bce_loss(x * obj_mask, tx * obj_mask)
        loss_y = self.bce_loss(y * obj_mask, ty * obj_mask)
        loss_w = self.mse_loss(w * obj_mask, tw * obj_mask)
        loss_h = self.mse_loss(h * obj_mask, th * obj_mask)
        loss_conf = self.bce_loss(pred_conf * obj_mask, obj_mask) + 0.5 * self.bce_loss(pred_conf * noobj_mask, noobj_mask * 0.0)
        loss_cls = self.bce_loss(pred_cls[obj_mask == 1], tcls[obj_mask == 1])

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return loss / nB






