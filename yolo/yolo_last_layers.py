# 3、yolo_loss（）
# YOLOv2的损失函数较YOLOv1也有比较大的改变，主要分为三大部分的损失，IOU损失，分类损失，坐标损失。IOU损失分为了no_objects_loss
# 和objects_loss，两者相比对objects_loss的惩罚更大。下面简单介绍一下和YOLOv1的区别。
# 3.1、confidence_loss：
# YOLOv2中，总共有845个anchor_boxes，与true_boxes匹配的用于预测pred_boxes，未与true_boxes匹配的anchor_boxes用于预测background。
# objects_loss（true_boxes所匹配的anchor_boxes）
# 与true_boxes所匹配的anchor_boxes去和预测的pred_boxes计算objects_loss。
# no_objects_loss（true_boxes未匹配的anchor_boxes）
# 1、未与true_boxes所匹配的anchor_boxes中，若与true_boxes的IOU>0.6，则无需计算loss。
# 2、未与true_boxes所匹配的anchor_boxes中，若与true_boxes的IOU<0.6，则计算no_objects_loss。
# 这里疑惑点比较多，也比较绕，不太好理解，自己当时也理解错了。后来自己理解：confidence是为了衡量anchor_boxes是否有物体的置信度，
# 对于负责预测前景（pred_boxes）的anchors_boxes来说，我们必须计算objects_loss；对于负责预测背景（background）的anchors_boxes来说，
# 若与true_boxes的IOU<0.6，我们需要计算no_objects_loss。这两条都好理解，因为都是各干各的活。但若与true_boxes的IOU>0.6时，
# 则不需要计算no_objects_loss。这是为什么呢？因为它给了我们惊喜，我们不忍苛责它。一个负责预测背景的anchor_boxes居然和true_boxes的IOU>0.6，
# 框的甚至比那些本来就负责预测前景的anchors要准，吃的是草，挤的是奶，怎么能再惩罚它呢？好了言归正传，
# 我个人觉得是因为被true_boxes的中心点可能在附近的gird cell里，但是true_boxes又比较大，
# 导致它和附近gird cell里的anchors_boxes的IOU很大，那么这部分造成的损失可以不进行计算，
# 毕竟它确实框的也准。就像faster rcnn中0.3<IOU<0.7的anchors一样不造成损失，因为这部分并不是重点需要优化的对象。
# 与YOLOv1不同的是修正系数的改变，YOLOv1中no_objects_loss和objects_loss分别是0.5和1，而YOLOv2中则是1和5。
# 3.2、classification_loss：
# 这部分和YOLOv1基本一致，就是经过softmax（）后，20维向量（数据集中分类种类为20种）的均方误差。
# 3.3、coordinates_loss：
# 这里较YOLOv1的改动较大，计算x,y的误差由相对于整个图像（416x416）的offset坐标误差的均方改变为相对于gird cell的offset
# （这个offset是取sigmoid函数得到的处于（0,1）的值）坐标误差的均方。也将修正系数由5改为了1 。
# 计算w,h的误差由w,h平方根的差的均方误差变为了，w,h与对true_boxes匹配的anchor_boxes的长宽的比值取log函数，
# 和YOLOv1的想法一样，对于相等的误差值，降低对大物体误差的惩罚，加大对小物体误差的惩罚。同时也将修正系数由5改为了1。

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

def build_targets_region(pred_boxes, target, anchors, num_anchors, nH, nW, noobject_scale, object_scale, sil_thresh, seen=15000):
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

class RegionLoss(nn.Module):
    def __init__(self, num_classes=2, anchors=[], num_anchors=5):
        super(RegionLoss, self).__init__()
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
            = build_targets_region(pred_boxes, target.data, self.anchors, nA, nH, nW, self.noobject_scale, self.object_scale, self.thresh)

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

        print('nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
        nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
        loss_conf.item(), loss_cls.item(), loss.item()))
        return loss

def build_targets_detection(target, gridsize, num_boxes, num_classes):
    nB = target.data.size(0)
    labels = torch.zeros(nB, 5*num_boxes + num_classes, gridsize, gridsize)
    for b in range(nB):
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            gc = target[b][t * 5]
            gx = target[b][t * 5 + 1] * gridsize
            gy = target[b][t * 5 + 2] * gridsize
            gw = target[b][t * 5 + 3] * gridsize
            gh = target[b][t * 5 + 4] * gridsize
            gridx = gx - int(gx)
            gridy = gy - int(gy)

            for b_n in range(num_boxes):
                labels[b, (b_n * 5):(b_n * 5 + 5), int(gy), int(gx)] = torch.tensor([gridx, gridy, gw, gh, 1])
                labels[b, num_boxes * 5 + int(gc), int(gy), int(gx)] = 1
    return labels


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=2, num_boxes=3, gridsize=7):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes # 20,80
        self.num_boxes = num_boxes #3
        self.gridsize = gridsize
        self.coord_scale = 5
        self.noobject_scale = 0.5
        self.object_scale = 1
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
        self.xyloss = nn.MSELoss(reduction="sum")
        self.whloss = nn.MSELoss(reduction="sum")
        self.objcof = nn.MSELoss(reduction="sum")
        self.noobjcof = nn.MSELoss(reduction="sum")
        self.classloss = nn.MSELoss(reduction="sum")

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, output, target):
        # output : B * ((4 + 1) * num_box + num_classes) * H * W
        output = output.view(-1, self.gridsize, self.gridsize, 5 * self.num_boxes + self.num_classes)
        # output = output.permute(0, 2, 3, 1).contiguous()
        target = target.permute(0, 3, 1, 2).contiguous().cuda()
        nB = output.data.size(0) #batch=4

        # 获取含有obj和不含obj的mask
        obj_mask = target[:, :, :, 4] == 1
        obj_mask = obj_mask.unsqueeze(-1).expand_as(target) #b 7 7 30
        noobj_mask = target[:, :, :, 4] == 0
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target) #b 7 7 30

        # 提取出对应mask的box和class
        obj_pred = output[obj_mask].view(-1, 5 * self.num_boxes + self.num_classes) #b*x 30
        bbox_pred = obj_pred[:, :5 * self.num_boxes].contiguous().view(-1, 5) #b*x*2 5
        class_pred = obj_pred[:, 5 * self.num_boxes:] #b*x 20
        obj_target = target[obj_mask].view(-1, 5 * self.num_boxes + self.num_classes)  # b*x 30
        bbox_target = obj_target[:, :5 * self.num_boxes].contiguous().view(-1, 5)  # b*x*2 5
        class_target = obj_target[:, 5 * self.num_boxes:]  # b*x 20

        # 没有目标的仅计算置信度
        noobj_pred = output[noobj_mask].view(-1, 5 * self.num_boxes + self.num_classes) #b*x 30
        noobj_target = target[noobj_mask].view(-1, 5 * self.num_boxes + self.num_classes)  # b*x 30

        noobj_cmask = torch.zeros(noobj_pred.size()).cuda()
        for i in range(self.num_boxes):
            ti = i + 1
            noobj_cmask[:, 5 * ti - 1] = 1

        noobj_pred_class = noobj_pred[noobj_cmask]
        noobj_target_class = noobj_target[noobj_cmask]
        noobj_conf_loss = F.mse_loss(noobj_pred_class, noobj_target_class, reduction='sum')

        # class loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        # 计算包含obj损失  即本来有，预测有  和本来有，预测无
        responce_mask = torch.zeros(bbox_target.size()).cuda()
        box_target_iou = torch.zeros(bbox_target.size()).cuda()
        for i in range(0, bbox_target.size()[0], self.num_boxes):

            # 预测 box
            box1 = bbox_pred[i:i + self.num_boxes]
            box1_xyxy = torch.zeros(box1.size()).cuda()
            box1_xyxy[:, :2] = box1[:, :2] / self.gridsize - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / self.gridsize + 0.5 * box1[:, 2:4]

            # label box
            box2 = bbox_target[i].view(-1, 5)
            box2_xyxy = torch.zeros(box2.size()).cuda()
            box2_xyxy[:, :2] = box2[:, :2] / self.gridsize - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / self.gridsize + 0.5 * box2[:, 2:4]

            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            responce_mask[i + max_index] = 1
            box_target_iou[i + max_index, 4] = max_iou

        # 获取包含obj与不包含obj的mask
        obj_responce_mask = responce_mask[:, :, :, 4] == 1
        obj_responce_mask = obj_responce_mask.unsqueeze(-1).expand_as(bbox_target)
        obj_notresponce_mask = responce_mask[:, :, :, 4] == 0
        obj_notresponce_mask = obj_notresponce_mask.unsqueeze(-1).expand_as(bbox_target)

        # 本来有，预测有
        box_pred_response = bbox_pred[obj_responce_mask].view(-1, 5)
        box_target_response = bbox_target[obj_responce_mask].view(-1, 5)
        xy_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum')
        wh_loss = F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), reduction='sum')
        loc_loss = xy_loss + wh_loss
        box_target_response_iou = box_target_iou[obj_responce_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')

        # 本来有，预测无
        box_pred_not_response = bbox_pred[obj_notresponce_mask].view(-1, 5)
        box_target_not_response = bbox_target[obj_notresponce_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        loss_sum = (self.coord_scale * loc_loss + contain_loss + self.noobject_scale * noobj_conf_loss + class_loss) / nB

        return loss_sum

def build_targets_yolov3(pred_boxes, target, anchors, num_anchors, nH, nW, noobject_scale, object_scale, sil_thresh, seen=15000):
    target = target.data
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
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            # gt_box = [0, 0, gw, gh]
            gt_box = torch.FloatTensor([0, 0, gw, gh])
            for n in range(nA):
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                # anchor_box = [0, 0, aw, ah]
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

class Yolov3Loss(nn.Module):
    def __init__(self, num_classes=2, anchors=[], num_anchors=3, stride=32):
        super(Yolov3Loss, self).__init__()
        self.num_classes = num_classes # 20,80
        self.num_anchors = num_anchors #5
        self.anchor_step = int(len(anchors) / num_anchors) #2
        self.stride = stride
        self.anchors = anchors #40,15, 90,45, 120,55, 190,65, 220,88
        # self.anchors = [anchor / self.stride for anchor in anchors]
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
        nH = output.data.size(2) #13 26 52
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
            = build_targets_yolov3(pred_boxes, target.data, self.anchors, nA, nH, nW, self.noobject_scale, self.object_scale, self.thresh)

        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum())

        tx = tx.cuda()
        ty = ty.cuda()
        tw = tw.cuda()
        th = th.cuda()
        tconf = tconf.cuda()
        tcls = tcls[cls_mask].view(-1).cuda()

        coord_mask = coord_mask.cuda()
        conf_mask = conf_mask.cuda().sqrt()
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC).cuda()
        cls = cls[cls_mask].view(-1, nC)
        cls_ = torch.sigmoid(cls)

        loss_x = self.coord_scale * nn.MSELoss(reduction='sum')(x * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.MSELoss(reduction='sum')(y * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * nn.MSELoss(reduction='sum')(w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss(reduction='sum')(h * coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(reduction='sum')(conf * conf_mask, tconf * conf_mask) / 2.0
        # loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction='sum')(cls, tcls)
        loss_cls = self.class_scale * nn.BCELoss(reduction='sum')(cls_, torch.zeros(cls_.shape).index_fill_(1, tcls.data.cpu().long(), 1.0).cuda())
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        print('nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
        nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
        loss_conf.item(), loss_cls.item(), loss.item()))
        return loss

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

        loss_xy = F.binary_cross_entropy(input=output[..., :2], target=obj_target[..., :2],
                                         weight=tgt_scale * tgt_scale, reduction='sum')
        loss_wh = F.mse_loss(input=output[..., 2:4], target=obj_target[..., 2:4], reduction='sum') / 2
        loss_obj = F.binary_cross_entropy(input=output[..., 4], target=obj_target[..., 4], reduction='sum')
        loss_cls = F.binary_cross_entropy(input=output[..., 5:], target=obj_target[..., 5:], reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss









