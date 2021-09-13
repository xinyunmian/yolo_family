# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import random
import sys

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    if bboxes.shape[0] == 0:
        return bboxes, 10000
    np.random.shuffle(bboxes)
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    if bboxes.shape[0] == 0:
        return bboxes, 10000

    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

    bboxes[:, 0] *= (net_w / sx)
    bboxes[:, 2] *= (net_w / sx)
    bboxes[:, 1] *= (net_h / sy)
    bboxes[:, 3] *= (net_h / sy)

    if flip:
        temp = net_w - bboxes[:, 0]
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = temp

    return bboxes, min_w_h


def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur, truth):
    try:
        img = mat
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        # crop
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)  # 交集

        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        # cv2.Mat sized
        itmethods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        itmethod = itmethods[random.randrange(5)]
        if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
            sized = cv2.resize(img, (w, h), interpolation=itmethod)
        else:
            cropped = np.zeros([sheight, swidth, 3], dtype="uint8")
            cropped[:, :, ] = np.mean(img, axis=(0, 1))
            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2], :] = img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2], :]
            # resize
            sized = cv2.resize(cropped, (w, h), interpolation=itmethod)

        # flip
        if flip:
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_BGR2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2BGR), 0, 255)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp
        if blur:
            if blur == 1:
                dst = cv2.GaussianBlur(sized, (17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            if blur == 1:
                img_rect = [0, 0, sized.shape[0], sized.shape[1]]
                for b in truth:
                    left = (b.x - b.w / 2.) * sized.shape[1]
                    width = b.w * sized.shape[1]
                    top = (b.y - b.h / 2.) * sized.shape[0]
                    height = b.h * sized.shape[0]
                    roi(left, top, width, height)
                    roi = roi & img_rect
                    dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
                                                                          roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        sized = mat

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes

def draw_bb(img, bboxes, cfg):
    imgmat = img.cpu().numpy().transpose((1, 2, 0))
    imgmat = imgmat * cfg.bgr_std + cfg.bgr_mean
    imgmat = imgmat.astype(np.uint8)
    for b in bboxes:
        b0 = int(b[0])
        b1 = int(b[1])
        b2 = int(b[2])
        b3 = int(b[3])
        imgmat = cv2.rectangle(imgmat, (b0, b1), (b2, b3), (0, 255, 0), 2)
    return imgmat

def draw_box(img, bboxes, cfg):
    imgmat = img.cpu().numpy().transpose((1, 2, 0))
    imgmat = imgmat * cfg.bgr_std + cfg.bgr_mean
    imgmat = imgmat.astype(np.uint8)
    h, w, c = imgmat.shape
    bboxes_label = np.reshape(bboxes, (-1, 5))
    for b in bboxes_label:
        cx = b[1] * w
        cy = b[2] * h
        iw = b[3] * w
        ih = b[4] * h
        xmin = int(cx - 0.5 * iw)
        ymin = int(cy - 0.5 * ih)
        xmax = int(cx + 0.5 * iw)
        ymax = int(cy + 0.5 * ih)
        imgmat = cv2.rectangle(imgmat, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return imgmat


class DataLoader_Yolo(Dataset):
    def __init__(self, data_path, config):
        super(DataLoader_Yolo, self).__init__()
        self.config = config

        truth = {}
        f = open(data_path, 'r', encoding='utf-8')
        for imgp in f.readlines():
            imgp = imgp.rstrip()
            imgmat = cv2.imread(imgp)
            hei, wid, _ = imgmat.shape
            labelp = imgp.replace('jpg','txt')
            truth[imgp] = []
            labs = open(labelp, 'r')
            for lab in labs.readlines():
                lab = lab.rstrip().split(" ")
                class_id = int(lab[0])
                cx = float(lab[1]) * wid
                cy = float(lab[2]) * hei
                iw = float(lab[3]) * wid
                ih = float(lab[4]) * hei
                xmin = cx - 0.5 * iw
                ymin = cy - 0.5 * ih
                xmax = cx + 0.5 * iw
                ymax = cy + 0.5 * ih

                truth[imgp].append([xmin, ymin, xmax, ymax, class_id])
        self.truth = truth
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        img_path = self.imgs[index]
        bboxes = np.array(self.truth.get(img_path), dtype=np.float)
        use_mixup = self.config.mixup
        if random.randint(0, 1):
            use_mixup = 0

        if use_mixup == 3:
            min_offset = 0.2
            cut_x = random.randint(int(self.config.netw * min_offset), int(self.config.netw * (1 - min_offset)))
            cut_y = random.randint(int(self.config.neth * min_offset), int(self.config.neth * (1 - min_offset)))

        r1, r2, r3, r4, r_scale = 0, 0, 0, 0, 0
        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
        gaussian_noise = 0

        out_img = np.zeros([self.config.neth, self.config.netw, 3], dtype="uint8")
        out_bboxes = []

        for i in range(use_mixup + 1):
            if i != 0:
                img_path = random.choice(list(self.truth.keys()))
                bboxes = np.array(self.truth.get(img_path), dtype=np.float)
            img = cv2.imread(img_path)
            if img is None:
                continue
            oh, ow, oc = img.shape
            dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.config.jitter, dtype=np.int) #裁剪比例0.3

            dhue = rand_uniform_strong(-self.config.hue, self.config.hue) #色调
            dsat = rand_scale(self.config.saturation) #饱和度
            dexp = rand_scale(self.config.exposure) #曝光度

            pleft = random.randint(-dw, dw)
            pright = random.randint(-dw, dw)
            ptop = random.randint(-dh, dh)
            pbot = random.randint(-dh, dh)

            flip = random.randint(0, 1) if self.config.flip else 0

            if (self.config.blur):
                tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
                if tmp_blur == 0:
                    blur = 0
                elif tmp_blur == 1:
                    blur = 1
                else:
                    blur = self.config.blur

            if self.config.gaussian and random.randint(0, 1):
                gaussian_noise = self.config.gaussian
            else:
                gaussian_noise = 0

            if self.config.letter_box:
                img_ar = ow / oh
                net_ar = self.config.netw / self.config.neth
                result_ar = img_ar / net_ar
                # print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if result_ar > 1:  # sheight - should be increased
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                    # print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                else:  # swidth - should be increased
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w
                    # printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot

            truth, min_w_h = fill_truth_detection(bboxes, self.config.boxes_maxnum, self.config.classes, flip, pleft, ptop, swidth,
                                                  sheight, self.config.netw, self.config.neth)
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                blur = min_w_h / 8

            ai = image_data_augmentation(img, self.config.netw, self.config.neth, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth)

            if use_mixup == 0:
                out_img = ai
                out_bboxes = truth
            if use_mixup == 1:
                if i == 0:
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5)
                    out_bboxes = np.concatenate([old_truth, truth], axis=0)
            elif use_mixup == 3:
                if flip:
                    tmp = pleft
                    pleft = pright
                    pright = tmp

                left_shift = int(min(cut_x, max(0, (-int(pleft) * self.config.netw / swidth))))
                top_shift = int(min(cut_y, max(0, (-int(ptop) * self.config.neth / sheight))))

                right_shift = int(min((self.config.netw - cut_x), max(0, (-int(pright) * self.config.netw / swidth))))
                bot_shift = int(min(self.config.neth - cut_y, max(0, (-int(pbot) * self.config.neth / sheight))))

                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.config.netw, self.config.neth, cut_x,
                                                       cut_y, i, left_shift, right_shift, top_shift, bot_shift)
                out_bboxes.append(out_bbox)
                # print(img_path)
        if use_mixup == 3:
            out_bboxes = np.concatenate(out_bboxes, axis=0)

        # out_img = ai
        # out_bboxes = truth
        out_bboxes1 = np.zeros([self.config.boxes_maxnum, 5])
        out_bboxes1[:min(out_bboxes.shape[0], self.config.boxes_maxnum)] = out_bboxes[:min(out_bboxes.shape[0], self.config.boxes_maxnum)]

        # data and label process
        target = self.transpose_label(out_img, out_bboxes1)
        imgt = out_img.astype(np.float32)
        imgt = (imgt - self.config.bgr_mean) / self.config.bgr_std
        imgt = imgt.transpose(2, 0, 1)
        imgt = torch.from_numpy(imgt)
        return imgt, target

    def transpose_label(self, img, boxes):
        height, width, channel = img.shape
        label = np.zeros((self.config.boxes_maxnum, 5))
        cc = 0
        for b in boxes:
            x1 = b[0]
            y1 = b[1]
            x2 = b[2]
            y2 = b[3]
            id = b[4]
            # label normalization
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            tw = (x2 - x1) / width
            th = (y2 - y1) / height
            label[cc, 0] = id
            label[cc, 1] = cx
            label[cc, 2] = cy
            label[cc, 3] = tw
            label[cc, 4] = th
            cc += 1
        label = np.reshape(label, (-1))
        return label

if __name__ == "__main__":
    from yolo_config import datacfg, get_img_list
    imgdir = "D:/data/imgs/rename/test"
    listp = "D:/data/imgs/rename/test.txt"
    # get_img_list(imgdir, listp, "jpg")

    random.seed(2020)
    np.random.seed(2020)
    datap = "D:/data/imgs/rename/test.txt"
    dataset = DataLoader_Yolo(datap, datacfg)
    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img, out_bboxes, datacfg)
        cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
        cv2.imshow('result2', a)
        cv2.waitKey(0)
