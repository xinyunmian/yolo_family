import os
import random
import sys
from train_config import traincfg as datacfg
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

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

class DataLoader_Yolov3(Dataset):
    def __init__(self, data_path, config):
        super(DataLoader_Yolov3, self).__init__()
        self.config = config
        truth = {}
        f = open(data_path, 'r', encoding='utf-8')
        for imgp in f.readlines():
            imgp = imgp.rstrip()
            imgmat = cv2.imread(imgp)
            hei, wid, _ = imgmat.shape
            labelp = imgp.replace('jpg','txt')
            labs = open(labelp, 'r')
            labline = labs.readlines()
            if len(labline)==0:
                continue
            else:
                truth[imgp] = []
                for lab in labline:
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
        img_moasic = self.config.moasic
        img_num = 4
        if random.randint(0, 1):
            img_moasic = 0
            img_num = 1

        if img_moasic == 1:
            img_num = 4
            min_offset = 0.2
            cut_x = random.randint(int(self.config.netw * min_offset), int(self.config.netw * (1 - min_offset)))
            cut_y = random.randint(int(self.config.neth * min_offset), int(self.config.neth * (1 - min_offset)))

        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0

        out_img = np.zeros([self.config.neth, self.config.netw, 3], dtype="uint8")
        out_bboxes = []

        for i in range(img_num):
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
            blur = self.config.blur

            if self.config.gaussian and random.randint(0, 1):
                gaussian_noise = self.config.gaussian
            else:
                gaussian_noise = 0

            if self.config.letter_box:
                img_ar = ow / oh
                net_ar = self.config.netw / self.config.neth
                result_ar = img_ar / net_ar
                if result_ar > 1:  # sheight - should be increased
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                else:  # swidth - should be increased
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w

            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot

            truth, min_w_h = fill_truth_detection(bboxes, self.config.boxes_maxnum, self.config.label_class, flip, pleft,
                                                  ptop, swidth, sheight, self.config.netw, self.config.neth)
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                blur = min_w_h / 8
            ai = image_data_augmentation(img, self.config.netw, self.config.neth, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth)

            if img_moasic == 0:
                out_img = ai
                out_bboxes = truth

            elif img_moasic == 1:
                if flip:
                    tmp = pleft
                    pleft = pright
                    pright = tmp
                left_shift = int(min(cut_x, max(0, (-int(pleft) * self.config.netw / swidth))))
                top_shift = int(min(cut_y, max(0, (-int(ptop) * self.config.neth / sheight))))
                right_shift = int(min((self.config.netw - cut_x), max(0, (-int(pright) * self.config.netw / swidth))))
                bot_shift = int(min(self.config.neth - cut_y, max(0, (-int(pbot) * self.config.neth / sheight))))
                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.config.netw, self.config.neth,
                                                       cut_x, cut_y, i, left_shift, right_shift, top_shift, bot_shift)
                out_bboxes.append(out_bbox)

        if img_moasic == 1:
            out_bboxes = np.concatenate(out_bboxes, axis=0)

        out_bboxes1 = np.zeros([out_bboxes.shape[0], 6])
        out_bboxes1[:, 1:] = out_bboxes

        # data and label process
        target = self.transpose_label(out_img, out_bboxes1)
        imgt = out_img.astype(np.float32)
        imgt = imgt / 255.0
        # imgt = (imgt - self.config.bgr_mean) / self.config.bgr_std
        imgt = imgt.transpose(2, 0, 1)
        imgt = torch.from_numpy(imgt)

        # a = draw_box(imgt, target, datacfg)
        # cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
        # cv2.imshow('result2', a)
        # cv2.waitKey(500)
        return imgt, target

    def transpose_label(self, img, boxes):
        height, width, channel = img.shape
        label = np.zeros((boxes.shape[0], 6))
        cc = 0
        for b in boxes:
            x1 = b[1]
            y1 = b[2]
            x2 = b[3]
            y2 = b[4]
            id = b[5]
            # label normalization
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            tw = (x2 - x1) / width
            th = (y2 - y1) / height

            if id == 0 or id == 1 or id == 2:
                label[cc, 1] = 0
            if id == 3 or id == 4 or id == 5:
                label[cc, 1] = 1

            label[cc, 2] = cx
            label[cc, 3] = cy
            label[cc, 4] = tw
            label[cc, 5] = th
            cc += 1
        return label

def collate_fn(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for i, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                annos[:, 0] = i
                targets.append(annos)
    return (torch.stack(imgs, 0), torch.cat(targets, 0))

def draw_box(img, bboxes):
    imgmat = img.cpu().numpy().transpose((1, 2, 0))
    imgmat = imgmat * 255.0
    imgmat = imgmat.astype(np.uint8)
    h, w, c = imgmat.shape
    bboxes_label = np.reshape(bboxes, (-1, 6))
    for b in bboxes_label:
        cls_id = str(int(b[1]))
        cx = b[2] * w
        cy = b[3] * h
        iw = b[4] * w
        ih = b[5] * h
        xmin = int(cx - 0.5 * iw)
        ymin = int(cy - 0.5 * ih)
        xmax = int(cx + 0.5 * iw)
        ymax = int(cy + 0.5 * ih)
        imgmat = cv2.rectangle(imgmat, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        imgmat = cv2.putText(imgmat, cls_id, (int(cx), ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
    return imgmat

if __name__ == "__main__":
    imgdir = "D:/data/imgs/rename/test"
    listp = "D:/data/imgs/facePicture/ears/error.txt"
    # get_img_list(imgdir, listp, "jpg")

    random.seed(2020)
    np.random.seed(2020)
    datap = "D:/data/imgs/facePicture/ears/earss.txt"
    dataset = DataLoader_Yolov3(datap, datacfg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    for batch_i, (imgs, targets) in enumerate(dataloader):
        targets = targets.cuda()

    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img, out_bboxes)
        cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
        cv2.imshow('result2', a)
        cv2.waitKey(0)


























