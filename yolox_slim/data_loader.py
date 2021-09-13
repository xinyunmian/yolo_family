import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
import math

import os
import sys
from tqdm import tqdm
from PIL import Image, ExifTags
import logging
import xml.etree.ElementTree as ET
from image_augment import TrainTransform, random_perspective, box_candidates
from yolox_utils import adjust_box_anns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class LoadVocData(data.Dataset):
    def __init__(self, img_path, cfg):
        super(LoadVocData, self).__init__()
        self.img_transform = TrainTransform(
                    max_labels=cfg.max_objs,
                    flip_prob=cfg.flip_prob,
                    hsv_prob=cfg.hsv_prob)
        self.img_size = cfg.insize

        # for read xml
        self.class_to_ind = dict(zip(cfg.cls_names, range(len(cfg.cls_names)))) # name -> id
        self.keep_difficult = True # 是否保留难识别的

        self.labels = []
        self.img_files = []
        f = open(img_path, 'r', encoding='utf-8')
        for imgp in f.readlines():
            imgp = imgp.rstrip()
            img_name = imgp.split("/")[-1]
            img_hz = img_name.split(".")[-1]
            labelp = imgp.replace(img_hz, 'xml')

            self.img_files.append(imgp)
            self.labels.append(labelp)

        self.annotations = [self.load_xml_annotations(_ids) for _ids in range(len(self.labels))]

    def __len__(self):
        return len(self.img_files)

    def get_origin_item(self, index):
        img = self.load_resized_img(index)
        target, img_info, _ = self.annotations[index]
        return img, target, img_info, index

    def __getitem__(self, index):
        # img = self.load_resized_img(index)
        # target, img_info, _ = self.annotations[index]
        img, target, img_info, img_id = self.get_origin_item(index)

        # 数据增强包括：hsv变换, 镜像, padding, box框(xy xy->cx cy w h)
        if self.img_transform is not None:
            img, target = self.img_transform(img, target, self.img_size)
        return img, target, img_info, img_id

    def load_resized_img(self, index):
        imgp = self.img_files[index]
        img = cv2.imread(imgp, cv2.IMREAD_COLOR)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        return resized_img

    # 读取xml label, 依据长边等比例resize，返回[xmin, ymin, xmax, ymax, label_ind]
    # 图像和box label未做归一化
    def load_xml_annotations(self, index):
        labelp = self.labels[index]

        # 读取xml label，保存格式为[xmin, ymin, xmax, ymax, label_ind]
        target = ET.parse(labelp).getroot()
        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)

def collate_fn(batch):
    img, label, img_info, ind = zip(*batch)  # transposed
    img = np.array(img)
    label = np.array(label)
    return torch.from_numpy(img), torch.from_numpy(label)

class Mosaic(data.Dataset):
    def __init__(self, dataset, mosaic=True, mixup=False,
                 preproc=None,
                 degrees=10.0, translate=0.1, shear=2.0, perspective=0.0,
                 mosaic_scale=(0.5, 1.5), mixup_scale=(0.5, 1.5),
                 mosaic_prob=1.0, mixup_prob=0.0):
        super(Mosaic, self).__init__()
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        if self.enable_mixup:
            self.mixup_prob = 1.0
        self.input_dim = self._dataset.img_size

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_h, input_w = self.input_dim[0], self.input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
            mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)
            for i_mosaic, index in enumerate(indices):
                img, _labels, _, _ = self._dataset.get_origin_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                # if i_mosaic == 0:
                #     mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coordinate(
                    i_mosaic, xc, yc, w, h, input_h, input_w)

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1
                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )

            if ( self.enable_mixup and not len(mosaic_labels) == 0 and random.random() < self.mixup_prob):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)

            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[2])

            self.showlabels(mix_img.copy(), padded_labels[:, 1:5])

            return mix_img, padded_labels, img_info, np.array([idx])

        else:
            img, label, img_info, idx = self._dataset.get_origin_item(idx)
            img, label = random_perspective(
                img,
                label,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
            )
            img, label = self.preproc(img, label, self.input_dim)
            self.showlabels(img.copy(), label[:, 1:5])
            return img, label, img_info, np.array([idx])

    def showlabels(self, img, boxs):
        img2 = img.transpose((1,2,0)).astype(np.uint8).copy()
        for box in boxs:
            # x, y, w, h = box[0] * img.shape[1], box[1] * img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]
            x, y, w, h = box[0], box[1], box[2], box[3]
            # cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
            cv2.rectangle(img2, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img2)
        cv2.waitKey(0)

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.annotations[cp_index][0]
        img, cp_labels, _, _ = self._dataset.get_origin_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

    def get_mosaic_coordinate(self, mosaic_index, xc, yc, w, h, input_h, input_w):
        # TODO update doc
        # index0 to top left part of image
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
        return (x1, y1, x2, y2), small_coord



if __name__ == "__main__":
    facelist = "D:/data/imgs/widerface_clean/testvoc_shuffle.txt"
    from config import yolox_cfg
    from get_model import CreatYolox
    from yolox_loss import YoloxLoss

    cal_loss = YoloxLoss(yolox_cfg)

    model = CreatYolox(cfg=yolox_cfg).model
    model.cuda().eval()
    model.decode = False

    data_train = LoadVocData(facelist, yolox_cfg)
    data_train = Mosaic(
        dataset=data_train,
        mosaic=False,
        # mixup=True,
        preproc=TrainTransform(max_labels=yolox_cfg.max_objs, flip_prob=yolox_cfg.flip_prob, hsv_prob=yolox_cfg.hsv_prob),
    )
    train_loader = data.DataLoader(data_train, batch_size=yolox_cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for i, (imgs, targets) in enumerate(train_loader):
        imgsii = imgs.cuda()
        outs = model(imgsii)

        lab = targets.cuda()
        loss = cal_loss(outs, lab)

        print("error")