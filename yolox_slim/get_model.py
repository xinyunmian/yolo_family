import os
import random
import numpy as np
from yolox_blocks import YOLOX, YOLOPAFPN, YOLOXHead
import torch.nn as nn

class CreatYolox():
    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.classes
        self.depth = cfg.net_depth
        self.width = cfg.net_width
        self.in_channels = cfg.in_channels
        self.dw = cfg.dwconv
        self.tr = cfg.for_train
        self.dc = cfg.decode_out

        self.backbone = YOLOPAFPN(self.depth, self.width, in_channels=self.in_channels, depthwise=self.dw)
        self.head = YOLOXHead(self.num_classes, self.width, in_channels=self.in_channels, depthwise=self.dw)
        self.model = YOLOX(self.backbone, self.head, trainning=self.tr, decode=self.dc)
        self.model.apply(self.init_yolo)
        self.model.head.initialize_biases(1e-2)

    def init_yolo(self, M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
