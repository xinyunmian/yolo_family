import time
import numpy as np
import cv2
from config import yolox_cfg
from get_model import CreatYolox
import torch
from collections import OrderedDict
import math

def adjust_learning_rate(epoch, optimizer):
    lr = yolox_cfg.lr
    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 600:
        lr = lr / 100000
    elif epoch > 550:
        lr = lr / 10000
    elif epoch > 150:
        lr = lr / 1000
    elif epoch > 80:
        lr = lr / 100
    elif epoch > 20:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr

def load_pretrain(net):
    print('Loading resume network...')
    state_dict = torch.load(yolox_cfg.resume_net)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

def train():
    net = CreatYolox(cfg=yolox_cfg).model
    if yolox_cfg.resume_net is not None:
        load_pretrain(net)

    net = net.cuda()
    print("to cuda done")


if __name__ == "__main__":
    train()






















