import numpy as np
from yolox_utils import COCO_CLASSES, MY_CLASSES

class yolox_config(object):
    #2.numeric parameters
    start_epoch = 0
    epochs = 501
    batch_size = 4
    insize = (640, 640)

    # augment params
    augment = 1
    moasic = 1.0
    mixup = 0
    max_objs = 500
    flip_prob = 0.5
    hsv_prob = 1.0
    perspective = 0.0  # image perspective (+/- fraction), range 0-0.001
    anchor_box_ratio = 4.0

    # net params
    strides = [8, 16, 32]
    classes = 1
    nanchors = 1

    # data path
    data_list = "D:/data/imgs/widerface_clean/testvoc_shuffle.txt"
    data_path = "D:/data/imgs/widerface_clean/testvoc"

    model_save = "D:/codes/pytorch_projects/yolo_face/weights"
    pretrain = True
    pretrain_weights = "D:/codes/pytorch_projects/yolo_face/weights/FaceShuffle_62.pth"

    lr = 0.001
    weight_decay = 0.0005

    # test
    for_train = True
    decode_out = True
    cls_names = MY_CLASSES #MY_CLASSES
    net_depth = 0.33
    net_width = 0.25
    dwconv = 1
    in_channels = [256, 512, 1024]
    padding_img = True
    input_sizes = (416, 416)  # net input 0: origin imgsize
    conf_thresh = 0.25
    nms_thresh = 0.3

yolox_cfg = yolox_config()