import numpy as np

class yolo3cfg(object):
    # tiny  [1, 2, 8, 8, 4]
    blocks = [1, 2, 8, 8, 4]
    #1.string parameters
    train_data = "D:/data/imgs/facePicture/face_bright/train"
    train_txt = "D:/data/imgs/facePicture/face_bright/face_bright.txt"
    model_save = "D:/codes/pytorch_projects/faceBright_detect/weights"

    #2.numeric parameters
    epochs = 501
    batch_size = 8
    netw = 320
    neth = 320
    boxes_maxnum = 50
    label_class = 6
    classes = 2
    nanchors = 3
    # anchors = [0.38,0.78,0.5,1.06,0.68,1.18,0.62,1.56,0.84,1.56,0.75,2.0,1.03,1.78,0.94,2.22,1.22,2.28]
    anchors = [12, 25, 16, 34, 21, 39, 21, 58, 25, 49, 27, 67, 32, 54, 36, 65, 37, 78]
    anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    strides = [8, 16, 32]
    out_channels = [16, 24, 32, 64, 128]
    # 1.25, 0.45, 2.8, 1.5, 3.75, 1.75, 5.9, 2.0, 6.8, 2.75 mark,ID
    # 12, 25, 16, 34, 20, 50, 22, 38, 24, 64, 27, 50, 30, 71, 33, 57, 39, 73 ears
    # [0.38,0.78, 0.5,1.06, 0.68,1.18, 0.62,1.56, 0.84,1.56, 0.75,2.0, 1.03,1.78, 0.94,2.22, 1.22,2.28] 320*320 ears
    # 3, 5, 8, 16, 17, 45, 35, 88, 55, 175, 110, 280 yolo3
    # 6, 12, 10, 18, 15, 25, 25, 36, 40, 68, 95, 154 yolo4

    # data augment
    data_list = "D:/data/imgs/facePicture/ears/error.txt"
    letter_box = 0
    flip = 0
    blur = 0
    gaussian = 0
    saturation = 1.5
    exposure = 1.5
    hue = .1
    jitter = 0.2
    moasic = 0

    model_save = "D:/codes/pytorch_projects/yolo/weights"
    lr = 0.001
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    conf_thresh = 0.2
    nms_thresh = 0.3

cfg3 = yolo3cfg()




















