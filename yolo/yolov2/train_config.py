import numpy as np

class train_yolo_config(object):
    #1.string parameters
    train_data = "D:/data/imgs/facePicture/face_bright/train"
    train_txt = "D:/data/imgs/facePicture/face_bright/face_bright.txt"
    val_data = "D:/data/imgs/facePicture/face_bright"
    val_txt = "D:/data/imgs/facePicture/face_bright/face_bright.txt"
    model_save = "D:/codes/pytorch_projects/faceBright_detect/weights"

    #2.numeric parameters
    epochs = 501
    batch_size = 16
    netw = 320
    neth = 320
    boxes_maxnum = 50
    label_class = 6
    classes = 2
    nanchors = 5
    anchors = [0.68,1.38,0.97,1.87,1.03,2.65,1.31,2.68,1.63,3.12]
    anchor_mask = [[3, 4, 5], [1, 2, 3]]
    stride = 32
    strides = [32, 16]
    # 1.25, 0.45, 2.8, 1.5, 3.75, 1.75, 5.9, 2.0, 6.8, 2.75 mark,ID
    # 0.31, 0.62, 0.45, 0.95, 0.55, 1.41, 0.62, 1.09, 0.78, 1.56 224*224 ears
    # [0.68,1.38,0.97,1.87,1.03,2.65,1.31,2.68,1.63,3.12] 448*448 ears
    # 3, 5, 8, 16, 17, 45, 35, 88, 55, 175, 110, 280 yolo3
    # 6, 12, 10, 18, 15, 25, 25, 36, 40, 68, 95, 154 yolo4

    # data augment
    data_list = "D:/data/imgs/facePicture/ears/earss.txt"
    letter_box = 0
    flip = 0
    blur = 0
    gaussian = 1
    saturation = 1.5
    exposure = 1.5
    hue = .1
    jitter = 0.2
    mixup = 3
    moasic = 0

    cfgfile = "D:/codes/pytorch_projects/yolo/weights/head_yolo4.cfg"
    weightfile = "D:/codes/pytorch_projects/yolo/weights/head_yolo4.weights"
    model_save = "D:/codes/pytorch_projects/yolo/weights"
    lr = 0.001
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    # test
    conf_thresh = 0.3
    nms_thresh = 0.3

traincfg = train_yolo_config()