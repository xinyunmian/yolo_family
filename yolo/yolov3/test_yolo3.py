import torch
import os
from yolov3.yolov3_slim import Yolo3LayerSlim
from yolov3.yolov3_config import cfg3 as traincfg
import cv2
import numpy as np
from yolo_process import get_boxes_yolo3, nms, plot_boxes_cv2
from save_params import save_feature_channel, pytorch_to_dpcoreParams
#cuda
device = "cpu"

obj_class = traincfg.classes
all_anchors = traincfg.anchors
anchor_mask = traincfg.anchor_mask
strides = traincfg.strides

def img_process(img):
    """将输入图片转换成网络需要的tensor
            Args:
                img_path: 人脸图片路径
            Returns:
                tensor： img(batch, channel, width, height)
    """
    im = cv2.resize(img, (traincfg.netw, traincfg.neth), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32)
    # im = (im - traincfg.bgr_mean) / traincfg.bgr_std
    im = im / 255.0
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    im = im.to(device)
    return im

def test_one(img_mat, dnet):
    img = img_process(img_mat)  # 数据处理，转为网络输入的形式
    out13, out26, out52 = dnet(img)
    outdata = []
    outdata.append(out52)
    outdata.append(out26)
    outdata.append(out13)
    boxes = get_boxes_yolo3(outdata, conf_thresh=traincfg.conf_thresh, num_classes=obj_class, anchors=all_anchors,
                            anchor_masks=anchor_mask, allnum_anchors=9, use_sigmoid=True, strides=strides)
    bboxes = []
    for bs in boxes:
        bboxes += bs

    bboxes = nms(bboxes, traincfg.nms_thresh)
    draw_img = plot_boxes_cv2(img_mat, bboxes)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow('result', draw_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    net = Yolo3LayerSlim(config=traincfg)
    weight_path = "yolo_800.pth"
    net_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(net_dict)
    net.eval()
    print('Finished loading model!')
    net = net.to(device)

    # from save_params import pytorch_to_dpcoreParams
    # saveparams = pytorch_to_dpcoreParams(net)
    # saveparams.forward("yolo_param_cfg.h", "yolo_param_src.h")

    img_path = "D:/codes/pytorch_projects/yolo/imgs/occ_clip(1).bmp"
    img = cv2.imread(img_path)
    test_one(img, net)











