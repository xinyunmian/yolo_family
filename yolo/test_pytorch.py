import torch
import os
from yolov2.mobile_yolo2 import yolo_mobile, yolo_type
from yolov2.train_config import traincfg
import cv2
import numpy as np
from yolo_process import get_boxes_yolo2, nms, plot_boxes_cv2, get_boxes_yolo2_type, plot_boxes_cv2_type
from save_params import save_feature_channel, pytorch_to_dpcoreParams
#cuda
device = "cuda"

net = yolo_type(nclass=2, nregress=1, nanchors=5)
weight_path = "weights/ears_2100.pth"
net_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
net.load_state_dict(net_dict)
net.eval()
print('Finished loading model!')
net = net.to(device)

# saveparams = pytorch_to_dpcoreParams(net)
# saveparams.forward("eartype_param_cfg.h", "eartype_param_src.h")

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
    outdata = dnet(img)

    output = outdata.data

    # boxes = get_boxes_yolo2(output, conf_thresh=traincfg.conf_thresh, num_classes=traincfg.classes, anchors=traincfg.anchors,
    #                         num_anchors=traincfg.nanchors, use_sigmoid=None, stride=1)[0]
    # bboxes = nms(boxes, traincfg.nms_thresh)
    # draw_img = plot_boxes_cv2(img_mat, bboxes)

    boxes = get_boxes_yolo2_type(output, conf_thresh=traincfg.conf_thresh, num_classes=traincfg.classes, anchors=traincfg.anchors,
                    num_anchors=traincfg.nanchors, use_sigmoid=None, stride=1)[0]
    bboxes = nms(boxes, traincfg.nms_thresh)
    draw_img = plot_boxes_cv2_type(img_mat, bboxes)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow('result', draw_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    img_path = "imgs/occ_clip3.bmp"
    img = cv2.imread(img_path)
    test_one(img, net)











