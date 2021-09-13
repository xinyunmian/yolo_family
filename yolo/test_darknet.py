import torch
import os
from darknet2pytorch import darknetCfg_to_pytorchModel
from yolov2.train_config import traincfg as datacfg
import cv2
from yolo_process import get_boxes_yolo2, get_boxes_yolo3, nms, plot_boxes_cv2
from save_params import save_feature_channel, pytorch_to_dpcoreParams
import numpy as np

def image2torch(img):
	width = img.width
	height = img.height
	img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
	img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
	img = img.view(1, 3, height, width)
	img = img.float().div(255.0)
	return img

model = darknetCfg_to_pytorchModel(datacfg.cfgfile, count=5, mode="test")

width = int(model.net_blocks[0]['width'])
height = int(model.net_blocks[0]['height'])
num_classes = int(model.net_blocks[-1]['classes'])
num_anchors = int(model.net_blocks[-1]['num'])
anchors = [float(i) for i in model.net_blocks[-1]['anchors'].split(',')]

model.load_weights(datacfg.weightfile)
# model.save_weights("IDpics.weights")
model.cuda()
model.eval()

# saveparams = pytorch_to_dpcoreParams(model)
# saveparams.forward("head3_param_cfg.h", "head3_param_src.h")

def test_one_yolo2(imgmat):
    sized = cv2.resize(imgmat, (width, height), interpolation=cv2.INTER_LINEAR)
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    # sized = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    sized = sized.astype(np.float32)
    sized = sized / 255.0
    # sized = (sized - datacfg.bgr_mean) / datacfg.bgr_std
    sized = sized.transpose(2, 0, 1)
    sized = torch.from_numpy(sized).unsqueeze(0)

    sized = sized.cuda()
    output = model(sized)
    # b, c, h, w = output.shape
    # save_feature_channel("txt/conv1p.txt", output, b, c, h, w)
    output = output.data
    boxes = get_boxes_yolo2(output, conf_thresh=datacfg.conf_thresh, num_classes=num_classes, anchors=anchors,
                            num_anchors=num_anchors, use_sigmoid=None, stride=datacfg.stride)[0]
    bboxes = nms(boxes, datacfg.nms_thresh)
    draw_img = plot_boxes_cv2(imgmat, bboxes)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow('result', draw_img)
    cv2.waitKey(0)

def test_one_yolo3(imgmat):
    sized = cv2.resize(imgmat, (width, height), interpolation=cv2.INTER_LINEAR)
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    # sized = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    sized = sized.astype(np.float32)
    sized = sized / 255.0
    # sized = (sized - datacfg.bgr_mean) / datacfg.bgr_std
    sized = sized.transpose(2, 0, 1)
    sized = torch.from_numpy(sized).unsqueeze(0)

    sized = sized.cuda()
    output = model(sized)
    # b, c, h, w = output.shape
    # save_feature_channel("txt/conv1p.txt", output, b, c, h, w)
    boxes = get_boxes_yolo3(output, conf_thresh=datacfg.conf_thresh, num_classes=num_classes, anchors=anchors,
                            anchor_masks=datacfg.anchor_mask, allnum_anchors=num_anchors, use_sigmoid=True, strides=datacfg.strides)
    bboxes = []
    for bs in boxes:
        bboxes += bs

    bboxes = nms(bboxes, datacfg.nms_thresh)
    draw_img = plot_boxes_cv2(imgmat, bboxes)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow('result', draw_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    img_path = "imgs/1.jpg"
    img = cv2.imread(img_path)
    test_one_yolo3(img)

















