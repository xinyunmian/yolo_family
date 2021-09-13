import time
import numpy as np
import cv2
from config import yolox_cfg
from get_model import CreatYolox
from yolox_utils import postprocess, vis
import torch

class ImgProcess:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, input_size):
        img, _ = self.preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

def test_one(net, imgpath, cfg):
    imgprocess = ImgProcess()
    imgmat = cv2.imread(imgpath)
    ratio = min(cfg.input_sizes[0] / imgmat.shape[0], cfg.input_sizes[1] / imgmat.shape[1])
    img, _ = imgprocess(imgmat, cfg.input_sizes)
    img = torch.from_numpy(img).unsqueeze(0)
    outputs = net(img)
    outputs = postprocess(outputs, cfg.classes, cfg.conf_thresh, cfg.nms_thresh, class_agnostic=True)
    output = outputs[0]
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(imgmat, bboxes, scores, cls, cfg.conf_thresh, cfg.cls_names)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', vis_res)
    cv2.waitKey(0)

if __name__ == "__main__":
    model = CreatYolox(cfg=yolox_cfg).model
    model.eval()

    weigts_path = "weights/yolox_nano.pth"
    ckpt = torch.load(weigts_path, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    test_imgp = "D:/codes/pytorch_projects/YOLOX/assets/c.jpg"
    test_one(model, test_imgp, yolox_cfg)
    print("load weights done !!!")


