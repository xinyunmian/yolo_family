from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torch.nn as nn
import math
from yolo_last_layers import *
import sys
from yolo_config import *
from save_params import save_feature_channel, pytorch_to_dpcoreParams
device = torch.device("cpu")

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, (H // hs) * (W // ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x

class maxpool1(nn.Module):
	def __init__(self):
		super(maxpool1, self).__init__()
	def forward(self, x):
		x_pad = F.pad(x, (0, 1, 0, 1), mode='replicate')
		x = F.max_pool2d(x_pad, 2, stride=1)
		return x

class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride
    def forward(self, x):
        assert (x.data.dim() == 4)
        B = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        x = x.view(B, C, H, 1, W, 1)
        x = x.expand(B, C, H, self.stride, W, self.stride).contiguous()
        x = x.view(B, C, H * self.stride, W * self.stride)
        return x

class Upsample_interpolate(nn.Module):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride
    def forward(self, x):
        assert (x.data.dim() == 4)
        B = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        out = F.interpolate(x, size=(H * self.stride, W * self.stride), mode='nearest')
        return out

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()
    def forward(self, x):
        return x

class darknetCfg_to_pytorchModel(nn.Module):
    def __init__(self, cfgfile, count=5, mode="train"):
        super(darknetCfg_to_pytorchModel, self).__init__()
        self.header_len = count
        self.header = torch.IntTensor([0, ] * self.header_len)
        self.seen = self.header[3]
        self.mode = mode
        self.det_strides = []
        self.net_blocks = parse_cfg(cfgfile)
        self.models = self.create_net(self.net_blocks)  # merge conv, bn,leaky
        self.loss = self.models[len(self.models) - 1]

    def create_net(self, blocks):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0

        for block in blocks:
            if block['type'] == 'net':
                init_width = int(block['width'])
                init_height = int(block['height'])
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                group = 1
                if "groups" in block:
                    group = int(block['groups'])
                activation = block['activation']
                model = nn.Sequential()
                #conv
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=group, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=group))
                # activate function
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), Mish())
                else:
                    print("convalution no activate {}".format(activation))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)

            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
                else:
                    model = maxpool1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)

            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)

            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)

            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(reduction='mean')
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(reduction='mean')
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(reduction='mean')
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)

            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                # models.append(Upsample_expand(stride))
                models.append(Upsample_interpolate(stride))

            # concat
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        prev_filters = out_filters[layers[0]]
                        prev_stride = out_strides[layers[0]]
                    else:
                        prev_filters = out_filters[layers[0]] // int(block['groups'])
                        prev_stride = out_strides[layers[0]] // int(block['groups'])
                elif len(layers) == 2:
                    assert (layers[0] == ind - 1 or layers[1] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:
                    assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + out_filters[layers[3]]
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())

            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())

            elif block['type'] == 'connected':
                filters = int(block['output'])
                stride = out_strides[-1]
                in_filters = prev_filters * (init_height//stride) * (init_width//stride)
                if block['activation'] == 'linear':
                    model = nn.Linear(in_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(in_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(in_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)

            elif block['type'] == 'dropout':
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                prob = float(block['probability'])
                model = nn.Dropout(p=prob)
                models.append(model)

            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(an) / 32.0 for an in anchors]
                loss.num_classes = int(block['classes']) #20
                loss.num_anchors = int(block['num']) #5
                loss.anchor_step = int(len(loss.anchors) // loss.num_anchors) #2
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(loss)
                self.det_strides.append(prev_stride)

            elif block['type'] == 'detection':
                stride = out_strides[-1]
                grids = init_height//stride
                loss = DetectionLoss()
                loss.gridsize = grids
                loss.num_classes = int(block['classes'])
                loss.num_boxes = int(block['num'])
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(loss)
                self.det_strides.append(prev_stride)

            elif block['type'] == 'yolo':
                if "scale_x_y" in block:
                    loss = Yolov4Loss()
                    num_anchors = int(block['num'])
                    num_classes = int(block['classes'])
                    all_anchors = [float(i) for i in block['anchors'].split(',')]
                    anchor_mask = [int(i) for i in block['mask'].split(',')]
                    anchor_step = int(len(all_anchors) // num_anchors)
                    scale_x_y = float(block['scale_x_y'])
                    ignore_thresh = float(block['ignore_thresh'])
                    loss.stride = prev_stride
                    anchors = []
                    for m in anchor_mask:
                        anchors += all_anchors[m * anchor_step: (m + 1) * anchor_step]
                    loss.anchors = anchors
                    loss.num_classes = num_classes  # 80
                    loss.num_anchors = len(anchor_mask)  # 3
                    loss.anchor_step = anchor_step  # 2
                    loss.scale_x_y = scale_x_y  # 1.05
                    loss.ignore_thresh = ignore_thresh  # 0.7
                    out_filters.append(prev_filters)
                    out_strides.append(prev_stride)
                    models.append(loss)
                    self.det_strides.append(prev_stride)

                else:
                    loss = Yolov3Loss()
                    num_anchors = int(block['num'])
                    num_classes = int(block['classes'])
                    all_anchors = [float(i) for i in block['anchors'].split(',')]
                    anchor_mask = [int(i) for i in block['mask'].split(',')]
                    anchor_step = int(len(all_anchors) // num_anchors)
                    loss.stride = prev_stride
                    # anchors = [(all_anchors[i], all_anchors[i + 1]) for i in range(0, len(all_anchors), 2)]
                    # anchors = [anchors[i] for i in anchor_mask]
                    anchors = []
                    for m in anchor_mask:
                        anchors += all_anchors[m * anchor_step: (m + 1) * anchor_step]

                    loss.anchors = anchors
                    loss.num_classes = num_classes  # 80
                    loss.num_anchors = len(anchor_mask)  # 3
                    loss.anchor_step = anchor_step  # 2
                    loss.object_scale = float(block['object_scale']) if "object_scale" in block else 5
                    loss.noobject_scale = float(block['noobject_scale']) if "noobject_scale" in block else 1
                    loss.class_scale = float(block['class_scale']) if "class_scale" in block else 1
                    loss.coord_scale = float(block['coord_scale']) if "coord_scale" in block else 1
                    loss.thresh = float(block['thresh']) if "thresh" in block else 0.6
                    # loss.object_scale = float(block['object_scale'])
                    # loss.noobject_scale = float(block['noobject_scale'])
                    # loss.class_scale = float(block['class_scale'])
                    # loss.coord_scale = float(block['coord_scale'])
                    # loss.thresh = float(block['thresh'])
                    out_filters.append(prev_filters)
                    out_strides.append(prev_stride)
                    models.append(loss)
                    self.det_strides.append(prev_stride)
        return models

    def print_network(self):
        print_cfg(self.net_blocks)

    # load weights
    def load_weights(self, weightfile):
        with open(weightfile, 'rb') as fp:
            # before yolo3, weights get from https://github.com/pjreddie/darknet count = 4.
            header = np.fromfile(fp, count=self.header_len, dtype=np.int32)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            buf = np.fromfile(fp, dtype=np.float32)
        start = 0
        ind = -2
        for block in self.net_blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] in ['net', 'maxpool', 'reorg', 'upsample', 'route', 'shortcut',
                                       'region', 'yolo', 'avgpool', 'softmax', 'cost', 'detection', 'dropout']:
                continue
            elif block['type'] in ['convolutional', 'local']:
                model = self.models[ind]
                try:
                    batch_normalize = int(block['batch_normalize'])
                except:
                    batch_normalize = False
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            else:
                print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
                sys.exit(0)

    # save weights
    def save_weights(self, outfile):
        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)
        ind = -1
        for blockId in range(1, len(self.net_blocks)):
            ind = ind + 1
            block = self.net_blocks[blockId]
            if block['type'] in ['convolutional', 'local']:
                model = self.models[ind]
                try:
                    batch_normalize = int(block['batch_normalize'])
                except:
                    batch_normalize = False
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fp, model[0])
                else:
                    save_fc(fp, model)
            elif block['type'] in ['net', 'maxpool', 'reorg', 'upsample', 'route', 'shortcut',
                                         'region', 'yolo', 'avgpool', 'softmax', 'cost', 'detection', 'dropout']:
                continue
            else:
                print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
                sys.exit(0)
        fp.close()

    def forward(self, x, target=None):
        ind = -2
        loss = 0
        res = []
        outputs = dict()
        for block in self.net_blocks:
            ind += 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'local', 'dropout']:
                x = self.models[ind](x)
                if ind >= 10000:
                    b, c, h, w = x.shape
                    save_feature_channel("txt/conv1p.txt", x, b, c, h, w)
                outputs[ind] = x
            elif block['type'] == 'connected':
                x = x.view(x.size(0), -1)
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                route_groups = int(block['groups']) if "groups" in block else 1
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if route_groups==1:
                        x = outputs[layers[0]]
                    elif route_groups==2:
                        x = outputs[layers[0]]
                        _, xc, _, _ = x.shape
                        x = x[:, xc // 2:, ...]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                elif activation == 'mish':
                    x = Mish()(x)
                outputs[ind] = x
            # yoloV1, yoloV2, yoloV3
            elif block['type'] in ['detection', 'region', 'yolo']:
                if self.mode == "train":
                    self.models[ind].seen = self.seen
                    loss += self.models[ind](x, target)
                else:
                    res.append(x)
            # for resnet, too lazy to realize, o(╥﹏╥)o
            elif block['type'] == 'cost':
                continue
            else:
                print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
                sys.exit(0)
        if self.mode == "train":
            return loss
        else:
            return x if len(res) < 2 else res


if __name__ == "__main__":
    from yolo_config import parse_cfg
    import struct
    weightfile = "weights/rings_necklaces_last.weights"
    cfgfile1 = "D:/codes/yolov4/Yolo4Projet/test/model/IDpics.cfg"
    cfgfile = "weights/rings_necklaces.cfg"

    model = darknetCfg_to_pytorchModel(cfgfile, mode="test")
    model.load_weights(weightfile)
    model.eval()
    # f = open('yolov3-tiny.wts', 'w')
    # f.write('{}\n'.format(len(model.state_dict().keys())))
    # for k, v in model.state_dict().items():
    #     vr = v.reshape(-1).cpu().numpy()
    #     f.write('{} {} '.format(k, len(vr)))
    #     for vv in vr:
    #         f.write(' ')
    #         f.write(struct.pack('>f', float(vv)).hex())
    #     f.write('\n')



    model.print_network()

    x = torch.randn(1, 3, 320, 320).cuda()
    label = torch.rand(1, 50 * 5).cuda()
    loss = model(x, label)
    print("ok")

























