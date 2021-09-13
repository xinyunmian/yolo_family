import torch.nn as nn
import torch.nn.functional as F
import torch


def load_darknet_weights(model, weights_path):
    import numpy as np
    # Open the weights file
    fp = open(weights_path, "rb")
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values
    # Needed to write header when saving weights
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    print("total len weights = ", weights.shape)
    fp.close()

    ptr = 0
    all_dict = model.state_dict()
    last_bn_weight = None
    last_conv = None
    for i, (k, v) in enumerate(all_dict.items()):
        if 'bn' in k:
            if 'weight' in k:
                last_bn_weight = v
            elif 'bias' in k:
                num_b = v.numel()
                vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                v.copy_(vv)
                print("bn_bias: ", ptr, num_b, k)
                ptr += num_b
                # weight
                v = last_bn_weight
                num_b = v.numel()
                vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                v.copy_(vv)
                print("bn_weight: ", ptr, num_b, k)
                ptr += num_b
                last_bn_weight = None
            elif 'running_mean' in k:
                num_b = v.numel()
                vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                v.copy_(vv)
                print("bn_mean: ", ptr, num_b, k)
                ptr += num_b
            elif 'running_var' in k:
                num_b = v.numel()
                vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                v.copy_(vv)
                print("bn_var: ", ptr, num_b, k)
                ptr += num_b
                # conv
                v = last_conv
                num_b = v.numel()
                vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                v.copy_(vv)
                print("conv wight: ", ptr, num_b, k)
                ptr += num_b
                last_conv = None
            else:
                raise Exception("Error for bn")
        elif 'conv' in k:
            if 'weight' in k:
                last_conv = v
            else:
                num_b = v.numel()
                vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                v.copy_(vv)
                print("conv bias: ", ptr, num_b, k)
                ptr += num_b
                # conv
                v = last_conv
                num_b = v.numel()
                vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                v.copy_(vv)
                print("conv wight: ", ptr, num_b, k)
                ptr += num_b
                last_conv = None
    print("Total ptr = ", ptr)
    print("real size = ", weights.shape)

class ResBlock(nn.Module):
    def __init__(self, inplanes=64, planes=[32, 64]):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class YoloDarknet(nn.Module):
    def __init__(self, ResBlockNum=[1, 1, 2, 2, 1]):
        super(YoloDarknet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], ResBlockNum[0])
        self.layer2 = self._make_layer([64, 128], ResBlockNum[1])
        self.layer3 = self._make_layer([128, 256], ResBlockNum[2])
        self.layer4 = self._make_layer([256, 512], ResBlockNum[3])
        self.layer5 = self._make_layer([512, 1024], ResBlockNum[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

    def _make_layer(self, planes, resNum):
        layers = nn.Sequential()
        layers.add_module("conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False))
        layers.add_module("bn", nn.BatchNorm2d(planes[1]))
        layers.add_module("relu", nn.LeakyReLU(0.1))

        self.inplanes = planes[1]
        for i in range(resNum):
            layers.add_module("res_{}".format(i), ResBlock(self.inplanes, planes))
        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out52 = self.layer3(x)
        out26 = self.layer4(out52)
        out13 = self.layer5(out26)

        return out52, out26, out13

class YoloLayerDarknet(nn.Module):
    def __init__(self, config):
        super(YoloLayerDarknet, self).__init__()
        self.config = config
        #  backbone
        self.backbone = YoloDarknet(ResBlockNum=config.blocks)
        _out_filters = self.backbone.layers_out_filters
        final_out = 3 * (5 + config.classes)

        # yolo detection layer
        #  embedding0 52*52
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out)
        #  embedding1(FPN) 26*26
        self.embedding1_cbr = self._make_cbr(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out)
        #  embedding2(FPN) 13*13
        self.embedding2_cbr = self._make_cbr(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out)

    def _make_cbr(self, _in, _out, ks):
        # cbl = conv + batch_norm + leaky_relu
        pad = (ks - 1) // 2 if ks else 0
        cbls = nn.Sequential()
        cbls.add_module("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False))
        cbls.add_module("bn", nn.BatchNorm2d(_out))
        cbls.add_module("relu", nn.LeakyReLU(0.1))
        return cbls

    def _make_embedding(self, filters_list, in_filters, out_filter):
        embeds = nn.Sequential()
        embeds.add_module("cbr1", self._make_cbr(in_filters, filters_list[0], 1))
        embeds.add_module("cbr2", self._make_cbr(filters_list[0], filters_list[1], 3))
        embeds.add_module("cbr3", self._make_cbr(filters_list[1], filters_list[0], 1))
        embeds.add_module("cbr4", self._make_cbr(filters_list[0], filters_list[1], 3))
        embeds.add_module("cbr5", self._make_cbr(filters_list[1], filters_list[0], 1))
        embeds.add_module("cbr6", self._make_cbr(filters_list[0], filters_list[1], 3))
        embeds.add_module("conv", nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True))
        return embeds

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x52, x26, x13 = self.backbone(x)
        #  yolo branch 0
        out13, out0_useFPN = _branch(self.embedding0, x13) # 13 * 13

        #  yolo branch 1
        x1_in = self.embedding1_cbr(out0_useFPN)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x26], 1)
        out26, out1_useFPN = _branch(self.embedding1, x1_in) # 26 * 26

        #  yolo branch 2
        x2_in = self.embedding2_cbr(out1_useFPN)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x52], 1)
        out52 = self.embedding2(x2_in)
        # out2, out2_branch = _branch(self.embedding2, x2_in)
        return out13, out26, out52


if __name__ == "__main__":
    from yolov3.yolov3_config import cfg3 as cfg
    net = YoloLayerDarknet(config=cfg)
    net.eval()

    torch.save(net.state_dict(), 'yolo.pth')
    x = torch.randn(1, 3, 416, 416)
    output1, output2, output3 = net(x)
    print(output1.size())



























