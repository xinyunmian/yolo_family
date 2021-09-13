import torch.nn as nn
import torch.nn.functional as F
import torch

class conv_bn(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv_dw(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conv_dw, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Yolo3Slim(nn.Module):
    def __init__(self, outc=[16, 24, 32, 64, 128]):
        super(Yolo3Slim, self).__init__()
        self.conv1 = conv_bn(3, outc[0], 2)
        self.conv11 = conv_dw(outc[0], outc[0], 1)
        self.conv111= conv_dw(outc[0], outc[0], 1)
        self.conv2 = conv_dw(outc[0], outc[1], 2)
        self.conv22 = conv_dw(outc[1], outc[1], 1)
        self.conv222 = conv_dw(outc[1], outc[1], 1)
        self.conv3 = conv_dw(outc[1], outc[2], 2)
        self.conv33 = conv_dw(outc[2], outc[2], 1)
        self.conv333 = conv_dw(outc[2], outc[2], 1)

        self.conv4 = conv_dw(outc[2], outc[3], 2)
        self.conv44 = conv_dw(outc[3], outc[3], 1)
        self.conv444 = conv_dw(outc[3], outc[3], 1)

        self.conv5 = conv_dw(outc[3], outc[4], 2)
        self.conv55 = conv_dw(outc[4], outc[4], 1)
        self.conv555 = conv_dw(outc[4], outc[4], 1)

        self.out_filters = outc

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.conv111(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.conv222(x)
        x = self.conv3(x)
        x = self.conv33(x)
        x = self.conv333(x)
        x52 = x
        x = self.conv4(x)
        x = self.conv44(x)
        x = self.conv444(x)
        x26 = x
        x = self.conv5(x)
        x = self.conv55(x)
        x = self.conv555(x)
        x13 = x
        return x52, x26, x13

class Yolo3LayerSlim(nn.Module):
    def __init__(self, config):
        super(Yolo3LayerSlim, self).__init__()
        self.config = config
        #  backbone
        self.backbone = Yolo3Slim(outc=config.out_channels)
        out_channels = self.backbone.out_filters
        final_out = 3 * (5 + config.classes)
        self.conv_out13 = self.make_embedding(out_channels[-1], final_out)

        self.conv_fpn1 = conv_dw(out_channels[-1], out_channels[-2], 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_out26 = self.make_embedding(out_channels[-2] + out_channels[-2], final_out)

        self.conv_fpn2 = conv_dw(out_channels[-2] + out_channels[-2], out_channels[-3], 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_out52 = self.make_embedding(out_channels[-3] + out_channels[-3], final_out)

    def make_embedding(self, infilter, outfilter):
        embeds = nn.Sequential()
        embeds.add_module("convdw", conv_dw(infilter, infilter, 1))
        embeds.add_module("convout", nn.Conv2d(infilter, outfilter, kernel_size=1, stride=1, padding=0, bias=False))
        return embeds

    def forward(self, x):
        x52, x26, x13 = self.backbone(x)

        out13 = self.conv_out13(x13)

        out1_useFPN = self.conv_fpn1(x13)
        x1 = self.upsample1(out1_useFPN)
        x1 = torch.cat([x1, x26], 1)
        out26 = self.conv_out26(x1)

        out2_useFPN = self.conv_fpn2(x1)
        x2 = self.upsample2(out2_useFPN)
        x2 = torch.cat([x2, x52], 1)
        out52 = self.conv_out52(x2)
        return out13, out26, out52


if __name__ == "__main__":
    from yolov3.yolov3_config import cfg3 as cfg
    from save_params import pytorch_to_dpcoreParams
    net = Yolo3LayerSlim(config=cfg)
    net.eval()
    saveparams = pytorch_to_dpcoreParams(net)
    saveparams.forward("yolo_param_cfg.h", "yolo_param_src.h")

    torch.save(net.state_dict(), 'yoloslim.pth')
    x = torch.randn(1, 3, 416, 416)
    output1, output2, output3 = net(x)
    print(output1.size())























