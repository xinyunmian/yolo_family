import torch.nn as nn
import torch.nn.functional as F
import torch

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

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

class conm_bn(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conm_bn, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conm_dw(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conm_dw, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = Mish()

        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = Mish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsampling, self).__init__()

        self.conv = conv_dw(in_channels, out_channels, 1)
        self.up = nn.Upsample(scale_factor=scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class SPP(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()

        self.maxpool1 = nn.MaxPool2d(kernel_size=pool_sizes[0], stride=1, padding=pool_sizes[0] // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=pool_sizes[1], stride=1, padding=pool_sizes[1] // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=pool_sizes[2], stride=1, padding=pool_sizes[2] // 2)

    def forward(self, x):
        features = []
        features.append(x)
        x1 = self.maxpool1(x)
        features.append(x1)
        x2 = self.maxpool2(x)
        features.append(x2)
        x3 = self.maxpool3(x)
        features.append(x3)
        feature = torch.cat(features, dim=1)
        return feature

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super(ResUnit, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.conv1 = conm_dw(in_channels, hidden_channels, 1)
        self.conv2 = conm_dw(hidden_channels, out_channels, 1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

class CSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSP, self).__init__()

        self.downsample_conv = conm_dw(in_channels, out_channels, 2)
        self.split_conv_down = conm_dw(out_channels, out_channels, 1)

        self.split_conv_res = conm_dw(out_channels, out_channels, 1)
        self.res_conv = nn.Sequential()
        self.res_conv.add_module("res", ResUnit(out_channels, out_channels, in_channels))
        self.res_conv.add_module("conv", conm_dw(out_channels, out_channels, 1))

        self.concat_conv = conm_dw(out_channels * 2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv_down(x)

        x1 = self.split_conv_res(x)
        x1 = self.res_conv(x1)

        concatx = torch.cat([x0, x1], dim=1)
        concatx = self.concat_conv(concatx)
        return concatx

class Yolo4Slim(nn.Module):
    def __init__(self, outc=[16, 24, 32, 48, 64, 128]):
        super(Yolo4Slim, self).__init__()
        self.conv = conm_bn(3, outc[0], 1)
        self.csp1 = CSP(outc[0], outc[1])
        self.csp11 = CSP(outc[1], outc[2])
        self.csp111 = CSP(outc[2], outc[3])

        self.csp2 = CSP(outc[3], outc[4])
        self.csp3 = CSP(outc[4], outc[5])

    def forward(self, x):
        features = []
        x = self.conv(x)
        x = self.csp1(x)
        x = self.csp11(x)
        x76 = self.csp111(x) # batch*48*76*76
        features.append(x76)

        x38 = self.csp2(x76) # batch*64*38*38
        features.append(x38)

        x19 = self.csp3(x38) # batch*128*19*19
        features.append(x19)
        return features

class Yolo4SlimNeck(nn.Module):
    def __init__(self, feature_channels=[48, 64, 128]):
        super(Yolo4SlimNeck, self).__init__()

        self.conv_spp0 = conm_dw(feature_channels[2], feature_channels[1], 1)
        self.spp = SPP()
        self.conv_spp1 = conm_dw(feature_channels[1] * 4, feature_channels[1], 1)

        self.upsample1 = Upsampling(feature_channels[1], feature_channels[1] // 2)
        self.conv_out38 = conm_dw(feature_channels[1], feature_channels[1] // 2, 1)
        self.conv_cat1 = conm_dw(feature_channels[1], feature_channels[1], 1)

        self.upsample2 = Upsampling(feature_channels[1], feature_channels[0] // 2)
        self.conv_out76 = conm_dw(feature_channels[0], feature_channels[0] // 2, 1)
        self.neck76 = conm_dw(feature_channels[0], feature_channels[0], 1)

        self.conv_down1 = conm_dw(feature_channels[0], feature_channels[1], 2)
        self.neck38 = conm_dw(feature_channels[1] * 2, feature_channels[1], 1)

        self.conv_down2 = conm_dw(feature_channels[1], feature_channels[1], 2)
        self.neck19 = conm_dw(feature_channels[1] * 2, feature_channels[2], 1)

    def forward(self, features):
        outputs = []
        feature76 = features[0]
        feature38 = features[1]
        feature19 = features[2]

        outspp = self.conv_spp0(feature19)
        outspp = self.spp(outspp)
        outspp = self.conv_spp1(outspp) #b*64*19*19

        out_up1 = self.upsample1(outspp) #b*32*38*38
        out38 = self.conv_out38(feature38) #b*32*38*38
        outcat1 = torch.cat([out38, out_up1], dim=1) #b*64*38*38
        outcat1 = self.conv_cat1(outcat1) #b*64*38*38

        out_up2 = self.upsample2(outcat1)  #b*24*76*76
        out76 = self.conv_out76(feature76)  #b*24*76*76
        outcat2 = torch.cat([out76, out_up2], dim=1)  #b*48*76*76
        neck76 = self.neck76(outcat2) #b*48*76*76
        outputs.append(neck76)

        out_down1 = self.conv_down1(neck76) #b*64*38*38
        out_down_cat1 = torch.cat([out_down1, outcat1], dim=1)  #b*128*38*38
        neck38 = self.neck38(out_down_cat1)  # b*64*38*38
        outputs.append(neck38)

        out_down2 = self.conv_down2(neck38)  # b*64*19*19
        out_down_cat2 = torch.cat([out_down2, outspp], dim=1)  # b*128*19*19
        neck19 = self.neck19(out_down_cat2)  # b*128*19*19
        outputs.append(neck19)
        return outputs

class Yolo4SlimHead(nn.Module):
    def __init__(self, feature_channels=[48, 64, 128], target_channel=255):
        super(Yolo4SlimHead, self).__init__()

        self.conv76 = conm_dw(feature_channels[0], feature_channels[0], 1)
        self.out76 = conm_dw(feature_channels[0], target_channel, 1)

        self.conv38 = conm_dw(feature_channels[1], feature_channels[1], 1)
        self.out38 = conm_dw(feature_channels[1], target_channel, 1)

        self.conv19 = conm_dw(feature_channels[2], feature_channels[2], 1)
        self.out19 = conm_dw(feature_channels[2], target_channel, 1)

    def forward(self, features):
        outputs = []

        feature76 = features[0]
        pre76 = self.conv76(feature76)
        pre76 = self.out76(pre76)
        outputs.append(pre76)

        feature38 = features[1]
        pre38 = self.conv38(feature38)
        pre38 = self.out38(pre38)
        outputs.append(pre38)

        feature19 = features[2]
        pre19 = self.conv19(feature19)
        pre19 = self.out19(pre19)
        outputs.append(pre19)
        return pre19, pre38, pre76

class Yolo4SlimNet(nn.Module):
    def __init__(self, config):
        super(Yolo4SlimNet, self).__init__()
        self.out_channels = config.out_channels
        self.feature_channels = self.out_channels[3:]
        self.target_channel = 3 * (5 + config.classes)

        self.backbone = Yolo4Slim(outc=self.out_channels)
        self.neck = Yolo4SlimNeck(feature_channels=self.feature_channels)
        self.head = Yolo4SlimHead(feature_channels=self.feature_channels, target_channel=self.target_channel)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        out19, out38, out76 = self.head(features)
        return out19, out38, out76

if __name__ == "__main__":
    from yolov4.yolov4_config import cfg4 as cfg
    net = Yolo4SlimNet(config=cfg)
    net.eval()

    torch.save(net.state_dict(), 'yolo4slim.pth')
    x = torch.randn(1, 3, 608, 608)
    out19, out38, out76 = net(x)
    print(out19.size())

















