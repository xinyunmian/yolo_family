import torch.nn as nn
import torch.nn.functional as F
import torch

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class conv_bn_activate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation="mish", bn=True, bias=False):
        super(conv_bn_activate, self).__init__()
        pad = (kernel_size - 1) // 2

        self.convbn = nn.Sequential()
        if bias:
            self.convbn.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.convbn.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.convbn.add_module("bn", nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.convbn.add_module("mish", Mish())
        elif activation == "relu":
            self.convbn.add_module("relu", nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.convbn.add_module("leaky", nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!!")

    def forward(self, x):
        for l in self.convbn:
            x = l(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsampling, self).__init__()

        self.conv = conv_bn_activate(in_channels, out_channels, 1, activation="leaky")
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

        self.conv1 = conv_bn_activate(in_channels, hidden_channels, 1)
        self.conv2 = conv_bn_activate(hidden_channels, out_channels, 3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

class CSPFirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstBlock, self).__init__()

        self.downsample_conv = conv_bn_activate(in_channels, out_channels, 3, stride=2)
        self.split_conv_down = conv_bn_activate(out_channels, out_channels, 1)

        self.split_conv_res = conv_bn_activate(out_channels, out_channels, 1)
        self.res_conv = nn.Sequential()
        self.res_conv.add_module("res", ResUnit(out_channels, out_channels, in_channels))
        self.res_conv.add_module("conv", conv_bn_activate(out_channels, out_channels, 1))

        self.concat_conv = conv_bn_activate(out_channels * 2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv_down(x)

        x1 = self.split_conv_res(x)
        x1 = self.res_conv(x1)

        concatx = torch.cat([x0, x1], dim=1)
        concatx = self.concat_conv(concatx)
        return concatx

class CSPBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlocks, self).__init__()

        self.downsample_conv = conv_bn_activate(in_channels, out_channels, 3, stride=2)
        self.split_conv_down = conv_bn_activate(out_channels, out_channels // 2, 1)

        self.split_conv_res = conv_bn_activate(out_channels, out_channels // 2, 1)
        self.res_conv_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.res_conv_blocks.add_module("res{}".format(i), ResUnit(out_channels // 2, out_channels // 2))
        self.res_conv_blocks.add_module("conv", conv_bn_activate(out_channels // 2, out_channels // 2, 1))

        self.concat_conv = conv_bn_activate(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv_down(x)

        x1 = self.split_conv_res(x)
        x1 = self.res_conv_blocks(x1)

        concatx = torch.cat([x0, x1], dim=1)
        concatx = self.concat_conv(concatx)
        return concatx

class Yolo4Backbone(nn.Module):
    def __init__(self, feature_channels=[64, 128, 256, 512, 1024]):
        super(Yolo4Backbone, self).__init__()
        self.out_channels = feature_channels[-3:]

        self.stem_conv = conv_bn_activate(3, 32, 3)
        self.csp1 = CSPFirstBlock(32, feature_channels[0])
        self.csp2 = CSPBlocks(feature_channels[0], feature_channels[1], 2)
        self.csp8_1 = CSPBlocks(feature_channels[1], feature_channels[2], 8) #76*76

        self.csp8_2 = CSPBlocks(feature_channels[2], feature_channels[3], 8) #38*38

        self.csp4 = CSPBlocks(feature_channels[3], feature_channels[4], 4) #19*19

    def forward(self, x):
        features = []
        x = self.stem_conv(x)
        x = self.csp1(x)
        x = self.csp2(x)
        x76 = self.csp8_1(x) # batch*256*76*76
        features.append(x76)


        x38 = self.csp8_2(x76) # batch*512*38*38
        features.append(x38)

        x19 = self.csp4(x38) # batch*1024*19*19
        features.append(x19)
        return features

class Yolo4Neck(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024]):
        super(Yolo4Neck, self).__init__()

        self.cbl3_spp0 = nn.Sequential(
            conv_bn_activate(feature_channels[2], feature_channels[2] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[2] // 2, feature_channels[2], 3, activation="leaky"),
            conv_bn_activate(feature_channels[2], feature_channels[2] // 2, 1, activation="leaky")
        )
        self.spp = SPP()
        self.cbl3_spp1 = nn.Sequential(
            conv_bn_activate(feature_channels[2] * 2, feature_channels[2] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[2] // 2, feature_channels[2], 3, activation="leaky"),
            conv_bn_activate(feature_channels[2], feature_channels[2] // 2, 1, activation="leaky")
        )
        self.upsample1 = Upsampling(feature_channels[2] // 2, feature_channels[1] // 2)
        self.cbl_out38 = conv_bn_activate(feature_channels[1], feature_channels[1] // 2, 1, activation="leaky")

        self.cbl5_out19_up1 = nn.Sequential(
            conv_bn_activate(feature_channels[1], feature_channels[1] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[1] // 2, feature_channels[1], 3, activation="leaky"),
            conv_bn_activate(feature_channels[1], feature_channels[1] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[1] // 2, feature_channels[1], 3, activation="leaky"),
            conv_bn_activate(feature_channels[1], feature_channels[1] // 2, 1, activation="leaky")
        )
        self.upsample2 = Upsampling(feature_channels[1] // 2, feature_channels[0] // 2)
        self.cbl_out76 = conv_bn_activate(feature_channels[0], feature_channels[0] // 2, 1, activation="leaky")
        self.cbl5_out19_up2 = nn.Sequential(
            conv_bn_activate(feature_channels[0], feature_channels[0] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[0] // 2, feature_channels[0], 3, activation="leaky"),
            conv_bn_activate(feature_channels[0], feature_channels[0] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[0] // 2, feature_channels[0], 3, activation="leaky"),
            conv_bn_activate(feature_channels[0], feature_channels[0] // 2, 1, activation="leaky")
        )

        self.cbl_down1 = conv_bn_activate(feature_channels[0] // 2, feature_channels[1] // 2, 3, 2, activation="leaky")
        self.cbl5_down1 = nn.Sequential(
            conv_bn_activate(feature_channels[1], feature_channels[1] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[1] // 2, feature_channels[1], 3, activation="leaky"),
            conv_bn_activate(feature_channels[1], feature_channels[1] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[1] // 2, feature_channels[1], 3, activation="leaky"),
            conv_bn_activate(feature_channels[1], feature_channels[1] // 2, 1, activation="leaky")
        )

        self.cbl_down2 = conv_bn_activate(feature_channels[1] // 2, feature_channels[2] // 2, 3, 2, activation="leaky")
        self.cbl5_down2 = nn.Sequential(
            conv_bn_activate(feature_channels[2], feature_channels[2] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[2] // 2, feature_channels[2], 3, activation="leaky"),
            conv_bn_activate(feature_channels[2], feature_channels[2] // 2, 1, activation="leaky"),
            conv_bn_activate(feature_channels[2] // 2, feature_channels[2], 3, activation="leaky"),
            conv_bn_activate(feature_channels[2], feature_channels[2] // 2, 1, activation="leaky")
        )

    def forward(self, features):
        outputs = []
        feature76 = features[0]
        feature38 = features[1]
        feature19 = features[2]

        outspp = self.cbl3_spp0(feature19)
        outspp = self.spp(outspp)
        outspp = self.cbl3_spp1(outspp)

        outspp_up = self.upsample1(outspp)
        out38 = self.cbl_out38(feature38)
        outspp_up_cat = torch.cat([out38, outspp_up], dim=1)
        outspp_up_cat = self.cbl5_out19_up1(outspp_up_cat)

        outspp_up_cat_up = self.upsample2(outspp_up_cat)
        out76 = self.cbl_out76(feature76)
        outspp_up_cat_up_cat = torch.cat([out76, outspp_up_cat_up], dim=1)
        outspp_up_cat_up_cat = self.cbl5_out19_up2(outspp_up_cat_up_cat)
        outputs.append(outspp_up_cat_up_cat)

        out76_down = self.cbl_down1(outspp_up_cat_up_cat)
        out38_cat = torch.cat([out76_down, outspp_up_cat], dim=1)
        out38_cat = self.cbl5_down1(out38_cat)
        outputs.append(out38_cat)

        out38_down = self.cbl_down2(out38_cat)
        out19_cat = torch.cat([out38_down, outspp], dim=1)
        out19_cat = self.cbl5_down2(out19_cat)
        outputs.append(out19_cat)
        return outputs

class Yolo4Head(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024], target_channel=255):
        super(Yolo4Head, self).__init__()

        self.predict76 = nn.Sequential(
                conv_bn_activate(feature_channels[0] // 2, feature_channels[0], 3),
                nn.Conv2d(feature_channels[0], target_channel, 1)
            )

        self.predict38 = nn.Sequential(
            conv_bn_activate(feature_channels[1] // 2, feature_channels[1], 3),
            nn.Conv2d(feature_channels[1], target_channel, 1)
        )

        self.predict19 = nn.Sequential(
            conv_bn_activate(feature_channels[2] // 2, feature_channels[2], 3),
            nn.Conv2d(feature_channels[2], target_channel, 1)
        )

    def forward(self, features):
        outputs = []
        feature76 = features[0]
        pre76 = self.predict76(feature76)
        outputs.append(pre76)

        feature38 = features[1]
        pre38 = self.predict38(feature38)
        outputs.append(pre38)

        feature19 = features[2]
        pre19 = self.predict19(feature19)
        outputs.append(pre19)
        return outputs

class Yolo4Net(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024]):
        super(Yolo4Net, self).__init__()

        self.backbone = Yolo4Backbone()
        self.neck = Yolo4Neck()
        self.head = Yolo4Head()

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        features = self.head(features)
        return features

if __name__ == "__main__":
    import numpy as np
    # from yolo_last_layers import ciou_yolov4
    #
    # output = torch.rand(1, 3, 19, 19, 85)
    # pred = output[..., :4].clone()
    # truth_box = torch.rand(8, 4)
    # pred_ious = ciou_yolov4(pred[0].view(-1, 4), truth_box, xyxy=False)
    # pred_best_iou, _ = pred_ious.max(dim=1)
    # pred_best_iou = (pred_best_iou > 0.1)
    # pred_best_iou = pred_best_iou.view(pred[0].shape[:3])
    # obj_mask = ~ pred_best_iou
    #
    #
    # ref_anchors = torch.rand(3, 4)
    # anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # anchor_ious_all = ciou_yolov4(truth_box, ref_anchors, CIOU=True)
    # best_n_all = anchor_ious_all.argmax(dim=1)
    # best_n = best_n_all % 3
    # best_n_mask = ((best_n_all == anch_masks[0][0]) |
    #                (best_n_all == anch_masks[0][1]) |
    #                (best_n_all == anch_masks[0][2]))
    #
    # con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
    #                    (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
    # con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
    #                    (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
    #
    # cha = con_br - con_tl
    # cha = torch.pow(cha, 2)
    # cha = cha.sum(dim=2)
    #
    # area_a = torch.prod(truth_box[:, 2:] - truth_box[:, :2], 1)
    #
    # boxa = truth_box[:, None, :2]
    # boxb = ref_anchors[:, :2]
    # tl = torch.max(boxa, boxb)
    #
    # boxc = truth_box[:, None, 2:]
    # boxd = ref_anchors[:, 2:]
    # br = torch.min(boxc, boxd)
    #
    # en = (tl < br)
    # en = en.type(tl.type())
    # en = en.prod(dim=2)
    #
    # rho2 = ((truth_box[:, None, 0] + truth_box[:, None, 2]) - (ref_anchors[:, 0] + ref_anchors[:, 2])) ** 2 / 4 + (
    #         (truth_box[:, None, 1] + truth_box[:, None, 3]) - (ref_anchors[:, 1] + ref_anchors[:, 3])) ** 2 / 4
    #
    #
    #
    #
    #
    # batch = 4
    # strides = [8, 16, 32]
    # anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
    # anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    #
    # masked_anchorss, ref_anchorss, grid_xs, grid_ys, anchor_ws, anchor_hs = [], [], [], [], [], []
    #
    # for i in range(3):
    #     all_anchors_grid = [(w / strides[i], h / strides[i]) for w, h in anchors]
    #     masked_anchors = np.array([all_anchors_grid[j] for j in anch_masks[i]], dtype=np.float32)
    #     ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
    #     ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
    #     ref_anchors = torch.from_numpy(ref_anchors)
    #     # calculate pred - xywh obj cls
    #     fsize = 608 // strides[i]
    #     grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1)
    #     grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2)
    #     anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2)
    #     anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2)
    #
    #     masked_anchorss.append(masked_anchors)
    #     ref_anchorss.append(ref_anchors)
    #     grid_xs.append(grid_x)
    #     grid_ys.append(grid_y)
    #     anchor_ws.append(anchor_w)
    #     anchor_hs.append(anchor_h)



    net = Yolo4Net()
    net.eval()
    # torch.save(net.state_dict(), 'facebright.pth')
    x = torch.randn(1, 3, 608, 608)
    y = net(x)
    print(y.size())










