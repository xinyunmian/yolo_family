import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        out = self.se[0](x)
        out = self.se[1](out)
        out = self.se[2](out)
        out = self.se[3](out)
        return x * self.se(x)

class mobilev3_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(mobilev3_Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class Head(nn.Module):
    def __init__(self, inchannel=96, outchannel=55):
        super(Head, self).__init__()
        self.inc = inchannel
        self.outc = outchannel
        self.conv1x1 = nn.Conv2d(self.inc, self.outc, 1, 1, 0, bias=False)
    def forward(self, x):
        out = self.conv1x1(x)
        return out

class yolo_mobilev3(nn.Module):
    def __init__(self, nclass=6, nanchors=5):
        super(yolo_mobilev3, self).__init__()
        self.outc = (5 + nclass) * nanchors
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.block1 = mobilev3_Block(3, 16, 32, 16, nn.ReLU(inplace=True), SeModule(16), 2)
        self.block2 = mobilev3_Block(3, 16, 32, 24, nn.ReLU(inplace=True), None, 2)
        self.block3 = mobilev3_Block(3, 24, 32, 24, nn.ReLU(inplace=True), None, 1)
        self.block4 = mobilev3_Block(5, 24, 64, 40, hswish(), SeModule(40), 2)
        self.block5 = mobilev3_Block(5, 40, 96, 40, hswish(), SeModule(40), 1)
        self.block6 = mobilev3_Block(5, 40, 96, 40, hswish(), SeModule(40), 1)
        self.block7 = mobilev3_Block(5, 40, 120, 48, hswish(), SeModule(48), 1)
        self.block8 = mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1)
        self.block9 = mobilev3_Block(5, 48, 256, 96, hswish(), SeModule(96), 2)
        self.block10 = mobilev3_Block(5, 96, 512, 96, hswish(), SeModule(96), 1)
        self.block11 = mobilev3_Block(5, 96, 512, 96, hswish(), SeModule(96), 1)

        self.yolo_head = Head(inchannel=96, outchannel=self.outc)

    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        # b, c, h, w = x.shape
        # save_feature_channel("txt/p/conv2.txt", x, b, c, h, w)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        out = self.yolo_head(x)
        return out

if __name__ == "__main__":
    import time
    net = yolo_mobilev3(nclass=6, nanchors=5)
    net.eval()
    torch.save(net.state_dict(), 'mobilev3.pth')
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y.size())

































