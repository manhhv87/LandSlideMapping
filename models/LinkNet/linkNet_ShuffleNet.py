import torch.nn as nn
from .backbones.ShuffleNetv2 import shufflenetv2_1_0


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(in_planes//4),
                                   nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                     # output = (input-1)stride+outputpadding -2padding+kernelsize
                                     nn.BatchNorm2d(in_planes//4),
                                     nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(out_planes),
                                   nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class LinkNet(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, in_ch=3, n_classes=21, pretrained=False):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()
        self.backbone = shufflenetv2_1_0(in_ch, pretrained=pretrained)

        self.decoder1 = Decoder(116, 24, 3, 2, 1, 1)
        self.decoder2 = Decoder(232, 116, 3, 2, 1, 1)
        self.decoder3 = Decoder(464, 232, 3, 2, 1, 1)
        self.decoder4 = Decoder(1024, 464, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(24, 16, 3, 2, 1, 1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(16, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initial block
        x1, x2, x3, x4, x5 = self.backbone(x)
        # print(len(x))
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)

        # Decoder blocks
        # d4 = e3 + self.decoder4(e4)
        d4 = x4 + self.decoder4(x5)
        d3 = x3 + self.decoder3(d4)
        d2 = x2 + self.decoder2(d3)
        # print(self.decoder1(d2).shape)
        d1 = x1 + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)

        return y
