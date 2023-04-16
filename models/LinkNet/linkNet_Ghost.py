import torch
import torch.nn as nn
from .backbones.GhostNet import ghostnet


class GhostNet(nn.Module):
    def __init__(self, in_channel=3, pretrained=False):
        super(GhostNet, self).__init__()
        model = ghostnet(in_channel=in_channel)
        if pretrained:
            print('loading-------')
            state_dict = torch.load("models/ghostnet_weights.pth")
            model.load_state_dict(state_dict)
        del model.global_pool
        del model.conv_head
        del model.act2
        del model.classifier
        del model.blocks[9]
        self.model = model
        # print(self.model)

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        feature_maps = []

        for idx, block in enumerate(self.model.blocks):
            x = block(x)
            if idx in [0, 2, 4, 6, 8]:
                feature_maps.append(x)
        return feature_maps[0:]


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
        self.backbone = GhostNet(in_channel=in_ch, pretrained=pretrained)

        self.decoder1 = Decoder(24, 16, 3, 2, 1, 1)
        self.decoder2 = Decoder(40, 24, 3, 2, 1, 1)
        self.decoder3 = Decoder(112, 40, 3, 2, 1, 1)
        self.decoder4 = Decoder(160, 112, 3, 2, 1, 1)

        # Classifier
        self.conv = nn.Conv2d(16, n_classes, 3, 1, 1)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initial block
        x = self.backbone(x)
        # print(len(x))
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # print(x[3].shape)
        # print(x[4].shape)

        # Decoder blocks
        # d4 = e3 + self.decoder4(e4)
        d4 = x[3] + self.decoder4(x[4])
        d3 = x[2] + self.decoder3(d4)
        d2 = x[1] + self.decoder2(d3)
        # print(self.decoder1(d2).shape)
        d1 = x[0] + self.decoder1(d2)

        # Classifier
        y = self.conv(d1)

        y = self.lsm(y)

        return y
