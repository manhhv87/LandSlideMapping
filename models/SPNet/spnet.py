###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .customize import StripPooling


class SPNet(nn.Module):
    def __init__(self, in_ch=14, n_classes=2, backbone='resnet50', pretrained=None, norm_layer=nn.BatchNorm2d, spm_on=True):
        super(SPNet, self).__init__()
        if backbone == 'resnet50':
            self.backbone = resnet50(in_ch=in_ch, pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = resnet101(in_ch=in_ch, pretrained=pretrained)
        else:
            self.backbone = None

        self.head = SPHead(2048, n_classes, norm_layer)
        # if aux:
        #    self.auxlayer = FCNHead(1024, num_classes, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        c4 = self.backbone(x)
        # print("Backbone:" + str(c4.shape))
        x = self.head(c4)
        # print("Head:"+str(x.shape))
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x


class SPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True)
                                         )
        self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                                         norm_layer(inter_channels // 2),
                                         nn.ReLU(True),
                                         nn.Dropout2d(0.1, False),
                                         nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x
