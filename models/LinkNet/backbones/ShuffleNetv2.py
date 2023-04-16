import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = t.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


# used in the block
def conv_1x1_bn(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


def conv_bn(in_c, out_c, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = nn.Sequential(
                # 3*3 dw conv, stride = 2
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 2
                nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )
        else:
            # in_c = out_c
            assert in_c == out_c

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 1
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True)
            )

    def forward(self, x):
        out = None
        if self.downsample:
            # if it is downsampling, we don't need to do channel split
            out = t.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            # channel split
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = t.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)


class ShuffleNetv2(nn.Module):
    def __init__(self, in_channel=3, net_type=1):
        super(ShuffleNetv2, self).__init__()
        # assert input_size % 32 == 0  # 因为一共会下采样32倍

        self.stage_repeat_num = [4, 8, 4]
        if net_type == 0.5:
            self.out_channels = [3, 24, 48, 96, 192, 1024]
        elif net_type == 1:
            self.out_channels = [3, 24, 116, 232, 464, 1024]
        elif net_type == 1.5:
            self.out_channels = [3, 24, 176, 352, 704, 1024]
        elif net_type == 2:
            self.out_channels = [3, 24, 244, 488, 976, 2948]
        else:
            print("the type is error, you should choose 0.5, 1, 1.5 or 2")

        # let's start building layers
        self.conv1 = nn.Conv2d(in_channel, self.out_channels[1], 3, 2, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_c = self.out_channels[1]

        self.stages2 = []
        self.stages3 = []
        self.stages4 = []
        for stage_idx in range(len(self.stage_repeat_num)):
            if (stage_idx == 0):
                out_c = self.out_channels[2 + stage_idx]
                repeat_num = self.stage_repeat_num[stage_idx]
                for i in range(repeat_num):
                    if i == 0:
                        self.stages2.append(ShuffleBlock(
                            in_c, out_c, downsample=True))
                    else:
                        self.stages2.append(ShuffleBlock(
                            in_c, in_c, downsample=False))
                    in_c = out_c
            elif (stage_idx == 1):
                out_c = self.out_channels[2 + stage_idx]
                repeat_num = self.stage_repeat_num[stage_idx]
                for i in range(repeat_num):
                    if i == 0:
                        self.stages3.append(ShuffleBlock(
                            in_c, out_c, downsample=True))
                    else:
                        self.stages3.append(ShuffleBlock(
                            in_c, in_c, downsample=False))
                    in_c = out_c
            else:
                out_c = self.out_channels[2 + stage_idx]
                repeat_num = self.stage_repeat_num[stage_idx]
                for i in range(repeat_num):
                    if i == 0:
                        self.stages4.append(ShuffleBlock(
                            in_c, out_c, downsample=True))
                    else:
                        self.stages4.append(ShuffleBlock(
                            in_c, in_c, downsample=False))
                    in_c = out_c
        self.stages2 = nn.Sequential(*self.stages2)
        self.stages3 = nn.Sequential(*self.stages3)
        self.stages4 = nn.Sequential(*self.stages4)

        in_c = self.out_channels[-2]
        out_c = self.out_channels[-1]
        # self.conv5 = conv_1x1_bn(in_c, out_c, 1)
        self.conv5_1 = conv_bn(in_c, out_c, 2)
        self.conv5_2 = conv_bn(out_c, out_c, 1)
        # self.g_avg_pool = nn.AvgPool2d(kernel_size=(int)(input_size / 32))  # 如果输入的是224，则此处为7
        # self.g_avg_pool = nn.AdaptiveAvgPool2d(1)  # 如果输入的是224，则此处为7

        # fc layer
        # self.fc = nn.Linear(out_c, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.stages2(x1)
        x3 = self.stages3(x2)
        x4 = self.stages4(x3)
        x5 = self.conv5_1(x4)
        x5 = self.conv5_2(x5)
        # x = self.g_avg_pool(x)
        # x = x.view(-1, self.out_channels[-1])
        # x = self.fc(x)
        return x1, x2, x3, x4, x5


def shufflenetv2_0_5(in_channel, pretrained=None):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShuffleNetv2(in_channel, 0.5)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def shufflenetv2_1_0(in_channel, pretrained=None):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShuffleNetv2(in_channel, 1)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def shufflenetv2_1_5(in_channel, pretrained=None):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShuffleNetv2(in_channel, 1.5)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def shufflenetv2_2_0(in_channel, pretrained=None):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShuffleNetv2(in_channel, 2)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model
