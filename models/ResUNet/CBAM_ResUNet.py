import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import warnings

from modules import init_weights

warnings.filterwarnings(action='ignore')


class basic_conv(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride=1, padding=0):
        super(basic_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_size)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction, gate_channels)
        )

        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = basic_conv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()

        self.ChannelGate = ChannelGate(gate_channels, reduction, pool_types)
        self.no_spatial = no_spatial

        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)

        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)

        return x_out


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_size, out_size, reduction=16, no_spatial=False):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

        self.cbam = CBAM(out_size, reduction=reduction, no_spatial=no_spatial)

        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_size)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.cbam(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x


class inconv(nn.Module):
    def __init__(self, in_size, out_size):
        super(inconv, self).__init__()

        self.conv = double_conv(in_size, out_size, no_spatial=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_size, out_size):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_size, out_size)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_size, out_size, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_size // 2, in_size // 2, 2, stride=2)

        self.conv = double_conv(in_size, out_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_size, out_size):
        super(outconv, self).__init__()

        self.conv = nn.Conv2d(in_size, out_size, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super(ASPP, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, dilation=rate[0]),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1,
                      padding=6, dilation=rate[1]),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1,
                      padding=12, dilation=rate[2]),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1,
                      padding=18, dilation=rate[3]),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c5 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c6 = nn.Sequential(
            nn.Conv2d(5*out_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Conv2d(out_c, out_c, kernel_size=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        x1 = self.c1(feature_map)
        x2 = self.c2(feature_map)
        x3 = self.c3(feature_map)
        x4 = self.c4(feature_map)

        x5 = self.avg_pool(feature_map)
        x5 = self.c5(x5)

        x5 = F.upsample(
            x5, size=(feature_map_h, feature_map_w), mode="bilinear")

        out = torch.cat([x1, x2, x3, x4, x5], 1)
        out = self.c6(out)
        out = self.conv(out)

        return out


class CBAM_ResUNet(nn.Module):
    def __init__(self, in_ch=14, n_classes=2, feature_scale=4, deep_supervision=False):
        super(CBAM_ResUNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.deep_supervision = deep_supervision

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.inc = inconv(self.in_ch, filters[0])
        self.down1 = down(filters[0], filters[1])
        self.down2 = down(filters[1], filters[2])
        self.down3 = down(filters[2], filters[3])

        # bridge
        # self.down4 = down(filters[3], filters[3])
        self.down4 = ASPP(filters[3], filters[3])

        # upsampling
        self.up1 = up(filters[4], filters[2])
        self.up2 = up(filters[3], filters[1])
        self.up3 = up(filters[2], filters[0])
        self.up4 = up(filters[1], filters[0])

        # final output
        self.fc = nn.Sequential(
            ASPP(filters[0], filters[0]),
            outconv(filters[0], n_classes)
        )

        self.dsoutc4 = outconv(filters[2], n_classes)
        self.dsoutc3 = outconv(filters[1], n_classes)
        self.dsoutc2 = outconv(filters[0], n_classes)
        self.dsoutc1 = outconv(filters[0], n_classes)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        # encoding
        x1 = self.inc(x)        # [32, 64, 128, 128]
        x2 = self.down1(x1)     # [32, 128, 64, 64]
        x3 = self.down2(x2)     # [32, 256, 32, 32]
        x4 = self.down3(x3)     # [32, 512, 16, 16]

        # bridge
        x5 = self.down4(x4)     # [32, 512, 8, 8]

        # decoding
        x44 = self.up1(x5, x4)      # [32, 256, 16, 16]
        x33 = self.up2(x44, x3)     # [32, 128, 32, 32]
        x22 = self.up3(x33, x2)     # [32, 64, 64, 64]
        x11 = self.up4(x22, x1)     # [32, 64, 128, 128]
        x0 = self.fc(x11)         # [32, 2, 128, 128]

        if self.deep_supervision:
            x11 = F.interpolate(self.dsoutc1(
                x11), x0.shape[2:], mode='bilinear')
            x22 = F.interpolate(self.dsoutc2(
                x22), x0.shape[2:], mode='bilinear')
            x33 = F.interpolate(self.dsoutc3(
                x33), x0.shape[2:], mode='bilinear')
            x44 = F.interpolate(self.dsoutc4(
                x44), x0.shape[2:], mode='bilinear')

            return x0, x11, x22, x33, x44
        else:
            return x0
