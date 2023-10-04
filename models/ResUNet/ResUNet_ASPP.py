import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings

from modules import init_weights

warnings.filterwarnings(action='ignore')


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


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

        self.conv = nn.Conv2d(out_c, out_c, kernel_size=1, stride=2)

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


class ResUNet_ASPP(nn.Module):
    def __init__(self, in_ch=14, n_classes=2, feature_scale=4):
        super(ResUNet_ASPP, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        self.input_layer = nn.Sequential(
            nn.Conv2d(self.in_ch, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(self.in_ch, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        # self.bridge = ResidualConv(filters[2], filters[3], 2, 1)
        self.bridge = ASPP(filters[2], filters[3])

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(
            filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(
            filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(
            filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], self.n_classes, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        # Bridge
        x4 = self.bridge(x3)

        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv3(x9)

        # final ouptut
        output = self.output_layer(x10)

        return output
