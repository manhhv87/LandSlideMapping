import torch
import torch.nn as nn
from einops import rearrange
from .vit import ViT


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, stride=1):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        base_channels = int(out_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, base_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.encoder1_1 = EncoderBottleneck(
            base_channels, base_channels * 4, base_channels)
        self.encoder1_2 = EncoderBottleneck(
            base_channels * 4, base_channels * 4, base_channels)
        self.encoder1_3 = EncoderBottleneck(
            base_channels * 4, base_channels * 4, base_channels)

        self.encoder2_1 = EncoderBottleneck(
            base_channels * 4, base_channels * 8, base_channels * 2, stride=2)
        self.encoder2_2 = EncoderBottleneck(
            base_channels * 8, base_channels * 8, base_channels * 2)
        self.encoder2_3 = EncoderBottleneck(
            base_channels * 8, base_channels * 8, base_channels * 2)
        self.encoder2_4 = EncoderBottleneck(
            base_channels * 8, base_channels * 8, base_channels * 2)

        self.encoder3_1 = EncoderBottleneck(
            base_channels * 8, base_channels * 16, base_channels * 4, stride=2)
        self.encoder3_2 = EncoderBottleneck(
            base_channels * 16, base_channels * 16, base_channels * 4)
        self.encoder3_3 = EncoderBottleneck(
            base_channels * 16, base_channels * 16, base_channels * 4)
        self.encoder3_4 = EncoderBottleneck(
            base_channels * 16, base_channels * 16, base_channels * 4)
        self.encoder3_5 = EncoderBottleneck(
            base_channels * 16, base_channels * 16, base_channels * 4)
        self.encoder3_6 = EncoderBottleneck(
            base_channels * 16, base_channels * 16, base_channels * 4)
        self.encoder3_7 = EncoderBottleneck(
            base_channels * 16, base_channels * 16, base_channels * 4)
        self.encoder3_8 = EncoderBottleneck(
            base_channels * 16, base_channels * 16, base_channels * 4)
        self.encoder3_9 = EncoderBottleneck(
            base_channels * 16, base_channels * 16, base_channels * 4)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 6,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.conv2 = nn.Conv2d(out_channels * 6, 512,
                               kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)

        x2 = self.encoder1_1(x2)
        x2 = self.encoder1_2(x2)
        x2 = self.encoder1_3(x2)

        x3 = self.encoder2_1(x2)
        x3 = self.encoder2_2(x3)
        x3 = self.encoder2_3(x3)
        x3 = self.encoder2_4(x3)

        x = self.encoder3_1(x3)
        x = self.encoder3_2(x)
        x = self.encoder3_3(x)
        x = self.encoder3_4(x)
        x = self.encoder3_5(x)
        x = self.encoder3_6(x)
        x = self.encoder3_7(x)
        x = self.encoder3_8(x)
        x = self.encoder3_9(x)
        # print('R50输出:'+str(x.shape))
        x = self.vit(x)
        # print('Transformer编码:' + str(x.shape))
        x = rearrange(x, "b (x y) c -> b c x y",
                      x=self.vit_img_dim, y=self.vit_img_dim)
        # print('恢复三维:' + str(x.shape))
        # print(x.shape)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        # print('通道变换:' + str(x.shape))
        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(
            int(out_channels * 3 / 2), int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(
            int(out_channels * 1 / 2), int(out_channels * 1 / 8))
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_cls = nn.Conv2d(
            int(out_channels * 1 / 8), class_num, kernel_size=3, padding=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        # print('第一次解码:' + str(x.shape))
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        # x = self.upsample(x)
        # print('第三次解码:' + str(x.shape))
        x = self.decoder4(x)

        x = self.conv_cls(x)
        # print('结果:' + str(x.shape))
        return x


class TransUNet(nn.Module):
    def __init__(self, img_dim=128, in_ch=3, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, n_classes=2):
        super().__init__()

        self.encoder = Encoder(img_dim, in_ch, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels, n_classes)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        # print('x:'+str(x.shape))
        # print('x1:'+str(x1.shape))
        # print('x2:'+str(x2.shape))
        # print('x3:'+str(x3.shape))
        x = self.decoder(x, x1, x2, x3)

        return x


'''
if __name__ == '__main__':
    import torch

    transunet = TransUNet(img_dim=128,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)

    print(sum(p.numel() for p in transunet.parameters()))
    print(transunet(torch.randn(1, 3, 128, 128)).shape)
'''
