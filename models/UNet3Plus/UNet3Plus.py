import torch
import torch.nn as nn

from .backbones.ResNet50 import resnet50
from .backbones.VGG16 import VGG16


class unetUp4(nn.Module):
    def __init__(self, filters, out_size):
        super(unetUp4, self).__init__()
        count = 5
        self.downsample1 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.downsample2 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.downsample3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(filters[0], out_size, 3, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_size, 3, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_size, 3, padding=1)
        self.conv4 = nn.Conv2d(filters[3], out_size, 3, padding=1)
        self.conv5 = nn.Conv2d(filters[4], out_size, 3, padding=1)
        self.conv = nn.Conv2d(filters[0]*count, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_bn = self.bn(h1_downsample_conv)
            h1_downsample_relu = self.relu(h1_downsample_bn)

            h2_downsample = self.downsample2(inputs2)
            h2_downsample_conv = self.conv2(h2_downsample)
            h2_downsample_bn = self.bn(h2_downsample_conv)
            h2_downsample_relu = self.relu(h2_downsample_bn)

            h3_downsample = self.downsample3(inputs3)
            h3_downsample_conv = self.conv3(h3_downsample)
            h3_downsample_bn = self.bn(h3_downsample_conv)
            h3_downsample_relu = self.relu(h3_downsample_bn)

            h5_upsample = self.upsample(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_bn = self.bn(h5_upsample_conv)
            h5_upsample_relu = self.relu(h5_upsample_bn)

            h4_conv = self.conv4(inputs4)
            h4_bn = self.bn(h4_conv)
            h4_relu = self.relu(h4_bn)

            h_concat = torch.cat((h1_downsample_relu, h2_downsample_relu,
                                 h3_downsample_relu, h4_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_bn = self.bn(h_conv)
            h_relu = self.relu(h_bn)
            return h_relu
        else:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_relu = self.relu(h1_downsample_conv)

            h2_downsample = self.downsample2(inputs2)
            # print("h2_downsample:" + str(h2_downsample.shape))
            h2_downsample_conv = self.conv2(h2_downsample)
            h2_downsample_relu = self.relu(h2_downsample_conv)

            h3_downsample = self.downsample3(inputs3)
            h3_downsample_conv = self.conv3(h3_downsample)
            h3_downsample_relu = self.relu(h3_downsample_conv)

            h5_upsample = self.upsample(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_relu = self.relu(h5_upsample_conv)

            h4_conv = self.conv4(inputs4)
            h4_relu = self.relu(h4_conv)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_downsample_relu, h3_downsample_relu, h4_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_relu = self.relu(h_conv)
            return h_relu


class unetUp3(nn.Module):
    def __init__(self, filters, out_size):
        super(unetUp3, self).__init__()
        count = 5
        self.downsample1 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.downsample2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv1 = nn.Conv2d(filters[0], out_size, 3, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_size, 3, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_size, 3, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv5 = nn.Conv2d(filters[4], out_size, 3, padding=1)
        self.conv = nn.Conv2d(filters[0] * count, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_bn = self.bn(h1_downsample_conv)
            h1_downsample_relu = self.relu(h1_downsample_bn)

            h2_downsample = self.downsample2(inputs2)
            h2_downsample_conv = self.conv2(h2_downsample)
            h2_downsample_bn = self.bn(h2_downsample_conv)
            h2_downsample_relu = self.relu(h2_downsample_bn)

            h3_conv = self.conv3(inputs3)
            h3_bn = self.bn(h3_conv)
            h3_relu = self.relu(h3_bn)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_bn = self.bn(h4_upsample_conv)
            h4_upsample_relu = self.relu(h4_upsample_bn)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_bn = self.bn(h5_upsample_conv)
            h5_upsample_relu = self.relu(h5_upsample_bn)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_downsample_relu, h3_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_bn = self.bn(h_conv)
            h_relu = self.relu(h_bn)
            return h_relu
        else:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_relu = self.relu(h1_downsample_conv)

            h2_downsample = self.downsample2(inputs2)
            h2_downsample_conv = self.conv2(h2_downsample)
            h2_downsample_relu = self.relu(h2_downsample_conv)

            h3_conv = self.conv3(inputs3)
            h3_relu = self.relu(h3_conv)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_relu = self.relu(h4_upsample_conv)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_relu = self.relu(h5_upsample_conv)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_downsample_relu, h3_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_relu = self.relu(h_conv)
            return h_relu


class unetUp2(nn.Module):
    def __init__(self, filters, out_size):
        super(unetUp2, self).__init__()
        count = 5
        self.downsample1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.conv1 = nn.Conv2d(filters[0], out_size, 3, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_size, 3, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv5 = nn.Conv2d(filters[4], out_size, 3, padding=1)
        self.conv = nn.Conv2d(filters[0] * count, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_bn = self.bn(h1_downsample_conv)
            h1_downsample_relu = self.relu(h1_downsample_bn)

            h2_conv = self.conv2(inputs2)
            h2_bn = self.bn(h2_conv)
            h2_relu = self.relu(h2_bn)

            h3_upsample = self.upsample3(inputs3)
            h3_upsample_conv = self.conv3(h3_upsample)
            h3_upsample_bn = self.bn(h3_upsample_conv)
            h3_upsample_relu = self.relu(h3_upsample_bn)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_bn = self.bn(h4_upsample_conv)
            h4_upsample_relu = self.relu(h4_upsample_bn)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv4(h5_upsample)
            h5_upsample_bn = self.bn(h5_upsample_conv)
            h5_upsample_relu = self.relu(h5_upsample_bn)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_relu, h3_upsample_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_bn = self.bn(h_conv)
            h_relu = self.relu(h_bn)
            return h_relu
        else:
            h1_downsample = self.downsample1(inputs1)
            h1_downsample_conv = self.conv1(h1_downsample)
            h1_downsample_relu = self.relu(h1_downsample_conv)

            h2_conv = self.conv2(inputs2)
            h2_relu = self.relu(h2_conv)

            h3_upsample = self.upsample3(inputs3)
            h3_upsample_conv = self.conv3(h3_upsample)
            h3_upsample_relu = self.relu(h3_upsample_conv)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_relu = self.relu(h4_upsample_conv)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_relu = self.relu(h5_upsample_conv)

            h_concat = torch.cat(
                (h1_downsample_relu, h2_relu, h3_upsample_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_relu = self.relu(h_conv)
            return h_relu


class unetUp1(nn.Module):
    def __init__(self, filters, out_size):
        super(unetUp1, self).__init__()
        count = 5
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.conv1 = nn.Conv2d(filters[0], out_size, 3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, 3, padding=1)
        self.conv5 = nn.Conv2d(filters[4], out_size, 3, padding=1)
        self.conv = nn.Conv2d(filters[0] * count, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            h1_conv = self.conv1(inputs1)
            h1_bn = self.bn(h1_conv)
            h1_relu = self.relu(h1_bn)

            h2_upsample = self.upsample2(inputs2)
            h2_upsample_conv = self.conv2(h2_upsample)
            h2_upsample_bn = self.bn(h2_upsample_conv)
            h2_upsample_relu = self.relu(h2_upsample_bn)

            h3_upsample = self.upsample3(inputs3)
            h3_upsample_conv = self.conv3(h3_upsample)
            h3_upsample_bn = self.bn(h3_upsample_conv)
            h3_upsample_relu = self.relu(h3_upsample_bn)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_bn = self.bn(h4_upsample_conv)
            h4_upsample_relu = self.relu(h4_upsample_bn)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_bn = self.bn(h5_upsample_conv)
            h5_upsample_relu = self.relu(h5_upsample_bn)

            h_concat = torch.cat(
                (h1_relu, h2_upsample_relu, h3_upsample_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_bn = self.bn(h_conv)
            h_relu = self.relu(h_bn)
            return h_relu
        else:
            h1_conv = self.conv1(inputs1)
            h1_relu = self.relu(h1_conv)

            h2_upsample = self.upsample2(inputs2)
            h2_upsample_conv = self.conv2(h2_upsample)
            h2_upsample_relu = self.relu(h2_upsample_conv)

            h3_upsample = self.upsample3(inputs3)
            h3_upsample_conv = self.conv3(h3_upsample)
            h3_upsample_relu = self.relu(h3_upsample_conv)

            h4_upsample = self.upsample4(inputs4)
            h4_upsample_conv = self.conv4(h4_upsample)
            h4_upsample_relu = self.relu(h4_upsample_conv)

            h5_upsample = self.upsample5(inputs5)
            h5_upsample_conv = self.conv5(h5_upsample)
            h5_upsample_relu = self.relu(h5_upsample_conv)

            h_concat = torch.cat(
                (h1_relu, h2_upsample_relu, h3_upsample_relu, h4_upsample_relu, h5_upsample_relu), 1)
            h_conv = self.conv(h_concat)
            h_relu = self.relu(h_conv)
            return h_relu


class UNet3Plus(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, pretrained=False, backbone='resnet50'):
        super(UNet3Plus, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(in_ch=in_ch, pretrained=pretrained)
        elif backbone == "resnet50":
            self.resnet = resnet50(in_ch=in_ch, pretrained=pretrained)
        else:
            raise ValueError(
                'Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        if backbone == 'vgg':
            out_filters = [64, 128, 256, 512, 1024]
            out_channels = 64
        elif backbone == "resnet50":
            out_filters = [64, 256, 512, 1024, 2048]
            out_channels = 64
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp4(out_filters, out_channels)
        # 128,128,256
        self.up_concat3 = unetUp3(out_filters, out_channels)
        # 256,256,128
        self.up_concat2 = unetUp2(out_filters, out_channels)
        # 512,512,64
        self.up_concat1 = unetUp1(out_filters, out_channels)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0],
                          kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], n_classes, 1)

        self.backbone = backbone

    def forward(self, inputs, bn=False):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # print("feat1:"+str(feat1.shape))
        # print("feat2:" + str(feat2.shape))
        # print("feat3:" + str(feat3.shape))
        # print("feat4:" + str(feat4.shape))
        # print("feat5:" + str(feat5.shape))
        up4 = self.up_concat4(feat1, feat2, feat3, feat4, feat5, bn)
        up3 = self.up_concat3(feat1, feat2, feat3, up4, feat5, bn)
        up2 = self.up_concat2(feat1, feat2, up3, up4, feat5, bn)
        up1 = self.up_concat1(feat1, up2, up3, up4, feat5, bn)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


class UNet3Plus_DeepSupervision(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg'):
        super(UNet3Plus, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1536]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError(
                'Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        out_channels = 64
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp4(out_filters, out_channels)
        # 128,128,256
        self.up_concat3 = unetUp3(out_filters, out_channels)
        # 256,256,128
        self.up_concat2 = unetUp2(out_filters, out_channels)
        # 512,512,64
        self.up_concat1 = unetUp1(out_filters, out_channels)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0],
                          kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs, bn):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat1, feat2, feat3, feat4, feat5, bn)
        up3 = self.up_concat3(feat1, feat2, feat3, up4, feat5, bn)
        up2 = self.up_concat2(feat1, feat2, up3, up4, feat5, bn)
        up1 = self.up_concat1(feat1, up2, up3, up4, feat5, bn)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


class UNet3Plus_DeepSupervision_CGM(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg'):
        super(UNet3Plus, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1536]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError(
                'Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        out_channels = 64
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp4(out_filters, out_channels)
        # 128,128,256
        self.up_concat3 = unetUp3(out_filters, out_channels)
        # 256,256,128
        self.up_concat2 = unetUp2(out_filters, out_channels)
        # 512,512,64
        self.up_concat1 = unetUp1(out_filters, out_channels)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0],
                          kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs, bn):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat1, feat2, feat3, feat4, feat5, bn)
        up3 = self.up_concat3(feat1, feat2, feat3, up4, feat5, bn)
        up2 = self.up_concat2(feat1, feat2, up3, up4, feat5, bn)
        up1 = self.up_concat1(feat1, up2, up3, up4, feat5, bn)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
