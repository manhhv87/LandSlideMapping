import torch
import torch.nn as nn

from .backbones.ResNet50 import resnet50
from .backbones.VGG16 import VGG16


class unetUp4(nn.Module):
    def __init__(self, input_size, out_size):
        super(unetUp4, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(input_size + 4*out_size, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, bn):
        if bn:
            inputs1_upsample = self.upsample(inputs1)
            output_concat = torch.cat(
                (inputs1_upsample, inputs2, inputs3, inputs4, inputs5), 1)

            output_concat_conv = self.conv(output_concat)
            output_concat_bn = self.bn(output_concat_conv)
            output_concat_relu = self.relu(output_concat_bn)

            return output_concat_relu
        else:
            inputs1_upsample = self.upsample(inputs1)
            output_concat = torch.cat(
                (inputs1_upsample, inputs2, inputs3, inputs4, inputs5), 1)

            output_concat_conv = self.conv(output_concat)
            output_concat_relu = self.relu(output_concat_conv)

            return output_concat_relu


class unetUp3(nn.Module):
    def __init__(self, input_size, out_size):
        super(unetUp3, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(input_size + 3*out_size, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, inputs4, bn):
        if bn:
            inputs1_upsample = self.upsample(inputs1)
            output_concat = torch.cat(
                (inputs1_upsample, inputs2, inputs3, inputs4), 1)

            output_concat_conv = self.conv(output_concat)
            output_concat_bn = self.bn(output_concat_conv)
            output_concat_relu = self.relu(output_concat_bn)

            return output_concat_relu
        else:
            inputs1_upsample = self.upsample(inputs1)
            output_concat = torch.cat(
                (inputs1_upsample, inputs2, inputs3, inputs4), 1)

            output_concat_conv = self.conv(output_concat)
            output_concat_relu = self.relu(output_concat_conv)

            return output_concat_relu


class unetUp2(nn.Module):
    def __init__(self, input_size, out_size):
        super(unetUp2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(input_size + 2*out_size, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, inputs3, bn):
        if bn:
            inputs1_upsample = self.upsample(inputs1)
            output_concat = torch.cat((inputs1_upsample, inputs2, inputs3), 1)

            output_concat_conv = self.conv(output_concat)
            output_concat_bn = self.bn(output_concat_conv)
            output_concat_relu = self.relu(output_concat_bn)

            return output_concat_relu
        else:
            inputs1_upsample = self.upsample(inputs1)
            output_concat = torch.cat((inputs1_upsample, inputs2, inputs3), 1)

            output_concat_conv = self.conv(output_concat)
            output_concat_relu = self.relu(output_concat_conv)

            return output_concat_relu


class unetUp1(nn.Module):
    def __init__(self, input_size, out_size):
        super(unetUp1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(input_size+out_size, out_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, bn):
        if bn:
            inputs1_upsample = self.upsample(inputs1)
            output_concat = torch.cat((inputs1_upsample, inputs2), 1)

            output_concat_conv = self.conv(output_concat)
            output_concat_bn = self.bn(output_concat_conv)
            output_concat_relu = self.relu(output_concat_bn)

            return output_concat_relu
        else:
            inputs1_upsample = self.upsample(inputs1)
            output_concat = torch.cat((inputs1_upsample, inputs2), 1)

            output_concat_conv = self.conv(output_concat)
            output_concat_relu = self.relu(output_concat_conv)

            return output_concat_relu


class UNet2Plus(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, pretrained=False, backbone='resnet50'):
        super(UNet2Plus, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(in_ch=in_ch, pretrained=pretrained)
        elif backbone == "resnet50":
            self.resnet = resnet50(in_ch=in_ch, pretrained=pretrained)
        else:
            raise ValueError(
                'Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))

        if backbone == 'vgg':
            out_filters = [64, 128, 256, 512, 1024]
        elif backbone == "resnet50":
            out_filters = [64, 256, 512, 1024, 2048]

        # upsampling
        # 64,64,512
        self.up_concat01 = unetUp1(out_filters[1], out_filters[0])
        self.up_concat11 = unetUp1(out_filters[2], out_filters[1])
        self.up_concat21 = unetUp1(out_filters[3], out_filters[2])
        self.up_concat31 = unetUp1(out_filters[4], out_filters[3])
        # 128,128,256
        self.up_concat02 = unetUp2(out_filters[1], out_filters[0])
        self.up_concat12 = unetUp2(out_filters[2], out_filters[1])
        self.up_concat22 = unetUp2(out_filters[3], out_filters[2])
        # 256,256,128
        self.up_concat03 = unetUp3(out_filters[1], out_filters[0])
        self.up_concat13 = unetUp3(out_filters[2], out_filters[1])
        # 512,512,64
        self.up_concat04 = unetUp4(out_filters[1], out_filters[0])

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
        x01 = self.up_concat01(feat2, feat1, True)
        x11 = self.up_concat11(feat3, feat2, True)
        x21 = self.up_concat21(feat4, feat3, True)
        x31 = self.up_concat31(feat5, feat4, True)

        x02 = self.up_concat02(x11, x01, feat1, True)
        x12 = self.up_concat12(x21, x11, feat2, True)
        x22 = self.up_concat22(x31, x21, feat3, True)

        x03 = self.up_concat03(x12, x02, x01, feat1, True)
        x13 = self.up_concat13(x22, x12, x11, feat2, True)

        x04 = self.up_concat04(x13, x03, x02, x01, feat1, True)

        if self.up_conv != None:
            x04 = self.up_conv(x04)

        final = self.final(x04)

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
