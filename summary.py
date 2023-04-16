# -----------------------------------------------------------------#
#   This part of the code is used to see the network structure
# -----------------------------------------------------------------#
import argparse

import torch
from thop import clever_format, profile
from torchsummary import summary

# from models.Deeplabv3_Plus.deeplabv3_plus import DeepLab

# from models.LinkNet.linkNet_ResNet import LinkNet
# from models.LinkNet.linkNet_Ghost import LinkNet
# from models.LinkNet.linkNet_ShuffleNet import LinkNet

# from models.MobileViT import get_model_from_name
# from models.SPNet.spnet import SPNet
# from models.TransUNet.transunet import TransUNet
# from models.UNet2Plus.UNet2Plus import UNet2Plus
# from models.UNet3Plus.UNet3Plus import UNet3Plus
# from models.SegFormer.SegFormer import Segformer
# from models.ResUNet.ResUNet import ResUNet
from models.ResUNet.CBAM_ResUNet import CBAM_ResUNet


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Semantic Segmentation network")

    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--input_shape", type=tuple_type, default="(128,128,14)",
                        help="Input images shape.")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = DeepLab(in_ch=args.input_shape[2], n_classes=args.num_classes).to(device)
    # model = LinkNet(in_ch=args.input_shape[2], n_classes=args.num_classes).to(device)

    # model = get_model_from_name['mobile_vit'](n_classes=args.num_classes).to(device)
    # model = SPNet(in_ch=args.input_shape[2], n_classes=args.num_classes).to(device)

    # model = TransUNet(img_dim=args.input_shape[0],
    #                   in_ch=args.input_shape[2],
    #                   n_classes=args.num_classes).to(device)

    # model = UNet2Plus(in_ch=args.input_shape[2], n_classes=args.num_classes).to(device)
    # model = UNet3Plus(in_ch=args.input_shape[2], n_classes=args.num_classes).to(device)
    # model = Segformer(in_ch=args.input_shape[2], n_classes=args.num_classes).to(device)

    # model = ResUNet(in_ch=args.input_shape[2], n_classes=args.num_classes).to(device)

    model = CBAM_ResUNet(
        in_ch=args.input_shape[2], n_classes=args.num_classes).to(device)

    summary(model, (args.input_shape[2],
            args.input_shape[0], args.input_shape[1]))

    dummy_input = torch.randn(
        1, args.input_shape[2], args.input_shape[0], args.input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)

    # ------------------------------------------------------------------------------#
    # flops * 2 is because the profile does not use convolution as two operations
    # Some papers use convolution to calculate multiplication and addition operations. multiply by 2
    # Some papers only consider the number of operations of multiplication, ignoring addition.
    # Do not multiply by 2 at this time
    # This code chooses to multiply by 2, refer to YOLOX.
    # --------------------------------------------------------#
    flops = flops * 2
    # flops, params = clever_format([flops, params], "%.2f")
    # print('Total params: %s' % (params))
    # print('Total GFLOPS: %s' % (flops))

    print("  %s  |  %s    " % ("Params(M)", "FLOPs(G)"))
    print("-------------|-------------")
    print("    %.2f    |   %.2f    " %
          (params / (1000 ** 2), flops / (1000 ** 3)))
