from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import os
from torch.utils import data
import torch.backends.cudnn as cudnn
import h5py

from utils.helpers import *
from dataset import LandslideDataSet
from modules import parse_args

from models.Deeplabv3_Plus.deeplabv3_plus import DeepLab

# from models.LinkNet.linkNet_ResNet import LinkNet
# from models.LinkNet.linkNet_Ghost import LinkNet
from models.LinkNet.linkNet_ShuffleNet import LinkNet

# from models.MobileViT import get_model_from_name
# from models.SPNet.spnet import SPNet
from models.TransUNet.transunet import TransUNet
# from models.UNet2Plus.UNet2Plus import UNet2Plus
# from models.UNet3Plus.UNet3Plus import UNet3Plus
# from models.SwinUNet.SwinUNet import SwinUNet
# from models.SegNet.SegNet import SegResNet
from models.ResUNet.ResUNet import ResUNet
from models.ResUNet.CBAM_ResUNet import CBAM_ResUNet


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

    snapshot_dir = args.snapshot_dir
    if os.path.exists(snapshot_dir) == False:
        os.makedirs(snapshot_dir)

    input_size = (args.input_shape[0], args.input_shape[1])

    cudnn.enabled = True
    cudnn.benchmark = True

    # Create network

    if args.net == 'DeepLab':
        model = DeepLab(
            in_ch=args.input_shape[2], n_classes=args.num_classes)
    elif args.net == 'LinkNet':
        model = LinkNet(
            in_ch=args.input_shape[2], n_classes=args.num_classes)
    elif args.net == 'TransUNet':
        model = TransUNet(
            in_ch=args.input_shape[2], n_classes=args.num_classes)
    elif args.net == 'ResUNet':
        model = ResUNet(
            in_ch=args.input_shape[2], n_classes=args.num_classes)
    elif args.net == 'CBAM_ResUNet':
        model = CBAM_ResUNet(
            in_ch=args.input_shape[2], n_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model = model.cuda()

    test_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.test_list, set_mask='masked'),
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    print('Testing..........')
    model.eval()

    for index, batch in enumerate(test_loader):
        image, _, _, name = batch
        image = image.float().cuda()
        name = name[0].split('.')[0].split('/')[-1].replace('image', 'mask')
        print(index+1, '/', len(test_loader), ': Testing ', name)

        with torch.no_grad():
            pred = model(image)

        _, pred = torch.max(
            interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')

        with h5py.File(snapshot_dir+name+'.h5', 'w') as hf:
            hf.create_dataset('mask', data=pred)


if __name__ == '__main__':
    main()
