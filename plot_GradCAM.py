from __future__ import absolute_import
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
import torchvision
import requests
import numpy as np
import torch.functional as F
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

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


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
            output = model(image)

    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    sem_classes = ['Non_LandSlide', 'LandSlide']
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    landslide_category = sem_class_to_idx["LandSlide"]
    landslide_mask = normalized_masks[0, :, :, :].argmax(
        axis=0).detach().cpu().numpy()
    landslide_mask_uint8 = 255 * np.uint8(landslide_mask == landslide_category)
    landslide_mask_float = np.float32(landslide_mask == landslide_category)

    target_layers = [model.fc]
    targets = [SemanticSegmentationTarget(
        landslide_category, landslide_mask_float)]

    with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=image, targets=targets)[0, :]
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=False)


if __name__ == '__main__':
    main()
