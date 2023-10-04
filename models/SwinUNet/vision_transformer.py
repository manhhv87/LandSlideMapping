# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUNet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_ch=3, n_classes=2, window_size=7, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1], num_heads=[3, 6, 12, 24]):
        super(SwinUNet, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.num_classes = n_classes
        self.window_size = window_size    
        self.depths = depths
        self.depths_decoder = depths_decoder
        self.num_heads = num_heads       
        
        # The size of the image, preferable a multiple of 2, and it must be able to divide WINDOW_SIZE
        self.swin_unet = SwinTransformerSys(img_size=self.img_size,
                                            patch_size=self.patch_size,         # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
                                            in_chans=self.in_ch,
                                            num_classes=self.num_classes,                                            
                                            window_size=self.window_size,       # the size of attention window per down/upsampling level
                                            depths=self.depths,                 # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
                                            depths_decoder=self.depths_decoder,
                                            num_heads=self.num_heads,           # number of attention heads per down/upsampling level 
                                            embed_dim=96,                       # number of channels in the first downsampling block; it is also the number of embedded dimens                                                                               
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.0,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k,
                                   v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(
                    pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(
                            k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
