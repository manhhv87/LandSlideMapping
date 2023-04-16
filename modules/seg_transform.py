import numpy as np
import fastai
from fastai.vision.all import *


class SegmentationAlbumentationsTransform(ItemTransform):
    def __init__(self, aug, **kwargs):
        super(SegmentationAlbumentationsTransform, self).__init__(**kwargs)
        self.aug = aug

    def encodes(self, x):
        if len(x) == 2:
            img, mask = x
            aug = self.aug(image=np.array(img), mask=np.array(mask))
            return {'image': TensorImage(aug['image']), 'mask': TensorMask(aug['mask'])}
        else:
            aug = self.aug(image=np.array(x))
            return {'image': TensorImage(aug['image'])}
