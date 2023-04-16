from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

from modules import parse_args

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def create_data_mask(path_to_img, path_to_ground_truth, path_to_pred_mask):

    data_temp = []
    data = []

    for file in glob.iglob(path_to_img):
        mask_name = file.split("/")[-1].replace('image', 'mask')
        path_to_gt = str(path_to_ground_truth) + mask_name

        with h5py.File(file) as hdf:
            image = np.array(hdf.get('img'))
            data_rgb = image[:, :, 3:0:-1]

            for i in range(0, data_rgb.shape[-1]):
                data_rgb[:, :, i] = data_rgb[:, :, i] / data_rgb[:, :, i].max()

        with h5py.File(path_to_gt) as hdf:
            mask_truth = np.array(hdf.get('mask'))
            mask_truth = np.float32(mask_truth)

        with h5py.File(path_to_pred_mask + 'CBAM_ResUNet/' + mask_name) as hdf:
            mask_CBAM_ResUNet = np.array(hdf.get('mask'))
            mask_CBAM_ResUNet = np.float32(mask_CBAM_ResUNet == 1)

        with h5py.File(path_to_pred_mask + 'ResUNet/' + mask_name) as hdf:
            mask_ResUNet = np.array(hdf.get('mask'))
            mask_ResUNet = np.float32(mask_ResUNet == 1)

        with h5py.File(path_to_pred_mask + 'DeepLab/' + mask_name) as hdf:
            mask_DeepLab = np.array(hdf.get('mask'))
            mask_DeepLab = np.float32(mask_DeepLab == 1)

        with h5py.File(path_to_pred_mask + 'LinkNet/' + mask_name) as hdf:
            mask_LinkNet = np.array(hdf.get('mask'))
            mask_LinkNet = np.float32(mask_LinkNet == 1)

        with h5py.File(path_to_pred_mask + 'TransUNet/' + mask_name) as hdf:
            mask_TransUNet = np.array(hdf.get('mask'))
            mask_TransUNet = np.float32(mask_TransUNet == 1)

        data_temp = [data_rgb, mask_truth, mask_CBAM_ResUNet,
                     mask_ResUNet, mask_DeepLab, mask_LinkNet, mask_TransUNet]
        data.append(data_temp)

    return data


def plot_figure(data, model_name):
    fig = plt.figure(figsize=(30, 20))
    nrows = len(data)
    ncols = len(data[0])

    ax = []
    idx_row = 0

    for i in range(0, nrows):
        idx_row += 1
        for j in range(0, ncols-1):
            ax.append(fig.add_subplot(nrows, ncols, i*ncols+1))
            plt.imshow(data[i][0])

            if idx_row == nrows:
                ax[-1].set_title(model_name[0], fontsize=24)  # set title

            plt.axis('off')

            ax.append(fig.add_subplot(nrows, ncols, i*ncols + j + 2))
            plt.imshow(data[i][j+1])

            if idx_row == nrows:
                ax[-1].set_title(model_name[j+1], fontsize=24)  # set title

            plt.axis('off')

    plt.savefig('/content/figure.png')


def main():
    args = parse_args()

    data = create_data_mask(
        args.path_to_img, args.path_to_ground_truth, args.path_to_pred_mask)

    model_name = ['RGB', 'Ground Truth', 'CBAM_ResUNet',
                  'ResUNet', 'DeepLab', 'LinkNet', 'TransUNet']

    plot_figure(data, model_name)


if __name__ == '__main__':
    main()
