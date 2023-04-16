import argparse

from utils.helpers import tuple_type


def parse_args():
    """Parse all the arguments provided from the CLI.

        Returns:
          A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train a Semantic Segmentation network")

    parser.add_argument("--net", type=str, default='CBAM_ResUNet',
                        help="network name.")
    parser.add_argument("--data_dir", type=str, default='./TrainData/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="training list file.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--input_shape", type=tuple_type, default="(128,128,14)",
                        help="Input images shape.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of images sent to the network in one step.")
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=0,
                        help='starting epoch')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread data-loading.")

    # cuda
    parser.add_argument('--cuda', dest='cuda', type=bool, default=True,
                        help='whether use CUDA')

    # multiple GPUs
    parser.add_argument('--mGPUs', dest='mGPUs', type=bool, default=False,
                        help='whether use multiple GPUs')
    parser.add_argument('--gpu_ids', dest='gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')

    parser.add_argument("--save_dir", type=str, default='./exp/',
                        help="where to save snapshots of the modules.")

    parser.add_argument("--k_fold", type=int, default=10,
                        help="number of fold for k-fold.")

    # config optimization
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='adamax',
                        help='training optimizer: adam, adamax, adamW, sgd')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3,
                        help='starting learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5,
                        help='weight_decay')

    parser.add_argument('--loss_func', dest='loss_func', type=str, default='dice',
                        help='loss function')
    parser.add_argument('--val_metric', dest='val_metric', type=str, default='val_loss',
                        help='metric to eval: val_loss, f1')

    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [lambda | step | multistep | exp | plateau | cosine | clr]')
    parser.add_argument('--warmup', type=int, default=10,
                        help='number of epochs before starting learning rate')

    parser.add_argument('--est', dest='est',  type=str, default='False',
                        help='early stopping')

    parser.add_argument("--snapshot_dir", type=str, default='./test_mask/',
                        help="where to save predicted maps.")
    parser.add_argument("--restore_from", type=str, default='./model.pth',
                        help="trained model.")

    parser.add_argument("--path_to_img", type=str, default='./pred_mask/img/image_*.h5',
                        help="path to image file.")
    parser.add_argument("--path_to_ground_truth", type=str, default='./pred_mask/mask/',
                        help="path to ground truth.")
    parser.add_argument("--path_to_pred_mask", type=str, default='./pred_mask/',
                        help="image name for pred.")

    parser.add_argument("--snapshot_CAM_dir", type=str, default='./CAM_map/',
                        help="where to save CAM maps.")

    return parser.parse_args()
