import argparse
import os
import shutil
import random
import glob2
import numpy as np
import pandas as pd


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_size_dataset(dir_path):
    count = 0

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count


def import_name(modulename, name):
    """ Import a named object from a module in the context of this function.
    """
    try:
        module = __import__(modulename, globals(), locals(), [name])
    except ImportError:
        return None
    return vars(module)[name]


def draw_curve(current_epoch, x_epoch, y_loss, y_err, fig, ax0, ax1):
    x_epoch.append(current_epoch + 1)
    ax0.plot(x_epoch, y_loss['train'], 'b-', linewidth=1.0, label='train')
    ax0.plot(x_epoch, y_loss['val'], '-r', linewidth=1.0, label='val')
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax1.plot(x_epoch, y_err['train'], '-b', linewidth=1.0, label='train')
    ax1.plot(x_epoch, y_err['val'], '-r', linewidth=1.0, label='val')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("error")
    if current_epoch == 0:
        ax0.legend(loc="upper right")
        ax1.legend(loc="upper right")
    fig.savefig(os.path.join('image/', 'train_curves.jpg'), dpi=600)


def split_fold(num_fold=10, test_image_number=380):
    print("Splitting for k-fold with %d fold" % num_fold)
    data_root = os.path.join(os.getcwd(), 'data')

    dir_names = []
    for fold in range(num_fold):
        dir_names.append('data/TrainData' + str(fold))

    for dir_name in dir_names:
        print("Creating fold " + dir_name)
        os.makedirs(dir_name)

        # making subdirectory train and test
        os.makedirs(os.path.join(os.getcwd(), dir_name, 'test'))
        os.makedirs(os.path.join(os.getcwd(), dir_name, 'train'))

        # locating to the test and train directory
        test_dir = os.path.join(os.getcwd(), dir_name, 'test')
        train_dir = os.path.join(os.getcwd(), dir_name, 'train')

        # making image and mask sub-dirs
        os.makedirs(os.path.join(test_dir, 'img'))
        os.makedirs(os.path.join(test_dir, 'mask'))
        os.makedirs(os.path.join(train_dir, 'img'))
        os.makedirs(os.path.join(train_dir, 'mask'))

        # read the image and mask directory
        image_files = os.listdir(os.path.join(os.getcwd(), 'data/img'))
        mask_files = os.listdir(os.path.join(os.getcwd(), 'data/mask'))

        # creating random file names for testing
        test_filenames = random.sample(image_files, test_image_number)

        for filename in test_filenames:
            img_data_root = os.path.join(data_root, 'img')
            msk_data_root = os.path.join(data_root, 'mask')

            img_dest = os.path.join(os.getcwd(), dir_name, 'test', 'img')
            msk_dest = os.path.join(os.getcwd(), dir_name, 'test', 'mask')

            img_file_path = os.path.join(img_data_root, filename)
            msk_file_path = os.path.join(
                msk_data_root, filename.replace('image', 'mask'))

            shutil.copy(img_file_path, img_dest)
            shutil.copy(msk_file_path, msk_dest)

        # saving files for training
        for other_filename in image_files:
            if other_filename in test_filenames:
                continue
            else:
                img_data_root = os.path.join(data_root, 'img')
                msk_data_root = os.path.join(data_root, 'mask')

                img_dest = os.path.join(os.getcwd(), dir_name, 'train', 'img')
                msk_dest = os.path.join(os.getcwd(), dir_name, 'train', 'mask')

                img_file_path = os.path.join(img_data_root, other_filename)
                msk_file_path = os.path.join(
                    msk_data_root, other_filename.replace('image', 'mask'))

                shutil.copy(img_file_path, img_dest)
                shutil.copy(msk_file_path, msk_dest)


def get_train_list(fold):
    all_files = []
    for ext in ["*.h5"]:
        images = glob2.glob(os.path.join(
            "data/TrainData" + str(fold) + "/train/img/", ext))
        all_files += images

    all_train_files = []
    for idx in np.arange(len(all_files)):
        image = str(all_files[idx]).split("/")
        image = os.path.join("TrainData" + str(fold) +
                             "/train/img/", str(image[4]))
        all_train_files.append(image)

    # Create train.txt
    with open("dataset/train.txt", "w") as f:
        for idx in np.arange(len(all_train_files)):
            f.write(all_train_files[idx] + '\n')


def get_test_list(fold):
    all_files = []
    for ext in ["*.h5"]:
        images = glob2.glob(os.path.join(
            "data/TrainData" + str(fold) + "/test/img/", ext))
        all_files += images

    all_test_files = []
    for idx in np.arange(len(all_files)):
        image = str(all_files[idx]).split("/")
        image = os.path.join("TrainData" + str(fold) +
                             "/test/img/", str(image[4]))
        all_test_files.append(image)

    # Create Test.txt
    with open("dataset/test.txt", "w") as f:
        for idx in np.arange(len(all_test_files)):
            f.write(all_test_files[idx] + '\n')


def get_train_test_list(fold):
    get_train_list(fold)
    get_test_list(fold)


# Put multiple dataframes into one xlsx sheet
def multiple_dfs(df_list, sheets, file_name, spaces):
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheets, startrow=row, startcol=0)
        row = row + len(dataframe.index) + spaces + 1
    writer.save()


# Put multiple dataframes across separate tabs/sheets
def dfs_tabs(df_list, sheet_list, file_name):
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0)
    writer.save()


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)
