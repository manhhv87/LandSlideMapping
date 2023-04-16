from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import xlsxwriter

from modules import parse_args
from modules import Trainer
import utils as utl


def main():
    args = parse_args()

    if args.cuda and args.mGPUs:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integer only')

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    # Splitting k-fold
    utl.split_fold(num_fold=args.k_fold, test_image_number=int(
        utl.get_size_dataset('./data/img') / args.k_fold))

    # avg_train_losses_all_fold = []
    # avg_valid_losses_all_fold = []
    jac_fold = []
    pre_fold = []
    rec_fold = []
    f1_fold = []

    jac_all_fold = []
    pre_all_fold = []
    rec_all_fold = []
    f1_all_fold = []

    for fold in range(args.k_fold):
        print("\nTraining on fold %d" % fold)

        # Creating train.txt and test.txt
        utl.get_train_test_list(fold)

        # create snapshots directory
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        trainer = Trainer(args)

        train_per_epoch = np.ceil(utl.get_size_dataset(
            "./data/TrainData" + str(fold) + "/train/img/") / args.batch_size)

        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            kbar = utl.Kbar(target=train_per_epoch, epoch=epoch,
                            num_epochs=args.epochs, width=25, always_stateful=False)

            avg_train_losses = trainer.training(kbar)
            early_stop, avg_valid_losses, jac, pre, rec, f1 = trainer.validation(
                kbar, args.save_dir, fold)

            jac_fold.append(jac)
            pre_fold.append(pre)
            rec_fold.append(rec)
            f1_fold.append(f1)

            if args.est == 'True':
                if early_stop:
                    print("Early stopping")
                    break

        # load the last checkpoint with the best model
        # model = trainer.get_model()
        # model.load_state_dict(torch.load('checkpoint.pt'))

        # plot_curve(avg_train_losses, avg_valid_losses)

        # avg_train_losses_all_fold.append(avg_train_losses)
        # avg_valid_losses_all_fold.append(avg_valid_losses)
        jac_all_fold.append(jac_fold)
        pre_all_fold.append(pre_fold)
        rec_all_fold.append(rec_fold)
        f1_all_fold.append(f1_fold)

        jac_fold = []
        pre_fold = []
        rec_fold = []
        f1_fold = []

    df1 = pd.DataFrame(jac_all_fold)
    df1.index = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    df2 = pd.DataFrame(pre_all_fold)
    df2.index = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    df3 = pd.DataFrame(rec_all_fold)
    df3.index = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    df4 = pd.DataFrame(f1_all_fold)
    df4.index = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

    # run function
    utl.dfs_tabs([df1, df2, df3, df4], ['IoU', 'Precision', 'Recall',
                                        'F1-score'], args.save_dir + 'results/landslide.xlsx')


if __name__ == '__main__':
    main()
