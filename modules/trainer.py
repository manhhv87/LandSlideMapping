import torch
import os
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import albumentations as A

from modules import SegmentationAlbumentationsTransform
from modules import get_loss_function
from modules import get_scheduler
from modules import EarlyStopping
from metrics import SegMetric
import utils as utl
from dataset import LandslideDataSet

from models.Deeplabv3_Plus.deeplabv3_plus import DeepLab    # Ok

from models.LinkNet.linkNet_ResNet import LinkNet           # Ok
# from models.LinkNet.linkNet_Ghost import LinkNet
# from models.LinkNet.linkNet_ShuffleNet import LinkNet

# from models.MobileViT import get_model_from_name
from models.SPNet.spnet import SPNet
from models.TransUNet.transunet import TransUNet            # OK
from models.UNet2Plus.UNet2Plus import UNet2Plus
from models.UNet3Plus.UNet3Plus import UNet3Plus
from models.SwinUNet.vision_transformer import SwinUNet
from models.SegNet.SegNet import SegResNet
from models.ResUNet.ResUNet import ResUNet                  # OK
from models.ResUNet.CBAM_ResUNet import CBAM_ResUNet        # OK
from models.ResUNet.ResUNet_ASPP import ResUNet_ASPP

# Now we will create a pipe of transformations
aug_pipe = A.Compose([
    # A.Resize(32,32),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
             A.GridDistortion(p=0.5),
             A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1)], p=0.8),
])

# Create our class with this aug_pipe
aug = SegmentationAlbumentationsTransform(aug_pipe)


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        self.train_loader = data.DataLoader(LandslideDataSet(self.args.data_dir, self.args.train_list, transform=aug, set_mask='masked'),
                                            batch_size=self.args.batch_size, shuffle=True,
                                            num_workers=self.args.num_workers, pin_memory=True)

        self.test_loader = data.DataLoader(LandslideDataSet(self.args.data_dir, self.args.test_list, set_mask='masked'),
                                           batch_size=self.args.batch_size, shuffle=False,
                                           num_workers=self.args.num_workers, pin_memory=True)

        if args.net == 'DeepLab':
            model = DeepLab(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'LinkNet':
            model = LinkNet(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'SPNet':
            model = SPNet(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'TransUNet':
            model = TransUNet(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'UNet2Plus':
            model = UNet2Plus(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'UNet3Plus':
            model = UNet3Plus(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        # elif args.net == 'Segformer':
        #     model = Segformer(
        #         in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'SwinUNet':
            model = SwinUNet(
                in_ch=args.input_shape[2], n_classes=args.num_classes, img_size=128, window_size=8, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1])
        elif args.net == 'ResUNet':
            model = ResUNet(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'SegNet':
            model = SegResNet(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'CBAM_ResUNet':
            model = CBAM_ResUNet(
                in_ch=args.input_shape[2], n_classes=args.num_classes)
        elif args.net == 'ResUNet_ASPP':
            model = ResUNet_ASPP(
                in_ch=args.input_shape[2], n_classes=args.num_classes)

        if args.optimizer == 'adam':
            opt = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamax':
            opt = torch.optim.Adamax(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            opt = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            opt = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)

        self.model = model
        self.optimizer = opt

        print("No. Parameters: {}".format(utl.count_params(self.model)))

        # Define criterion
        self.criterion = get_loss_function(self.args)
        self.pla_lr_scheduler = get_scheduler(self.optimizer, self.args)

        if self.args.est == 'True':
            self.early_stopping = EarlyStopping(
                patience=15, delta=1e-3, verbose=True)

        # Define SegMeter
        self.segmetric = SegMetric()

        # multiple mGPUs
        if self.args.mGPUs:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.args.gpu_ids)

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()

        # to track the average training loss per epoch as the model trains
        self.avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        self.avg_valid_losses = []

        self.early_stop = False
        self.best_pred = -np.Inf

    def training(self, kbar):
        # to track the training loss as the model trains
        train_losses = []
        # prep model for training
        self.model.train()

        for batch_id, batch in enumerate(self.train_loader):
            image, target, _, _ = batch

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()

            inputs = Variable(image)
            labels = Variable(target)

            # forward pass: compute predicted outputs by passing inputs to the model
            preds = self.model(inputs)

            # calculate the loss
            loss = self.criterion(preds, labels.long())

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward(torch.ones_like(loss))

            # perform a single optimization step (parameter update)
            self.optimizer.step()

            # record training loss
            train_losses.append(loss.item())

            kbar.update(batch_id, values=[("loss", np.average(train_losses))])

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        self.avg_train_losses.append(train_loss)

        # clear lists to track next epoch
        train_losses = []
        return self.avg_train_losses

    def validation(self, kbar, args, fold):
        # to track the validation loss as the model trains
        valid_losses = []
        # prep model for evaluation
        self.model.eval()
        # self.evaluator.reset()
        self.segmetric.reset()

        for _, batch in enumerate(self.test_loader):
            image, target, _, _ = batch

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                probs = self.model(image)

            # calculate the loss
            loss = self.criterion(probs, target.long())

            # record validation loss
            valid_losses.append(loss.item())

            probs = probs.data.cpu().numpy()
            preds = np.argmax(probs, axis=1)

            # Add batch sample into evaluator
            self.segmetric.add_batch(
                torch.from_numpy(preds).cuda(), target.long(), mode='multiclass', num_classes=2)

        # print training/validation statistics
        # calculate average loss over an epoch
        valid_loss = np.average(valid_losses)
        self.avg_valid_losses.append(valid_loss)

        self.pla_lr_scheduler.step()

        # Fast test during the training
        acc = self.segmetric.accuracy(reduction='macro', class_idv=True)
        iou = self.segmetric.iou_score(reduction='macro', class_idv=True)
        pre = self.segmetric.positive_predictive_value(
            reduction='macro', class_idv=True)
        rec = self.segmetric.sensitivity(reduction='macro', class_idv=True)
        f1 = self.segmetric.f1_score(reduction='macro', class_idv=True)

        kbar.add(1, values=[("val_loss", valid_loss), ("Acc", acc[1]),
                            ("IoU", iou[1]), ('precision', pre[1]), ('recall', rec[1]), ('f1', f1[1])])

        # free up memory
        torch.cuda.empty_cache()

        if self.args.est == 'True':
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if self.args.val_metric == 'val_loss':
                self.early_stopping(valid_loss, self.model)
            elif self.args.val_metric == 'f1':
                self.early_stopping(f1[1], self.model)
            else:
                raise Exception("Sorry, don't know metrics")

            if self.early_stopping.early_stop:
                self.early_stop = True

        else:
            if f1[1] > self.best_pred:
                print('\nF1 score improved from %0.5f to %0.5f, saving model...' %
                      (self.best_pred, f1[1]))

                # Save model
                torch.save(self.model.state_dict(), os.path.join(
                    args + 'models', 'model_' + str(fold) + '.pth'))

                # Update best validation mIoU
                self.best_pred = f1[1]

            else:
                print('\nF1 score (%.05f) did not improve from %0.5f' %
                      (f1[1], self.best_pred))

        # clear lists to track next epoch
        valid_losses = []

        return self.early_stop, self.avg_valid_losses, iou, pre, rec, f1
