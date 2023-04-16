import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Optional, List
import numpy as np

import losses._functional as _F

from losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["DiceLoss",
           "FocalLoss",
           "JaccardLoss",
           "JDTLoss",
           "LovaszLoss",
           "MCCLoss",
           "SoftBCEWithLogitsLoss",
           "SoftCrossEntropyLoss",
           "TverskyLoss",
           "ComboLoss"]


class DiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = _F.to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                # N,H*W -> N,H*W, C
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * \
                    mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(y_pred, y_true.type_as(
            y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return _F.soft_dice_score(output, target, smooth, eps, dims)


class FocalLoss(_Loss):
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target. 
                   alpha > 0.5 penalises false negatives more than false positives, by default None
            gamma: Power factor for dampening weight (focal strength), by default 2.
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            _F.focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:

            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss


class JaccardLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(JaccardLoss, self).__init__()

        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = _F.to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = _F.soft_jaccard_score(
            y_pred,
            y_true.type(y_pred.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


"""
The losses are proposed in [1] Jaccard Metric Losses: 
Optimizing the Jaccard Index with Soft Labels <https://arxiv.org/abs/2302.05666>.
"""


class JDTLoss(_Loss):
    def __init__(self, loss="Jaccard", per_image=False, log_loss=False, from_logits=True, T=1.0, smooth=1.0, gamma=1.0, threshold=0.01, class_weight=None, ignore_index=None):
        super().__init__()
        """
        Args:
            loss (str): Compute the `"Jaccard"` loss.
            per_image (bool): Compute the loss per image or per batch.
            log_loss (bool): Compute the log loss or not.
            from_logits (bool): Inputs are logits or probabilities.
            T (float): Temperature to smooth predicted probabilities.
            smooth (float): A float number to avoid NaN error.
            gamma (float): When `gamma` > 1, the loss focuses more on less accurate predictions that have been misclassified.
            threshold (float): Threshold to select active classes.
            class_weight (torch.Tensor | list[float] | None): Weight of each class. If specified, its size should be equal to the number of classes.
            ignore_index (int | None): The class index to be ignored.
        """

        self.loss = loss
        self.per_image = per_image
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.T = T
        self.smooth = smooth
        self.gamma = gamma
        self.threshold = threshold
        self.class_weight = class_weight
        self.ignore_index = ignore_index

    def compute_active_classes(self, prob, label, classes):
        """
        Args:
            prob (torch.Tensor):        Its shape should be (C, -1).
            label (torch.Tensor):       Its shape should be (C, -1).
            classes (str | list[int]):  When it is ‘str‘, it is the mode to compute active classes. 
                                        When it is ‘list[int]‘, it is the list of class indices to compute the loss.
            threshold (float):          Threshold to select active classes.
         """
        all_classes = torch.arange(prob.shape[0]).cuda()   # tensor([0, 1])

        if classes == "All":
            active_classes = all_classes
        elif classes == "Present":
            active_classes = torch.argmax(label, dim=0).unique()
        elif classes == "Prob":
            active_classes = all_classes[torch.amax(
                prob, dim=1) > self.threshold]
        elif classes == "Label":
            active_classes = all_classes[torch.amax(
                label, dim=1) > self.threshold]
        elif classes == "Both":
            active_classes = all_classes[torch.amax(
                prob + label, dim=1) > self.threshold]
        else:
            active_classes = torch.tensor(classes)

        return active_classes

    def compute_loss(self, prob, label):
        if self.loss == "Jaccard":
            cardinality = torch.sum(prob + label, dim=1)
            difference = torch.sum(torch.abs(prob - label), dim=1)
            intersection = cardinality - difference
            union = cardinality + difference
            score = (intersection + self.smooth) / (union + self.smooth)

        if self.log_loss:
            losses = -torch.log(score)
        else:
            losses = 1.0 - score

        if self.gamma > 1.0:
            losses **= (1.0 / self.gamma)

        if self.class_weight is not None:
            losses *= self.class_weight

        return losses

    def forward_per_image(self, prob, label, not_ignore, classes):
        num_classes, batch_size = prob.shape[:2]

        losses = ctn = 0

        for i in range(batch_size):
            not_ignore_i = not_ignore[:, i, :]
            prob_i = prob[:, i, :][not_ignore_i].reshape(num_classes, -1)
            label_i = label[:, i, :][not_ignore_i].reshape(num_classes, -1)

            if prob_i.size(1) < 1:
                continue

            active_classes = self.compute_active_classes(
                prob_i, label_i, classes)

            if active_classes.size(0) < 1:
                continue

            losses_i = self.compute_loss(prob_i, label_i)
            losses += losses_i[active_classes].mean()
            ctn += 1

        if ctn == 0:
            return 0. * prob.sum()

        return losses / ctn

    def forward_per_batch(self, prob, label, not_ignore, classes):
        """
        In distributed training, `forward_per_batch` computes the loss per GPU-batch instead of per whole batch.
        """

        num_classes = prob.shape[0]

        prob = prob.reshape(num_classes, -1)
        label = label.reshape(num_classes, -1)
        not_ignore = not_ignore.reshape(num_classes, -1)

        prob = prob[not_ignore].reshape(num_classes, -1)
        label = label[not_ignore].reshape(num_classes, -1)

        if prob.size(1) < 1:
            return 0. * prob.sum()

        active_classes = self.compute_active_classes(prob, label, classes)

        if active_classes.size(0) < 1:
            return 0. * prob.sum()

        losses = self.compute_loss(prob, label)

        return losses[active_classes].mean()

    def forward(self, logits, label, not_ignore=None, classes="Label"):
        """
        Args:
            logits (torch.Tensor): Logits or probabilities. Its shape should be (B, C, D1, D2, ...).
            label (torch.Tensor): When it is hard label, its shape should be (B, D1, D2, ...).
                                  When it is soft label, its shape should be (B, C, D1, D2, ...).
            not_ignore (torch.Tensor | None): (1) If `self.ignore_index` is `None`, it can be `None`.
                                              (2) If `self.ignore_index` is not `None` and `label` is hard label, it can be `None`.
                                              (3) In all other cases, its shape should be (B, D1, D2, ...) and its dtype should be `torch.bool`.
            classes (str | list[int]): When it is `str`, it is the mode to compute active classes.
                                       When it is `list[int]`, it is the list of class indices to compute the loss. 
        """

        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long

        # (B, C, D1, D2, ...) -> (C, B, D1 * D2 * ...)
        logits = logits.view(batch_size, num_classes, -1).permute(1, 0, 2)

        if self.from_logits:
            prob = (logits / self.T).log_softmax(dim=0).exp()
        else:
            prob = logits

        if self.ignore_index is None and not_ignore is None:
            not_ignore = torch.ones_like(prob).to(torch.bool)
        elif self.ignore_index is not None and not_ignore is None and hard_label:
            not_ignore = (label != self.ignore_index).view(
                batch_size, -1).unsqueeze(0).expand(num_classes, batch_size, -1)
        else:
            not_ignore = not_ignore.view(
                batch_size, -1).unsqueeze(0).expand(num_classes, batch_size, -1)

        if hard_label:
            label = label.view(batch_size, -1)
            label = F.one_hot(torch.clamp(label, 0, num_classes - 1),
                              num_classes=num_classes).permute(2, 0, 1)
        else:
            label = label.view(batch_size, num_classes, -1).permute(1, 0, 2)

        if self.class_weight is not None and not torch.is_tensor(self.class_weight):
            self.class_weight = prob.new_tensor(self.class_weight)

        assert prob.shape == label.shape == not_ignore.shape
        assert classes in ["All", "Present", "Prob", "Label", "Both"] or all(
            (isinstance(c, int) and 0 <= c < num_classes) for c in classes)
        assert self.class_weight is None or self.class_weight.size(
            0) == num_classes

        if self.per_image:
            return self.forward_per_image(prob, label, not_ignore, classes)
        else:
            return self.forward_per_batch(prob, label, not_ignore, classes)


class LovaszLoss(_Loss):
    def __init__(
        self,
        mode: str,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        from_logits: bool = True,
    ):
        """Lovasz loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, y_pred, y_true):
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            loss = _F._lovasz_hinge(
                y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        elif self.mode == MULTICLASS_MODE:
            y_pred = y_pred.softmax(dim=1)
            loss = _F._lovasz_softmax(
                y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        else:
            raise ValueError("Wrong mode {}.".format(self.mode))
        return loss


class MCCLoss(_Loss):
    def __init__(self, eps: float = 1e-5):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary mode.

        Args:
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class

        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MCC loss

        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """

        bs = y_true.shape[0]

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss


class SoftBCEWithLogitsLoss(nn.Module):

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + \
                y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred,
            soft_targets,
            self.weight,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftCrossEntropyLoss(nn.Module):

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        ignore_index: Optional[int] = -100,
        dim: int = 1,
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return _F.label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )


class TverskyLoss(DiceLoss):
    """Tversky loss for image segmentation task.
    Where FP and FN is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes input is raw logits
        smooth:
        ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``

    Return:
        loss: torch.Tensor

    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
    ):

        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(mode, classes, log_loss, from_logits, smooth, ignore_index, eps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return _F.soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)


class MCCLoss(nn.Module):
    def __init__(
        self,
        mode: str,
        from_logits: bool = True,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            from_logits: If True, assumes input is raw logits
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}

        super(MCCLoss, self).__init__()
        self.mode = mode        
        self.from_logits = from_logits
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MCC loss

        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                # N,H*W -> N,H*W, C
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * \
                    mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss


class ComboLoss(_Loss):
    def __init__(self, alpha=0.5, dice_kwargs={}, mcc_kwargs={}):
        super(ComboLoss, self).__init__()

        self.alpha = alpha

        self.dice_loss = DiceLoss(**dice_kwargs)
        self.mcc_loss = MCCLoss(**mcc_kwargs)

    def forward(self, output, target):
        return self.alpha * self.mcc_loss(output, target) + (1-self.alpha)*self.dice_loss(output, target)