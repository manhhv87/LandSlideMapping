import torch
import numpy as np
from typing import Optional, List, Tuple, Union

import metrics._functional as _F
import warnings


class SegMetric(object):
    def __init__(self):
        self.tp_lst = []
        self.fp_lst = []
        self.fn_lst = []
        self.tn_lst = []

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def add_batch(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
        mode: str,
        ignore_index: Optional[int] = None,
        threshold: Optional[Union[float, List[float]]] = None,
        num_classes: Optional[int] = None
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Compute true positive, false positive, false negative, true negative 'pixels'
        for each image and each class.
        Args:
            output (Union[torch.LongTensor, torch.FloatTensor]): Model output with following
                shapes and types depending on the specified ``mode``:
                'binary'
                    shape (N, 1, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``
                'multilabel'
                    shape (N, C, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``
                'multiclass'
                    shape (N, ...) and ``torch.LongTensor``
            target (torch.LongTensor): Targets with following shapes depending on the specified ``mode``:
                'binary'
                    shape (N, 1, ...)
                'multilabel'
                    shape (N, C, ...)
                'multiclass'
                    shape (N, ...)
            mode (str): One of ``'binary'`` | ``'multilabel'`` | ``'multiclass'``
            ignore_index (Optional[int]): Label to ignore on for metric computation.
                **Not** supproted for ``'binary'`` and ``'multilabel'`` modes.  Defaults to None.
            threshold (Optional[float, List[float]]): Binarization threshold for
                ``output`` in case of ``'binary'`` or ``'multilabel'`` modes. Defaults to None.
            num_classes (Optional[int]): Number of classes, necessary attribute
                only for ``'multiclass'`` mode. Class values should be in range 0..(num_classes - 1).
                If ``ignore_index`` is specified it should be outside the classes range, e.g. ``-1`` or
                ``255``.
        Raises:
            ValueError: in case of misconfiguration.
        Returns:
            Tuple[torch.LongTensor]: true_positive, false_positive, false_negative,
                true_negative tensors (N, C) shape each.
        """

        if torch.is_floating_point(target):
            raise ValueError(
                f"Target should be one of the integer types, got {target.dtype}.")

        if torch.is_floating_point(output) and threshold is None:
            raise ValueError(
                f"Output should be one of the integer types if ``threshold`` is not None, got {output.dtype}."
            )

        if torch.is_floating_point(output) and mode == "multiclass":
            raise ValueError(
                f"For ``multiclass`` mode ``output`` should be one of the integer types, got {output.dtype}.")

        if mode not in {"binary", "multiclass", "multilabel"}:
            raise ValueError(
                f"``mode`` should be in ['binary', 'multiclass', 'multilabel'], got mode={mode}.")

        if mode == "multiclass" and threshold is not None:
            raise ValueError(
                "``threshold`` parameter does not supported for this 'multiclass' mode")

        if output.shape != target.shape:
            raise ValueError(
                "Dimensions should match, but ``output`` shape is not equal to ``target`` "
                + f"shape, {output.shape} != {target.shape}"
            )

        if mode != "multiclass" and ignore_index is not None:
            raise ValueError(
                f"``ignore_index`` parameter is not supproted for '{mode}' mode")

        if mode == "multiclass" and num_classes is None:
            raise ValueError(
                "``num_classes`` attribute should be not ``None`` for 'multiclass' mode.")

        if ignore_index is not None and 0 <= ignore_index <= num_classes - 1:
            raise ValueError(
                f"``ignore_index`` should be outside the class values range, but got class values in range "
                f"0..{num_classes - 1} and ``ignore_index={ignore_index}``. Hint: if you have ``ignore_index = 0``"
                f"consirder subtracting ``1`` from your target and model output to make ``ignore_index = -1``"
                f"and relevant class values started from ``0``."
            )

        if mode == "multiclass":
            tp_count, fp_count, fn_count, tn_count = _F._get_stats_multiclass(
                output, target, num_classes, ignore_index)

            self.tp_lst.append(tp_count)
            self.fp_lst.append(fp_count)
            self.fn_lst.append(fn_count)
            self.tn_lst.append(tn_count)

        else:
            if threshold is not None:
                output = torch.where(output >= threshold, 1, 0)
                target = torch.where(target >= threshold, 1, 0)

            tp_count, fp_count, fn_count, tn_count = _F._get_stats_multilabel(
                output, target)

            self.tp_lst.append(tp_count)
            self.fp_lst.append(fp_count)
            self.fn_lst.append(fn_count)
            self.tn_lst.append(tn_count)

        self.tp = torch.vstack(tuple(self.tp_lst))
        self.fp = torch.vstack(tuple(self.fp_lst))
        self.fn = torch.vstack(tuple(self.fn_lst))
        self.tn = torch.vstack(tuple(self.tn_lst))

        return self.tp, self.fp, self.fn, self.tn

    def fbeta_score(
        self,
        beta: float = 1.0,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """F beta score"""
        return _F._compute_metric(
            _F._fbeta_score,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            beta=beta,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def f1_score(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """F1 score"""
        return _F._compute_metric(
            _F._fbeta_score,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            beta=1.0,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def iou_score(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """IoU score or Jaccard index"""  # noqa
        return _F._compute_metric(
            _F._iou_score,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def accuracy(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Accuracy"""
        return _F._compute_metric(
            _F._accuracy,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def sensitivity(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Sensitivity, recall, hit rate, or true positive rate (TPR)"""
        return _F._compute_metric(
            _F._sensitivity,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def specificity(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Specificity, selectivity or true negative rate (TNR)"""
        return _F._compute_metric(
            _F._specificity,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def mcc(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Matthews correlation coefficient (MCC)"""
        return _F._compute_metric(
            _F._mcc,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def balanced_accuracy(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Balanced accuracy"""
        return _F._compute_metric(
            _F._balanced_accuracy,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def positive_predictive_value(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Precision or positive predictive value (PPV)"""
        return _F._compute_metric(
            _F._positive_predictive_value,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def negative_predictive_value(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Negative predictive value (NPV)"""
        return _F._compute_metric(
            _F._negative_predictive_value,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def false_negative_rate(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Miss rate or false negative rate (FNR)"""
        return _F._compute_metric(
            _F._false_negative_rate,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def false_positive_rate(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Fall-out or false positive rate (FPR)"""
        return _F._compute_metric(
            _F._false_positive_rate,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def false_discovery_rate(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """False discovery rate (FDR)"""  # noqa
        return _F._compute_metric(
            _F._false_discovery_rate,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def false_omission_rate(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """False omission rate (FOR)"""  # noqa
        return _F._compute_metric(
            _F._false_omission_rate,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def positive_likelihood_ratio(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Positive likelihood ratio (LR+)"""
        return _F._compute_metric(
            _F._positive_likelihood_ratio,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def negative_likelihood_ratio(
        self,
        reduction: Optional[str] = None,
        class_weights: Optional[List[float]] = None,
        class_idv: Optional[bool] = False,
        zero_division: Union[str, float] = 1.0,
    ) -> torch.Tensor:
        """Negative likelihood ratio (LR-)"""
        return _F._compute_metric(
            _F._negative_likelihood_ratio,
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            reduction=reduction,
            class_weights=class_weights,
            class_idv=class_idv,
            zero_division=zero_division,
        )

    def reset(self):
        self.tp_lst = []
        self.fp_lst = []
        self.fn_lst = []
        self.tn_lst = []

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
