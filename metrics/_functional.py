import torch
import warnings
from typing import Optional, List, Tuple, Union


__all__ = [
    "_get_stats_multiclass",
    "_get_stats_multilabel",
    "_compute_metric",
    "_fbeta_score",
    "_iou_score",
    "_accuracy",
    "_sensitivity",
    "_specificity",
    "_mcc",
    "_balanced_accuracy",
    "_positive_predictive_value",
    "_negative_predictive_value",
    "_false_negative_rate",
    "_false_positive_rate",
    "_false_discovery_rate",
    "_false_omission_rate",
    "_positive_likelihood_ratio",
    "_negative_likelihood_ratio",
]


@torch.no_grad()
def _get_stats_multiclass(
    output: torch.LongTensor,
    target: torch.LongTensor,
    num_classes: int,
    ignore_index: Optional[int],
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:

    batch_size, *dims = output.shape
    num_elements = torch.prod(torch.tensor(dims)).long()

    if ignore_index is not None:
        ignore = target == ignore_index
        output = torch.where(ignore, -1, output)
        target = torch.where(ignore, -1, target)
        ignore_per_sample = ignore.view(batch_size, -1).sum(1)

    tp_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fp_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fn_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    tn_count = torch.zeros(batch_size, num_classes, dtype=torch.long)

    for i in range(batch_size):
        target_i = target[i]
        output_i = output[i]
        mask = output_i == target_i
        matched = torch.where(mask, target_i, -1)
        tp = torch.histc(matched.float(), bins=num_classes,
                         min=0, max=num_classes - 1)
        fp = torch.histc(output_i.float(), bins=num_classes,
                         min=0, max=num_classes - 1) - tp
        fn = torch.histc(target_i.float(), bins=num_classes,
                         min=0, max=num_classes - 1) - tp
        tn = num_elements - tp - fp - fn
        if ignore_index is not None:
            tn = tn - ignore_per_sample[i]
        tp_count[i] = tp.long()
        fp_count[i] = fp.long()
        fn_count[i] = fn.long()
        tn_count[i] = tn.long()

    return tp_count, fp_count, fn_count, tn_count


@torch.no_grad()
def _get_stats_multilabel(
    output: torch.LongTensor,
    target: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:

    batch_size, num_classes, *dims = target.shape
    output = output.view(batch_size, num_classes, -1)
    target = target.view(batch_size, num_classes, -1)

    tp = (output * target).sum(2)
    fp = output.sum(2) - tp
    fn = target.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

    return tp, fp, fn, tn


###################################################################################################
# Metrics computation
###################################################################################################


def _handle_zero_division(x, zero_division):
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x


def _compute_metric(
    metric_fn,
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    class_idv: Optional[bool] = False,
    zero_division="warn",
    **metric_kwargs,
) -> float:

    if class_weights is None and reduction is not None and "weighted" in reduction:
        raise ValueError(
            f"Class weights should be provided for `{reduction}` reduction")

    class_weights = class_weights if class_weights is not None else 1.0
    class_weights = torch.tensor(class_weights).to(tp.device)
    class_weights = class_weights / class_weights.sum()

    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)

    elif reduction == "macro":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        if class_idv == False:
            score = (score * class_weights)
        else:
            score = score * class_weights

    elif reduction == "weighted":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        if class_idv == False:
            score = (score * class_weights).sum()
        else:
            score = score * class_weights

    elif reduction == "micro-imagewise":
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        if class_idv == False:
            score = score.mean()
        else:
            score = score

    elif reduction == "macro-imagewise" or reduction == "weighted-imagewise":
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        if class_idv == False:
            score = (score.mean(0) * class_weights).mean()
        else:
            score = (score.mean(0) * class_weights)

    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, weighted, micro-imagewise,"
            + "macro-imagesize, weighted-imagewise, none, None]"
        )

    return score


# Logic for metric computation, all metrics are with the same interface

def _fbeta_score(tp, fp, fn, tn, beta=1):
    beta_tp = (1 + beta**2) * tp
    beta_fn = (beta**2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score


def _iou_score(tp, fp, fn, tn):
    return tp / (tp + fp + fn)


def _accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def _sensitivity(tp, fp, fn, tn):
    return tp / (tp + fn)


def _specificity(tp, fp, fn, tn):
    return tn / (tn + fp)


def _mcc(tp, fp, fn, tn):
    num = tp * tn - fp * fn
    den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den


def _balanced_accuracy(tp, fp, fn, tn):
    return (_sensitivity(tp, fp, fn, tn) + _specificity(tp, fp, fn, tn)) / 2


def _positive_predictive_value(tp, fp, fn, tn):
    return tp / (tp + fp)


def _negative_predictive_value(tp, fp, fn, tn):
    return tn / (tn + fn)


def _false_negative_rate(tp, fp, fn, tn):
    return fn / (fn + tp)


def _false_positive_rate(tp, fp, fn, tn):
    return fp / (fp + tn)


def _false_discovery_rate(tp, fp, fn, tn):
    return 1 - _positive_predictive_value(tp, fp, fn, tn)


def _false_omission_rate(tp, fp, fn, tn):
    return 1 - _negative_predictive_value(tp, fp, fn, tn)


def _positive_likelihood_ratio(tp, fp, fn, tn):
    return _sensitivity(tp, fp, fn, tn) / _false_positive_rate(tp, fp, fn, tn)


def _negative_likelihood_ratio(tp, fp, fn, tn):
    return _false_negative_rate(tp, fp, fn, tn) / _specificity(tp, fp, fn, tn)


_doc = """
    Args:
        tp (torch.LongTensor): tensor of shape (N, C), true positive cases
        fp (torch.LongTensor): tensor of shape (N, C), false positive cases
        fn (torch.LongTensor): tensor of shape (N, C), false negative cases
        tn (torch.LongTensor): tensor of shape (N, C), true negative cases
        reduction (Optional[str]): Define how to aggregate metric between classes and images:
            - 'micro'
                Sum true positive, false positive, false negative and true negative pixels over
                all images and all classes and then compute score.
            - 'macro'
                Sum true positive, false positive, false negative and true negative pixels over
                all images for each label, then compute score for each label separately and average labels scores.
                This does not take label imbalance into account.
            - 'weighted'
                Sum true positive, false positive, false negative and true negative pixels over
                all images for each label, then compute score for each label separately and average
                weighted labels scores.
            - 'micro-imagewise'
                Sum true positive, false positive, false negative and true negative pixels for **each image**,
                then compute score for **each image** and average scores over dataset. All images contribute equally
                to final score, however takes into accout class imbalance for each image.
            - 'macro-imagewise'
                Compute score for each image and for each class on that image separately, then compute average score
                on each image over labels and average image scores over dataset. Does not take into account label
                imbalance on each image.
            - 'weighted-imagewise'
                Compute score for each image and for each class on that image separately, then compute weighted average
                score on each image over labels and average image scores over dataset.
            - 'none' or ``None``
                Same as ``'macro-imagewise'``, but without any reduction.
            For ``'binary'`` case ``'micro' = 'macro' = 'weighted'`` and
            ``'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'``.
            Prefixes ``'micro'``, ``'macro'`` and ``'weighted'`` define how the scores for classes will be aggregated,
            while postfix ``'imagewise'`` defines how scores between the images will be aggregated.
        class_weights (Optional[List[float]]): list of class weights for metric
            aggregation, in case of `weighted*` reduction is chosen. Defaults to None.
        zero_division (Union[str, float]): Sets the value to return when there is a zero division,
            i.e. when all predictions and labels are negative. If set to “warn”, this acts as 0,
            but warnings are also raised. Defaults to 1.
    
    Returns:
        torch.Tensor: if ``'reduction'`` is not ``None`` or ``'none'`` returns scalar metric,
            else returns tensor of shape (N, C)
    
    References:
        https://en.wikipedia.org/wiki/Confusion_matrix
"""
