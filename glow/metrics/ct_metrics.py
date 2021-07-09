from __future__ import annotations

from functools import partial
from typing import NamedTuple, Protocol

import torch

# ---------------------- catalyst's way to compute dice ----------------------


class _MetricCallable(Protocol):
    def __call__(self, tp: torch.Tensor, fp: torch.Tensor,
                 fn: torch.Tensor) -> torch.Tensor:
        ...


def _get_region_based_metrics(outputs: torch.Tensor,
                              targets: torch.Tensor,
                              metric_fn: _MetricCallable,
                              threshold: float | None = None,
                              micro: bool = False) -> torch.Tensor:
    if threshold is not None:
        outputs = (outputs > threshold).float()

    # ! Only one-hot or binary labels are accepted
    assert outputs.shape == targets.shape

    # Compute parts of 2x2 confusion matrix
    # Batch dimension is always ignored
    dims = [0, *range(2, outputs.ndim)]  # Do not average over class dimension

    tp = (outputs * targets).sum(dims)  # tp: [C]
    fp = (outputs * (1 - targets)).sum(dims)  # fp: [C]
    fn = (targets * (1 - outputs)).sum(dims)  # fn: [C]

    if micro:  # Score aggregated stats
        # TP becomes CM.trace, and FP = FN = CM.sum - CM.trace
        #  Dice -> CM.trace / CM.sum = acc
        #  IoU -> CM.trace / (2 * CM.sum - CM.trace) = acc / (2 - acc)
        return metric_fn(tp.sum(), fp.sum(), fn.sum())  # [1]

    return metric_fn(tp, fp, fn).mean()  # [1], average over class scores


def _dice(tp: torch.Tensor,
          fp: torch.Tensor,
          fn: torch.Tensor,
          eps: float = 1e-7) -> torch.Tensor:
    union = tp + fp + fn
    return (2 * tp + eps * (union == 0).float()) / (2 * tp + fp + fn + eps)


def _iou(tp: torch.Tensor,
         fp: torch.Tensor,
         fn: torch.Tensor,
         eps: float = 1e-7) -> torch.Tensor:
    union = tp + fp + fn
    return (tp + eps * (union == 0).float()) / (union + eps)


def dice(outputs: torch.Tensor,
         targets: torch.Tensor,
         threshold: float = None,
         micro: bool = False,
         eps: float = 1e-7) -> torch.Tensor:
    return _get_region_based_metrics(
        outputs,
        targets,
        metric_fn=partial(_dice, eps=eps),
        threshold=threshold,
        micro=micro)


# --------------------------- My way. Rewrite this ---------------------------


def confusion_matrix(output: torch.Tensor,
                     target: torch.Tensor,
                     ignore_index: int | None = -1,
                     grad: bool = False,
                     samplewise: bool = False) -> torch.Tensor:
    b, c = output.shape[:2]
    output = output.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
    target = target.view(b, -1)  # [B, N]

    if ignore_index is not None:
        target = target.clone()
        target[target == ignore_index] = c  # Make ignored index last

    if grad:
        if samplewise:  # Compute metric for each sample in batch
            cm = output.new_zeros((b, c + 1, c))
            target = target[..., None]  # [B, N, 1]
        else:  # Flatten batch over first dim
            cm = output.new_zeros((c + 1, c))
            target = target.view(-1, 1)  # [B x N, 1]
            output = output.reshape(-1, c)  # [B x N, C]

        cm.scatter_add_(-2, target.expand_as(output), output.softmax(-1))
    else:
        ones = target.new_ones([1] * target.ndim).expand_as(target)

        batch = torch.arange(b, dtype=torch.long, device=target.device)
        batch = batch[:, None]  # [B, 1]
        if samplewise:
            batch = batch.expand_as(target)  # [B, N]

        cm = target.new_zeros((b, c + 1, c))
        cm.index_put_([batch, target, output.argmax(-1)],
                      ones,
                      accumulate=True)
        cm = (cm if samplewise else cm.sum(0)).double()

    return cm[..., :c, :]  # Strip ignored index


def cm_to_binary(cmat: torch.Tensor) -> torch.Tensor:
    cmat = torch.stack([cmat[..., :, 0], cmat[..., :, 1:].sum(-1)], dim=-1)
    return torch.stack([cmat[..., 0, :], cmat[..., 1:, :].sum(-2)], dim=-2)


def dice_from_cm(cm: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(cm.dtype).eps

    inter = cm.diagonal(dim1=-2, dim2=-1)
    mass = cm.sum(-2) + cm.sum(-1)
    return 2 * ((inter + eps) / (mass + eps)).mean()


class Dice(NamedTuple):
    """Computes Dice score.

    If grad is set, computes score with gradient support.
    If binary is set, computes score for 0 vs not-0 case.
    If samplewise is set, computes scores for each sample in batch.
    """
    grad: bool = False
    samplewise: bool = False
    binary: bool = False
    ignore_index: int | None = None

    def __call__(self, output: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        cm = confusion_matrix(output, target, self.ignore_index, self.grad,
                              self.samplewise)
        if self.binary:
            cm = cm_to_binary(cm)
        return dice_from_cm(cm)


class ConfusionMatrix(NamedTuple):
    ignore_index: int | None = None

    def __call__(self, output):
        pred, true, *_ = output
        classes = pred.shape[1]

        pred = pred.argmax(1).view(-1)
        true = true.view(-1)
        if self.ignore_index is not None:
            pred = pred[true != self.ignore_index]
            true = true[true != self.ignore_index]

        cm = torch.zeros(classes, classes, dtype=torch.long)
        cm = cm.index_put_((true, pred), torch.tensor(1), accumulate=True)

        cm = cm.double()
        return cm / cm.sum()
