# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.losses import weighted_loss, smooth_l1_loss, SmoothL1Loss
from torchvision.ops import box_iou


@weighted_loss
def kl_smooth_l1_loss(
    pred: Tensor,
    target: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    loss = torch.where(loss < beta, 0.5 * loss.pow(2) / beta, loss - 0.5 * beta)
    loss = loss.mul(target[..., 2:].norm(dim=1, keepdim=True).neg().sigmoid().detach())
    return loss


@weighted_loss
def scale_loss(
    pred: Tensor, target: Tensor, *, loss: Tensor = 1, eps: float = 1e-4
) -> Tensor:
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    width_target = target[..., 2:] - target[..., :2]
    width_pred = pred[..., 2:] - pred[..., :2]
    scale = (width_pred - width_target).pow(2).sum(dim=-1).add(eps) / enclose_wh.pow(
        2
    ).sum(dim=-1).add(eps)
    return loss + scale


def wiou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""`Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
    Box Regression <https://arxiv.org/abs/2301.10051>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    gious = box_iou(pred, target).diag()

    pred_cnt = (pred[..., :2] + pred[..., 2:4]) / 2
    target_cnt = (target[..., :2] + target[..., 2:4]) / 2
    center_dist = torch.norm(pred_cnt - target_cnt, dim=-1).pow(2)
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    wh_box = torch.norm(enclose_wh, dim=-1).pow(2)
    ratio = torch.exp(center_dist.add(eps) / wh_box.detach().add(eps))

    if fp16:
        gious = gious.to(torch.float16)
        ratio = ratio.to(torch.float16)

    return (1 - gious) * ratio


@MODELS.register_module()
class WIoULoss(nn.Module):
    r"""`Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
    Box Regression <https://arxiv.org/abs/2301.10051>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        alpha: float = 1.9,
        delta: int = 3,
        momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        self.register_buffer("iou_mean", torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        **kwargs
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        return self.loss_weight * scale_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            loss=self.smoothed_loss(pred, target),
            **kwargs
        )

    def smoothed_loss(self, pred, target):
        L_iou = wiou_loss(pred, target)
        self.iou_mean.mul_(self.momentum).add_(
            (1 - self.momentum) * L_iou.detach().mean()
        )
        beta = L_iou.detach() / self.iou_mean
        divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
        return (beta / divisor).mul(L_iou)


@weighted_loss
def wise_l1_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""`Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
    Box Regression <https://arxiv.org/abs/2301.10051>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    pred_cnt = (pred[..., :2] + pred[..., 2:4]) / 2
    target_cnt = (target[..., :2] + target[..., 2:4]) / 2
    center_dist = torch.norm(pred_cnt - target_cnt, dim=-1).pow(2)
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, :2] + pred[:, 2:], target[:, :2] + target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    wh_box = torch.norm(enclose_wh, dim=-1).pow(2)
    ratio = torch.exp(center_dist.add(eps) / wh_box.detach().add(eps))

    if fp16:
        ratio = ratio.to(torch.float16)

    return ratio.mul(smooth_l1_loss(pred, target, beta=1.0))


@MODELS.register_module()
class KLSmoothL1Loss(nn.Module):
    """KL smooth L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(
        self, beta: float = 1.0, reduction: str = "mean", loss_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        **kwargs
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * kl_smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss_bbox


@MODELS.register_module()
class WiseL1Loss(nn.Module):
    r"""`Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
    Box Regression <https://arxiv.org/abs/2301.10051>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        alpha: float = 1.9,
        delta: int = 3,
        momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        self.register_buffer("iou_mean", torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        **kwargs
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        return self.loss_weight * self.smoothed_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )

    def smoothed_loss(self, pred, target, weight, reduction, avg_factor, **kwargs):
        L_iou = wise_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, **kwargs
        )
        self.iou_mean.mul_(self.momentum).add_(
            (1 - self.momentum) * L_iou.detach().mean()
        )
        beta = L_iou.detach() / self.iou_mean
        divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
        return (beta / divisor).mul(L_iou)


@MODELS.register_module()
class IoURegressionLoss(SmoothL1Loss):

    def forward(
        self,
        bbox_preds,
        bbox_targets,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        shape_ratio_preds = bbox_preds[..., 2] / bbox_preds[..., 3].add(1e-8)
        shape_ratio_targets = bbox_targets[..., 2] / bbox_targets[..., 3].add(1e-8)
        return (
            super()
            .forward(
                shape_ratio_preds,
                shape_ratio_targets,
                weight,
                avg_factor,
                reduction_override,
                **kwargs
            )
            .mul(0)
        )
