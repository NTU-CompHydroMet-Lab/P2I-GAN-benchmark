"""Advanced loss functions for precipitation GAN benchmarking."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

__all__ = [
    "ReconstructionLoss",
    "gan_loss",
    "transform",
    "weighted_l1_distance",
    "softmax_temperature",
    "kl_divergence",
    "compute_forward_difference",
    "spatial_pool",
    "temporal_difference_matching_loss",
    "shock_map",
    "shock_map_loss",
    "k1_loss",
    "AdversarialLoss",
    "WeightedL1Metric",
    "K1LossMetric",
    "ShockDifferenceMetric",
]


class ReconstructionLoss:
    """Weighted combination of hole and valid region reconstruction loss."""

    def __init__(self, hole_weight: float = 1.0, valid_weight: float = 1.0):
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        hole = F.l1_loss(prediction * mask, target * mask)
        valid = F.l1_loss(prediction * (1 - mask), target * (1 - mask))
        loss = self.hole_weight * hole + self.valid_weight * valid
        return loss, {"hole": float(hole.detach()), "valid": float(valid.detach())}


def transform(x: torch.Tensor) -> torch.Tensor:
    """Convert normalized rainfall back to mm scale (legacy helper)."""
    return 0.036 * torch.pow(10, (x * 255.0 / 3.0) * 0.0625)


def weighted_l1_distance(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Weighted L1 distance from NowcastNet paper."""
    a, b, c = 0.50, 5.14, 0.12
    x_max = 0.70

    x_max_tensor = torch.tensor(x_max, device=x_true.device, dtype=x_true.dtype)
    w_max = a * torch.exp(b * x_max_tensor) + c
    w = a * torch.exp(b * x_true) + c
    weight = torch.where(x_true > x_max_tensor, w_max, w)
    return torch.mean(weight * torch.abs(x_pred - x_true))


def softmax_temperature(tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature-scaled softmax over flattened spatial dims."""
    size = tensor.size()
    tensor = tensor.view(size[0], size[1], -1)
    softmax = F.softmax(tensor / temperature, dim=-1)
    return softmax.view(size)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence between temporal difference distributions."""
    p = p.view(p.size(0), p.size(1), -1)
    q = q.view(q.size(0), q.size(1), -1)
    return F.kl_div(p.log(), q, reduction="batchmean")


def compute_forward_difference(series: torch.Tensor) -> torch.Tensor:
    """Forward temporal difference."""
    return series[:, 1:] - series[:, :-1]


def spatial_pool(x: torch.Tensor) -> torch.Tensor:
    """Down-sample spatially to reduce pixel-level sensitivity."""
    pool = nn.MaxPool2d(kernel_size=5, stride=4, padding=2)
    return pool(x)


def temporal_difference_matching_loss(
    pred_diff: torch.Tensor,
    true_diff: torch.Tensor,
    true: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted matching of temporal gradients."""
    pred_diff_pool = spatial_pool(pred_diff)
    true_diff_pool = spatial_pool(true_diff)
    base = spatial_pool(true[:, :-1])
    weight = 1.0 / (1.0 + beta * base + eps)

    loss = (pred_diff_pool - true_diff_pool) ** 2
    weighted_loss = loss * weight
    return weighted_loss.mean()


def _kernels(dtype: torch.dtype, device: torch.device):
    kx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3) / 8
    ky = kx.transpose(-1, -2).contiguous()
    kl = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)
    return kx, ky, kl


def _conv_reflect(x4: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    x4 = F.pad(x4, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x4, kernel, padding=0)


def shock_map(x: torch.Tensor, beta: float = 30.0, eps: float = 1e-12) -> torch.Tensor:
    """Compute shock map for structural difference regularization."""
    batch, frames, height, width = x.shape
    x4 = x.reshape(batch * frames, 1, height, width)
    kx, ky, kl = _kernels(x4.dtype, x4.device)
    gx = _conv_reflect(x4, kx)
    gy = _conv_reflect(x4, ky)
    grad = torch.sqrt(gx * gx + gy * gy + eps)
    lap = _conv_reflect(x4, kl)
    return (torch.tanh(beta * lap) * grad).reshape(batch, frames, height, width)


def shock_map_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    beta: float = 30.0,
    tau: float = 25.0,
    border_ignore: int = 0,
    pool: int = 2,
    eps: float = 1e-12,
) -> torch.Tensor:
    del tau  # legacy arg retained for compatibility
    if pool > 1:
        pred = F.max_pool2d(pred, kernel_size=pool, stride=pool)
        true = F.max_pool2d(true, kernel_size=pool, stride=pool)

    shock_pred = shock_map(pred, beta, eps)
    shock_true = shock_map(true, beta, eps)
    diff = F.relu(torch.abs(shock_true) - torch.abs(shock_pred))

    if border_ignore > 0:
        m = border_ignore
        diff = diff[..., m:-m, m:-m]

    return diff


def k1_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    temp_alpha: float,
    k1_alpha: float,
) -> torch.Tensor:
    """Full K1 loss combining weighted pool + KL + shock map terms."""
    device = pred.device

    pool_loss = weighted_l1_distance(pred, true)
    reg_loss = torch.tensor(0.0, device=device)
    shock_loss = torch.tensor(0.0, device=device)

    if k1_alpha > 0:
        pred_diff = compute_forward_difference(pred)
        true_diff = compute_forward_difference(true)
        pred_prob = softmax_temperature(pred_diff, 0.1)
        true_prob = softmax_temperature(true_diff, 0.1)
        reg_loss = kl_divergence(pred_prob, true_prob)

    if temp_alpha == 0:
        shock_diff = shock_map_loss(pred, true, beta=0.02, border_ignore=2, pool=1)
        shock_loss = shock_diff.mean()

    return pool_loss + k1_alpha * reg_loss + temp_alpha * shock_loss


class AdversarialLoss(nn.Module):
    """Multi-mode adversarial loss."""

    def __init__(self, loss_type: str = "nsgan", target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.loss_type = loss_type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if loss_type == "nsgan":
            self.criterion = nn.BCELoss()
        elif loss_type == "lsgan":
            self.criterion = nn.MSELoss()
        elif loss_type == "hinge":
            self.criterion = nn.ReLU()
        else:
            raise ValueError(f"Unsupported GAN loss type: {loss_type}")

    def forward(self, outputs: torch.Tensor, is_real: bool, is_disc: bool | None = None) -> torch.Tensor:
        if self.loss_type == "hinge":
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            return (-outputs).mean()

        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        return self.criterion(outputs, labels)


_GAN_LOSS_CACHE: Dict[Tuple[str, float, float], AdversarialLoss] = {}


def gan_loss(
    logits: torch.Tensor,
    target_is_real: bool,
    *,
    loss_type: str = "nsgan",
    is_disc: bool = False,
    target_real_label: float = 1.0,
    target_fake_label: float = 0.0,
) -> torch.Tensor:
    """GAN loss helper mirroring :class:`AdversarialLoss`."""

    key = (loss_type, target_real_label, target_fake_label)
    loss_fn = _GAN_LOSS_CACHE.get(key)
    if loss_fn is None:
        loss_fn = AdversarialLoss(loss_type, target_real_label, target_fake_label)
        _GAN_LOSS_CACHE[key] = loss_fn

    if loss_fn.real_label.device != logits.device:
        loss_fn = loss_fn.to(logits.device)
        _GAN_LOSS_CACHE[key] = loss_fn  # cache moved module

    return loss_fn(logits, target_is_real, is_disc=is_disc)


class _BaseLossMetric(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _accumulate(self, loss: torch.Tensor) -> None:
        self.loss_sum += loss.detach()
        self.n_obs += torch.tensor(1.0, device=loss.device)

    def compute(self) -> torch.Tensor:
        return self.loss_sum / torch.clamp(self.n_obs, min=1.0)


class WeightedL1Metric(_BaseLossMetric):
    """TorchMetrics wrapper for weighted L1 distance."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        loss = weighted_l1_distance(preds, target)
        self._accumulate(loss)


class K1LossMetric(_BaseLossMetric):
    """TorchMetrics wrapper for k1 loss."""

    def __init__(self, temp_alpha: float = 1.0, k1_alpha: float = 0.0):
        super().__init__()
        self.temp_alpha = temp_alpha
        self.k1_alpha = k1_alpha

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        loss = k1_loss(preds, target, self.temp_alpha, self.k1_alpha)
        self._accumulate(loss)


class ShockDifferenceMetric(_BaseLossMetric):
    """Logs mean shock-map discrepancy."""

    def __init__(self, beta: float = 0.02, border_ignore: int = 2, pool: int = 1):
        super().__init__()
        self.beta = beta
        self.border_ignore = border_ignore
        self.pool = pool

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        diff = shock_map_loss(
            preds,
            target,
            beta=self.beta,
            border_ignore=self.border_ignore,
            pool=self.pool,
        )
        self._accumulate(diff.mean())
