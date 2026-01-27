from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .io import align_length, crop_center, ensure_thw, select_by_mask


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def pss(pred: np.ndarray,
        gt: np.ndarray,
        bins: int = 50,
        min_value: float = 0.5,
        value_range: Optional[Tuple[float, float]] = None) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.size == 0 or gt.size == 0:
        return float("nan")

    if value_range is None:
        both = np.concatenate([pred.ravel(), gt.ravel()])
        both = both[np.isfinite(both)]
        if min_value is not None:
            both = both[both > min_value]
        if both.size == 0:
            return float("nan")
        vmin = float(both.min())
        vmax = float(both.max())
        if vmin == vmax:
            vmax = vmin + 1e-6
        value_range = (vmin, vmax)

    t = pred.shape[0]
    scores = []
    for i in range(t):
        p = pred[i].ravel()
        g = gt[i].ravel()
        p = p[np.isfinite(p)]
        g = g[np.isfinite(g)]
        if min_value is not None:
            p = p[p > min_value]
            g = g[g > min_value]
        if p.size == 0 or g.size == 0:
            continue
        p_hist, _ = np.histogram(p, bins=bins, range=value_range)
        g_hist, _ = np.histogram(g, bins=bins, range=value_range)
        p_freq = p_hist / (p_hist.sum() + 1e-12)
        g_freq = g_hist / (g_hist.sum() + 1e-12)
        scores.append(float(np.minimum(p_freq, g_freq).sum()))

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def ssim2d(a: np.ndarray, b: np.ndarray, c1: float = 0.01**2, c2: float = 0.03**2) -> float:
    mu_a = a.mean()
    mu_b = b.mean()
    sig_a = ((a - mu_a) ** 2).mean()
    sig_b = ((b - mu_b) ** 2).mean()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return float(num / (den + 1e-10))


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float()


def _ensure_btchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:  # [T,H,W]
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.ndim == 4:  # [B,T,H,W]
        x = x.unsqueeze(2)
    return x


def _pool8(x: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = x.shape
    y = F.avg_pool2d(x.reshape(b * t, c, h, w), kernel_size=8, stride=8)
    return y.view(b, t, c, y.shape[-2], y.shape[-1])


def _ssim2d_torch(a: torch.Tensor, b: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    mu_a = a.mean()
    mu_b = b.mean()
    sig_a = ((a - mu_a) ** 2).mean()
    sig_b = ((b - mu_b) ** 2).mean()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return num / (den + 1e-10)


def _tssim_series(x: torch.Tensor, lag: int) -> torch.Tensor:
    vals = []
    for t in range(lag, x.shape[1]):
        lst = [_ssim2d_torch(x[b, t, 0], x[b, t - lag, 0]) for b in range(x.shape[0])]
        vals.append(torch.stack(lst))
    return torch.stack(vals, dim=0).T


def ssim_spatial(pred: np.ndarray, gt: np.ndarray, use_pool8: bool = True) -> float:
    pred = _ensure_btchw(_to_tensor(pred))
    gt = _ensure_btchw(_to_tensor(gt))
    if use_pool8:
        pred = _pool8(pred)
        gt = _pool8(gt)
    bsz, t = pred.shape[0], pred.shape[1]
    vals = []
    for b in range(bsz):
        for ti in range(t):
            vals.append(_ssim2d_torch(pred[b, ti, 0], gt[b, ti, 0]))
    return float(torch.stack(vals).mean().item())


def delta_tssim(pred: np.ndarray, gt: np.ndarray, lag: int = 1, use_pool8: bool = True) -> float:
    pred = _ensure_btchw(_to_tensor(pred))
    gt = _ensure_btchw(_to_tensor(gt))
    if pred.shape[1] <= lag:
        return float("nan")
    if use_pool8:
        pred = _pool8(pred)
        gt = _pool8(gt)
    t_pred = _tssim_series(pred, lag)
    t_gt = _tssim_series(gt, lag)
    return float((t_pred - t_gt).mean().item())


def nse(pred: np.ndarray, gt: np.ndarray) -> float:
    num = np.sum((pred - gt) ** 2)
    den = np.sum((gt - np.mean(gt)) ** 2)
    return float(1.0 - num / (den + 1e-10))


def transform_mmhr(arr: np.ndarray, divide_by_3: bool = True) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.maximum(arr, 0.001)
    if divide_by_3:
        arr = arr / 3.0
    # Prevent overflow: 10**x overflows beyond ~1e308 for float64.
    exp = np.clip(arr * 0.0625, a_min=None, a_max=38.0)
    out = 10 ** exp * 0.036
    return np.clip(out, 0.0, 200.0)


def categorical_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float) -> Dict[str, float]:
    pred_bin = pred >= threshold
    gt_bin = gt >= threshold
    hits = np.logical_and(pred_bin, gt_bin).sum()
    misses = np.logical_and(~pred_bin, gt_bin).sum()
    false_alarms = np.logical_and(pred_bin, ~gt_bin).sum()
    correct_negatives = np.logical_and(~pred_bin, ~gt_bin).sum()
    pod = hits / (hits + misses + 1e-10)
    far = false_alarms / (hits + false_alarms + 1e-10)
    csi = hits / (hits + misses + false_alarms + 1e-10)
    n_total = hits + misses + false_alarms + correct_negatives
    if n_total > 0:
        hss = 2 * (hits * correct_negatives - misses * false_alarms) / (
            (misses**2 + false_alarms**2 + 2 * hits * correct_negatives +
             (misses + false_alarms) * (hits + correct_negatives) + 1e-10)
        )
    else:
        hss = float("nan")
    return {"POD": float(pod), "FAR": float(far), "CSI": float(csi), "HSS": float(hss)}


def _apply_mask_mode(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> Dict[str, np.ndarray]:
    if mode == "radar":
        pred_sel = select_by_mask(pred, mask, invert=True)
        gt_sel = select_by_mask(gt, mask, invert=True)
    elif mode == "gauge":
        pred_sel = select_by_mask(pred, mask, invert=False)
        gt_sel = select_by_mask(gt, mask, invert=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return {"pred": pred_sel, "gt": gt_sel}


def run_exp1(preds: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
             truth: Union[np.ndarray, Dict[str, np.ndarray]],
             mask: np.ndarray,
             mode: str,
             crop_size: int,
             thresholds: Tuple[float, ...] = (0.5, 2.0, 4.0, 8.0),
             use_pool8: bool = True,
             divide_by_3: bool = True) -> Dict[str, Dict[str, float]]:
    if isinstance(truth, dict):
        event_keys = list(truth.keys())
        if not event_keys:
            return {}

        truth_list = [truth[k] for k in event_keys if truth.get(k) is not None]
        truth_arr = np.concatenate(truth_list, axis=0) if truth_list else np.empty((0,))
        preds_concat: Dict[str, np.ndarray] = {}
        for name, pred_src in preds.items():
            if not isinstance(pred_src, dict):
                continue
            pred_list = [pred_src[k] for k in event_keys if pred_src.get(k) is not None]
            pred_arr = np.concatenate(pred_list, axis=0) if pred_list else np.empty((0,))
            preds_concat[name] = pred_arr

        preds = preds_concat
        truth = truth_arr

    results: Dict[str, Dict[str, float]] = {}
    truth = transform_mmhr(truth, divide_by_3=divide_by_3)
    truth = crop_center(truth, crop_size)

    for name, pred in preds.items():
        pred = transform_mmhr(pred, divide_by_3=divide_by_3)
        pred, truth_aligned = align_length(pred, truth)
        pred = crop_center(pred, crop_size)
        masked = _apply_mask_mode(pred, truth_aligned, mask, mode)
        pred_sel = masked["pred"]
        gt_sel = masked["gt"]

        results[name] = {
            "MAE": mae(pred_sel, gt_sel),
            "RMSE": rmse(pred_sel, gt_sel),
            "PSS": pss(pred_sel, gt_sel),
            "SSIM": ssim_spatial(pred, truth_aligned, use_pool8=use_pool8),
            "DTSSIM_L1": delta_tssim(pred, truth_aligned, lag=1, use_pool8=use_pool8),
            "DTSSIM_L2": delta_tssim(pred, truth_aligned, lag=2, use_pool8=use_pool8),
            "NSE": nse(pred_sel, gt_sel),
        }
        for thr in thresholds:
            key = f"CAT_{thr:g}"
            results[name][key] = categorical_metrics(pred_sel, gt_sel, thr)

    return results
