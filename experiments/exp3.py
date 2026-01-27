from __future__ import annotations

import os
from typing import Dict, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

from .exp1 import transform_mmhr
from .io import align_length, crop_center, ensure_dir, select_by_mask


def nse(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    m = np.isfinite(pred) & np.isfinite(gt)
    if not np.any(m):
        return float("nan")
    pred = pred[m]
    gt = gt[m]
    num = np.sum((pred - gt) ** 2)
    den = np.sum((gt - np.mean(gt)) ** 2)
    return float(1.0 - num / (den + 1e-10))


def _select_values(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "radar":
        pred_sel = select_by_mask(pred, mask, invert=True)
        gt_sel = select_by_mask(gt, mask, invert=True)
    elif mode == "gauge":
        pred_sel = select_by_mask(pred, mask, invert=False)
        gt_sel = select_by_mask(gt, mask, invert=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return pred_sel.ravel(), gt_sel.ravel()


def _nse_per_frame(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    t = min(pred.shape[0], gt.shape[0])
    out = np.full(t, np.nan, dtype=np.float64)
    for i in range(t):
        pred_sel, gt_sel = _select_values(pred[i:i + 1], gt[i:i + 1], mask, mode)
        out[i] = nse(pred_sel, gt_sel)
    return out


SCATTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd', '#8c564b']


def _residual_multi_row(pred_list, true, labels,
                        colors=SCATTER_COLORS,
                        figsize=(18, 3), dpi=600,
                        lim_x=(0, 32), lim_y=(-24, 8),
                        xticks=(0, 4, 8, 12, 16, 20, 24, 28, 32),
                        yticks=(-24, -20, -16, -12, -8, -4, 0, 4, 8),
                        max_points=2000, alpha=0.25, s=6,
                        min_value: float = 0.1,
                        seed=42, save_path: str | None = None) -> None:
    rng = np.random.default_rng(seed)
    n = len(pred_list)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        axes = [axes]

    tf = np.asarray(true, np.float64).ravel()
    mask_t = np.isfinite(tf)
    tf = tf[mask_t]
    if tf.size == 0:
        plt.close(fig)
        return

    for i, (pred, label) in enumerate(zip(pred_list, labels)):
        ax = axes[i]
        pf = np.asarray(pred, np.float64).ravel()
        pf = pf[mask_t[:pf.size]] if pf.size <= mask_t.size else pf
        m = np.isfinite(pf) & np.isfinite(tf[:pf.size])
        x = tf[:pf.size][m]
        y = pf[m] - tf[:pf.size][m]
        keep = x >= min_value
        x = x[keep]
        y = y[keep]
        if x.size > max_points:
            idx = rng.choice(x.size, size=max_points, replace=False)
            x, y = x[idx], y[idx]

        ax.scatter(x, y, s=s, alpha=alpha, color=colors[i],
                   edgecolors='none', rasterized=False, zorder=1)
        if x.size >= 2:
            slope, intercept, r, _, _ = stats.linregress(x, y)
            x_line = np.linspace(lim_x[0], lim_x[1], 200)
            ax.plot(x_line, np.zeros_like(x_line), color='gray', ls=':', lw=1.0)
            ax.plot(x_line, intercept + slope * x_line, 'k--', lw=1.0)
            ax.text(0.04, 0.82, f"R²={r**2:.3f}\nslope={slope:.3f}",
                    transform=ax.transAxes, fontsize=11)

        ax.set_title(label, fontsize=13, fontweight='bold', pad=4)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', labelsize=10, width=1.0, length=3)
        ax.grid(False)
        ax.axhline(0, color='black', lw=1.0, ls='--')

        if i == 0:
            ax.set_ylabel("Residual (Pred - Obs, mm/h)", fontsize=12)
        ax.set_xlabel("Obs (mm/h)", fontsize=12)

    plt.tight_layout(pad=1.0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def _scatter_multi_row(pred_list, true, labels,
                       colors=SCATTER_COLORS,
                       figsize=(18, 3), dpi=600,
                       lim=(0, 32),
                       xticks=(0, 4, 8, 12, 16, 20, 24, 28, 32),
                       yticks=(0, 4, 8, 12, 16, 20, 24, 28, 32),
                       max_points=2000,
                       alpha=0.25, s=6,
                       min_value: float = 0.1,
                       seed=42, save_path: str | None = None) -> None:
    rng = np.random.default_rng(seed)
    n = len(pred_list)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        axes = [axes]

    x1 = np.linspace(lim[0], lim[1], 200)
    tf = np.asarray(true, np.float64).ravel()
    mask_t = np.isfinite(tf)
    tf = tf[mask_t]
    if tf.size == 0:
        plt.close(fig)
        return

    for i, (pred, label) in enumerate(zip(pred_list, labels)):
        ax = axes[i]
        pf = np.asarray(pred, np.float64).ravel()
        pf = pf[mask_t[:pf.size]] if pf.size <= mask_t.size else pf
        m = np.isfinite(pf) & np.isfinite(tf[:pf.size])
        x = tf[:pf.size][m]
        y = pf[m]
        keep = x >= min_value
        x = x[keep]
        y = y[keep]
        if x.size > max_points:
            idx = rng.choice(x.size, size=max_points, replace=False)
            x, y = x[idx], y[idx]

        ax.scatter(x, y, s=s, alpha=alpha, color=colors[i],
                   edgecolors='none', rasterized=False, zorder=1)
        if x.size >= 2:
            slope, intercept, r, _, _ = stats.linregress(x, y)
            ax.plot(x1, x1, color='gray', ls=':', lw=1.0)
            ax.plot(x1, intercept + slope * x1, 'k--', lw=1.0)
            ax.text(0.04, 0.82, f"R²={r**2:.3f}\nslope={slope:.3f}",
                    transform=ax.transAxes, fontsize=11)

        ax.set_title(label, fontsize=13, fontweight='bold', pad=4)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal', 'box')
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', labelsize=10, width=1.0, length=3)
        ax.grid(False)

        if i == 0:
            ax.set_ylabel("Pred (mm/h)", fontsize=12)
        ax.set_xlabel("Obs (mm/h)", fontsize=12)

    plt.tight_layout(pad=1.0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def _logfreq_plot(pred_list, true, labels,
                  colors=SCATTER_COLORS,
                  figsize=(6, 4), dpi=300,
                  lim=(0, 32), bins=64,
                  save_path: str | None = None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    bin_edges = np.linspace(lim[0], lim[1], bins + 1)

    tf = np.asarray(true, np.float64).ravel()
    tf = tf[np.isfinite(tf)]
    if tf.size == 0:
        plt.close(fig)
        return
    hist_t, _ = np.histogram(tf, bins=bin_edges)
    freq_t = hist_t / max(hist_t.sum(), 1)
    ax.semilogy(bin_edges[:-1], freq_t, color='black', lw=1.6, label='Obs')

    for pred, label, color in zip(pred_list, labels, colors):
        pf = np.asarray(pred, np.float64).ravel()
        pf = pf[np.isfinite(pf)]
        if pf.size == 0:
            continue
        hist_p, _ = np.histogram(pf, bins=bin_edges)
        freq_p = hist_p / max(hist_p.sum(), 1)
        ax.semilogy(bin_edges[:-1], freq_p, color=color, lw=1.6, label=label)

    ax.set_xlim(*lim)
    ax.set_xlabel("Rainfall (mm/h)")
    ax.set_ylabel("Relative Frequency (log scale)")
    ax.grid(False)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _nse_boxplot(nse_by_method: Dict[str, List[float]],
                 out_path: str) -> None:
    methods = list(nse_by_method.keys())
    data = [nse_by_method[m] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    box = ax.boxplot(
        data,
        labels=methods,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        boxprops=dict(linewidth=1.2, color='black'),
        medianprops=dict(linewidth=2.0, color='black'),
        whiskerprops=dict(linewidth=1.2, color='black'),
        capprops=dict(linewidth=1.2, color='black'),
        flierprops=dict(marker='o', markersize=3, alpha=0.5, color='gray'),
        meanprops=dict(marker='D', markerfacecolor='white',
                       markeredgecolor='black', markersize=5),
    )
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('NSE', fontsize=14)
    ax.set_xlabel('Methods', fontsize=13)
    ax.set_title(f'NSE Comparison ({len(data[0])} Rain Events)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.2, 1.0)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def run_exp3(preds: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
             truth: Union[np.ndarray, Dict[str, np.ndarray]],
             mask: np.ndarray,
             mode: str,
             crop_size: int,
             out_dir: str) -> Dict[str, float]:
    ensure_dir(out_dir)
    metrics: Dict[str, float] = {}
    already_transformed = False

    nse_by_method: Dict[str, List[float]] = {}
    if isinstance(truth, dict):
        event_keys = list(truth.keys())
        truth_list = []
        preds_concat: Dict[str, List[np.ndarray]] = {}
        for name in preds.keys():
            nse_by_method[name] = []
            preds_concat[name] = []

        for event_key in event_keys:
            truth_ev = truth.get(event_key)
            if truth_ev is None:
                continue
            truth_ev = transform_mmhr(truth_ev, divide_by_3=True)
            truth_ev = crop_center(truth_ev, crop_size)
            truth_list.append(truth_ev)

            for name, pred_src in preds.items():
                if not isinstance(pred_src, dict):
                    continue
                pred_ev = pred_src.get(event_key)
                if pred_ev is None:
                    continue
                pred_ev = transform_mmhr(pred_ev, divide_by_3=True)
                pred_ev, truth_aligned = align_length(pred_ev, truth_ev)
                pred_ev = crop_center(pred_ev, crop_size)
                nse_frames = _nse_per_frame(pred_ev, truth_aligned, mask, mode)
                nse_frames = np.where(np.isfinite(nse_frames), np.maximum(nse_frames, 0.0), np.nan)
                nse_event = float(np.nanmean(nse_frames)) if np.any(np.isfinite(nse_frames)) else float("nan")
                nse_by_method[name].append(nse_event)
                preds_concat[name].append(pred_ev)
                truth_ev = truth_aligned

        truth = np.concatenate(truth_list, axis=0) if truth_list else np.empty((0,))
        preds = {
            name: np.concatenate(frames, axis=0) if frames else np.empty((0,))
            for name, frames in preds_concat.items()
        }
        already_transformed = True

    if not already_transformed:
        truth = transform_mmhr(truth, divide_by_3=True)
    truth = crop_center(truth, crop_size)

    pred_list = []
    labels = []
    for name, pred in preds.items():
        if not already_transformed:
            pred = transform_mmhr(pred, divide_by_3=True)
        pred, truth_aligned = align_length(pred, truth)
        pred = crop_center(pred, crop_size)
        pred_sel, gt_sel = _select_values(pred, truth_aligned, mask, mode)
        metrics[f"NSE_{name}"] = nse(pred_sel, gt_sel)
        pred_list.append(pred.ravel())
        labels.append(name)

    gt_flat = truth.ravel()
    _scatter_multi_row(
        pred_list,
        gt_flat,
        labels,
        figsize=(18, 3),
        lim=(0, 32),
        dpi=600,
        max_points=2000,
        alpha=0.6,
        s=10,
        min_value=0.1,
        save_path=os.path.join(out_dir, "scatter_panels.pdf"),
    )
    _residual_multi_row(
        pred_list,
        gt_flat,
        labels,
        figsize=(18, 3),
        dpi=600,
        max_points=2000,
        alpha=0.6,
        s=10,
        min_value=0.1,
        save_path=os.path.join(out_dir, "residual_panels.pdf"),
    )
    if nse_by_method:
        _nse_boxplot(nse_by_method, os.path.join(out_dir, "nse_boxplot.pdf"))
        _logfreq_plot(
            pred_list,
            gt_flat,
            [name for name in preds.keys()],
            figsize=(6, 4),
            dpi=300,
            lim=(0, 32),
            bins=64,
            save_path=os.path.join(out_dir, "logfreq.pdf"),
        )
    return metrics
