from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from PIL import Image

from .exp1 import transform_mmhr
from .io import align_length, crop_center, ensure_dir, ensure_thw, load_mask, mask_for_input, save_text

# Hard-coded palette for paper-style plots (matches notebook).
_PAPER_BOUNDS = [0, 0.5, 1, 2, 4, 8, 16, 200]
_PAPER_COLORS = [
    "#000000",
    "#46327e",
    "#277f8e",
    "#4ac16d",
    "#a0da39",
    "#fde725",
    "#ffffff",
]
_PAPER_SUB = 20


def _to_uint8(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    frame = np.clip(frame, vmin, vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = (frame - vmin) / (vmax - vmin)
    return (norm * 255.0).astype(np.uint8)


def _save_frames(frames: np.ndarray, out_dir: str, vmin: float, vmax: float, prefix: str) -> None:
    ensure_dir(out_dir)
    frames = ensure_thw(frames)
    for i in range(frames.shape[0]):
        img = Image.fromarray(_to_uint8(frames[i], vmin, vmax))
        img.save(os.path.join(out_dir, f"{prefix}_{i:03d}.png"))


def _save_gif(frames: np.ndarray, out_path: str, vmin: float, vmax: float, fps: int) -> None:
    frames = ensure_thw(frames)
    imgs = [Image.fromarray(_to_uint8(frames[i], vmin, vmax)) for i in range(frames.shape[0])]
    if not imgs:
        return
    duration = int(1000 / max(fps, 1))
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

def _save_combo_gif(frames_map: Dict[str, np.ndarray],
                    out_path: str,
                    cmap,
                    norm,
                    fps: int,
                    input_mask: np.ndarray | None = None,
                    title: str | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    labels = list(frames_map.keys())
    frames_list = [ensure_thw(frames_map[k]) for k in labels]
    n = min(f.shape[0] for f in frames_list) if frames_list else 0
    if n <= 0:
        return
    mask_points = None
    if input_mask is not None:
        mask_points = np.argwhere(input_mask.astype(bool))

    imgs = []
    for t in range(n):
        fig, axes = plt.subplots(1, len(labels), figsize=(3.1 * len(labels), 3.8), dpi=350)
        fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.02)
        if len(labels) == 1:
            axes = [axes]
        for ax, label, frames in zip(axes, labels, frames_list):
            if label.lower() == "input" and input_mask is not None:
                ax.imshow(np.zeros_like(frames[t]), cmap="gray", vmin=0.0, vmax=1.0)
                if mask_points is not None and mask_points.size > 0:
                    vals = frames[t][input_mask.astype(bool)]
                    ax.scatter(
                        mask_points[:, 1],
                        mask_points[:, 0],
                        c=vals,
                        cmap=cmap,
                        norm=norm,
                        s=18,
                        edgecolors="#dddddd",
                        linewidths=0.4,
                        zorder=5,
                    )
            else:
                ax.imshow(frames[t], cmap=cmap, norm=norm)
            ax.set_title(label, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes,
            orientation="horizontal",
            fraction=0.08,
            pad=0.18,
            ticks=_PAPER_BOUNDS,
        )
        tick_labels = [f"{b:g}" for b in _PAPER_BOUNDS[:-1]] + [""]
        cbar.set_ticklabels(tick_labels)
        cbar.set_label("Rainfall (mm/h)", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        if title:
            frame_label = f"{title} | Frame {t + 1}/{n}"
            fig.suptitle(frame_label, fontsize=12)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        imgs.append(Image.fromarray(buf))
        plt.close(fig)

    duration = int(1000 / max(fps, 1))
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


def _list_event_keys(path: str) -> List[str]:
    try:
        import zarr
    except Exception:
        return []

    z = zarr.open(path, mode="r")
    keys = []
    if hasattr(z, "group_keys"):
        keys = list(z.group_keys())
    if not keys and hasattr(z, "keys"):
        keys = [k for k in z.keys() if isinstance(k, str)]
    if not keys:
        return []

    def key_num(k: str) -> Tuple[int, str]:
        m = re.search(r"event[_-]?(\\d+)", k, re.IGNORECASE)
        return (int(m.group(1)) if m else 10**9, k)

    return [k for k in sorted(keys, key=key_num)]


def _load_event_array(path: str, event_key: str) -> np.ndarray:
    try:
        import zarr
    except Exception:
        raise FileNotFoundError(f"zarr not available to load event {event_key} from {path}")

    z = zarr.open(path, mode="r")
    if event_key in z:
        return np.asarray(z[event_key])
    raise FileNotFoundError(f"Missing event {event_key} in {path}")


def run_exp2(preds: Dict[str, Union[str, np.ndarray]],
             truth: Union[str, np.ndarray],
             observation: Union[str, np.ndarray],
             mask_train: np.ndarray,
             out_dir: str,
             crop_size: int,
             frames: int | None,
             vmin: float,
             vmax: float,
             gif_fps: int,
             divide_by_3: bool = True) -> None:
    if isinstance(truth, str) and isinstance(observation, str):
        event_keys = _list_event_keys(truth)
        if not event_keys:
            raise FileNotFoundError(f"No event groups found in {truth}")
        event_keys = event_keys[:20]
        range_lines = []
        cmap, norm, _ = _build_paper_cmap()
        max_frames = 30
        for idx, event_key in enumerate(event_keys, start=1):
            truth_ev = _load_event_array(truth, event_key)
            obs_ev = _load_event_array(observation, event_key)
            truth_ev = transform_mmhr(truth_ev, divide_by_3=divide_by_3)
            obs_ev = transform_mmhr(obs_ev, divide_by_3=divide_by_3)
            truth_ev = crop_center(truth_ev, crop_size)[:max_frames]
            obs_ev = crop_center(obs_ev, crop_size)[:max_frames]
            mask_bool = mask_train.astype(bool)
            masked_input = obs_ev * mask_bool[None, ...]

            preds_ev: Dict[str, np.ndarray] = {}
            for name, pred_src in preds.items():
                if isinstance(pred_src, str):
                    pred_ev = _load_event_array(pred_src, event_key)
                else:
                    pred_ev = pred_src
                pred_ev = transform_mmhr(pred_ev, divide_by_3=divide_by_3)
                pred_ev, truth_aligned = align_length(pred_ev, truth_ev)
                pred_ev = crop_center(pred_ev, crop_size)[:max_frames]
                preds_ev[name] = pred_ev
                truth_ev = truth_aligned

            total_frames = truth_ev.shape[0]
            for pred_ev in preds_ev.values():
                total_frames = min(total_frames, pred_ev.shape[0])
            truth_ev = truth_ev[:total_frames]
            masked_input = masked_input[:total_frames]
            for name in list(preds_ev.keys()):
                preds_ev[name] = preds_ev[name][:total_frames]

            range_lines.append(
                f"{event_key}: frames 1-{total_frames} (count={total_frames})"
            )
            combo_frames = {
                "Input": masked_input,
                "Truth": truth_ev,
            }
            for name, pred_ev in preds_ev.items():
                combo_frames[name] = pred_ev

            title = f"{event_key} | total frames {total_frames}"
            out_path = os.path.join(out_dir, f"comparison_{event_key}.gif")
            _save_combo_gif(combo_frames, out_path, cmap, norm, gif_fps,
                            input_mask=mask_train, title=title)

        save_text(os.path.join(out_dir, "event_ranges.txt"), range_lines)
        return

    truth_arr = np.asarray(truth)
    obs_arr = np.asarray(observation)
    truth_arr = transform_mmhr(truth_arr, divide_by_3=divide_by_3)
    obs_arr = transform_mmhr(obs_arr, divide_by_3=divide_by_3)
    truth_arr = crop_center(truth_arr, crop_size)
    obs_arr = crop_center(obs_arr, crop_size)
    mask_bool = mask_train.astype(bool)
    masked_input = obs_arr * mask_bool[None, ...]

    preds_aligned: Dict[str, np.ndarray] = {}
    for name, pred in preds.items():
        pred_arr = np.asarray(pred)
        pred_arr = transform_mmhr(pred_arr, divide_by_3=divide_by_3)
        pred_arr, truth_aligned = align_length(pred_arr, truth_arr)
        pred_arr = crop_center(pred_arr, crop_size)
        preds_aligned[name] = pred_arr
        truth_arr = truth_aligned

    total_frames = truth_arr.shape[0]
    for pred_arr in preds_aligned.values():
        total_frames = min(total_frames, pred_arr.shape[0])
    truth_arr = truth_arr[:total_frames]
    masked_input = masked_input[:total_frames]
    for name in list(preds_aligned.keys()):
        preds_aligned[name] = preds_aligned[name][:total_frames]

    combo_frames = {
        "Input": masked_input,
        "Truth": truth_arr,
    }
    for name, pred_arr in preds_aligned.items():
        combo_frames[name] = pred_arr
    cmap, norm, _ = _build_paper_cmap()
    title = f"Event 01 | total frames {total_frames}"
    _save_combo_gif(combo_frames, os.path.join(out_dir, "comparison_event_01.gif"),
                    cmap, norm, gif_fps, input_mask=mask_train, title=title)


def _build_paper_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    def hex_to_rgb01(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lerp(a, b, t):
        return tuple((1 - t) * ai + t * bi for ai, bi in zip(a, b))

    fine_bounds = []
    for i in range(len(_PAPER_BOUNDS) - 1):
        seg = np.linspace(_PAPER_BOUNDS[i], _PAPER_BOUNDS[i + 1], _PAPER_SUB + 1, endpoint=False)
        if i == 0:
            fine_bounds.extend(seg.tolist())
        else:
            fine_bounds.extend(seg[1:].tolist())
    fine_bounds.append(_PAPER_BOUNDS[-1])
    fine_bounds = np.asarray(fine_bounds, float)

    rgb_base = [hex_to_rgb01(h) for h in _PAPER_COLORS]
    grad_colors = []
    for i in range(len(rgb_base) - 1):
        c0, c1 = rgb_base[i], rgb_base[i + 1]
        for k in range(_PAPER_SUB):
            t = k / float(_PAPER_SUB - 1)
            grad_colors.append(lerp(c0, c1, t))
    grad_colors.append(rgb_base[-1])
    while len(grad_colors) < len(fine_bounds) - 1:
        grad_colors.append(rgb_base[-1])

    cmap = ListedColormap(grad_colors, name=f"seg{_PAPER_SUB}_smooth")
    norm = BoundaryNorm(fine_bounds, cmap.N, clip=True)
    return cmap, norm, fine_bounds


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top:top + size, left:left + size]


def _load_event_images(folders: Dict[str, str],
                       method_order: Iterable[str],
                       event_id: int,
                       select_idx: Iterable[int],
                       crop_size: int) -> Tuple[np.ndarray, List[str]]:
    rain_str = f"rain{event_id}"
    sample_folder = os.path.join(folders.get("Gauge", ""), rain_str)
    if not os.path.isdir(sample_folder):
        raise FileNotFoundError(f"Missing sample folder: {sample_folder}")
    all_pngs = sorted(
        [f for f in os.listdir(sample_folder) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    images = []
    labels = []
    for method in method_order:
        folder = folders.get(method, "")
        labels.append(method)
        frames = []
        if not folder:
            frames = [np.zeros((crop_size, crop_size), dtype=np.float32) for _ in select_idx]
            images.append(frames)
            continue
        for idx in select_idx:
            if idx >= len(all_pngs):
                frames.append(np.zeros((crop_size, crop_size), dtype=np.float32))
                continue
            path = os.path.join(folder, rain_str, all_pngs[idx])
            if not os.path.isfile(path):
                frames.append(np.zeros((crop_size, crop_size), dtype=np.float32))
                continue
            arr = np.array(Image.open(path).convert("F")).astype(np.float32) / 3.0
            arr = 10 ** (arr * 0.0625) * 0.036
            arr = _center_crop(arr, crop_size)
            frames.append(arr)
        images.append(frames)
    return np.asarray(images), labels


def _draw_block(ax_grid, images, method_order, select_idx, mask, mask_points, cmap, norm):
    last_im = None
    for t in range(images.shape[1]):
        for m in range(images.shape[0]):
            ax = ax_grid[t, m]
            label = method_order[m]
            if label == "RadarMasked":
                ax.imshow(np.zeros_like(images[m, t]), cmap="gray", vmin=0.0, vmax=1.0)
                vals = images[m, t][mask == 1]
                ax.scatter(mask_points[:, 1], mask_points[:, 0],
                           c=vals, cmap=cmap, norm=norm,
                           s=24, edgecolors="#dddddd", linewidths=0.4, zorder=5)
                last_im = ax.images[-1] if ax.images else None
            else:
                last_im = ax.imshow(images[m, t], cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            if method_order[m] == "Gauge":
                vals = images[m, t][mask == 1]
                ax.scatter(mask_points[:, 1], mask_points[:, 0],
                           c=vals, cmap=cmap, norm=norm,
                           s=38, edgecolors="black", linewidths=0.7, zorder=5)

            if t == 0:
                ax.set_title(method_order[m], fontsize=13)

            if m == 0:
                time_label = f"{t * 5} min"
                ax.text(-0.12, 0.5, time_label, transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, rotation=90)
    return last_im


def _event_key(event_id: int) -> str:
    return f"event_{int(event_id):02d}"


def run_exp2_paper(folders: Dict[str, str],
                   method_order: Iterable[str],
                   events: Iterable[Dict[str, object]],
                   mask_path: str,
                   crop_size: int,
                   out_dir: str,
                   output_pdf: str,
                   crop_pdf: bool = False,
                   crop_output: str = "cropped_stitched.pdf",
                   crop_y_ranges: Tuple[Tuple[float, float], ...] = ((0.019, 0.5), (0.58, 1.0)),
                   crop_zoom: float = 3.0,
                   crop_margin_left: float = 0.0,
                   crop_margin_right: float = 0.0) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    cmap, norm, _ = _build_paper_cmap()

    mask = load_mask(mask_path)
    mask = _center_crop(mask, crop_size)
    mask_points = np.argwhere(mask == 1)

    method_order = list(method_order)
    events = list(events)
    ncols = len(method_order)
    nrows_each = len(events[0]["select_idx"])
    total_rows = (nrows_each + 1) * len(events)

    fig = plt.figure(figsize=(2.1 * ncols, 1.9 * total_rows))
    gs = GridSpec(nrows=total_rows, ncols=ncols, figure=fig,
                  top=0.93, bottom=0.06, wspace=0.05, hspace=0.02)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.20, 0.88, 0.60, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", ticks=_PAPER_BOUNDS)
    tick_labels = [f"{b:g}" for b in _PAPER_BOUNDS[:-1]] + [""]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Rainfall (mm/h)", fontsize=13, labelpad=3)
    cbar.ax.tick_params(labelsize=11)

    row_cursor = 0
    for event in events:
        ax_title = fig.add_subplot(gs[row_cursor, :])
        ax_title.axis("off")
        ax_title.text(-0.015, 0.2, event["title"],
                      fontsize=14, fontweight="bold", ha="left", va="center",
                      transform=ax_title.transAxes)
        row_cursor += 1

        ax_grid = np.empty((nrows_each, ncols), dtype=object)
        for r in range(nrows_each):
            for c in range(ncols):
                ax_grid[r, c] = fig.add_subplot(gs[row_cursor + r, c])

        imgs, labels = _load_event_images(folders, method_order, event["event_id"],
                                          event["select_idx"], crop_size)
        _ = _draw_block(ax_grid, imgs, labels, event["select_idx"], mask, mask_points, cmap, norm)
        row_cursor += nrows_each

    fig_path = os.path.join(out_dir, output_pdf)
    plt.tight_layout(rect=[0, 0, 1, 0.7])
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    if crop_pdf:
        _crop_pdf_panels(fig_path, os.path.join(out_dir, crop_output),
                         y_ranges=crop_y_ranges, zoom=crop_zoom,
                         margin_left=crop_margin_left, margin_right=crop_margin_right)


def run_exp2_paper_zarr(observation_path: str,
                        methods: Dict[str, str],
                        events: Iterable[Dict[str, object]],
                        mask_path: str,
                        crop_size: int,
                        out_dir: str,
                        output_pdf: str,
                        method_order: Iterable[str] | None = None,
                        crop_pdf: bool = False,
                        crop_output: str = "cropped_stitched.pdf",
                        crop_y_ranges: Tuple[Tuple[float, float], ...] = ((0.019, 0.5), (0.58, 1.0)),
                        crop_zoom: float = 3.0,
                        crop_margin_left: float = 0.0,
                        crop_margin_right: float = 0.0) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    cmap, norm, _ = _build_paper_cmap()

    mask = load_mask(mask_path)
    mask = _center_crop(mask, crop_size)
    mask_points = np.argwhere(mask == 1)

    if method_order is None:
        method_order = tuple(["RadarMasked", "Nimrod"] + list(methods.keys()))
    method_order = list(method_order)
    events = list(events)
    ncols = len(method_order)
    nrows_each = len(events[0]["select_idx"])
    total_rows = (nrows_each + 1) * len(events)

    fig = plt.figure(figsize=(2.4 * ncols, 1.9 * total_rows))
    gs = GridSpec(nrows=total_rows, ncols=ncols, figure=fig,
                  top=0.93, bottom=0.06, wspace=0.04, hspace=0.02)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.20, 0.88, 0.60, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", ticks=_PAPER_BOUNDS)
    tick_labels = [f"{b:g}" for b in _PAPER_BOUNDS[:-1]] + [""]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Rainfall (mm/h)", fontsize=13, labelpad=3)
    cbar.ax.tick_params(labelsize=11)

    row_cursor = 0
    for event in events:
        ax_title = fig.add_subplot(gs[row_cursor, :])
        ax_title.axis("off")
        ax_title.text(-0.015, 0.2, event["title"],
                      fontsize=14, fontweight="bold", ha="left", va="center",
                      transform=ax_title.transAxes)
        row_cursor += 1

        ax_grid = np.empty((nrows_each, ncols), dtype=object)
        for r in range(nrows_each):
            for c in range(ncols):
                ax_grid[r, c] = fig.add_subplot(gs[row_cursor + r, c])

        event_key = _event_key(int(event["event_id"]))
        select_idx = list(event["select_idx"])
        obs_ev = _load_event_array(observation_path, event_key)
        obs_ev = transform_mmhr(obs_ev, divide_by_3=True)
        obs_ev = crop_center(obs_ev, crop_size)

        images = []
        labels = []
        for method in method_order:
            labels.append(method)
            frames = []
            if method == "RadarMasked":
                source = obs_ev
            elif method == "Nimrod":
                source = obs_ev
            else:
                method_path = methods.get(method)
                if not method_path:
                    source = None
                else:
                    pred_ev = _load_event_array(method_path, event_key)
                    pred_ev = transform_mmhr(pred_ev, divide_by_3=True)
                    pred_ev = crop_center(pred_ev, crop_size)
                    source = pred_ev

            for idx in select_idx:
                if source is None or idx >= source.shape[0]:
                    frames.append(np.zeros((crop_size, crop_size), dtype=np.float32))
                else:
                    frames.append(source[idx])
            images.append(frames)

        imgs = np.asarray(images)
        _ = _draw_block(ax_grid, imgs, labels, select_idx, mask, mask_points, cmap, norm)
        row_cursor += nrows_each

    fig_path = os.path.join(out_dir, output_pdf)
    plt.tight_layout(rect=[0, 0, 1, 0.7])
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    if crop_pdf:
        _crop_pdf_panels(fig_path, os.path.join(out_dir, crop_output),
                         y_ranges=crop_y_ranges, zoom=crop_zoom,
                         margin_left=crop_margin_left, margin_right=crop_margin_right)


def _crop_pdf_panels(pdf_path: str,
                     output_path: str,
                     y_ranges: Tuple[Tuple[float, float], ...],
                     zoom: float,
                     margin_left: float,
                     margin_right: float) -> None:
    import fitz

    def crop_regions_from_pdf_rel(page, rects_rel, zoom):
        (x0, y0, x1, y1) = page.rect
        rects_pts = []
        for (rx0, ry0, rx1, ry1) in rects_rel:
            rects_pts.append((
                x0 + rx0 * (x1 - x0),
                y0 + ry0 * (y1 - y0),
                x0 + rx1 * (x1 - x0),
                y0 + ry1 * (y1 - y0),
            ))
        images = []
        mat = fitz.Matrix(zoom, zoom)
        for (px0, py0, px1, py1) in rects_pts:
            clip = fitz.Rect(px0, py0, px1, py1)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return images

    def stitch_images(imgs, direction="v", gap=8, bg=(255, 255, 255)):
        if direction == "h":
            h = max(im.height for im in imgs)
            w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
            canvas = Image.new("RGB", (w, h), bg)
            x = 0
            for im in imgs:
                y = (h - im.height) // 2
                canvas.paste(im, (x, y))
                x += im.width + gap
        else:
            w = max(im.width for im in imgs)
            h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
            canvas = Image.new("RGB", (w, h), bg)
            y = 0
            for im in imgs:
                x = (w - im.width) // 2
                canvas.paste(im, (x, y))
                y += im.height + gap
        return canvas

    doc = fitz.open(pdf_path)
    page = doc[0]
    rects_rel = []
    for (y0, y1) in y_ranges:
        y0c = max(0.0, min(1.0, y0))
        y1c = max(0.0, min(1.0, y1))
        if y1c <= y0c:
            continue
        rects_rel.append((margin_left, y0c, 1 - margin_right, y1c))
    parts = crop_regions_from_pdf_rel(page, rects_rel, zoom=zoom)
    doc.close()
    if not parts:
        return
    stitched = stitch_images(parts, direction="v", gap=8, bg=(255, 255, 255))
    stitched.save(output_path)
