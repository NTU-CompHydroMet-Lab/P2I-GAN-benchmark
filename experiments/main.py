from __future__ import annotations

import os

from .config import build_config, get_mode_config
from .exp1 import run_exp1
from .exp2 import run_exp2, run_exp2_paper, run_exp2_paper_zarr
from .exp3 import run_exp3
from .io import ensure_dir, load_mask, load_zarr_array, save_config_snapshot, save_json, save_text


def _crop_mask(mask, size):
    h, w = mask.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return mask[top:top + size, left:left + size]


def main() -> None:
    cfg = build_config()
    mode_cfg = get_mode_config(cfg)

    results_root = os.path.join(cfg.save_dir, cfg.experiment_name)
    ensure_dir(results_root)
    save_config_snapshot(os.path.join(results_root, "config.json"), cfg)

    observation = load_zarr_array(mode_cfg.observation_path)
    truth = load_zarr_array(mode_cfg.truth_path, return_events=True)

    preds = {name: load_zarr_array(path, return_events=True) for name, path in mode_cfg.methods.items()}

    mask_train = load_mask(mode_cfg.mask_train_path)
    mask_test = load_mask(mode_cfg.mask_test_path)
    mask_train = _crop_mask(mask_train, cfg.crop_size)
    mask_test = _crop_mask(mask_test, cfg.crop_size)

    if cfg.run_exp1:
        exp1_dir = os.path.join(results_root, "exp1")
        ensure_dir(exp1_dir)
        mask_for_mode = mask_train if cfg.mode == "radar" else mask_test
        metrics = run_exp1(
            preds,
            truth,
            mask_for_mode,
            cfg.mode,
            cfg.crop_size,
            use_pool8=cfg.exp1_pool8,
            divide_by_3=True,
        )
        save_json(os.path.join(exp1_dir, "metrics.json"), metrics)
        lines = []
        if (
            metrics
            and isinstance(next(iter(metrics.values())), dict)
            and isinstance(next(iter(next(iter(metrics.values())).values())), dict)
        ):
            # event -> method -> metrics
            for event_key, methods in metrics.items():
                lines.append(f"[{event_key}]")
                for name, vals in methods.items():
                    lines.append(f"{name}:")
                    for k, v in vals.items():
                        if isinstance(v, dict):
                            lines.append(f"  {k}:")
                            for sub_k, sub_v in v.items():
                                lines.append(f"    {sub_k}: {sub_v:.6f}")
                        else:
                            lines.append(f"  {k}: {v:.6f}")
                    lines.append("")
                lines.append("")
        else:
            for name, vals in metrics.items():
                lines.append(f"[{name}]")
                for k, v in vals.items():
                    if isinstance(v, dict):
                        lines.append(f"{k}:")
                        for sub_k, sub_v in v.items():
                            lines.append(f"  {sub_k}: {sub_v:.6f}")
                    else:
                        lines.append(f"{k}: {v:.6f}")
                lines.append("")
        save_text(os.path.join(exp1_dir, "metrics.txt"), lines)

    if cfg.run_exp2_gif:
        exp2_gif_dir = os.path.join(results_root, "exp2_gif")
        ensure_dir(exp2_gif_dir)
        run_exp2(
            preds=mode_cfg.methods,
            truth=mode_cfg.truth_path,
            observation=mode_cfg.observation_path,
            mask_train=mask_train,
            out_dir=exp2_gif_dir,
            crop_size=cfg.crop_size,
            frames=None,
            vmin=cfg.visualization_vmin,
            vmax=cfg.visualization_vmax,
            gif_fps=cfg.gif_fps,
            divide_by_3=True,
            mode=cfg.mode,
        )

    if cfg.run_exp2_pdf:
        exp2_pdf_dir = os.path.join(results_root, "exp2_pdf")
        ensure_dir(exp2_pdf_dir)
        mask_path = cfg.exp2_paper_mask_path or mode_cfg.mask_train_path
        run_exp2_paper_zarr(
            observation_path=mode_cfg.observation_path,
            methods=mode_cfg.methods,
            events=cfg.exp2_paper_events,
            mask_path=mask_path,
            crop_size=cfg.crop_size,
            out_dir=exp2_pdf_dir,
            output_pdf=cfg.exp2_paper_output_pdf,
            method_order=("RadarMasked", "Nimrod", *mode_cfg.methods.keys()),
            crop_pdf=False,
            crop_output=cfg.exp2_paper_crop_output,
            crop_y_ranges=((0.019, 0.5), (0.58, 1.0)),
            crop_zoom=3.0,
            crop_margin_left=0.0,
            crop_margin_right=0.0,
        )

    if cfg.run_exp3:
        exp3_dir = os.path.join(results_root, "exp3")
        ensure_dir(exp3_dir)
        mask_for_mode = mask_train if cfg.mode == "radar" else mask_test
        metrics = run_exp3(preds, truth, mask_for_mode, cfg.mode, cfg.crop_size, exp3_dir)
        save_json(os.path.join(exp3_dir, "metrics.json"), metrics)
        lines = [f"{k}: {v:.6f}" for k, v in metrics.items()]
        save_text(os.path.join(exp3_dir, "metrics.txt"), lines)


if __name__ == "__main__":
    main()
