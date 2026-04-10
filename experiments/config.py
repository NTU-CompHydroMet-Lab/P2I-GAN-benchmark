from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ModeConfig:
    observation_path: str
    methods: Dict[str, str]
    mask_train_path: str
    mask_test_path: str
    nimrod_path: str

@dataclass
class ExperimentConfig:
    experiment_name: str = "exp1_pm_v3_radar"
    description: str = "PM evaluation test if pm loss enhence model performance."
    save_dir: str = "results"
    mode: str = "radar"  # "radar" or "gauge"
    run_exp1: bool = True
    run_exp2_gif: bool = False
    run_exp2_pdf: bool = False
    run_exp3: bool = False
    crop_size: int = 128
    visualization_vmin: float = 0.0
    visualization_vmax: float = 32.0
    gif_fps: int = 6
    exp1_pool8: bool = True
    exp2_paper_output_pdf: str = "two_events_stacked_titles.pdf"
    exp2_paper_crop_output: str = "cropped_stitched.pdf"
    exp2_paper_mask_path: Optional[str] = None
    exp2_paper_radar_method_order: Tuple[str, ...] = (
        "Radar",
        "P2I-GAN+",
        "P2I-GAN",
        "DK",
        "STDK",
        "OK",
        "RBF",
    )
    exp2_paper_gauge_method_order: Tuple[str, ...] = (
        "Gauge",
        "Radar",
        "P2I-GAN+",
        "P2I-GAN",
        "DK",
        "STDK",
        "KED",
        "KRE",
    )

    exp2_paper_events: Tuple[Dict[str, object], ...] = (
        {
            "event_id": 5,
            "select_idx": (25, 26, 27),
            "title": "Stratiform . Start Time : 2021-05-20 11:00:00 UTC",
        },
        {
            "event_id": 19,
            "select_idx": (19, 20, 21),
            "title": "Convective . Start Time : 2022-10-31 19:35:00 UTC",
        },
    )

    # exp2_paper_events: Tuple[Dict[str, object], ...] = (
    #     {
    #         "event_id": 4,
    #         "select_idx": (20, 21, 22),
    #         "title": "Stratiform . Start Time : 2021-05-08 10:40:00 UTC",
    #     },
    #     {
    #         "event_id": 10,
    #         "select_idx": (26, 27, 28),
    #         "title": "Convective . Start Time : 2021-12-07 10:10:00 UTC",
    #     },
    # )
    exp2_paper_folders: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, ModeConfig] = field(default_factory=dict)

def build_config() -> ExperimentConfig:
    radar_mode = ModeConfig(
        observation_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/nimrod/nimrod_test.zarr"
        ),
        nimrod_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/nimrod/nimrod_test.zarr"
        ),
        methods={
            "P2I-GAN+": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/p2igan_gan_baseline_v3_pm_nimrod.zarr"
            ),
            "P2I-GAN": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/p2igan_baseline_v3_pm_nimrod.zarr"
            ),
            "DK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/dk_v3_pm_nimrod.zarr"
            ),
            "STDK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/stdk_v3_pm_nimrod.zarr"
            ),
            "OK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/ok_nimrod.zarr"
            ),
            "RBF": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/rbf_nimrod.zarr"
            ),
        },
        mask_train_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/masks/gauge_mask_128_train.txt"
        ),
        mask_test_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/masks/gauge_mask_128_test.txt"
        ),
    )

    gauge_mode = ModeConfig(
        observation_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/midas/midas_test.zarr"
        ),
        nimrod_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/nimrod/nimrod_test.zarr"
        ),
        methods={
            "P2I-GAN+": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/p2igan_gan_baseline_v3_pm_gauge.zarr"
            ),
            "P2I-GAN": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/p2igan_baseline_v3_pm_gauge.zarr"
            ),
            "DK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/dk_v3_pm_gauge.zarr"
            ),
            "STDK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/stdk_v3_pm_gauge.zarr"
            ),
            "KRE": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/kre_gauge.zarr"
            ),
            "KED": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/ked_gauge.zarr"
            ),
        },
        mask_train_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/masks/gauge_mask_128_train.txt"
        ),
        mask_test_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/masks/gauge_mask_128_test.txt"
        ),
    )

    cfg = ExperimentConfig()
    cfg.data = {"radar": radar_mode, "gauge": gauge_mode}
    return cfg


def get_mode_config(cfg: ExperimentConfig) -> ModeConfig:
    mode_cfg = cfg.data.get(cfg.mode)
    if mode_cfg is None:
        raise ValueError(f"Unknown mode: {cfg.mode}")
    return mode_cfg
