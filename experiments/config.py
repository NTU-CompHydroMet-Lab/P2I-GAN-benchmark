from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ModeConfig:
    observation_path: str
    truth_path: str
    methods: Dict[str, str]
    mask_train_path: str
    mask_test_path: str


@dataclass
class ExperimentConfig:
    experiment_name: str = "test_exp_2_gauge"
    description: str = "Tests for P2I-GAN Benchmarking Experiments"
    save_dir: str = "results"
    mode: str = "gauge"  # "radar" or "gauge"
    run_exp1: bool = False
    run_exp2_gif: bool = True
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
    exp2_paper_method_order: Tuple[str, ...] = (
        "Gauge",
        "Radar",
        # "P2I-GAN+",
        "P2I-GAN",
        "DK",
        "STDK",
        # "KED",
        # "KRE",
    )
    exp2_paper_events: Tuple[Dict[str, object], ...] = (
        {
            "event_id": 8,
            "select_idx": (65, 66, 67),
            "title": "Stratiform . Start Time : 2021-10-02 10:25:00 UTC",
        },
        {
            "event_id": 12,
            "select_idx": (0, 1, 2),
            "title": "Convective . Start Time : 2022-03-16 11:00:00 UTC",
        },
    )
    exp2_paper_folders: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, ModeConfig] = field(default_factory=dict)

def build_config() -> ExperimentConfig:
    radar_mode = ModeConfig(
        observation_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/nimrod/nimrod_test.zarr"
        ),
        truth_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/nimrod/nimrod_test.zarr"
        ),
        methods={
            "P2IGAN": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/p2igan_nimrod.zarr"
            ),
            "DK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/dk_nimrod.zarr"
            ),
            "STDK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/stdk_nimrod.zarr"
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
        truth_path=(
            "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/nimrod/nimrod_test.zarr"
        ),
        methods={
            "P2IGAN": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/p2igan_gauge.zarr"
            ),
            "DK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/dk_gauge.zarr"
            ),
            "STDK": (
                "/home/NAS/homes/brick-10015/P2I-GAN-benchmark/datasets/infer/stdk_gauge.zarr"
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
