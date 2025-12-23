"""Model registry for the benchmark."""

from typing import Any, Dict

import torch.nn as nn

from .p2igan import P2IDiscriminator, P2IGenerator
from .simple import SimpleDiscriminator, SimpleGenerator


def build_generator(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "simple").lower()
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", in_channels)
    base_channels = model_cfg.get("base_channels", 64)

    if model_name == "p2igan":
        return P2IGenerator(cfg)

    return SimpleGenerator(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)


def build_discriminator(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "simple").lower()
    in_channels = model_cfg.get("in_channels", 1)
    base_channels = model_cfg.get("base_channels", 64)

    if model_name == "p2igan":
        data_cfg = cfg.get("data_loader") or cfg.get("data", {}).get("train", {})
        sample_length = data_cfg.get("sample_length", 16)
        seq_channels = in_channels * sample_length
        return P2IDiscriminator(in_channels=seq_channels)

    return SimpleDiscriminator(in_channels=in_channels, base_channels=base_channels)


__all__ = [
    "build_generator",
    "build_discriminator",
    "SimpleGenerator",
    "SimpleDiscriminator",
    "P2IGenerator",
    "P2IDiscriminator",
]
