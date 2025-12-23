from __future__ import annotations

import torch
import torch.nn as nn


def _conv3d_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    layers = [
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    return nn.Sequential(*layers)


class SimpleGenerator(nn.Module):
    """Tiny generator used as the default benchmark baseline."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        doubled_in = in_channels * 2  # concatenate masked frames and mask along channel dim
        hidden = base_channels

        self.encoder = nn.Sequential(
            _conv3d_block(doubled_in, hidden),
            _conv3d_block(hidden, hidden * 2, stride=2),
            _conv3d_block(hidden * 2, hidden * 4, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden * 4, hidden * 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(hidden * 2, hidden, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # dataset normalized to [0, 1]
        )

    def forward(self, masked_video: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # inputs are (B, T, C, H, W); convert to (B, C, T, H, W)
        masked_cl = masked_video.permute(0, 2, 1, 3, 4)
        masks_cl = masks.permute(0, 2, 1, 3, 4)
        x = torch.cat([masked_cl, masks_cl], dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.permute(0, 2, 1, 3, 4)


class SimpleDiscriminator(nn.Module):
    """Minimal 3D discriminator with global pooling head."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        hidden = base_channels
        self.features = nn.Sequential(
            _conv3d_block(in_channels, hidden, stride=2),
            _conv3d_block(hidden, hidden * 2, stride=2),
            _conv3d_block(hidden * 2, hidden * 4, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden * 4, 1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = video.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
        feats = self.features(x)
        return self.head(feats)
