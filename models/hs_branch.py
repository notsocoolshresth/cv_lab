from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class HSBranch3D(nn.Module):
    """
    HS branch for spectral-spatial feature extraction.

    Expected input: (B, 100, 64, 64)
    Internal 3D view: (B, 1, 100, 64, 64)
    """

    def __init__(
        self,
        in_bands: int = 100,
        embed_dim: int = 256,
        stem_channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3 = stem_channels
        self.in_bands = in_bands

        self.stem = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=3, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        self.res_block = Residual3DBlock(c3)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(c3, embed_dim)

    def forward(self, hs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hs.ndim != 4:
            raise ValueError(f"Expected HS tensor of shape (B, C, H, W), got {tuple(hs.shape)}")
        if hs.size(1) != self.in_bands:
            raise ValueError(f"Expected {self.in_bands} HS bands, got {hs.size(1)}")

        x = hs.unsqueeze(1)
        x = self.stem(x)
        x = self.res_block(x)

        pooled = self.pool(self.dropout(x)).flatten(1)
        embedding = self.proj(pooled)

        b, c, d, h, w = x.shape
        tokens = x.reshape(b, c, d * h * w).transpose(1, 2)

        return {
            "embedding": embedding,
            "tokens": tokens,
            "feature_map": x,
        }
