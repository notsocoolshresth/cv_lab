from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


class MSBranch(nn.Module):
    """
    Lightweight multispectral branch.

    Input: (B, 5, 64, 64)
    Outputs:
      - embedding: (B, embed_dim)
      - tokens: (B, T, C) where T=H'*W'
    """

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 256,
        widths: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        w1, w2, w3 = widths

        self.stage1 = ConvSEBlock(in_channels, w1, stride=1)
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ConvSEBlock(w1, w2, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ConvSEBlock(w2, w3, stride=1)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(w3, embed_dim),
        )

    def forward(self, ms: torch.Tensor) -> Dict[str, torch.Tensor]:
        if ms.ndim != 4:
            raise ValueError(f"Expected MS tensor of shape (B, C, H, W), got {tuple(ms.shape)}")
        if ms.size(1) != 5:
            raise ValueError(f"Expected 5 MS bands, got {ms.size(1)}")

        x = self.pool1(self.stage1(ms))
        x = self.pool2(self.stage2(x))
        x = self.stage3(x)

        embedding = self.head(self.dropout(x))

        b, c, h, w = x.shape
        tokens = x.reshape(b, c, h * w).transpose(1, 2)

        return {
            "embedding": embedding,
            "tokens": tokens,
            "feature_map": x,
        }
