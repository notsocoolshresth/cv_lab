from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


class MultiModalAugmentation:
    """
    Cross-modal augmentation with shared spatial transform.

    Input sample keys expected:
      - ms: (5, H, W)
      - hs: (C_hs, H, W)
      - rgb: (3, H, W)
      - label: int
    """

    def __init__(
        self,
        out_size: int = 64,
        crop_size: int = 56,
        p_flip: float = 0.5,
        p_rot: float = 0.75,
        p_crop: float = 0.5,
        p_spectral_jitter: float = 0.5,
        spectral_jitter_sigma: float = 0.01,
        p_band_dropout: float = 0.5,
        band_dropout_ratio: float = 0.1,
        p_spectral_shift: float = 0.3,
        max_spectral_shift: int = 2,
    ) -> None:
        self.out_size = out_size
        self.crop_size = crop_size
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.p_crop = p_crop

        self.p_spectral_jitter = p_spectral_jitter
        self.spectral_jitter_sigma = spectral_jitter_sigma
        self.p_band_dropout = p_band_dropout
        self.band_dropout_ratio = band_dropout_ratio
        self.p_spectral_shift = p_spectral_shift
        self.max_spectral_shift = max_spectral_shift

    def _apply_shared_spatial(self, ms: torch.Tensor, hs: torch.Tensor, rgb: torch.Tensor):
        if random.random() < self.p_flip:
            ms = torch.flip(ms, dims=[2])
            hs = torch.flip(hs, dims=[2])
            rgb = torch.flip(rgb, dims=[2])

        if random.random() < self.p_flip:
            ms = torch.flip(ms, dims=[1])
            hs = torch.flip(hs, dims=[1])
            rgb = torch.flip(rgb, dims=[1])

        if random.random() < self.p_rot:
            k = random.choice([1, 2, 3])
            ms = torch.rot90(ms, k, dims=[1, 2])
            hs = torch.rot90(hs, k, dims=[1, 2])
            rgb = torch.rot90(rgb, k, dims=[1, 2])

        if random.random() < self.p_crop and self.crop_size < ms.shape[-1]:
            _, h, w = ms.shape
            ch = cw = self.crop_size
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)

            ms = ms[:, top : top + ch, left : left + cw]
            hs = hs[:, top : top + ch, left : left + cw]
            rgb = rgb[:, top : top + ch, left : left + cw]

            ms = F.interpolate(ms.unsqueeze(0), size=(self.out_size, self.out_size), mode="bilinear", align_corners=False).squeeze(0)
            hs = F.interpolate(hs.unsqueeze(0), size=(self.out_size, self.out_size), mode="bilinear", align_corners=False).squeeze(0)
            rgb = F.interpolate(rgb.unsqueeze(0), size=(self.out_size, self.out_size), mode="bilinear", align_corners=False).squeeze(0)

        return ms, hs, rgb

    def _apply_hs_spectral(self, hs: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_spectral_jitter:
            hs = hs + torch.randn_like(hs) * self.spectral_jitter_sigma

        if random.random() < self.p_band_dropout:
            n_bands = hs.shape[0]
            n_drop = max(1, int(n_bands * self.band_dropout_ratio))
            drop_idx = torch.randperm(n_bands)[:n_drop]
            hs[drop_idx] = 0.0

        if random.random() < self.p_spectral_shift and self.max_spectral_shift > 0:
            shift = random.randint(-self.max_spectral_shift, self.max_spectral_shift)
            if shift != 0:
                hs = torch.roll(hs, shifts=shift, dims=0)

        return hs

    def __call__(self, sample: Dict[str, torch.Tensor | int | str | bool]):
        ms = sample["ms"]
        hs = sample["hs"]
        rgb = sample["rgb"]

        ms, hs, rgb = self._apply_shared_spatial(ms, hs, rgb)
        hs = self._apply_hs_spectral(hs)

        sample["ms"] = ms
        sample["hs"] = hs
        sample["rgb"] = rgb
        return sample


def hs_mixup(
    hs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup for HS tensors only: returns mixed_hs, y_a, y_b, lam."""
    if alpha <= 0:
        return hs, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(hs.size(0), device=hs.device)
    mixed = lam * hs + (1 - lam) * hs[idx]
    return mixed, labels, labels[idx], float(lam)
