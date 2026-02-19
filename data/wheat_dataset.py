from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import tifffile as tiff
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
VALID_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass
class ModalityStats:
    mean: np.ndarray
    std: np.ndarray


def _is_image_file(name: str) -> bool:
    return name.lower().endswith(VALID_EXTENSIONS)


def _parse_label_from_name(name: str) -> int:
    if "_hyper_" not in name:
        return -1
    cls_name = name.split("_hyper_")[0]
    return CLASS_MAP.get(cls_name, -1)


def _read_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tiff.imread(path).astype(np.float32)
    else:
        arr = np.array(Image.open(path)).astype(np.float32)

    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image at {path}, got shape {arr.shape}")
    return arr


def _to_chw(arr: np.ndarray) -> np.ndarray:
    return arr.transpose(2, 0, 1)


class WheatMultiModalDataset(Dataset):
    """
    Unified HS+MS+RGB dataset.

    - Aligns samples by identical filename across modalities.
    - Filters black/corrupt samples (mean < black_threshold) when requested.
    - Applies HS clean-band slicing [10:110] by default (100 bands).
    - Optionally resizes HS spatially from 32x32 to 64x64.
    - Supports train-derived per-band z-score normalization.
    """

    def __init__(
        self,
        ms_dir: str,
        hs_dir: str,
        rgb_dir: str,
        is_train: bool = True,
        filter_black: bool = True,
        black_threshold: float = 1.0,
        hs_band_range: Tuple[int, int] = (10, 110),
        hs_resize_to: Optional[Tuple[int, int]] = (64, 64),
        stats: Optional[Dict[str, ModalityStats]] = None,
        transform=None,
    ) -> None:
        self.ms_dir = ms_dir
        self.hs_dir = hs_dir
        self.rgb_dir = rgb_dir
        self.is_train = is_train
        self.filter_black = filter_black
        self.black_threshold = black_threshold
        self.hs_band_range = hs_band_range
        self.hs_resize_to = hs_resize_to
        self.stats = stats
        self.transform = transform

        self.samples = self._build_index()

    def _build_index(self) -> List[Dict[str, object]]:
        def _build_stem_map(dir_path: str) -> Dict[str, str]:
            stem_to_path = {}
            for f in os.listdir(dir_path):
                if not _is_image_file(f):
                    continue
                stem, _ = os.path.splitext(f)
                stem_to_path[stem] = os.path.join(dir_path, f)
            return stem_to_path

        ms_map = _build_stem_map(self.ms_dir)
        hs_map = _build_stem_map(self.hs_dir)
        rgb_map = _build_stem_map(self.rgb_dir)

        common = sorted(set(ms_map.keys()) & set(hs_map.keys()) & set(rgb_map.keys()))
        samples: List[Dict[str, object]] = []

        for stem in common:
            ms_path = ms_map[stem]
            hs_path = hs_map[stem]
            rgb_path = rgb_map[stem]

            label = _parse_label_from_name(stem) if self.is_train else -1

            is_black = self._is_black_sample(ms_path, hs_path, rgb_path)
            if self.filter_black and is_black:
                continue

            samples.append(
                {
                    "fname": os.path.basename(ms_path),
                    "ms_path": ms_path,
                    "hs_path": hs_path,
                    "rgb_path": rgb_path,
                    "label": label,
                    "is_black": is_black,
                }
            )

        return samples

    def _is_black_sample(self, ms_path: str, hs_path: str, rgb_path: str) -> bool:
        try:
            ms = _read_image(ms_path)
            hs = _read_image(hs_path)
            rgb = _read_image(rgb_path)
        except Exception:
            return True

        return (
            float(ms.mean()) < self.black_threshold
            or float(hs.mean()) < self.black_threshold
            or float(rgb.mean()) < self.black_threshold
        )

    @staticmethod
    def _slice_hs_bands(hs_hwc: np.ndarray, band_range: Tuple[int, int]) -> np.ndarray:
        start, end = band_range
        end = min(end, hs_hwc.shape[2])
        start = max(0, min(start, end - 1))
        return hs_hwc[:, :, start:end]

    def _resize_hs(self, hs_chw: np.ndarray) -> np.ndarray:
        if self.hs_resize_to is None:
            return hs_chw

        x = torch.from_numpy(hs_chw).unsqueeze(0)
        x = F.interpolate(x, size=self.hs_resize_to, mode="bilinear", align_corners=False)
        return x.squeeze(0).numpy()

    def _normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        if self.stats is None or key not in self.stats:
            return x
        mean = self.stats[key].mean[:, None, None]
        std = self.stats[key].std[:, None, None]
        return (x - mean) / (std + 1e-8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int | bool]:
        s = self.samples[idx]

        ms = _read_image(s["ms_path"])
        hs = _read_image(s["hs_path"])
        rgb = _read_image(s["rgb_path"])

        hs = self._slice_hs_bands(hs, self.hs_band_range)

        ms = _to_chw(ms)
        hs = _to_chw(hs)
        rgb = _to_chw(rgb)

        hs = self._resize_hs(hs)

        ms = self._normalize(ms, "ms")
        hs = self._normalize(hs, "hs")
        rgb = self._normalize(rgb, "rgb")

        sample = {
            "ms": torch.from_numpy(ms).float(),
            "hs": torch.from_numpy(hs).float(),
            "rgb": torch.from_numpy(rgb).float(),
            "label": int(s["label"]),
            "fname": str(s["fname"]),
            "is_black": bool(s["is_black"]),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @classmethod
    def compute_train_stats(
        cls,
        ms_dir: str,
        hs_dir: str,
        rgb_dir: str,
        hs_band_range: Tuple[int, int] = (10, 110),
        hs_resize_to: Optional[Tuple[int, int]] = (64, 64),
        black_threshold: float = 1.0,
    ) -> Dict[str, ModalityStats]:
        tmp_ds = cls(
            ms_dir=ms_dir,
            hs_dir=hs_dir,
            rgb_dir=rgb_dir,
            is_train=True,
            filter_black=True,
            black_threshold=black_threshold,
            hs_band_range=hs_band_range,
            hs_resize_to=hs_resize_to,
            stats=None,
        )

        def _collect(key: str) -> np.ndarray:
            arrs = []
            for i in range(len(tmp_ds)):
                x = tmp_ds[i][key].numpy()
                arrs.append(x)
            return np.stack(arrs, axis=0)

        ms_all = _collect("ms")
        hs_all = _collect("hs")
        rgb_all = _collect("rgb")

        return {
            "ms": ModalityStats(mean=ms_all.mean(axis=(0, 2, 3)), std=ms_all.std(axis=(0, 2, 3))),
            "hs": ModalityStats(mean=hs_all.mean(axis=(0, 2, 3)), std=hs_all.std(axis=(0, 2, 3))),
            "rgb": ModalityStats(mean=rgb_all.mean(axis=(0, 2, 3)), std=rgb_all.std(axis=(0, 2, 3))),
        }

    @staticmethod
    def from_split_root(
        root: str,
        split: str,
        stats: Optional[Dict[str, ModalityStats]] = None,
        is_train: Optional[bool] = None,
        **kwargs,
    ) -> "WheatMultiModalDataset":
        base = os.path.join(root, split)
        return WheatMultiModalDataset(
            ms_dir=os.path.join(base, "MS"),
            hs_dir=os.path.join(base, "HS"),
            rgb_dir=os.path.join(base, "RGB"),
            is_train=(split == "train") if is_train is None else is_train,
            stats=stats,
            **kwargs,
        )
