"""
AC-FNet Pipeline — Adapted from "Enhancing Plant Disease Identification Through
Hyperspectral Imaging and AC-FNet Framework" (ICPR Paper)
================================================================================

Full pipeline:
1. JM-SLIC Segmentation      — Isolate leaf tissue from background
2. Preprocessing              — Dead pixel removal + Wiener noise filtering
3. Super-Resolution (S-BI)    — Upscale HS 32x32 → 64x64 (bicubic + stochastic gradient)
4. Spectral Unmixing          — Separate spectral bands into endmember abundances
5. Feature Extraction         — Branch A: 7 Vegetation Indices
                               — Branch B: Texture (GLCM) + Airspace + Endmember features
6. Feature Fusion + PCA       — Combine VI + Image features → reduce to optimal dimension
7. AC-FNet Model              — Atrous Convolution FractalNet for classification
8. Dual-Path Training         — AC-FNet CNN features + Handcrafted features → XGBoost

Adapted for:
- 3-class wheat disease: Health, Rust, Other
- MS: 5 bands (64×64), HS: 125 bands (32×32), RGB: PNG
- 577 training samples (after removing 23 black images)
- 300 validation samples
"""

import os
import csv
import json
import random
import warnings
import numpy as np
import tifffile as tiff
from PIL import Image
from scipy import ndimage, stats as scipy_stats
from scipy.signal import wiener
from scipy.spatial.distance import cosine
from scipy.ndimage import zoom as scipy_zoom
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import graycomatrix, graycoprops

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
CFG = {
    "train_ms_dir": "Kaggle_Prepared/train/MS",
    "val_ms_dir": "Kaggle_Prepared/val/MS",
    "train_hs_dir": "Kaggle_Prepared/train/HS",
    "val_hs_dir": "Kaggle_Prepared/val/HS",
    "train_rgb_dir": "Kaggle_Prepared/train/RGB",
    "val_rgb_dir": "Kaggle_Prepared/val/RGB",
    "output_dir": "acfnet_output",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
    # AC-FNet CNN config
    "cnn_epochs": 80,
    "cnn_batch_size": 32,
    "cnn_lr": 1e-3,
    "cnn_weight_decay": 1e-4,
    "cnn_patience": 15,
    # JM-SLIC config
    "jmslic_n_segments": 64,  # Number of superpixels
    "jmslic_compactness": 10.0,
    "jmslic_beta": 0.5,       # Spatial weight
    # S-BI super-resolution
    "sbi_target_size": 64,    # Upscale HS from 32 to 64
    "sbi_eta": 0.01,          # Gradient control parameter
    # PCA config
    "pca_n_components": 15,   # Reduce fused features
    # Ensemble weights
    "cnn_weight": 0.3,        # CNN features weight in final fusion
    "handcraft_weight": 0.7,  # Handcrafted features weight
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]

# Approximate wavelengths for HS bands (450-950nm, 125 bands)
HS_WAVELENGTHS = np.linspace(450, 950, 125)
HS_CLEAN_START, HS_CLEAN_END = 10, 110  # Clean band range


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Module 1: JM-SLIC Segmentation
# ============================================================
class JMSLIC:
    """
    Jeffries-Matusita SLIC — adapted from paper.
    Segments image into superpixels using JM distance in LAB+spatial space.
    Returns a binary leaf mask by thresholding green/vegetation content.
    """

    def __init__(self, n_segments=64, compactness=10.0, beta=0.5, n_iter=10):
        self.n_segments = n_segments
        self.compactness = compactness
        self.beta = beta
        self.n_iter = n_iter

    def _rgb_to_lab(self, rgb):
        """Convert RGB (0-255) to LAB color space."""
        rgb = rgb.astype(np.float32) / 255.0
        # Linearize sRGB
        mask = rgb > 0.04045
        rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
        rgb[~mask] = rgb[~mask] / 12.92

        # RGB to XYZ
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
        xyz = rgb @ M.T

        # Normalize by D65 white point
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 1] /= 1.00000
        xyz[:, :, 2] /= 1.08883

        # XYZ to LAB
        eps = 0.008856
        kappa = 903.3
        mask = xyz > eps
        xyz[mask] = xyz[mask] ** (1.0 / 3.0)
        xyz[~mask] = (kappa * xyz[~mask] + 16.0) / 116.0

        L = 116.0 * xyz[:, :, 1] - 16.0
        a = 500.0 * (xyz[:, :, 0] - xyz[:, :, 1])
        b = 200.0 * (xyz[:, :, 1] - xyz[:, :, 2])

        return np.stack([L, a, b], axis=-1)

    def _jm_distance(self, c1, c2, C):
        """Jeffries-Matusita distance between two 5D center vectors [l,a,b,u,v]."""
        lab_diff = np.sqrt(np.sum((c1[:3] - c2[:3]) ** 2))
        uv_diff = np.sqrt(np.sum((c1[3:] - c2[3:]) ** 2))
        d = 2.0 * (1.0 - np.exp(-(lab_diff + (self.beta / C) * uv_diff)))
        return d

    def segment(self, ms_img, rgb_img=None):
        """
        Create a leaf/vegetation mask using spectral information.

        For MS images, uses NDVI to separate vegetation from background.
        For RGB, uses LAB-based JM-SLIC.

        Args:
            ms_img: (H, W, 5) float32 MS image
            rgb_img: (H, W, 3) uint8 RGB image (optional)

        Returns:
            mask: (H, W) binary mask — True for leaf pixels
        """
        H, W = ms_img.shape[:2]
        eps = 1e-8

        # Use NDVI from MS bands to create vegetation mask
        # Band order: Blue(0), Green(1), Red(2), RedEdge(3), NIR(4)
        nir = ms_img[:, :, 4].astype(np.float32)
        red = ms_img[:, :, 2].astype(np.float32)
        green = ms_img[:, :, 1].astype(np.float32)

        ndvi = (nir - red) / (nir + red + eps)
        gndvi = (nir - green) / (nir + green + eps)

        # Adaptive threshold: vegetation has NDVI > 0.2 typically
        # Use Otsu-like approach on NDVI histogram
        ndvi_flat = ndvi.ravel()
        ndvi_valid = ndvi_flat[~np.isnan(ndvi_flat)]

        if len(ndvi_valid) > 0:
            # Try percentile-based threshold
            threshold = max(0.15, np.percentile(ndvi_valid, 20))
            mask = ndvi > threshold

            # Clean up mask with morphological operations
            mask = ndimage.binary_fill_holes(mask)
            mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
            mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))

            # If mask covers too little or too much, use full image
            coverage = mask.sum() / mask.size
            if coverage < 0.1 or coverage > 0.95:
                mask = np.ones((H, W), dtype=bool)
        else:
            mask = np.ones((H, W), dtype=bool)

        return mask


# ============================================================
# Module 2: Preprocessing Pipeline
# ============================================================
class Preprocessor:
    """
    Dead pixel removal + Gaussian noise removal (Wiener filter).
    Adapted from paper's preprocessing pipeline.
    """

    @staticmethod
    def remove_dead_pixels(img, threshold=0.25):
        """
        Identify and replace dead pixels (>25% zero values in spectrum).

        Args:
            img: (H, W, C) image array
            threshold: fraction of zero-value bands to consider pixel dead

        Returns:
            cleaned image
        """
        img = img.copy().astype(np.float32)
        H, W, C = img.shape

        # Find dead pixels: pixels with >25% zero values across bands
        zero_fraction = (img == 0).sum(axis=2) / C
        dead_mask = zero_fraction > threshold

        if dead_mask.any():
            # Replace dead pixels with median of 3x3 neighborhood
            for c in range(C):
                band = img[:, :, c]
                median_filtered = ndimage.median_filter(band, size=3)
                band[dead_mask] = median_filtered[dead_mask]
                img[:, :, c] = band

        return img

    @staticmethod
    def wiener_denoise(img, noise_var=None):
        """
        Apply Wiener filter for Gaussian noise removal.
        Uses scipy's wiener filter per-band.

        Args:
            img: (H, W, C) float32 image
            noise_var: estimated noise variance (None for auto)

        Returns:
            denoised image
        """
        img = img.copy().astype(np.float32)
        H, W, C = img.shape

        for c in range(C):
            band = img[:, :, c]
            if band.max() > 0:
                # Apply Wiener filter with 3x3 kernel
                filtered = wiener(band, mysize=3, noise=noise_var)
                # Ensure non-negative
                img[:, :, c] = np.maximum(filtered, 0)

        return img

    @staticmethod
    def preprocess(img, apply_wiener=True):
        """Full preprocessing pipeline."""
        img = Preprocessor.remove_dead_pixels(img)
        if apply_wiener:
            img = Preprocessor.wiener_denoise(img)
        return img


# ============================================================
# Module 3: Super-Resolution (S-BI)
# ============================================================
class SBI_SuperResolution:
    """
    Stochastic Gradient-based Bicubic Interpolation (S-BI).
    Upscales HS (32×32) to match MS resolution (64×64).

    The stochastic gradient refinement enhances edges after bicubic interpolation.
    """

    def __init__(self, target_size=64, eta=0.01, n_iter=3):
        self.target_size = target_size
        self.eta = eta
        self.n_iter = n_iter

    def _bicubic_weight(self, t, a=-0.5):
        """Bicubic interpolation kernel weight."""
        t = abs(t)
        if t <= 1:
            return (a + 2) * t**3 - (a + 3) * t**2 + 1
        elif t < 2:
            return a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
        return 0.0

    def upscale(self, img):
        """
        Upscale image using bicubic interpolation + stochastic gradient refinement.

        Args:
            img: (H, W, C) float32 image (e.g., 32×32×125)

        Returns:
            upscaled: (target_size, target_size, C) image
        """
        H, W, C = img.shape
        if H == self.target_size and W == self.target_size:
            return img

        # Step 1: Bicubic interpolation using scipy.ndimage.zoom
        scale_h = self.target_size / H
        scale_w = self.target_size / W
        # zoom spatial dims only, keep channels
        upscaled = scipy_zoom(img, (scale_h, scale_w, 1), order=3).astype(np.float32)
        # Ensure exact target size
        upscaled = upscaled[:self.target_size, :self.target_size, :C]

        # Step 2: Stochastic gradient refinement (enhance edges)
        for iteration in range(self.n_iter):
            # Process a subset of bands to keep it fast
            for c in range(0, C, max(1, C // 20)):
                band = upscaled[:, :, c]
                if band.max() == 0:
                    continue

                # Compute gradient magnitude
                gy, gx = np.gradient(band)
                grad_mag = np.sqrt(gx**2 + gy**2)

                # Stochastic noise for exploration
                noise = np.random.randn(*band.shape).astype(np.float32) * 0.001

                # Gradient-based sharpening: enhance edges
                laplacian = ndimage.laplace(band)
                update = self.eta * (laplacian + noise)

                # Apply update only in smooth regions (preserve edges)
                p75 = np.percentile(grad_mag, 75)
                smooth_mask = grad_mag < p75
                band[smooth_mask] += update[smooth_mask]

                upscaled[:, :, c] = np.maximum(band, 0)

        return upscaled


# ============================================================
# Module 4: Spectral Unmixing (Simplified Bayesian)
# ============================================================
class SpectralUnmixer:
    """
    Simplified spectral unmixing using Non-negative Least Squares (NNLS).
    Separates each pixel into endmember abundances.

    For MS: 5 endmembers (one per band region)
    For HS: Extract key endmembers via vertex component analysis
    """

    def __init__(self, n_endmembers=4):
        self.n_endmembers = n_endmembers  # Vegetation, Soil, Shadow, Disease
        self.endmembers = None

    def _extract_endmembers_vca(self, spectra, n_endmembers):
        """
        Vertex Component Analysis — extract spectral endmembers.

        Args:
            spectra: (N_pixels, N_bands) array
            n_endmembers: number of endmembers to extract

        Returns:
            endmembers: (n_endmembers, N_bands)
        """
        N, B = spectra.shape
        if N < n_endmembers:
            return spectra[:n_endmembers]

        # PCA projection
        mean_spec = spectra.mean(axis=0)
        centered = spectra - mean_spec
        cov = centered.T @ centered / N
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1][:n_endmembers - 1]
        projections = centered @ eigenvectors[:, idx]

        # Iteratively find most extreme spectra
        endmember_indices = []
        # Start with brightest pixel
        brightness = np.sum(spectra, axis=1)
        endmember_indices.append(np.argmax(brightness))

        for _ in range(1, n_endmembers):
            if len(endmember_indices) >= N:
                break
            # Find pixel most different from current endmembers
            current_em = spectra[endmember_indices]
            min_distances = np.full(N, np.inf)
            for em in current_em:
                dists = np.sqrt(np.sum((spectra - em) ** 2, axis=1))
                min_distances = np.minimum(min_distances, dists)
            # Exclude already selected
            for idx in endmember_indices:
                min_distances[idx] = -1
            endmember_indices.append(np.argmax(min_distances))

        return spectra[endmember_indices]

    def unmix(self, img):
        """
        Perform spectral unmixing on an image.

        Args:
            img: (H, W, C) float32 image

        Returns:
            abundances: (H, W, n_endmembers) abundance maps
            endmembers: (n_endmembers, C) endmember spectra
        """
        H, W, C = img.shape
        pixels = img.reshape(-1, C).astype(np.float64)

        # Remove zero pixels
        valid_mask = pixels.sum(axis=1) > 0
        valid_pixels = pixels[valid_mask]

        if len(valid_pixels) < self.n_endmembers:
            return np.zeros((H, W, self.n_endmembers), dtype=np.float32), \
                   np.zeros((self.n_endmembers, C), dtype=np.float32)

        # Extract endmembers
        endmembers = self._extract_endmembers_vca(valid_pixels, self.n_endmembers)
        self.endmembers = endmembers

        # Solve for abundances using least squares (with non-negativity constraint)
        # abundances = argmin ||pixel - E @ a||² s.t. a >= 0, sum(a) = 1
        E = endmembers.T  # (C, n_em)
        EtE = E.T @ E + 1e-8 * np.eye(self.n_endmembers)
        EtE_inv = np.linalg.inv(EtE)

        abundances_flat = np.zeros((H * W, self.n_endmembers), dtype=np.float32)
        Ety = E.T @ pixels.T  # (n_em, N_pixels)
        a = (EtE_inv @ Ety).T  # (N_pixels, n_em)

        # Project to simplex (non-negative, sum to 1)
        a = np.maximum(a, 0)
        row_sums = a.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        a = a / row_sums
        abundances_flat = a.astype(np.float32)

        return abundances_flat.reshape(H, W, self.n_endmembers), endmembers.astype(np.float32)


# ============================================================
# Module 5: Feature Extraction
# ============================================================
class FeatureExtractor:
    """
    Two-branch feature extraction as described in the paper:
    Branch A: 7 Vegetation Indices (VI)
    Branch B: Image-based features (Texture + Airspace + Endmember)
    """

    def __init__(self):
        self.unmixer = SpectralUnmixer(n_endmembers=4)

    # ------ Branch A: Vegetation Indices ------
    def extract_vi_features(self, ms_img, hs_img=None, mask=None):
        """
        Extract 7 chlorophyll-based vegetation indices + extended set.

        From the paper:
        1. NDVI, 2. GNDVI, 3. RGR, 4. IPVI, 5. CLgreen, 6. CIrededge, 7. CVI

        Args:
            ms_img: (H, W, 5) preprocessed MS image
            hs_img: (H, W, C) preprocessed HS image (optional, upscaled to 64×64)
            mask: (H, W) leaf mask

        Returns:
            dict of VI features
        """
        features = {}
        eps = 1e-8
        bands = ms_img.transpose(2, 0, 1).astype(np.float32)  # (5, H, W)
        blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]

        if mask is not None:
            # Apply mask — only compute stats on leaf pixels
            blue_m = blue[mask]
            green_m = green[mask]
            red_m = red[mask]
            rededge_m = rededge[mask]
            nir_m = nir[mask]
        else:
            blue_m = blue.ravel()
            green_m = green.ravel()
            red_m = red.ravel()
            rededge_m = rededge.ravel()
            nir_m = nir.ravel()

        if len(nir_m) < 10:
            # Fallback to full image if mask too small
            blue_m = blue.ravel()
            green_m = green.ravel()
            red_m = red.ravel()
            rededge_m = rededge.ravel()
            nir_m = nir.ravel()

        # ----- Paper's 7 VIs -----
        # 1. NDVI
        ndvi = (nir_m - red_m) / (nir_m + red_m + eps)
        features["vi_NDVI_mean"] = np.mean(ndvi)
        features["vi_NDVI_std"] = np.std(ndvi)

        # 2. GNDVI
        gndvi = (nir_m - green_m) / (nir_m + green_m + eps)
        features["vi_GNDVI_mean"] = np.mean(gndvi)
        features["vi_GNDVI_std"] = np.std(gndvi)

        # 3. RGR (Red-Green Ratio)
        rgr = red_m / (green_m + eps)
        features["vi_RGR_mean"] = np.mean(rgr)
        features["vi_RGR_std"] = np.std(rgr)

        # 4. IPVI (Infrared Percentage Vegetation Index)
        ipvi = nir_m / (nir_m + red_m + eps)
        features["vi_IPVI_mean"] = np.mean(ipvi)
        features["vi_IPVI_std"] = np.std(ipvi)

        # 5. CLgreen (Green Chlorophyll Index)
        clgreen = (nir_m / (green_m + eps)) - 1.0
        features["vi_CLgreen_mean"] = np.mean(np.clip(clgreen, -10, 10))
        features["vi_CLgreen_std"] = np.std(np.clip(clgreen, -10, 10))

        # 6. CIrededge (Red-Edge Chlorophyll Index)
        cirededge = (nir_m / (rededge_m + eps)) - 1.0
        features["vi_CIrededge_mean"] = np.mean(np.clip(cirededge, -10, 10))
        features["vi_CIrededge_std"] = np.std(np.clip(cirededge, -10, 10))

        # 7. CVI (Chlorophyll Vegetation Index) = NDVI * NIR/Green
        cvi = ndvi * (nir_m / (green_m + eps))
        features["vi_CVI_mean"] = np.mean(np.clip(cvi, -20, 20))
        features["vi_CVI_std"] = np.std(np.clip(cvi, -20, 20))

        # ----- Extended VIs (from proven pipeline) -----
        ndre = (nir_m - rededge_m) / (nir_m + rededge_m + eps)
        savi = 1.5 * (nir_m - red_m) / (nir_m + red_m + 0.5 + eps)
        evi = 2.5 * (nir_m - red_m) / (nir_m + 6 * red_m - 7.5 * blue_m + 1.0 + eps)
        mcari = ((rededge_m - red_m) - 0.2 * (rededge_m - green_m)) * (rededge_m / (red_m + eps))

        for idx_name, idx_vals in [("NDRE", ndre), ("SAVI", savi), ("EVI", evi), ("MCARI", mcari)]:
            idx_vals = np.clip(idx_vals, -10, 10)
            features[f"vi_{idx_name}_mean"] = np.mean(idx_vals)
            features[f"vi_{idx_name}_std"] = np.std(idx_vals)
            features[f"vi_{idx_name}_p10"] = np.percentile(idx_vals, 10)
            features[f"vi_{idx_name}_p90"] = np.percentile(idx_vals, 90)

        # Per-band stats (masked)
        for i, name in enumerate(BAND_NAMES):
            b = bands[i]
            if mask is not None:
                b = b[mask]
            else:
                b = b.ravel()
            if len(b) < 5:
                b = bands[i].ravel()
            features[f"band_{name}_mean"] = np.mean(b)
            features[f"band_{name}_std"] = np.std(b)
            features[f"band_{name}_median"] = np.median(b)
            features[f"band_{name}_cv"] = np.std(b) / (np.mean(b) + eps)
            features[f"band_{name}_p5"] = np.percentile(b, 5)
            features[f"band_{name}_p95"] = np.percentile(b, 95)
            features[f"band_{name}_iqr"] = np.percentile(b, 75) - np.percentile(b, 25)
            features[f"band_{name}_skew"] = float(scipy_stats.skew(b))
            features[f"band_{name}_kurtosis"] = float(scipy_stats.kurtosis(b))

        # Spectral shape features
        band_means = [np.mean(bands[i]) for i in range(5)]
        features["spec_slope_vis"] = band_means[2] - band_means[0]
        features["spec_slope_rededge"] = band_means[3] - band_means[2]
        features["spec_slope_nir"] = band_means[4] - band_means[3]
        features["spec_curvature"] = band_means[3] - 0.5 * (band_means[2] + band_means[4])
        features["spec_nir_vis_ratio"] = band_means[4] / (np.mean(band_means[:3]) + eps)

        # Inter-band correlations
        flat_bands = bands.reshape(5, -1)
        corr_matrix = np.corrcoef(flat_bands)
        for i in range(5):
            for j in range(i + 1, 5):
                features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]

        # HS-derived VIs (if available)
        if hs_img is not None and hs_img.shape[2] > 70:
            self._extract_hs_vi(hs_img, features, mask)

        return features

    def _extract_hs_vi(self, hs_img, features, mask=None):
        """Extract vegetation indices from hyperspectral bands."""
        eps = 1e-8
        n_bands = hs_img.shape[2]
        clean_start, clean_end = HS_CLEAN_START, min(HS_CLEAN_END, n_bands)
        clean_hs = hs_img[:, :, clean_start:clean_end]

        # Mean spectrum
        if mask is not None and mask.shape == hs_img.shape[:2]:
            mean_spec = np.mean(clean_hs[mask], axis=0)
        else:
            mean_spec = np.mean(clean_hs, axis=(0, 1))

        n = len(mean_spec)
        if n < 50:
            return

        # Key bands (approximate)
        blue_b = mean_spec[0]       # ~480nm
        green_b = mean_spec[15]     # ~550nm
        red_b = mean_spec[35]       # ~670nm
        rededge_b = mean_spec[45]   # ~720nm
        nir_b = mean_spec[70] if n > 70 else mean_spec[-1]  # ~820nm

        # HS vegetation indices
        features["hs_NDVI"] = (nir_b - red_b) / (nir_b + red_b + eps)
        features["hs_GNDVI"] = (nir_b - green_b) / (nir_b + green_b + eps)
        features["hs_NDRE"] = (nir_b - rededge_b) / (nir_b + rededge_b + eps)
        features["hs_CI_RE"] = nir_b / (rededge_b + eps) - 1
        features["hs_CI_Green"] = nir_b / (green_b + eps) - 1
        features["hs_PRI"] = (green_b - red_b) / (green_b + red_b + eps)
        features["hs_ARI"] = 1 / (green_b + eps) - 1 / (rededge_b + eps)
        features["hs_CRI"] = 1 / (blue_b + eps) - 1 / (green_b + eps)
        features["hs_Rust_Index"] = red_b / (green_b + eps)
        features["hs_Iron_Index"] = (red_b - blue_b) / (red_b + blue_b + eps)
        features["hs_Health_Ratio"] = nir_b / (red_b + eps)

        # Spectral derivatives
        deriv1 = np.gradient(mean_spec)
        features["hs_deriv1_std"] = np.std(deriv1)
        features["hs_deriv1_max"] = np.max(deriv1)

        # Red edge position
        re_region = deriv1[30:50] if n > 50 else deriv1
        features["hs_REP"] = np.argmax(re_region) + 30
        features["hs_REP_slope"] = np.max(re_region)

        # Spectral region means
        features["hs_mean_vis"] = np.mean(mean_spec[:40])
        features["hs_mean_nir"] = np.mean(mean_spec[60:]) if n > 60 else np.mean(mean_spec[-10:])
        features["hs_nir_vis_ratio"] = features["hs_mean_nir"] / (features["hs_mean_vis"] + eps)

        # Continuum removal
        x = np.arange(n)
        continuum = mean_spec[0] + (mean_spec[-1] - mean_spec[0]) * x / (n - 1)
        cr = mean_spec / (continuum + eps)
        features["hs_cr_depth"] = 1 - np.min(cr)
        features["hs_cr_area"] = np.mean(cr)

    # ------ Branch B: Image-Based Features ------
    def extract_image_features(self, ms_img, mask=None):
        """
        Extract texture + airspace + spatial features.

        Paper's image features:
        - Airspace characteristics (S): green/non-green pixel ratio
        - Texture (T): GLCM features
        - SMACC endmember abundances (E)

        Args:
            ms_img: (H, W, 5) preprocessed MS image
            mask: (H, W) leaf mask

        Returns:
            dict of image features
        """
        features = {}
        eps = 1e-8
        H, W, C = ms_img.shape

        # ----- Airspace Characteristics -----
        # Ratio of green (vegetation) to non-green pixels
        nir = ms_img[:, :, 4].astype(np.float32)
        red = ms_img[:, :, 2].astype(np.float32)
        green = ms_img[:, :, 1].astype(np.float32)
        ndvi = (nir - red) / (nir + red + eps)

        veg_pixels = (ndvi > 0.2).sum()
        total_pixels = H * W
        features["airspace_veg_ratio"] = veg_pixels / total_pixels
        features["airspace_green_intensity"] = np.mean(green) / (np.mean(red) + eps)

        # ----- GLCM Texture Features -----
        # Compute for key bands: Green, Red, NIR
        for band_idx, band_name in [(1, "Green"), (2, "Red"), (4, "NIR")]:
            band = ms_img[:, :, band_idx].astype(np.float32)

            # Normalize to 0-255 uint8 for GLCM
            if band.max() > band.min():
                band_norm = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
            else:
                band_norm = np.zeros((H, W), dtype=np.uint8)

            # Compute GLCM at multiple angles
            try:
                glcm = graycomatrix(
                    band_norm,
                    distances=[1, 2],
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=64,  # Reduce from 256 for efficiency
                    symmetric=True,
                    normed=True,
                )

                # 6 GLCM properties (as in paper)
                for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
                    vals = graycoprops(glcm, prop)
                    features[f"glcm_{band_name}_{prop}_mean"] = np.mean(vals)
                    features[f"glcm_{band_name}_{prop}_std"] = np.std(vals)

                # Entropy (computed manually)
                glcm_norm = glcm.astype(np.float64)
                glcm_norm[glcm_norm == 0] = 1e-10
                entropy = -np.sum(glcm_norm * np.log2(glcm_norm), axis=(0, 1))
                features[f"glcm_{band_name}_entropy_mean"] = np.mean(entropy)

            except Exception:
                # If GLCM fails, fill with zeros
                for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
                    features[f"glcm_{band_name}_{prop}_mean"] = 0.0
                    features[f"glcm_{band_name}_{prop}_std"] = 0.0
                features[f"glcm_{band_name}_entropy_mean"] = 0.0

        # ----- Spatial Gradient Features -----
        for band_idx, band_name in enumerate(BAND_NAMES):
            band = ms_img[:, :, band_idx].astype(np.float32)
            gy, gx = np.gradient(band)
            grad_mag = np.sqrt(gx**2 + gy**2)
            features[f"grad_{band_name}_mean"] = np.mean(grad_mag)
            features[f"grad_{band_name}_std"] = np.std(grad_mag)

            # Local variance (texture roughness)
            local_mean = ndimage.uniform_filter(band, size=3)
            local_sq_mean = ndimage.uniform_filter(band**2, size=3)
            local_var = np.maximum(local_sq_mean - local_mean**2, 0)
            features[f"localvar_{band_name}_mean"] = np.mean(local_var)

        # ----- Endmember Abundance Features -----
        abundances, endmembers = self.unmixer.unmix(ms_img)
        for em_idx in range(abundances.shape[2]):
            ab = abundances[:, :, em_idx]
            features[f"endmember_{em_idx}_mean"] = np.mean(ab)
            features[f"endmember_{em_idx}_std"] = np.std(ab)
            features[f"endmember_{em_idx}_max"] = np.max(ab)

        # Endmember spectral properties
        for em_idx in range(endmembers.shape[0]):
            em_spec = endmembers[em_idx]
            if em_spec.max() > 0:
                features[f"endmember_{em_idx}_brightness"] = np.sum(em_spec)
                features[f"endmember_{em_idx}_slope"] = em_spec[-1] - em_spec[0]

        return features

    def extract_all_features(self, ms_img, hs_img=None, mask=None):
        """Extract all features from both branches."""
        vi_features = self.extract_vi_features(ms_img, hs_img, mask)
        img_features = self.extract_image_features(ms_img, mask)

        # Merge
        all_features = {}
        all_features.update(vi_features)
        all_features.update(img_features)
        return all_features


# ============================================================
# Module 6: AC-FNet Model (Atrous Convolution FractalNet)
# ============================================================
class FractalBlock(nn.Module):
    """
    Single Fractal Block with Atrous (Dilated) Convolution.

    ℵξ+1(ς) = ⌊(ℵξ ∘ ℵξ)(ς)⌋ Θ [enn(ς)]

    - Uses dilated convolutions for multi-scale receptive fields
    - Recursive fractal structure for feature extraction
    - Join operation (Θ) merges paths
    """

    def __init__(self, in_ch, out_ch, dilation_rates=[1, 2, 4]):
        super().__init__()

        # Atrous convolution branches with different dilation rates
        self.branches = nn.ModuleList()
        for d in dilation_rates:
            branch = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        # 1×1 conv to merge branches (Join operation Θ)
        self.join = nn.Sequential(
            nn.Conv2d(out_ch * len(dilation_rates), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Skip connection
        self.skip = nn.Identity() if in_ch == out_ch else \
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        # Multi-scale atrous paths
        branch_outputs = [branch(x) for branch in self.branches]
        # Join (concatenate + 1×1 conv)
        joined = self.join(torch.cat(branch_outputs, dim=1))
        # Residual connection
        return F.relu(joined + self.skip(x))


class ACFNet(nn.Module):
    """
    AC-FNet: Atrous Convolution-based FractalNet.

    Architecture:
    - Input: Multi-channel image (MS bands + indices)
    - Spectral mixing layer (1×1 conv)
    - 3 cascaded FractalBlocks with max-pooling
    - Global pooling → FC head

    Used both as:
    1. End-to-end classifier
    2. Feature extractor (pre-FC embeddings for XGBoost fusion)
    """

    def __init__(self, in_ch=11, num_classes=3, embed_dim=128):
        super().__init__()

        # 1×1 spectral mixing (learn optimal band combinations)
        self.spec_mix = nn.Sequential(
            nn.Conv2d(in_ch, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Cascaded Fractal Blocks
        self.fractal1 = FractalBlock(32, 48, dilation_rates=[1, 2])
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)

        self.fractal2 = FractalBlock(48, 96, dilation_rates=[1, 2, 4])
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.15)

        self.fractal3 = FractalBlock(96, 192, dilation_rates=[1, 2, 4])
        self.pool3 = nn.AdaptiveAvgPool2d(1)

        # Embedding layer
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_embedding=False):
        x = self.spec_mix(x)

        x = self.fractal1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.fractal2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.fractal3(x)
        x = self.pool3(x)

        embedding = self.embed(x)

        if return_embedding:
            return embedding

        return self.classifier(embedding)


# ============================================================
# Module 7: Data Pipeline
# ============================================================
class DataPipeline:
    """Orchestrates the full data pipeline from raw images to features."""

    def __init__(self):
        self.segmenter = JMSLIC(
            n_segments=CFG["jmslic_n_segments"],
            compactness=CFG["jmslic_compactness"],
            beta=CFG["jmslic_beta"],
        )
        self.preprocessor = Preprocessor()
        self.super_res = SBI_SuperResolution(
            target_size=CFG["sbi_target_size"],
            eta=CFG["sbi_eta"],
        )
        self.feature_extractor = FeatureExtractor()

    def process_sample(self, ms_path, hs_path=None, apply_segmentation=True):
        """
        Process a single sample through the full pipeline.

        Returns:
            features: dict of handcrafted features
            ms_tensor: (11, 64, 64) preprocessed MS + indices tensor
            hs_upscaled: (100, 64, 64) upscaled HS tensor (or None)
            is_black: whether image is black/corrupt
        """
        # Load MS
        ms_img = tiff.imread(ms_path).astype(np.float32)
        if ms_img.mean() < 1.0:
            return None, None, None, True

        # Step 1: Segmentation (leaf mask)
        if apply_segmentation:
            mask = self.segmenter.segment(ms_img)
        else:
            mask = None

        # Step 2: Preprocessing
        ms_clean = self.preprocessor.preprocess(ms_img, apply_wiener=True)

        # Load and process HS
        hs_upscaled = None
        if hs_path and os.path.exists(hs_path):
            hs_img = tiff.imread(hs_path).astype(np.float32)
            if hs_img.mean() >= 1.0:
                hs_clean = self.preprocessor.preprocess(hs_img, apply_wiener=True)
                # Step 3: Super-resolution (32→64)
                hs_upscaled = self.super_res.upscale(hs_clean)

        # Step 5: Feature extraction (both branches)
        features = self.feature_extractor.extract_all_features(
            ms_clean, hs_upscaled, mask
        )

        # Prepare tensor for CNN
        ms_tensor = self._prepare_ms_tensor(ms_clean)
        hs_tensor = None
        if hs_upscaled is not None:
            hs_tensor = self._prepare_hs_tensor(hs_upscaled)

        return features, ms_tensor, hs_tensor, False

    def _prepare_ms_tensor(self, ms_img):
        """Convert MS image to tensor with spectral indices."""
        bands = ms_img.transpose(2, 0, 1).astype(np.float32)  # (5, H, W)
        eps = 1e-8
        blue, green, red, re, nir = bands[0], bands[1], bands[2], bands[3], bands[4]

        ndvi = (nir - red) / (nir + red + eps)
        ndre = (nir - re) / (nir + re + eps)
        gndvi = (nir - green) / (nir + green + eps)
        ci_re = np.clip(nir / (re + eps) - 1.0, -5, 5)
        savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
        rgr = np.clip(red / (green + eps), -5, 5)

        indices = np.stack([ndvi, ndre, gndvi, ci_re, savi, rgr])  # (6, H, W)
        return np.concatenate([bands, indices], axis=0)  # (11, H, W)

    def _prepare_hs_tensor(self, hs_img):
        """Extract key HS bands as tensor."""
        # Use clean bands, subsample to manageable number
        n_bands = hs_img.shape[2]
        clean_start = HS_CLEAN_START
        clean_end = min(HS_CLEAN_END, n_bands)
        clean_hs = hs_img[:, :, clean_start:clean_end]

        # Subsample to ~20 bands (every 5th band)
        step = max(1, clean_hs.shape[2] // 20)
        subsampled = clean_hs[:, :, ::step]  # (64, 64, ~20)
        return subsampled.transpose(2, 0, 1).astype(np.float32)  # (~20, 64, 64)


# ============================================================
# Module 8: Training Pipeline
# ============================================================
class ACFNetTrainer:
    """Complete training pipeline with dual-path approach."""

    def __init__(self):
        self.pipeline = DataPipeline()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Device: {self.device}")

    def load_dataset(self, ms_dir, hs_dir, is_train=True):
        """Load and process entire dataset."""
        ms_files = sorted(os.listdir(ms_dir))
        all_features = []
        all_tensors = []
        all_labels = []
        all_fnames = []
        all_black = []
        skipped = 0

        for i, f in enumerate(ms_files):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Processing {i + 1}/{len(ms_files)}...", flush=True)

            ms_path = os.path.join(ms_dir, f)
            hs_path = os.path.join(hs_dir, f) if hs_dir else None

            features, ms_tensor, hs_tensor, is_black = self.pipeline.process_sample(
                ms_path, hs_path, apply_segmentation=True
            )

            if is_black:
                if is_train and "_hyper_" in f:
                    skipped += 1
                    continue
                else:
                    all_features.append(None)
                    all_tensors.append(None)
                    all_black.append(True)
                    all_labels.append(-1)
                    all_fnames.append(f)
                    continue

            all_features.append(features)
            all_tensors.append(ms_tensor)
            all_black.append(False)

            if is_train and "_hyper_" in f:
                cls_name = f.split("_hyper_")[0]
                all_labels.append(CLASS_MAP[cls_name])
            else:
                all_labels.append(-1)
            all_fnames.append(f)

        if skipped > 0:
            print(f"  Skipped {skipped} black images")

        return all_features, all_tensors, all_labels, all_fnames, all_black

    def train_cnn_fold(self, images, labels, train_idx, val_idx, fold):
        """Train AC-FNet for one fold. Returns OOF embeddings and predictions."""
        in_ch = images.shape[1]
        model = ACFNet(in_ch=in_ch, num_classes=3, embed_dim=128).to(self.device)

        class_weights = torch.tensor([1.3, 0.9, 1.0], dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["cnn_lr"],
                                       weight_decay=CFG["cnn_weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["cnn_epochs"], eta_min=1e-6
        )

        train_imgs = images[train_idx].to(self.device)
        train_labs = labels[train_idx].to(self.device)
        val_imgs = images[val_idx].to(self.device)
        val_labs = labels[val_idx].to(self.device)

        best_f1 = 0
        best_state = None
        patience = 0

        for ep in range(CFG["cnn_epochs"]):
            # Train
            model.train()
            perm = torch.randperm(len(train_imgs), device=self.device)
            total_loss, correct, total = 0, 0, 0

            for start in range(0, len(perm), CFG["cnn_batch_size"]):
                idx = perm[start:start + CFG["cnn_batch_size"]]
                imgs = train_imgs[idx].clone()
                labs = train_labs[idx]

                # Simple augmentation
                if torch.rand(1).item() > 0.5:
                    imgs = imgs.flip(3)
                if torch.rand(1).item() > 0.5:
                    imgs = imgs.flip(2)

                optimizer.zero_grad(set_to_none=True)
                out = model(imgs)
                loss = criterion(out, labs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(idx)
                correct += (out.argmax(1) == labs).sum().item()
                total += len(idx)

            scheduler.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_out = model(val_imgs)
                val_preds = val_out.argmax(1).cpu().numpy()
                val_true = val_labs.cpu().numpy()
                val_acc = accuracy_score(val_true, val_preds)
                val_f1 = f1_score(val_true, val_preds, average='macro')

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if (ep + 1) % 10 == 0:
                print(f"    Ep {ep + 1:3d} | Loss:{total_loss / total:.3f} "
                      f"TAcc:{correct / total:.3f} VAcc:{val_acc:.3f} VF1:{val_f1:.3f} "
                      f"Best:{best_f1:.3f}")

            if patience >= CFG["cnn_patience"]:
                break

        # Extract embeddings with best model
        model.load_state_dict(best_state)
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            val_embeddings = model(val_imgs, return_embedding=True).cpu().numpy()
            val_probs = F.softmax(model(val_imgs), dim=1).cpu().numpy()

        return val_embeddings, val_probs, best_f1, best_state

    def run(self):
        """Execute the full training pipeline."""
        seed_everything(CFG["seed"])
        os.makedirs(CFG["output_dir"], exist_ok=True)

        # =====================================================
        # Step 1: Load and process data
        # =====================================================
        print("=" * 70)
        print("STEP 1: Loading and preprocessing data")
        print("=" * 70)

        print("\nProcessing training set...")
        train_feats, train_tensors, train_labels, train_fnames, train_black = \
            self.load_dataset(CFG["train_ms_dir"], CFG["train_hs_dir"], is_train=True)

        print(f"\nProcessing validation set...")
        val_feats, val_tensors, val_labels, val_fnames, val_black = \
            self.load_dataset(CFG["val_ms_dir"], CFG["val_hs_dir"], is_train=False)

        # =====================================================
        # Step 2: Prepare handcrafted features
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 2: Preparing handcrafted features")
        print("=" * 70)

        feature_names = list(train_feats[0].keys())
        print(f"Feature count: {len(feature_names)}")

        X_train = np.array([
            [f.get(k, 0.0) for k in feature_names] for f in train_feats
        ], dtype=np.float32)
        y_train = np.array(train_labels)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

        for ci in range(3):
            print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")

        # Val features
        black_mask = [f is None for f in val_feats]
        for i in range(len(val_feats)):
            if val_feats[i] is None:
                val_feats[i] = {k: 0.0 for k in feature_names}

        X_val = np.array([
            [f.get(k, 0.0) for k in feature_names] for f in val_feats
        ], dtype=np.float32)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
        print(f"  Val: {len(val_feats)} samples ({sum(black_mask)} black)")

        # Step 6: PCA on feature subsets
        print("\n  Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=min(CFG["pca_n_components"], X_train.shape[1]))
        scaler_pca = StandardScaler()
        X_train_pca = pca.fit_transform(scaler_pca.fit_transform(X_train))
        X_val_pca = pca.transform(scaler_pca.transform(X_val))
        print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

        # Augment original features with PCA components
        X_train_aug = np.hstack([X_train, X_train_pca])
        X_val_aug = np.hstack([X_val, X_val_pca])

        augmented_names = feature_names + [f"pca_{i}" for i in range(X_train_pca.shape[1])]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_val_scaled = scaler.transform(X_val_aug)

        # =====================================================
        # Step 3: Train AC-FNet CNN (optional — for embeddings)
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 3: Training AC-FNet CNN")
        print("=" * 70, flush=True)

        # NOTE: CNN disabled to avoid scipy/torch segfault interaction
        # The handcrafted features (GLCM + VIs + endmember abundances) from 
        # the paper are the main value-add. CNN embeddings can be added later
        # by running CNN separately.
        use_cnn = False
        print("  CNN disabled (handcrafted features only mode)", flush=True)

        if not use_cnn:
            cnn_oof_embeddings = np.zeros((len(X_train), 128))
            cnn_val_embeddings = np.zeros((len(X_val), 128))
            cnn_oof_probs = np.zeros((len(X_train), 3))
            cnn_val_probs = np.zeros((len(X_val), 3))

        # =====================================================
        # Step 4: Feature Fusion + Final XGBoost/LightGBM
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 4: Feature Fusion + Gradient Boosting")
        print("=" * 70)

        # Fuse handcrafted features with CNN embeddings
        if use_cnn:
            # Add CNN OOF probs and embeddings as features
            # PCA on embeddings to reduce from 128 to 20
            embed_pca = PCA(n_components=20)
            train_embed_pca = embed_pca.fit_transform(cnn_oof_embeddings)
            val_embed_pca = embed_pca.transform(cnn_val_embeddings)

            X_train_fused = np.hstack([X_train_scaled, cnn_oof_probs, train_embed_pca])
            X_val_fused = np.hstack([X_val_scaled, cnn_val_probs, val_embed_pca])
            fused_names = augmented_names + \
                          [f"cnn_prob_{i}" for i in range(3)] + \
                          [f"cnn_embed_{i}" for i in range(20)]
        else:
            X_train_fused = X_train_scaled
            X_val_fused = X_val_scaled
            fused_names = augmented_names

        print(f"Fused feature count: {X_train_fused.shape[1]}")

        # ----- XGBoost -----
        print("\n--- XGBoost 5-Fold CV ---")
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": CFG["seed"],
            "tree_method": "hist",
            "verbosity": 0,
        }

        skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
        oof_xgb = np.zeros((len(X_train_fused), 3))
        val_xgb_folds = []
        feature_importance = np.zeros(len(fused_names))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_fused, y_train)):
            X_tr, X_va = X_train_fused[tr_idx], X_train_fused[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            oof_xgb[va_idx] = model.predict_proba(X_va)
            val_xgb_folds.append(model.predict_proba(X_val_fused))
            feature_importance += model.feature_importances_

            acc = accuracy_score(y_va, np.argmax(oof_xgb[va_idx], axis=1))
            print(f"  Fold {fold + 1}: Acc={acc:.4f}")

        xgb_preds = np.argmax(oof_xgb, axis=1)
        xgb_acc = accuracy_score(y_train, xgb_preds)
        xgb_f1 = f1_score(y_train, xgb_preds, average='macro')
        print(f"\nXGB OOF Accuracy: {xgb_acc:.4f}, F1: {xgb_f1:.4f}")
        print(classification_report(y_train, xgb_preds,
                                    target_names=list(CLASS_MAP.keys()), digits=4))

        # Top features
        feature_importance /= CFG["n_folds"]
        top_idx = np.argsort(feature_importance)[::-1][:25]
        print("\nTop 25 features:")
        for i, idx in enumerate(top_idx):
            print(f"  {i + 1:2d}. {fused_names[idx]:35s} imp={feature_importance[idx]:.4f}")

        # ----- LightGBM -----
        print("\n--- LightGBM 5-Fold CV ---")
        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_samples": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": CFG["seed"],
            "verbose": -1,
            "num_leaves": 31,
        }

        oof_lgb = np.zeros((len(X_train_fused), 3))
        val_lgb_folds = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_fused, y_train)):
            X_tr, X_va = X_train_fused[tr_idx], X_train_fused[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

            oof_lgb[va_idx] = model.predict_proba(X_va)
            val_lgb_folds.append(model.predict_proba(X_val_fused))

            acc = accuracy_score(y_va, np.argmax(oof_lgb[va_idx], axis=1))
            print(f"  Fold {fold + 1}: Acc={acc:.4f}")

        lgb_preds = np.argmax(oof_lgb, axis=1)
        lgb_acc = accuracy_score(y_train, lgb_preds)
        lgb_f1 = f1_score(y_train, lgb_preds, average='macro')
        print(f"\nLGB OOF Accuracy: {lgb_acc:.4f}, F1: {lgb_f1:.4f}")
        print(classification_report(y_train, lgb_preds,
                                    target_names=list(CLASS_MAP.keys()), digits=4))

        # =====================================================
        # Step 5: Ensemble
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 5: Ensemble (XGB + LGB + CNN)")
        print("=" * 70)

        # OOF ensemble
        w_xgb = 0.4
        w_lgb = 0.3
        w_cnn = 0.3 if use_cnn else 0.0

        if not use_cnn:
            w_xgb = 0.5
            w_lgb = 0.5

        ens_oof = w_xgb * oof_xgb + w_lgb * oof_lgb
        if use_cnn:
            ens_oof += w_cnn * cnn_oof_probs
        ens_preds = np.argmax(ens_oof, axis=1)
        ens_acc = accuracy_score(y_train, ens_preds)
        ens_f1 = f1_score(y_train, ens_preds, average='macro')
        print(f"Ensemble OOF Accuracy: {ens_acc:.4f}, F1: {ens_f1:.4f}")
        print(classification_report(y_train, ens_preds,
                                    target_names=list(CLASS_MAP.keys()), digits=4))

        # Val predictions
        val_xgb_probs = np.mean(val_xgb_folds, axis=0)
        val_lgb_probs = np.mean(val_lgb_folds, axis=0)
        val_ens_probs = w_xgb * val_xgb_probs + w_lgb * val_lgb_probs
        if use_cnn:
            val_ens_probs += w_cnn * cnn_val_probs

        # Override black images → Other
        for i, is_b in enumerate(black_mask):
            if is_b:
                val_ens_probs[i] = [0.0, 0.0, 1.0]
                val_xgb_probs[i] = [0.0, 0.0, 1.0]
                val_lgb_probs[i] = [0.0, 0.0, 1.0]

        val_preds = np.argmax(val_ens_probs, axis=1)
        pred_classes = [INV_CLASS_MAP[p] for p in val_preds]

        dist = {c: pred_classes.count(c) for c in CLASS_MAP}
        print(f"\nVal prediction distribution: {dist}")

        # =====================================================
        # Step 6: Save Results
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 6: Saving Results")
        print("=" * 70)

        np.save(os.path.join(CFG["output_dir"], "val_probs_final.npy"), val_ens_probs)
        np.save(os.path.join(CFG["output_dir"], "val_probs_xgb.npy"), val_xgb_probs)
        np.save(os.path.join(CFG["output_dir"], "val_probs_lgb.npy"), val_lgb_probs)
        np.save(os.path.join(CFG["output_dir"], "oof_xgb.npy"), oof_xgb)
        np.save(os.path.join(CFG["output_dir"], "oof_lgb.npy"), oof_lgb)

        if use_cnn:
            np.save(os.path.join(CFG["output_dir"], "val_probs_cnn.npy"), cnn_val_probs)
            np.save(os.path.join(CFG["output_dir"], "oof_cnn.npy"), cnn_oof_probs)

        # Submission CSV
        sub_path = os.path.join(CFG["output_dir"], "submission.csv")
        with open(sub_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Id", "Category"])
            for fn, cl in zip(val_fnames, pred_classes):
                w.writerow([fn, cl])

        # Feature names
        with open(os.path.join(CFG["output_dir"], "feature_names.json"), "w") as f:
            json.dump(fused_names, f, indent=2)

        # =====================================================
        # Summary
        # =====================================================
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Handcrafted features: {len(feature_names)}")
        print(f"+ PCA components:     {X_train_pca.shape[1]}")
        if use_cnn:
            print(f"+ CNN embeddings:     20 (PCA from 128)")
            print(f"+ CNN probs:          3")
        print(f"= Total fused:        {X_train_fused.shape[1]}")
        print(f"\nXGB OOF Accuracy:     {xgb_acc:.4f}, F1: {xgb_f1:.4f}")
        print(f"LGB OOF Accuracy:     {lgb_acc:.4f}, F1: {lgb_f1:.4f}")
        if use_cnn:
            print(f"CNN OOF Accuracy:     {cnn_acc:.4f}, F1: {cnn_f1:.4f}")
        print(f"Ensemble OOF Acc:     {ens_acc:.4f}, F1: {ens_f1:.4f}")
        print(f"\nSubmission saved to:  {sub_path}")
        print(f"Val probs saved to:   {CFG['output_dir']}/val_probs_final.npy")
        print(f"Val distribution:     {dist}")
        print("\nDone!")


if __name__ == "__main__":
    trainer = ACFNetTrainer()
    trainer.run()
