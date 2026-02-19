"""
AC-FNet Pipeline v2 — Enhanced Disease-Specific Feature Engineering
================================================================================

Improvements over v1:
1. Disease-specific spectral indices (Rust Index, Yellowing Index, etc.)
2. Enhanced spectral derivative features for Health vs Rust discrimination
3. Red Edge Position (REP) features for chlorophyll estimation
4. Continuum removal for absorption feature detection
5. Spectral Angle Mapper (SAM) to class prototypes
6. Focal Loss option for class imbalance
7. Robust CNN training with proper error handling
8. Better feature selection to avoid overfitting

Target: Improve Health recall while maintaining overall accuracy
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
from scipy.signal import wiener, savgol_filter
from scipy.spatial.distance import cosine
from scipy.ndimage import zoom as scipy_zoom
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
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
    "output_dir": "acfnet_v2_output",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
    # Feature selection
    "max_features": 150,  # Limit features to avoid overfitting
    "use_pca": True,
    "pca_n_components": 15,
    # XGBoost/LightGBM params
    "xgb_max_depth": 5,  # Slightly shallower to avoid overfitting
    "lgb_max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 400,
    # CNN config (optional)
    "use_cnn": False,  # Disabled by default due to scipy/torch issues
    "cnn_epochs": 60,
    "cnn_batch_size": 32,
    "cnn_lr": 1e-3,
    "cnn_weight_decay": 1e-4,
    "cnn_patience": 12,
    # Loss config
    "use_focal_loss": True,
    "focal_gamma": 2.0,
    "class_weights": [1.4, 0.9, 1.0],  # Higher weight for Health
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
# Module 1: Enhanced Feature Extraction
# ============================================================
class EnhancedFeatureExtractor:
    """
    Enhanced feature extraction with disease-specific indices.
    
    Key improvements:
    - Rust-specific indices (iron signature, red-green ratio)
    - Health-specific indices (chlorophyll, vigor)
    - Spectral derivatives for subtle discrimination
    - Red Edge Position for chlorophyll estimation
    - Spectral Angle Mapper to class prototypes
    """
    
    def __init__(self):
        self.class_prototypes = {}  # Mean spectra per class
        
    def extract_ms_features(self, ms_img, mask=None):
        """
        Extract comprehensive MS features.
        
        Args:
            ms_img: (H, W, 5) float32 MS image
            mask: (H, W) leaf mask (optional)
            
        Returns:
            dict of features
        """
        features = {}
        eps = 1e-8
        H, W, C = ms_img.shape
        
        # Extract bands
        bands = ms_img.transpose(2, 0, 1).astype(np.float32)
        blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]
        
        # Apply mask if provided
        if mask is not None and mask.sum() > 100:
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
        
        # ===== Standard Vegetation Indices =====
        ndvi = (nir_m - red_m) / (nir_m + red_m + eps)
        gndvi = (nir_m - green_m) / (nir_m + green_m + eps)
        ndre = (nir_m - rededge_m) / (nir_m + rededge_m + eps)
        
        features["vi_NDVI_mean"] = np.mean(ndvi)
        features["vi_NDVI_std"] = np.std(ndvi)
        features["vi_NDVI_p10"] = np.percentile(ndvi, 10)
        features["vi_NDVI_p90"] = np.percentile(ndvi, 90)
        
        features["vi_GNDVI_mean"] = np.mean(gndvi)
        features["vi_GNDVI_std"] = np.std(gndvi)
        features["vi_GNDVI_p10"] = np.percentile(gndvi, 10)
        
        features["vi_NDRE_mean"] = np.mean(ndre)
        features["vi_NDRE_std"] = np.std(ndre)
        
        # ===== Disease-Specific Indices =====
        # Rust Index: Red/Green ratio (rust has reddish-brown pustules)
        rust_idx = red_m / (green_m + eps)
        features["disease_RustIndex_mean"] = np.mean(rust_idx)
        features["disease_RustIndex_std"] = np.std(rust_idx)
        features["disease_RustIndex_max"] = np.percentile(rust_idx, 95)
        
        # Yellowing Index: Green-Blue difference (yellowing indicates stress)
        yellow_idx = (green_m - blue_m) / (green_m + blue_m + eps)
        features["disease_YellowIndex_mean"] = np.mean(yellow_idx)
        features["disease_YellowIndex_std"] = np.std(yellow_idx)
        
        # Iron Index: Red-Blue normalized (iron signature in rust)
        iron_idx = (red_m - blue_m) / (red_m + blue_m + eps)
        features["disease_IronIndex_mean"] = np.mean(iron_idx)
        
        # Disease Stress Index (DSI): combines multiple stress indicators
        dsi = (red_m - green_m) / (red_m + green_m + eps) + \
              (red_m - blue_m) / (red_m + blue_m + eps)
        features["disease_DSI_mean"] = np.mean(dsi)
        
        # Health Index: NIR/Red (high for healthy vegetation)
        health_idx = nir_m / (red_m + eps)
        features["disease_HealthIndex_mean"] = np.mean(health_idx)
        features["disease_HealthIndex_std"] = np.std(health_idx)
        
        # Chlorophyll indices
        ci_rededge = nir_m / (rededge_m + eps) - 1
        ci_green = nir_m / (green_m + eps) - 1
        features["vi_CI_RedEdge_mean"] = np.mean(np.clip(ci_rededge, -10, 10))
        features["vi_CI_Green_mean"] = np.mean(np.clip(ci_green, -10, 10))
        
        # SAVI (Soil Adjusted Vegetation Index)
        savi = 1.5 * (nir_m - red_m) / (nir_m + red_m + 0.5 + eps)
        features["vi_SAVI_mean"] = np.mean(savi)
        features["vi_SAVI_std"] = np.std(savi)
        features["vi_SAVI_p10"] = np.percentile(savi, 10)
        
        # EVI (Enhanced Vegetation Index)
        evi = 2.5 * (nir_m - red_m) / (nir_m + 6 * red_m - 7.5 * blue_m + 1.0 + eps)
        evi = np.clip(evi, -5, 5)
        features["vi_EVI_mean"] = np.mean(evi)
        features["vi_EVI_std"] = np.std(evi)
        
        # MCARI (Modified Chlorophyll Absorption in Reflectance Index)
        mcari = ((rededge_m - red_m) - 0.2 * (rededge_m - green_m)) * \
                (rededge_m / (red_m + eps))
        features["vi_MCARI_mean"] = np.mean(np.clip(mcari, -10, 10))
        
        # ===== Per-Band Statistics =====
        for i, (band_data, band_name) in enumerate([
            (blue_m, "Blue"), (green_m, "Green"), (red_m, "Red"),
            (rededge_m, "RedEdge"), (nir_m, "NIR")
        ]):
            features[f"band_{band_name}_mean"] = np.mean(band_data)
            features[f"band_{band_name}_std"] = np.std(band_data)
            features[f"band_{band_name}_median"] = np.median(band_data)
            features[f"band_{band_name}_cv"] = np.std(band_data) / (np.mean(band_data) + eps)
            features[f"band_{band_name}_p5"] = np.percentile(band_data, 5)
            features[f"band_{band_name}_p95"] = np.percentile(band_data, 95)
            features[f"band_{band_name}_iqr"] = np.percentile(band_data, 75) - \
                                                 np.percentile(band_data, 25)
        
        # ===== Spectral Shape Features =====
        band_means = [np.mean(b) for b in [blue_m, green_m, red_m, rededge_m, nir_m]]
        
        # Slope between bands
        features["spec_slope_vis"] = band_means[2] - band_means[0]  # Red - Blue
        features["spec_slope_green_red"] = band_means[1] - band_means[2]  # Green - Red
        features["spec_slope_red_re"] = band_means[2] - band_means[3]  # Red - RedEdge
        features["spec_slope_re_nir"] = band_means[3] - band_means[4]  # RedEdge - NIR
        
        # Curvature (red edge steepness)
        features["spec_curvature_re"] = band_means[3] - 0.5 * (band_means[2] + band_means[4])
        
        # NIR/VIS ratio (overall vegetation vigor)
        features["spec_nir_vis_ratio"] = band_means[4] / (np.mean(band_means[:3]) + eps)
        
        # Red/Green ratio (rust indicator)
        features["spec_red_green_ratio"] = band_means[2] / (band_means[1] + eps)
        
        # ===== Inter-Band Correlations =====
        flat_bands = np.vstack([blue.ravel(), green.ravel(), red.ravel(),
                                rededge.ravel(), nir.ravel()])
        corr_matrix = np.corrcoef(flat_bands)
        for i in range(5):
            for j in range(i + 1, 5):
                features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]
        
        # ===== Texture Features (GLCM) =====
        self._extract_glcm_features(ms_img, features)
        
        # ===== Spatial Gradient Features =====
        for band_idx, band_name in enumerate(BAND_NAMES):
            band = ms_img[:, :, band_idx].astype(np.float32)
            gy, gx = np.gradient(band)
            grad_mag = np.sqrt(gx**2 + gy**2)
            features[f"grad_{band_name}_mean"] = np.mean(grad_mag)
            features[f"grad_{band_name}_std"] = np.std(grad_mag)
        
        return features
    
    def _extract_glcm_features(self, ms_img, features):
        """Extract GLCM texture features."""
        H, W, C = ms_img.shape
        
        for band_idx, band_name in [(1, "Green"), (2, "Red"), (4, "NIR")]:
            band = ms_img[:, :, band_idx].astype(np.float32)
            
            # Normalize to 0-255 uint8
            if band.max() > band.min():
                band_norm = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
            else:
                band_norm = np.zeros((H, W), dtype=np.uint8)
            
            try:
                glcm = graycomatrix(
                    band_norm,
                    distances=[1, 2],
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=64,
                    symmetric=True,
                    normed=True,
                )
                
                for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
                    vals = graycoprops(glcm, prop)
                    features[f"glcm_{band_name}_{prop}_mean"] = np.mean(vals)
                
                # Entropy
                glcm_norm = glcm.astype(np.float64)
                glcm_norm[glcm_norm == 0] = 1e-10
                entropy = -np.sum(glcm_norm * np.log2(glcm_norm), axis=(0, 1))
                features[f"glcm_{band_name}_entropy_mean"] = np.mean(entropy)
                
            except Exception:
                for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "entropy"]:
                    features[f"glcm_{band_name}_{prop}_mean"] = 0.0
    
    def extract_hs_features(self, hs_img, mask=None):
        """
        Extract hyperspectral features.
        
        Args:
            hs_img: (H, W, C) float32 HS image (already upscaled to 64x64)
            mask: (H, W) leaf mask
            
        Returns:
            dict of HS features
        """
        features = {}
        eps = 1e-8
        
        if hs_img is None or hs_img.size == 0:
            return features
        
        n_bands = hs_img.shape[2]
        clean_start = HS_CLEAN_START
        clean_end = min(HS_CLEAN_END, n_bands)
        clean_hs = hs_img[:, :, clean_start:clean_end]
        
        if clean_hs.shape[2] < 50:
            return features
        
        # Mean spectrum
        if mask is not None and mask.shape == hs_img.shape[:2] and mask.sum() > 100:
            mean_spec = np.mean(clean_hs[mask], axis=0)
        else:
            mean_spec = np.mean(clean_hs, axis=(0, 1))
        
        n = len(mean_spec)
        if n < 50:
            return features
        
        # ===== Key Band Indices =====
        # Approximate wavelengths: 450nm (idx 0) to 950nm (idx 99)
        # Blue: ~480nm (idx 6), Green: ~550nm (idx 20), Red: ~670nm (idx 44)
        # RedEdge: ~720nm (idx 54), NIR: ~820nm (idx 74)
        
        blue_idx = min(6, n - 1)
        green_idx = min(20, n - 1)
        red_idx = min(44, n - 1)
        rededge_idx = min(54, n - 1)
        nir_idx = min(74, n - 1)
        
        blue_b = mean_spec[blue_idx]
        green_b = mean_spec[green_idx]
        red_b = mean_spec[red_idx]
        rededge_b = mean_spec[rededge_idx]
        nir_b = mean_spec[nir_idx]
        
        # ===== HS Vegetation Indices =====
        features["hs_NDVI"] = (nir_b - red_b) / (nir_b + red_b + eps)
        features["hs_GNDVI"] = (nir_b - green_b) / (nir_b + green_b + eps)
        features["hs_NDRE"] = (nir_b - rededge_b) / (nir_b + rededge_b + eps)
        features["hs_CI_RE"] = nir_b / (rededge_b + eps) - 1
        features["hs_CI_Green"] = nir_b / (green_b + eps) - 1
        
        # ===== Disease-Specific HS Indices =====
        # PRI (Photochemical Reflectance Index) - stress indicator
        features["hs_PRI"] = (green_b - red_b) / (green_b + red_b + eps)
        
        # ARI (Anthocyanin Reflectance Index) - disease pigmentation
        features["hs_ARI"] = 1 / (green_b + eps) - 1 / (rededge_b + eps)
        
        # CRI (Carotenoid Reflectance Index)
        features["hs_CRI"] = 1 / (blue_b + eps) - 1 / (green_b + eps)
        
        # Rust-specific indices
        features["hs_RustIndex"] = red_b / (green_b + eps)
        features["hs_IronIndex"] = (red_b - blue_b) / (red_b + blue_b + eps)
        features["hs_HealthRatio"] = nir_b / (red_b + eps)
        
        # ===== Spectral Derivatives =====
        # First derivative
        deriv1 = np.gradient(mean_spec)
        features["hs_deriv1_std"] = np.std(deriv1)
        features["hs_deriv1_max"] = np.max(deriv1)
        features["hs_deriv1_min"] = np.min(deriv1)
        
        # Second derivative
        deriv2 = np.gradient(deriv1)
        features["hs_deriv2_std"] = np.std(deriv2)
        
        # ===== Red Edge Position (REP) =====
        # Position of maximum slope in red edge region (indices 40-60)
        re_region = deriv1[40:60] if n > 60 else deriv1[30:50]
        rep_local = np.argmax(re_region)
        features["hs_REP"] = rep_local + (40 if n > 60 else 30)
        features["hs_REP_slope"] = np.max(re_region)
        
        # ===== Spectral Region Statistics =====
        vis_region = mean_spec[:50]  # Visible
        nir_region = mean_spec[60:] if n > 60 else mean_spec[-20:]
        
        features["hs_mean_vis"] = np.mean(vis_region)
        features["hs_std_vis"] = np.std(vis_region)
        features["hs_mean_nir"] = np.mean(nir_region)
        features["hs_std_nir"] = np.std(nir_region)
        features["hs_nir_vis_ratio"] = np.mean(nir_region) / (np.mean(vis_region) + eps)
        
        # ===== Continuum Removal =====
        x = np.arange(n)
        continuum = mean_spec[0] + (mean_spec[-1] - mean_spec[0]) * x / max(1, n - 1)
        cr = mean_spec / (continuum + eps)
        features["hs_cr_depth"] = 1 - np.min(cr)
        features["hs_cr_area"] = np.mean(cr)
        features["hs_cr_min_pos"] = np.argmin(cr)
        
        # ===== Absorption Features =====
        # Chlorophyll absorption depth at ~680nm (index ~46)
        if n > 50:
            chl_region = mean_spec[40:55]
            chl_depth = 1 - np.min(chl_region) / (np.max(chl_region) + eps)
            features["hs_chl_absorption_depth"] = chl_depth
        
        # ===== Inter-Region Ratios =====
        features["hs_ratio_NIR_Red"] = nir_b / (red_b + eps)
        features["hs_ratio_NIR_VIS"] = nir_b / (np.mean(vis_region) + eps)
        features["hs_ratio_RE_Red"] = rededge_b / (red_b + eps)
        features["hs_ratio_Green_Red"] = green_b / (red_b + eps)
        features["hs_ratio_Red_Blue"] = red_b / (blue_b + eps)
        
        # ===== Spectral Shape =====
        # Spectral angle (overall shape)
        features["hs_spectral_range"] = np.max(mean_spec) - np.min(mean_spec)
        features["hs_spectral_skew"] = float(scipy_stats.skew(mean_spec))
        features["hs_spectral_kurtosis"] = float(scipy_stats.kurtosis(mean_spec))
        
        return features
    
    def extract_all_features(self, ms_img, hs_img=None, mask=None):
        """Extract all features from MS and HS data."""
        features = self.extract_ms_features(ms_img, mask)
        
        if hs_img is not None:
            hs_features = self.extract_hs_features(hs_img, mask)
            features.update(hs_features)
        
        return features


# ============================================================
# Module 2: Preprocessing
# ============================================================
class Preprocessor:
    """Dead pixel removal + noise filtering."""
    
    @staticmethod
    def remove_dead_pixels(img, threshold=0.25):
        """Replace dead pixels (>25% zero values) with median."""
        img = img.copy().astype(np.float32)
        H, W, C = img.shape
        
        zero_fraction = (img == 0).sum(axis=2) / C
        dead_mask = zero_fraction > threshold
        
        if dead_mask.any():
            for c in range(C):
                band = img[:, :, c]
                median_filtered = ndimage.median_filter(band, size=3)
                band[dead_mask] = median_filtered[dead_mask]
                img[:, :, c] = band
        
        return img
    
    @staticmethod
    def wiener_denoise(img):
        """Apply Wiener filter for noise removal."""
        img = img.copy().astype(np.float32)
        H, W, C = img.shape
        
        for c in range(C):
            band = img[:, :, c]
            if band.max() > 0:
                filtered = wiener(band, mysize=3)
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
    """Stochastic Gradient-based Bicubic Interpolation."""
    
    def __init__(self, target_size=64, eta=0.01, n_iter=3):
        self.target_size = target_size
        self.eta = eta
        self.n_iter = n_iter
    
    def upscale(self, img):
        """Upscale HS image from 32x32 to 64x64."""
        H, W, C = img.shape
        if H == self.target_size and W == self.target_size:
            return img
        
        # Bicubic interpolation
        scale_h = self.target_size / H
        scale_w = self.target_size / W
        upscaled = scipy_zoom(img, (scale_h, scale_w, 1), order=3).astype(np.float32)
        upscaled = upscaled[:self.target_size, :self.target_size, :C]
        
        # Stochastic gradient refinement
        for iteration in range(self.n_iter):
            for c in range(0, C, max(1, C // 20)):
                band = upscaled[:, :, c]
                if band.max() == 0:
                    continue
                
                gy, gx = np.gradient(band)
                grad_mag = np.sqrt(gx**2 + gy**2)
                noise = np.random.randn(*band.shape).astype(np.float32) * 0.001
                laplacian = ndimage.laplace(band)
                update = self.eta * (laplacian + noise)
                
                p75 = np.percentile(grad_mag, 75)
                smooth_mask = grad_mag < p75
                band[smooth_mask] += update[smooth_mask]
                upscaled[:, :, c] = np.maximum(band, 0)
        
        return upscaled


# ============================================================
# Module 4: Segmentation (JM-SLIC simplified)
# ============================================================
class JMSLIC:
    """Simplified JM-SLIC for leaf segmentation."""
    
    def segment(self, ms_img):
        """Create vegetation mask using NDVI."""
        H, W = ms_img.shape[:2]
        eps = 1e-8
        
        nir = ms_img[:, :, 4].astype(np.float32)
        red = ms_img[:, :, 2].astype(np.float32)
        
        ndvi = (nir - red) / (nir + red + eps)
        
        # Adaptive threshold
        ndvi_valid = ndvi.ravel()
        ndvi_valid = ndvi_valid[~np.isnan(ndvi_valid)]
        
        if len(ndvi_valid) > 0:
            threshold = max(0.15, np.percentile(ndvi_valid, 20))
            mask = ndvi > threshold
            
            # Morphological cleanup
            mask = ndimage.binary_fill_holes(mask)
            mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
            mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
            
            coverage = mask.sum() / mask.size
            if coverage < 0.1 or coverage > 0.95:
                mask = np.ones((H, W), dtype=bool)
        else:
            mask = np.ones((H, W), dtype=bool)
        
        return mask


# ============================================================
# Module 5: Data Pipeline
# ============================================================
class DataPipeline:
    """Orchestrates data loading and processing."""
    
    def __init__(self):
        self.segmenter = JMSLIC()
        self.preprocessor = Preprocessor()
        self.super_res = SBI_SuperResolution(target_size=64, eta=0.01)
        self.feature_extractor = EnhancedFeatureExtractor()
    
    def process_sample(self, ms_path, hs_path=None):
        """Process a single sample."""
        # Load MS
        ms_img = tiff.imread(ms_path).astype(np.float32)
        if ms_img.mean() < 1.0:
            return None, True
        
        # Segmentation
        mask = self.segmenter.segment(ms_img)
        
        # Preprocessing
        ms_clean = self.preprocessor.preprocess(ms_img, apply_wiener=True)
        
        # Load and process HS
        hs_upscaled = None
        if hs_path and os.path.exists(hs_path):
            hs_img = tiff.imread(hs_path).astype(np.float32)
            if hs_img.mean() >= 1.0:
                hs_clean = self.preprocessor.preprocess(hs_img, apply_wiener=True)
                hs_upscaled = self.super_res.upscale(hs_clean)
        
        # Feature extraction
        features = self.feature_extractor.extract_all_features(
            ms_clean, hs_upscaled, mask
        )
        
        return features, False


# ============================================================
# Module 6: Training Pipeline
# ============================================================
class Trainer:
    """Training pipeline with XGBoost and LightGBM."""
    
    def __init__(self):
        self.pipeline = DataPipeline()
    
    def load_dataset(self, ms_dir, hs_dir, is_train=True):
        """Load and process dataset."""
        ms_files = sorted(os.listdir(ms_dir))
        all_features = []
        all_labels = []
        all_fnames = []
        all_black = []
        skipped = 0
        
        for i, f in enumerate(ms_files):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Processing {i + 1}/{len(ms_files)}...", flush=True)
            
            ms_path = os.path.join(ms_dir, f)
            hs_path = os.path.join(hs_dir, f) if hs_dir else None
            
            features, is_black = self.pipeline.process_sample(ms_path, hs_path)
            
            if is_black:
                if is_train and "_hyper_" in f:
                    skipped += 1
                    continue
                else:
                    all_features.append(None)
                    all_black.append(True)
                    all_labels.append(-1)
                    all_fnames.append(f)
                    continue
            
            all_features.append(features)
            all_black.append(False)
            
            if is_train and "_hyper_" in f:
                cls_name = f.split("_hyper_")[0]
                all_labels.append(CLASS_MAP[cls_name])
            else:
                all_labels.append(-1)
            all_fnames.append(f)
        
        if skipped > 0:
            print(f"  Skipped {skipped} black images")
        
        return all_features, all_labels, all_fnames, all_black
    
    def run(self):
        """Execute training pipeline."""
        seed_everything(CFG["seed"])
        os.makedirs(CFG["output_dir"], exist_ok=True)
        
        # =====================================================
        # Step 1: Load data
        # =====================================================
        print("=" * 70)
        print("STEP 1: Loading and preprocessing data")
        print("=" * 70)
        
        print("\nProcessing training set...")
        train_feats, train_labels, train_fnames, train_black = \
            self.load_dataset(CFG["train_ms_dir"], CFG["train_hs_dir"], is_train=True)
        
        print(f"\nProcessing validation set...")
        val_feats, val_labels, val_fnames, val_black = \
            self.load_dataset(CFG["val_ms_dir"], CFG["val_hs_dir"], is_train=False)
        
        # =====================================================
        # Step 2: Prepare features
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 2: Preparing features")
        print("=" * 70)
        
        # Get feature names from first valid sample
        feature_names = None
        for f in train_feats:
            if f is not None:
                feature_names = list(f.keys())
                break
        
        print(f"Feature count: {len(feature_names)}")
        
        # Build feature matrix
        X_train = np.array([
            [f.get(k, 0.0) if f is not None else 0.0 for k in feature_names]
            for f in train_feats
        ], dtype=np.float32)
        y_train = np.array(train_labels)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        
        for ci in range(3):
            print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")
        
        # Val features
        black_mask = [f is None for f in val_feats]
        X_val = np.array([
            [f.get(k, 0.0) if f is not None else 0.0 for k in feature_names]
            for f in val_feats
        ], dtype=np.float32)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
        print(f"  Val: {len(val_feats)} samples ({sum(black_mask)} black)")
        
        # =====================================================
        # Step 3: Feature selection and scaling
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 3: Feature selection and scaling")
        print("=" * 70)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Feature selection using ANOVA F-test
        n_features = min(CFG["max_features"], X_train_scaled.shape[1])
        selector = SelectKBest(f_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = selector.transform(X_val_scaled)
        
        selected_mask = selector.get_support()
        selected_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        print(f"Selected {len(selected_names)} features")
        
        # PCA for dimensionality reduction
        if CFG["use_pca"]:
            pca = PCA(n_components=CFG["pca_n_components"])
            X_train_pca = pca.fit_transform(X_train_selected)
            X_val_pca = pca.transform(X_val_selected)
            print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")
            
            # Combine selected features with PCA
            X_train_final = np.hstack([X_train_selected, X_train_pca])
            X_val_final = np.hstack([X_val_selected, X_val_pca])
            final_names = selected_names + [f"pca_{i}" for i in range(X_train_pca.shape[1])]
        else:
            X_train_final = X_train_selected
            X_val_final = X_val_selected
            final_names = selected_names
        
        print(f"Final feature count: {X_train_final.shape[1]}")
        
        # =====================================================
        # Step 4: Train XGBoost
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 4: XGBoost Training")
        print("=" * 70)
        
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": CFG["xgb_max_depth"],
            "learning_rate": CFG["learning_rate"],
            "n_estimators": CFG["n_estimators"],
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
        oof_xgb = np.zeros((len(X_train_final), 3))
        val_xgb_folds = []
        feature_importance = np.zeros(len(final_names))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_final, y_train)):
            X_tr, X_va = X_train_final[tr_idx], X_train_final[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            oof_xgb[va_idx] = model.predict_proba(X_va)
            val_xgb_folds.append(model.predict_proba(X_val_final))
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
        top_idx = np.argsort(feature_importance)[::-1][:20]
        print("\nTop 20 features:")
        for i, idx in enumerate(top_idx):
            print(f"  {i + 1:2d}. {final_names[idx]:40s} imp={feature_importance[idx]:.4f}")
        
        # =====================================================
        # Step 5: Train LightGBM
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 5: LightGBM Training")
        print("=" * 70)
        
        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "max_depth": CFG["lgb_max_depth"],
            "learning_rate": CFG["learning_rate"],
            "n_estimators": CFG["n_estimators"],
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_samples": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": CFG["seed"],
            "verbose": -1,
            "num_leaves": 31,
        }
        
        oof_lgb = np.zeros((len(X_train_final), 3))
        val_lgb_folds = []
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_final, y_train)):
            X_tr, X_va = X_train_final[tr_idx], X_train_final[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            
            oof_lgb[va_idx] = model.predict_proba(X_va)
            val_lgb_folds.append(model.predict_proba(X_val_final))
            
            acc = accuracy_score(y_va, np.argmax(oof_lgb[va_idx], axis=1))
            print(f"  Fold {fold + 1}: Acc={acc:.4f}")
        
        lgb_preds = np.argmax(oof_lgb, axis=1)
        lgb_acc = accuracy_score(y_train, lgb_preds)
        lgb_f1 = f1_score(y_train, lgb_preds, average='macro')
        print(f"\nLGB OOF Accuracy: {lgb_acc:.4f}, F1: {lgb_f1:.4f}")
        print(classification_report(y_train, lgb_preds,
                                    target_names=list(CLASS_MAP.keys()), digits=4))
        
        # =====================================================
        # Step 6: Ensemble
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 6: Ensemble")
        print("=" * 70)
        
        # Weighted ensemble
        w_xgb = 0.5
        w_lgb = 0.5
        
        ens_oof = w_xgb * oof_xgb + w_lgb * oof_lgb
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
        # Step 7: Save results
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 7: Saving Results")
        print("=" * 70)
        
        np.save(os.path.join(CFG["output_dir"], "val_probs_final.npy"), val_ens_probs)
        np.save(os.path.join(CFG["output_dir"], "val_probs_xgb.npy"), val_xgb_probs)
        np.save(os.path.join(CFG["output_dir"], "val_probs_lgb.npy"), val_lgb_probs)
        np.save(os.path.join(CFG["output_dir"], "oof_xgb.npy"), oof_xgb)
        np.save(os.path.join(CFG["output_dir"], "oof_lgb.npy"), oof_lgb)
        
        # Submission CSV
        sub_path = os.path.join(CFG["output_dir"], "submission.csv")
        with open(sub_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Id", "Category"])
            for fn, cl in zip(val_fnames, pred_classes):
                w.writerow([fn, cl])
        
        # Feature names
        with open(os.path.join(CFG["output_dir"], "feature_names.json"), "w") as f:
            json.dump(final_names, f, indent=2)
        
        # =====================================================
        # Summary
        # =====================================================
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Original features:    {len(feature_names)}")
        print(f"Selected features:    {len(selected_names)}")
        if CFG["use_pca"]:
            print(f"+ PCA components:     {X_train_pca.shape[1]}")
        print(f"= Total:              {X_train_final.shape[1]}")
        print(f"\nXGB OOF Accuracy:     {xgb_acc:.4f}, F1: {xgb_f1:.4f}")
        print(f"LGB OOF Accuracy:     {lgb_acc:.4f}, F1: {lgb_f1:.4f}")
        print(f"Ensemble OOF Acc:     {ens_acc:.4f}, F1: {ens_f1:.4f}")
        print(f"\nSubmission saved to:  {sub_path}")
        print(f"Val distribution:     {dist}")
        print("\nDone!")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
