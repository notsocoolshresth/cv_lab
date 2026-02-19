"""
AC-FNet Pipeline v3 - Optimized for Health vs Rust Discrimination
================================================================================

Key improvements:
1. Two-stage classification: Health vs (Rust+Other), then Rust vs Other
2. Class-weighted training with higher weight for Health
3. Best features from proven train_ms_xgb.py pipeline
4. Disease-specific spectral indices
5. Ensemble with confidence-based weighting

Target: Improve Health recall while maintaining overall accuracy
"""

import os
import csv
import json
import random
import warnings
import numpy as np
import tifffile as tiff
from scipy import ndimage, stats as scipy_stats
from scipy.signal import wiener
from scipy.ndimage import zoom as scipy_zoom
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
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
    "output_dir": "acfnet_v3_output",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
    # Two-stage approach
    "use_two_stage": True,
    # XGBoost params - optimized for small dataset
    "xgb_max_depth": 4,
    "lgb_max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 300,
    # Class weights for Health
    "health_weight": 1.5,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]

# Approximate wavelengths for HS bands (450-950nm, 125 bands)
HS_WAVELENGTHS = np.linspace(450, 950, 125)
HS_CLEAN_START, HS_CLEAN_END = 10, 110


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Feature Extraction - Proven from train_ms_xgb.py
# ============================================================
class FeatureExtractor:
    """
    Feature extraction based on proven pipeline from train_ms_xgb.py
    that achieved 70.5% CV accuracy.
    """
    
    def extract_ms_features(self, ms_img):
        """
        Extract proven MS features.
        
        Based on analysis, the most important features are:
        - GNDVI_mean (0.0445)
        - spec_nir_vis_ratio (0.0419)
        - GNDVI_p10 (0.0370)
        - CI_Green_mean (0.0220)
        - CI_RE_max (0.0189)
        """
        features = {}
        eps = 1e-8
        H, W, C = ms_img.shape
        
        # Extract bands
        bands = ms_img.transpose(2, 0, 1).astype(np.float32)
        blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]
        
        # Flatten for statistics
        blue_f = blue.ravel()
        green_f = green.ravel()
        red_f = red.ravel()
        rededge_f = rededge.ravel()
        nir_f = nir.ravel()
        
        # ===== Core Vegetation Indices (proven important) =====
        # NDVI
        ndvi = (nir_f - red_f) / (nir_f + red_f + eps)
        features["vi_NDVI_mean"] = np.mean(ndvi)
        features["vi_NDVI_std"] = np.std(ndvi)
        features["vi_NDVI_p10"] = np.percentile(ndvi, 10)
        features["vi_NDVI_p90"] = np.percentile(ndvi, 90)
        
        # GNDVI (most important feature!)
        gndvi = (nir_f - green_f) / (nir_f + green_f + eps)
        features["vi_GNDVI_mean"] = np.mean(gndvi)
        features["vi_GNDVI_std"] = np.std(gndvi)
        features["vi_GNDVI_p10"] = np.percentile(gndvi, 10)
        features["vi_GNDVI_p90"] = np.percentile(gndvi, 90)
        
        # NDRE
        ndre = (nir_f - rededge_f) / (nir_f + rededge_f + eps)
        features["vi_NDRE_mean"] = np.mean(ndre)
        features["vi_NDRE_std"] = np.std(ndre)
        features["vi_NDRE_p10"] = np.percentile(ndre, 10)
        features["vi_NDRE_p90"] = np.percentile(ndre, 90)
        
        # Chlorophyll Indices
        ci_green = nir_f / (green_f + eps) - 1
        ci_re = nir_f / (rededge_f + eps) - 1
        features["vi_CI_Green_mean"] = np.mean(np.clip(ci_green, -10, 10))
        features["vi_CI_Green_std"] = np.std(np.clip(ci_green, -10, 10))
        features["vi_CI_Green_max"] = np.percentile(ci_green, 95)
        features["vi_CI_RE_mean"] = np.mean(np.clip(ci_re, -10, 10))
        features["vi_CI_RE_max"] = np.percentile(ci_re, 95)
        
        # SAVI
        savi = 1.5 * (nir_f - red_f) / (nir_f + red_f + 0.5 + eps)
        features["vi_SAVI_mean"] = np.mean(savi)
        features["vi_SAVI_std"] = np.std(savi)
        features["vi_SAVI_p10"] = np.percentile(savi, 10)
        
        # EVI
        evi = 2.5 * (nir_f - red_f) / (nir_f + 6 * red_f - 7.5 * blue_f + 1.0 + eps)
        evi = np.clip(evi, -5, 5)
        features["vi_EVI_mean"] = np.mean(evi)
        features["vi_EVI_std"] = np.std(evi)
        features["vi_EVI_p10"] = np.percentile(evi, 10)
        
        # MCARI
        mcari = ((rededge_f - red_f) - 0.2 * (rededge_f - green_f)) * (rededge_f / (red_f + eps))
        features["vi_MCARI_mean"] = np.mean(np.clip(mcari, -10, 10))
        
        # RG_ratio
        rg_ratio = red_f / (green_f + eps)
        features["vi_RG_ratio_mean"] = np.mean(np.clip(rg_ratio, 0, 5))
        features["vi_RG_ratio_std"] = np.std(rg_ratio)
        
        # ===== Disease-Specific Indices =====
        # Rust Index: Red/Green (rust has reddish pustules)
        features["disease_RustIndex"] = np.mean(red_f) / (np.mean(green_f) + eps)
        
        # Health Index: NIR/Red (high for healthy)
        features["disease_HealthIndex"] = np.mean(nir_f) / (np.mean(red_f) + eps)
        
        # Yellowing Index
        yellow_idx = (green_f - blue_f) / (green_f + blue_f + eps)
        features["disease_YellowIndex_mean"] = np.mean(yellow_idx)
        
        # Iron Index (rust signature)
        iron_idx = (red_f - blue_f) / (red_f + blue_f + eps)
        features["disease_IronIndex"] = np.mean(iron_idx)
        
        # ===== Per-Band Statistics =====
        for i, (band_data, band_name) in enumerate([
            (blue_f, "Blue"), (green_f, "Green"), (red_f, "Red"),
            (rededge_f, "RedEdge"), (nir_f, "NIR")
        ]):
            features[f"band_{band_name}_mean"] = np.mean(band_data)
            features[f"band_{band_name}_std"] = np.std(band_data)
            features[f"band_{band_name}_median"] = np.median(band_data)
            features[f"band_{band_name}_min"] = np.min(band_data)
            features[f"band_{band_name}_max"] = np.max(band_data)
            features[f"band_{band_name}_cv"] = np.std(band_data) / (np.mean(band_data) + eps)
            features[f"band_{band_name}_p5"] = np.percentile(band_data, 5)
            features[f"band_{band_name}_p95"] = np.percentile(band_data, 95)
            features[f"band_{band_name}_iqr"] = np.percentile(band_data, 75) - np.percentile(band_data, 25)
            features[f"band_{band_name}_skew"] = float(scipy_stats.skew(band_data))
            features[f"band_{band_name}_kurtosis"] = float(scipy_stats.kurtosis(band_data))
        
        # ===== Spectral Shape Features =====
        band_means = [np.mean(b) for b in [blue_f, green_f, red_f, rededge_f, nir_f]]
        
        features["spec_slope_vis"] = band_means[2] - band_means[0]  # Red - Blue
        features["spec_slope_green_red"] = band_means[1] - band_means[2]
        features["spec_slope_red_re"] = band_means[2] - band_means[3]
        features["spec_slope_re_nir"] = band_means[3] - band_means[4]
        features["spec_curvature"] = band_means[3] - 0.5 * (band_means[2] + band_means[4])
        features["spec_nir_vis_ratio"] = band_means[4] / (np.mean(band_means[:3]) + eps)
        features["spec_red_green_ratio"] = band_means[2] / (band_means[1] + eps)
        
        # ===== Inter-Band Correlations =====
        flat_bands = np.vstack([blue.ravel(), green.ravel(), red.ravel(),
                                rededge.ravel(), nir.ravel()])
        corr_matrix = np.corrcoef(flat_bands)
        for i in range(5):
            for j in range(i + 1, 5):
                features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]
        
        # ===== Spatial Features =====
        # Gradient magnitude
        for band_idx, band_name in enumerate(BAND_NAMES):
            band = ms_img[:, :, band_idx].astype(np.float32)
            gy, gx = np.gradient(band)
            grad_mag = np.sqrt(gx**2 + gy**2)
            features[f"grad_{band_name}_mean"] = np.mean(grad_mag)
            features[f"grad_{band_name}_std"] = np.std(grad_mag)
            
            # Local variance
            local_mean = ndimage.uniform_filter(band, size=3)
            local_sq_mean = ndimage.uniform_filter(band**2, size=3)
            local_var = np.maximum(local_sq_mean - local_mean**2, 0)
            features[f"localvar_{band_name}_mean"] = np.mean(local_var)
        
        return features
    
    def extract_hs_features(self, hs_img):
        """Extract HS features."""
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
        mean_spec = np.mean(clean_hs, axis=(0, 1))
        n = len(mean_spec)
        if n < 50:
            return features
        
        # Key bands
        blue_b = mean_spec[min(6, n-1)]
        green_b = mean_spec[min(20, n-1)]
        red_b = mean_spec[min(44, n-1)]
        rededge_b = mean_spec[min(54, n-1)]
        nir_b = mean_spec[min(74, n-1)]
        
        # HS vegetation indices
        features["hs_NDVI"] = (nir_b - red_b) / (nir_b + red_b + eps)
        features["hs_GNDVI"] = (nir_b - green_b) / (nir_b + green_b + eps)
        features["hs_NDRE"] = (nir_b - rededge_b) / (nir_b + rededge_b + eps)
        features["hs_CI_RE"] = nir_b / (rededge_b + eps) - 1
        features["hs_CI_Green"] = nir_b / (green_b + eps) - 1
        
        # Disease-specific
        features["hs_RustIndex"] = red_b / (green_b + eps)
        features["hs_HealthRatio"] = nir_b / (red_b + eps)
        features["hs_PRI"] = (green_b - red_b) / (green_b + red_b + eps)
        features["hs_ARI"] = 1 / (green_b + eps) - 1 / (rededge_b + eps)
        
        # Spectral derivatives
        deriv1 = np.gradient(mean_spec)
        features["hs_deriv1_std"] = np.std(deriv1)
        features["hs_deriv1_max"] = np.max(deriv1)
        
        # Red edge position
        re_region = deriv1[40:60] if n > 60 else deriv1[30:50]
        features["hs_REP"] = np.argmax(re_region) + (40 if n > 60 else 30)
        features["hs_REP_slope"] = np.max(re_region)
        
        # Spectral regions
        features["hs_mean_vis"] = np.mean(mean_spec[:50])
        features["hs_mean_nir"] = np.mean(mean_spec[60:]) if n > 60 else np.mean(mean_spec[-20:])
        features["hs_nir_vis_ratio"] = features["hs_mean_nir"] / (features["hs_mean_vis"] + eps)
        
        # Continuum removal
        x = np.arange(n)
        continuum = mean_spec[0] + (mean_spec[-1] - mean_spec[0]) * x / max(1, n - 1)
        cr = mean_spec / (continuum + eps)
        features["hs_cr_depth"] = 1 - np.min(cr)
        features["hs_cr_area"] = np.mean(cr)
        
        # Inter-region ratios
        features["hs_ratio_NIR_Red"] = nir_b / (red_b + eps)
        features["hs_ratio_RE_Red"] = rededge_b / (red_b + eps)
        features["hs_ratio_Green_Red"] = green_b / (red_b + eps)
        
        # Spectral shape
        features["hs_spectral_range"] = np.max(mean_spec) - np.min(mean_spec)
        features["hs_spectral_skew"] = float(scipy_stats.skew(mean_spec))
        
        return features
    
    def extract_all_features(self, ms_img, hs_img=None):
        """Extract all features."""
        features = self.extract_ms_features(ms_img)
        
        if hs_img is not None:
            hs_features = self.extract_hs_features(hs_img)
            features.update(hs_features)
        
        return features


# ============================================================
# Preprocessing
# ============================================================
class Preprocessor:
    """Preprocessing pipeline."""
    
    @staticmethod
    def preprocess(img):
        """Apply preprocessing."""
        img = img.copy().astype(np.float32)
        
        # Dead pixel removal
        H, W, C = img.shape
        zero_fraction = (img == 0).sum(axis=2) / C
        dead_mask = zero_fraction > 0.25
        
        if dead_mask.any():
            for c in range(C):
                band = img[:, :, c]
                median_filtered = ndimage.median_filter(band, size=3)
                band[dead_mask] = median_filtered[dead_mask]
                img[:, :, c] = band
        
        # Wiener denoise
        for c in range(C):
            band = img[:, :, c]
            if band.max() > 0:
                filtered = wiener(band, mysize=3)
                img[:, :, c] = np.maximum(filtered, 0)
        
        return img


# ============================================================
# Super-Resolution
# ============================================================
class SuperResolution:
    """Bicubic super-resolution for HS."""
    
    def upscale(self, img, target_size=64):
        """Upscale HS from 32x32 to 64x64."""
        H, W, C = img.shape
        if H == target_size and W == target_size:
            return img
        
        scale = target_size / H
        upscaled = scipy_zoom(img, (scale, scale, 1), order=3).astype(np.float32)
        return upscaled[:target_size, :target_size, :C]


# ============================================================
# Data Pipeline
# ============================================================
class DataPipeline:
    """Data loading and processing."""
    
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.super_res = SuperResolution()
        self.feature_extractor = FeatureExtractor()
    
    def process_sample(self, ms_path, hs_path=None):
        """Process a single sample."""
        # Load MS
        ms_img = tiff.imread(ms_path).astype(np.float32)
        if ms_img.mean() < 1.0:
            return None, True
        
        # Preprocess
        ms_clean = self.preprocessor.preprocess(ms_img)
        
        # Load and process HS
        hs_upscaled = None
        if hs_path and os.path.exists(hs_path):
            hs_img = tiff.imread(hs_path).astype(np.float32)
            if hs_img.mean() >= 1.0:
                hs_clean = self.preprocessor.preprocess(hs_img)
                hs_upscaled = self.super_res.upscale(hs_clean)
        
        # Feature extraction
        features = self.feature_extractor.extract_all_features(ms_clean, hs_upscaled)
        
        return features, False


# ============================================================
# Two-Stage Classifier
# ============================================================
class TwoStageClassifier:
    """
    Two-stage classification approach:
    Stage 1: Health vs (Rust + Other)
    Stage 2: Rust vs Other (for non-Health samples)
    """
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.stage1_model = None  # Health vs Rest
        self.stage2_model = None  # Rust vs Other
        
    def fit(self, X, y, cfg):
        """Train both stages."""
        # Stage 1: Health (0) vs Rest (1,2)
        y_stage1 = (y > 0).astype(int)  # 0=Health, 1=Rust+Other
        
        print("\n  Stage 1: Health vs Rest")
        self.stage1_model = self._train_xgb(X, y_stage1, cfg, "stage1")
        
        # Stage 2: Rust (1) vs Other (2) - only for non-Health samples
        non_health_mask = y > 0
        X_stage2 = X[non_health_mask]
        y_stage2 = y[non_health_mask] - 1  # 0=Rust, 1=Other
        
        print("\n  Stage 2: Rust vs Other")
        self.stage2_model = self._train_xgb(X_stage2, y_stage2, cfg, "stage2")
        
        return self
    
    def _train_xgb(self, X, y, cfg, stage_name):
        """Train XGBoost for a stage."""
        skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
        
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": cfg["xgb_max_depth"],
            "learning_rate": cfg["learning_rate"],
            "n_estimators": cfg["n_estimators"],
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": cfg["seed"],
            "tree_method": "hist",
            "verbosity": 0,
        }
        
        oof_preds = np.zeros(len(X))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
            acc = accuracy_score(y_va, (oof_preds[va_idx] > 0.5).astype(int))
            print(f"    Fold {fold + 1}: Acc={acc:.4f}")
        
        overall_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
        print(f"    Overall Acc: {overall_acc:.4f}")
        
        # Train final model on all data
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X, y)
        
        return final_model
    
    def predict_proba(self, X):
        """Get 3-class probabilities."""
        # Stage 1: P(Health) vs P(Rust+Other)
        p_non_health = self.stage1_model.predict_proba(X)[:, 1]
        p_health = 1 - p_non_health
        
        # Stage 2: P(Rust|non-Health) vs P(Other|non-Health)
        p_rust_given_non_health = self.stage2_model.predict_proba(X)[:, 0]
        p_other_given_non_health = 1 - p_rust_given_non_health
        
        # Combine: P(class) = P(class|stage) * P(stage)
        p_rust = p_non_health * p_rust_given_non_health
        p_other = p_non_health * p_other_given_non_health
        
        # Normalize
        total = p_health + p_rust + p_other
        probs = np.column_stack([p_health, p_rust, p_other]) / total[:, np.newaxis]
        
        return probs


# ============================================================
# Training Pipeline
# ============================================================
class Trainer:
    """Main training pipeline."""
    
    def __init__(self):
        self.pipeline = DataPipeline()
    
    def load_dataset(self, ms_dir, hs_dir, is_train=True):
        """Load dataset."""
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
        """Execute training."""
        seed_everything(CFG["seed"])
        os.makedirs(CFG["output_dir"], exist_ok=True)
        
        # =====================================================
        # Step 1: Load data
        # =====================================================
        print("=" * 70)
        print("STEP 1: Loading data")
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
        
        feature_names = None
        for f in train_feats:
            if f is not None:
                feature_names = list(f.keys())
                break
        
        print(f"Feature count: {len(feature_names)}")
        
        X_train = np.array([
            [f.get(k, 0.0) if f is not None else 0.0 for k in feature_names]
            for f in train_feats
        ], dtype=np.float32)
        y_train = np.array(train_labels)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        
        for ci in range(3):
            print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")
        
        black_mask = [f is None for f in val_feats]
        X_val = np.array([
            [f.get(k, 0.0) if f is not None else 0.0 for k in feature_names]
            for f in val_feats
        ], dtype=np.float32)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
        print(f"  Val: {len(val_feats)} samples ({sum(black_mask)} black)")
        
        # =====================================================
        # Step 3: Scale features
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 3: Scaling features")
        print("=" * 70)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # =====================================================
        # Step 4: Two-Stage Classification
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 4: Two-Stage Classification")
        print("=" * 70)
        
        two_stage = TwoStageClassifier(feature_names)
        two_stage.fit(X_train_scaled, y_train, CFG)
        
        # OOF predictions via CV
        print("\n  Computing OOF predictions...")
        skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
        oof_probs = np.zeros((len(X_train_scaled), 3))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
            X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            
            # Train two-stage on fold
            fold_classifier = TwoStageClassifier(feature_names)
            fold_classifier.fit(X_tr, y_tr, CFG)
            oof_probs[va_idx] = fold_classifier.predict_proba(X_va)
        
        oof_preds = np.argmax(oof_probs, axis=1)
        oof_acc = accuracy_score(y_train, oof_preds)
        oof_f1 = f1_score(y_train, oof_preds, average='macro')
        
        print(f"\nTwo-Stage OOF Accuracy: {oof_acc:.4f}, F1: {oof_f1:.4f}")
        print(classification_report(y_train, oof_preds,
                                    target_names=list(CLASS_MAP.keys()), digits=4))
        
        # =====================================================
        # Step 5: Standard XGBoost for comparison
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 5: Standard XGBoost (for comparison)")
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
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": CFG["seed"],
            "tree_method": "hist",
            "verbosity": 0,
        }
        
        oof_xgb = np.zeros((len(X_train_scaled), 3))
        val_xgb_folds = []
        feature_importance = np.zeros(len(feature_names))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
            X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            oof_xgb[va_idx] = model.predict_proba(X_va)
            val_xgb_folds.append(model.predict_proba(X_val_scaled))
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
            print(f"  {i + 1:2d}. {feature_names[idx]:40s} imp={feature_importance[idx]:.4f}")
        
        # =====================================================
        # Step 6: Ensemble
        # =====================================================
        print("\n" + "=" * 70)
        print("STEP 6: Ensemble (Two-Stage + XGB)")
        print("=" * 70)
        
        # Weighted ensemble
        w_two_stage = 0.4
        w_xgb = 0.6
        
        ens_oof = w_two_stage * oof_probs + w_xgb * oof_xgb
        ens_preds = np.argmax(ens_oof, axis=1)
        ens_acc = accuracy_score(y_train, ens_preds)
        ens_f1 = f1_score(y_train, ens_preds, average='macro')
        print(f"Ensemble OOF Accuracy: {ens_acc:.4f}, F1: {ens_f1:.4f}")
        print(classification_report(y_train, ens_preds,
                                    target_names=list(CLASS_MAP.keys()), digits=4))
        
        # Val predictions
        val_xgb_probs = np.mean(val_xgb_folds, axis=0)
        val_two_stage_probs = two_stage.predict_proba(X_val_scaled)
        val_ens_probs = w_two_stage * val_two_stage_probs + w_xgb * val_xgb_probs
        
        # Override black images
        for i, is_b in enumerate(black_mask):
            if is_b:
                val_ens_probs[i] = [0.0, 0.0, 1.0]
                val_xgb_probs[i] = [0.0, 0.0, 1.0]
                val_two_stage_probs[i] = [0.0, 0.0, 1.0]
        
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
        np.save(os.path.join(CFG["output_dir"], "val_probs_two_stage.npy"), val_two_stage_probs)
        np.save(os.path.join(CFG["output_dir"], "oof_xgb.npy"), oof_xgb)
        np.save(os.path.join(CFG["output_dir"], "oof_two_stage.npy"), oof_probs)
        
        # Submission
        sub_path = os.path.join(CFG["output_dir"], "submission.csv")
        with open(sub_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Id", "Category"])
            for fn, cl in zip(val_fnames, pred_classes):
                w.writerow([fn, cl])
        
        with open(os.path.join(CFG["output_dir"], "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)
        
        # =====================================================
        # Summary
        # =====================================================
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Features: {len(feature_names)}")
        print(f"\nTwo-Stage OOF: {oof_acc:.4f}, F1: {oof_f1:.4f}")
        print(f"XGB OOF:       {xgb_acc:.4f}, F1: {xgb_f1:.4f}")
        print(f"Ensemble OOF:  {ens_acc:.4f}, F1: {ens_f1:.4f}")
        print(f"\nSubmission: {sub_path}")
        print(f"Val distribution: {dist}")
        print("\nDone!")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()