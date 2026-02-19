"""
Enhanced HS Feature Engineering - Extracting Discriminative Features from 125-band HS
======================================================================================

Key improvements:
1. Spectral derivatives (1st, 2nd, 3rd) - capture subtle spectral shape
2. Continuum removal - highlight absorption features
3. Red Edge Position (REP) - precise vegetation health indicator
4. Spectral Angle Mapper (SAM) to class prototypes
5. Absorption band depth features
6. Disease-specific spectral signatures
7. PCA on full spectrum
8. Spectral texture features
9. Water and chlorophyll absorption features
10. Custom rust-detection indices
"""

import os
import csv
import json
import warnings
import numpy as np
import tifffile as tiff
from scipy import ndimage, stats as scipy_stats
from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

CFG = {
    "train_ms_dir": "Kaggle_Prepared/train/MS",
    "val_ms_dir": "Kaggle_Prepared/val/MS",
    "train_hs_dir": "Kaggle_Prepared/train/HS",
    "val_hs_dir": "Kaggle_Prepared/val/HS",
    "output_dir": "hs_enhanced",
    "n_folds": 5,
    "seed": 42,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]

# Approximate wavelengths for HS bands (450-950nm range, 100 clean bands)
HS_WAVELENGTHS = np.linspace(450, 950, 100)


def extract_ms_features(ms_img):
    """Extract MS features (same as proven approach)."""
    ms_img = ms_img.astype(np.float32)
    features = {}
    eps = 1e-8

    bands = ms_img.transpose(2, 0, 1)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]

    # Raw mean spectrum
    mean_spectrum = np.array([np.mean(b) for b in [blue, green, red, rededge, nir]])
    for i, name in enumerate(BAND_NAMES):
        features[f"ms_raw_{name}"] = mean_spectrum[i]

    # Per-band statistics
    for i, name in enumerate(BAND_NAMES):
        b = bands[i].ravel()
        features[f"ms_{name}_mean"] = np.mean(b)
        features[f"ms_{name}_std"] = np.std(b)
        features[f"ms_{name}_min"] = np.min(b)
        features[f"ms_{name}_max"] = np.max(b)
        features[f"ms_{name}_cv"] = np.std(b) / (np.mean(b) + eps)

    # Vegetation indices
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)
    gndvi = (nir - green) / (nir + green + eps)
    ci_re = nir / (rededge + eps) - 1
    ci_green = nir / (green + eps) - 1
    rg_ratio = red / (green + eps)
    health_ratio = nir / (red + eps)
    iron_idx = (red - blue) / (red + blue + eps)
    evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)

    indices = {
        "NDVI": ndvi, "NDRE": ndre, "GNDVI": gndvi,
        "CI_RE": ci_re, "CI_Green": ci_green,
        "RG_ratio": rg_ratio, "Health_Ratio": health_ratio,
        "Iron_Idx": iron_idx, "EVI": evi, "SAVI": savi,
    }

    for idx_name, idx_map in indices.items():
        idx_map = np.clip(idx_map, -10, 10)
        v = idx_map.ravel()
        features[f"ms_{idx_name}_mean"] = np.mean(v)
        features[f"ms_{idx_name}_std"] = np.std(v)
        features[f"ms_{idx_name}_min"] = np.min(v)
        features[f"ms_{idx_name}_max"] = np.max(v)

    # Spectral shape
    features["ms_slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["ms_slope_re"] = mean_spectrum[3] - mean_spectrum[2]
    features["ms_nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)

    return features


def extract_hs_features(hs_img, class_prototypes=None):
    """Extract enhanced HS features from 125-band hyperspectral image."""
    hs_img = hs_img.astype(np.float32)
    features = {}
    eps = 1e-8
    
    n_bands = hs_img.shape[2]
    
    # Use clean bands (10-110)
    clean_start, clean_end = 10, min(110, n_bands - 1)
    clean_hs = hs_img[:, :, clean_start:clean_end]
    
    # Mean spectrum (key feature)
    mean_spectrum = np.mean(clean_hs, axis=(0, 1))
    n = len(mean_spectrum)
    
    # ====== 1. Basic Statistics ======
    features["hs_mean_reflectance"] = np.mean(mean_spectrum)
    features["hs_std_reflectance"] = np.std(mean_spectrum)
    features["hs_max_reflectance"] = np.max(mean_spectrum)
    features["hs_min_reflectance"] = np.min(mean_spectrum)
    features["hs_range_reflectance"] = features["hs_max_reflectance"] - features["hs_min_reflectance"]
    features["hs_skewness"] = float(scipy_stats.skew(mean_spectrum))
    features["hs_kurtosis"] = float(scipy_stats.kurtosis(mean_spectrum))
    
    # ====== 2. Spectral Derivatives (Key for subtle differences) ======
    if n > 3:
        # First derivative (rate of change)
        deriv1 = np.gradient(mean_spectrum)
        features["hs_deriv1_mean"] = np.mean(deriv1)
        features["hs_deriv1_std"] = np.std(deriv1)
        features["hs_deriv1_max"] = np.max(deriv1)
        features["hs_deriv1_min"] = np.min(deriv1)
        features["hs_deriv1_range"] = features["hs_deriv1_max"] - features["hs_deriv1_min"]
        
        # Second derivative (curvature)
        deriv2 = np.gradient(deriv1)
        features["hs_deriv2_mean"] = np.mean(deriv2)
        features["hs_deriv2_std"] = np.std(deriv2)
        features["hs_deriv2_max"] = np.max(deriv2)
        features["hs_deriv2_min"] = np.min(deriv2)
        
        # Third derivative
        deriv3 = np.gradient(deriv2)
        features["hs_deriv3_std"] = np.std(deriv3)
    
    # ====== 3. Red Edge Position (REP) - Critical for vegetation health ======
    # Red edge is around 680-750nm (bands ~30-50 in our range)
    if n > 50:
        red_edge_region = mean_spectrum[30:50]
        red_edge_deriv = deriv1[30:50] if 'deriv1' in dir() else np.gradient(red_edge_region)
        
        # REP: position of maximum derivative in red edge region
        rep_idx = np.argmax(red_edge_deriv) + 30
        features["hs_rep_position"] = rep_idx
        features["hs_rep_value"] = mean_spectrum[rep_idx] if rep_idx < n else mean_spectrum[40]
        features["hs_rep_slope"] = np.max(red_edge_deriv)
        
        # Red edge well depth
        re_min = np.min(mean_spectrum[25:35])  # Red minimum
        re_max = np.max(mean_spectrum[45:55])  # NIR maximum
        features["hs_re_depth"] = re_max - re_min
        features["hs_re_ratio"] = re_max / (re_min + eps)
    
    # ====== 4. Spectral Regions ======
    # Blue (450-495nm): bands 0-10
    # Green (495-570nm): bands 10-25
    # Red (620-700nm): bands 35-50
    # Red Edge (700-750nm): bands 50-60
    # NIR (750-950nm): bands 60-100
    
    regions = {
        "blue": (0, 10),
        "green": (10, 25),
        "yellow": (25, 35),
        "red": (35, 50),
        "rededge": (50, 60),
        "nir": (60, 100),
    }
    
    region_means = {}
    for region_name, (start, end) in regions.items():
        if end <= n:
            region_vals = mean_spectrum[start:end]
            region_means[region_name] = np.mean(region_vals)
            features[f"hs_{region_name}_mean"] = np.mean(region_vals)
            features[f"hs_{region_name}_std"] = np.std(region_vals)
            features[f"hs_{region_name}_max"] = np.max(region_vals)
            features[f"hs_{region_name}_min"] = np.min(region_vals)
    
    # ====== 5. Enhanced Vegetation Indices ======
    if n > 70:
        # Get specific bands
        blue_b = mean_spectrum[5]    # ~480nm
        green_b = mean_spectrum[17]  # ~550nm
        yellow_b = mean_spectrum[30] # ~610nm
        red_b = mean_spectrum[42]    # ~670nm
        rede_b = mean_spectrum[52]   # ~720nm
        nir_b = mean_spectrum[75]    # ~820nm
        
        # Standard indices
        features["hs_NDVI"] = (nir_b - red_b) / (nir_b + red_b + eps)
        features["hs_NDRE"] = (nir_b - rede_b) / (nir_b + rede_b + eps)
        features["hs_GNDVI"] = (nir_b - green_b) / (nir_b + green_b + eps)
        features["hs_CI_RE"] = nir_b / (rede_b + eps) - 1
        features["hs_CI_Green"] = nir_b / (green_b + eps) - 1
        
        # Disease-specific indices
        # Rust causes increased red reflectance
        features["hs_Rust_Index"] = red_b / (green_b + eps)
        features["hs_Rust_Index2"] = (red_b - yellow_b) / (red_b + yellow_b + eps)
        
        # Health indicators
        features["hs_Health_Index"] = nir_b / (red_b + eps)
        features["hs_Chlorophyll_Index"] = (nir_b / red_b) - 1
        
        # Water index
        features["hs_Water_Index"] = nir_b / (rede_b + eps)
        
        # Plant Stress Index
        features["hs_PSI"] = (rede_b - red_b) / (rede_b + red_b + eps)
        
        # Photochemical Reflectance Index (PRI) - stress indicator
        features["hs_PRI"] = (green_b - red_b) / (green_b + red_b + eps)
        
        # Anthocyanin Reflectance Index (ARI) - disease pigmentation
        features["hs_ARI"] = 1 / (green_b + eps) - 1 / (rede_b + eps)
        
        # Carotenoid Reflectance Index (CRI)
        features["hs_CRI"] = 1 / (blue_b + eps) - 1 / (green_b + eps)
        
        # Iron oxide index (rust has iron signature)
        features["hs_Iron_Index"] = (red_b - blue_b) / (red_b + blue_b + eps)
        
        # Yellowing index
        features["hs_Yellowing"] = (green_b - blue_b) / (green_b + blue_b + eps)
        
        # Disease stress index
        features["hs_DSI"] = (nir_b - rede_b) / (nir_b + red_b + eps)
        
        # Modified chlorophyll absorption ratio
        features["hs_MCARI"] = ((rede_b - red_b) - 0.2 * (rede_b - green_b)) * (rede_b / (red_b + eps))
        
        # Optimized soil-adjusted vegetation index
        features["hs_OSAVI"] = (nir_b - red_b) / (nir_b + red_b + 0.16 + eps)
        
        # Triangular vegetation index
        features["hs_TVI"] = 0.5 * (120 * (nir_b - green_b) - 200 * (red_b - green_b))
    
    # ====== 6. Inter-Region Ratios ======
    if len(region_means) >= 4:
        features["hs_NIR_Red_ratio"] = region_means.get("nir", 0) / (region_means.get("red", 1) + eps)
        features["hs_NIR_Vis_ratio"] = region_means.get("nir", 0) / (np.mean([region_means.get("blue", 1), region_means.get("green", 1), region_means.get("red", 1)]) + eps)
        features["hs_RE_Red_ratio"] = region_means.get("rededge", 0) / (region_means.get("red", 1) + eps)
        features["hs_Green_Red_ratio"] = region_means.get("green", 0) / (region_means.get("red", 1) + eps)
        features["hs_Red_Blue_ratio"] = region_means.get("red", 0) / (region_means.get("blue", 1) + eps)
    
    # ====== 7. Absorption Features ======
    if n > 60:
        # Chlorophyll absorption (around 680nm, band ~45)
        chloro_trough = np.min(mean_spectrum[40:50])
        chloro_left = mean_spectrum[35]
        chloro_right = mean_spectrum[52]
        features["hs_chloro_depth"] = (np.mean([chloro_left, chloro_right]) - chloro_trough) / (np.mean([chloro_left, chloro_right]) + eps)
        features["hs_chloro_min"] = chloro_trough
        
        # Water absorption (around 970nm, but we only go to 950nm)
        # Use NIR plateau variability instead
        nir_plateau = mean_spectrum[70:90]
        features["hs_nir_plateau_mean"] = np.mean(nir_plateau)
        features["hs_nir_plateau_std"] = np.std(nir_plateau)
    
    # ====== 8. Continuum Removal Features ======
    if n > 10:
        # Simple convex hull approximation
        # Find local maxima and interpolate
        from scipy.interpolate import interp1d
        
        # Simple continuum: connect first and last points
        x = np.arange(n)
        continuum = mean_spectrum[0] + (mean_spectrum[-1] - mean_spectrum[0]) * x / (n - 1)
        
        # Continuum removed spectrum
        cr_spectrum = mean_spectrum / (continuum + eps)
        features["hs_cr_mean"] = np.mean(cr_spectrum)
        features["hs_cr_min"] = np.min(cr_spectrum)
        features["hs_cr_depth"] = 1 - np.min(cr_spectrum)  # Absorption depth
        
        # Area under continuum-removed curve
        features["hs_cr_area"] = np.sum(cr_spectrum) / n
    
    # ====== 9. Spectral Angle Mapper (SAM) to Class Prototypes ======
    if class_prototypes is not None and 'hs' in class_prototypes:
        for cls_name, proto in class_prototypes['hs'].items():
            # Match lengths
            min_len = min(len(mean_spectrum), len(proto))
            spec = mean_spectrum[:min_len]
            prot = proto[:min_len]
            
            # Spectral angle
            dot = np.dot(spec, prot)
            norm_prod = np.linalg.norm(spec) * np.linalg.norm(prot)
            if norm_prod > 0:
                angle = np.arccos(np.clip(dot / norm_prod, -1, 1))
                features[f"hs_SAM_{cls_name}"] = angle
            
            # Euclidean distance
            features[f"hs_EucDist_{cls_name}"] = np.linalg.norm(spec - prot)
            
            # Spectral Information Divergence
            p = spec / (np.sum(spec) + eps)
            q = prot / (np.sum(prot) + eps)
            sid = np.sum(p * np.log2((p + eps) / (q + eps))) + np.sum(q * np.log2((q + eps) / (p + eps)))
            features[f"hs_SID_{cls_name}"] = sid
    
    # ====== 10. PCA Features ======
    # Use mean spectrum values as PCA-like features
    if n >= 10:
        # First 10 principal components approximation
        for i in range(10):
            features[f"hs_pc_{i}"] = mean_spectrum[i * (n // 10)]
    
    # ====== 11. Spectral Texture ======
    # Pixel-wise spectral variability
    pixel_means = np.mean(clean_hs, axis=2)  # Mean across bands per pixel
    features["hs_pixel_mean_var"] = np.std(pixel_means)
    features["hs_pixel_mean_range"] = np.max(pixel_means) - np.min(pixel_means)
    
    # Spectral angle variability across pixels
    if clean_hs.shape[0] > 1 and clean_hs.shape[1] > 1:
        # Sample some pixels
        sample_pixels = clean_hs[::8, ::8, :].reshape(-1, n)  # Every 8th pixel
        if len(sample_pixels) > 1:
            # Compute spectral angles between pixels
            angles = []
            ref_spectrum = mean_spectrum / (np.linalg.norm(mean_spectrum) + eps)
            for px in sample_pixels[:20]:  # Limit computation
                px_norm = px / (np.linalg.norm(px) + eps)
                angle = np.arccos(np.clip(np.dot(ref_spectrum, px_norm), -1, 1))
                angles.append(angle)
            features["hs_spectral_angle_std"] = np.std(angles)
    
    # ====== 12. Area Under Curve Features ======
    features["hs_auc_vis"] = np.sum(mean_spectrum[:50])  # Visible region
    features["hs_auc_nir"] = np.sum(mean_spectrum[50:])  # NIR region
    features["hs_auc_ratio"] = features["hs_auc_nir"] / (features["hs_auc_vis"] + eps)
    
    return features


def compute_class_prototypes(ms_dir, hs_dir):
    """Compute mean spectral signatures for each class."""
    ms_protos = {cls: [] for cls in CLASS_MAP}
    hs_protos = {cls: [] for cls in CLASS_MAP}
    
    for f in sorted(os.listdir(ms_dir)):
        if "_hyper_" not in f:
            continue
        fp = os.path.join(ms_dir, f)
        img = tiff.imread(fp).astype(np.float32)
        if img.mean() < 1.0:
            continue
        cls_name = f.split("_hyper_")[0]
        mean_spec = np.array([np.mean(img[:,:,i]) for i in range(5)])
        ms_protos[cls_name].append(mean_spec)
    
    for f in sorted(os.listdir(hs_dir)):
        if "_hyper_" not in f:
            continue
        fp = os.path.join(hs_dir, f)
        img = tiff.imread(fp).astype(np.float32)
        if img.mean() < 1.0:
            continue
        cls_name = f.split("_hyper_")[0]
        n_bands = img.shape[2]
        clean = img[:, :, 10:min(110, n_bands-1)]
        mean_spec = np.mean(clean, axis=(0, 1))
        hs_protos[cls_name].append(mean_spec)
    
    final_ms = {cls: np.mean(vals, axis=0) for cls, vals in ms_protos.items() if vals}
    final_hs = {cls: np.mean(vals, axis=0) for cls, vals in hs_protos.items() if vals}
    
    return {'ms': final_ms, 'hs': final_hs}


def extract_all(ms_dir, hs_dir, prototypes=None):
    """Extract all features from all files."""
    ms_files = sorted(os.listdir(ms_dir))
    all_features, all_labels, all_fnames = [], [], []
    skipped = 0
    
    for ms_f in ms_files:
        ms_fp = os.path.join(ms_dir, ms_f)
        ms_img = tiff.imread(ms_fp).astype(np.float32)
        
        hs_img = None
        hs_fp = os.path.join(hs_dir, ms_f)
        if os.path.exists(hs_fp):
            hs_img = tiff.imread(hs_fp).astype(np.float32)
            if hs_img.mean() < 1.0:
                hs_img = None
        
        if ms_img.mean() < 1.0:
            skipped += 1
            if "_hyper_" in ms_f:
                continue
            else:
                all_features.append(None)
                all_labels.append(-1)
                all_fnames.append(ms_f)
                continue
        
        # Extract features
        feats = extract_ms_features(ms_img)
        if hs_img is not None:
            hs_feats = extract_hs_features(hs_img, prototypes)
            feats.update(hs_feats)
        
        all_features.append(feats)
        
        if "_hyper_" in ms_f:
            cls_name = ms_f.split("_hyper_")[0]
            all_labels.append(CLASS_MAP[cls_name])
        else:
            all_labels.append(-1)
        all_fnames.append(ms_f)
    
    if skipped > 0:
        print(f"  Skipped/flagged {skipped} black images")
    return all_features, all_labels, all_fnames


def main():
    np.random.seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    print("Computing class prototypes...")
    prototypes = compute_class_prototypes(CFG["train_ms_dir"], CFG["train_hs_dir"])
    print("  MS prototypes:")
    for cls, proto in prototypes['ms'].items():
        print(f"    {cls}: {proto}")
    print("  HS prototypes length:")
    for cls, proto in prototypes['hs'].items():
        print(f"    {cls}: {len(proto)} bands")

    print("\nExtracting training features...")
    train_feats, train_labels, train_fnames = extract_all(
        CFG["train_ms_dir"], CFG["train_hs_dir"], prototypes
    )
    print(f"  {len(train_feats)} samples")

    feature_names = list(train_feats[0].keys())
    X_train = np.array([[f.get(k, 0.0) for k in feature_names] for f in train_feats], dtype=np.float32)
    y_train = np.array(train_labels)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

    print(f"  {X_train.shape[1]} features extracted")
    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")

    print("\nExtracting validation features...")
    val_feats, val_labels, val_fnames = extract_all(
        CFG["val_ms_dir"], CFG["val_hs_dir"], prototypes
    )
    black_mask = [f is None for f in val_feats]
    
    for i in range(len(val_feats)):
        if val_feats[i] is None:
            val_feats[i] = {k: 0.0 for k in feature_names}

    X_val = np.array([[f.get(k, 0.0) for k in feature_names] for f in val_feats], dtype=np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"  {len(val_feats)} samples ({sum(black_mask)} black)")

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ================================================================
    # Training
    # ================================================================
    print(f"\n{'='*70}")
    print("Training Models")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

    # XGBoost
    print("\n--- XGBoost ---")
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
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_xgb_preds = np.argmax(oof_xgb, axis=1)
    xgb_acc = accuracy_score(y_train, oof_xgb_preds)
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")

    # Top features
    feature_importance /= CFG["n_folds"]
    top_idx = np.argsort(feature_importance)[::-1][:20]
    print("\nTop 20 features:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1:2d}. {feature_names[idx]:30s} importance={feature_importance[idx]:.4f}")

    # LightGBM
    print("\n--- LightGBM ---")
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

    oof_lgb = np.zeros((len(X_train_scaled), 3))
    val_lgb_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        oof_lgb[va_idx] = model.predict_proba(X_va)
        val_lgb_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, np.argmax(oof_lgb[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_lgb_preds = np.argmax(oof_lgb, axis=1)
    lgb_acc = accuracy_score(y_train, oof_lgb_preds)
    print(f"LGB OOF Accuracy: {lgb_acc:.4f}")

    # Ensemble
    print(f"\n{'='*70}")
    print("Ensemble")
    print(f"{'='*70}")

    ens_oof = 0.5 * oof_xgb + 0.5 * oof_lgb
    ens_preds = np.argmax(ens_oof, axis=1)
    ens_acc = accuracy_score(y_train, ens_preds)
    print(f"Ensemble OOF Accuracy: {ens_acc:.4f}")
    print(classification_report(y_train, ens_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # Validation Predictions
    # ================================================================
    print(f"\n{'='*70}")
    print("VALIDATION PREDICTIONS")
    print(f"{'='*70}")

    val_xgb_probs = np.mean(val_xgb_folds, axis=0)
    val_lgb_probs = np.mean(val_lgb_folds, axis=0)
    val_probs = 0.5 * val_xgb_probs + 0.5 * val_lgb_probs
    
    val_preds = np.argmax(val_probs, axis=1)

    # Override black images
    for i, is_b in enumerate(black_mask):
        if is_b:
            val_preds[i] = 2

    pred_classes = [INV_CLASS_MAP[p] for p in val_preds]
    dist = {c: pred_classes.count(c) for c in CLASS_MAP}
    print(f"Val prediction distribution: {dist}")

    # ================================================================
    # Save Results
    # ================================================================
    np.save(os.path.join(CFG["output_dir"], "val_probs_final.npy"), val_probs)

    sub_path = os.path.join(CFG["output_dir"], "submission.csv")
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for fn, cl in zip(val_fnames, pred_classes):
            w.writerow([fn, cl])

    with open(os.path.join(CFG["output_dir"], "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\nSaved to {CFG['output_dir']}/")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total features: {len(feature_names)}")
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")
    print(f"LGB OOF Accuracy: {lgb_acc:.4f}")
    print(f"Ensemble OOF Accuracy: {ens_acc:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
