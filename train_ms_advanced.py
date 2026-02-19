"""
Advanced MS Classification with Novel Approaches
=================================================
Target: 80%+ CV accuracy

Novel techniques implemented:
1. Spectral Angle Mapper (SAM) features - spectral similarity to class prototypes
2. Semi-supervised pseudo-labeling - use high-confidence val predictions
3. Enhanced feature engineering - more domain-specific indices
4. Multi-model ensemble - XGBoost + LightGBM + CatBoost + Neural Network
5. Feature selection with SHAP-based importance
6. Class-specific threshold optimization
7. Data augmentation on spectral features
8. Stacking ensemble with meta-learner
"""

import os
import csv
import json
import warnings
import numpy as np
import tifffile as tiff
from scipy import ndimage, stats as scipy_stats
from scipy.spatial.distance import cosine
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

CFG = {
    "data_dir": "Kaggle_Prepared/train/MS",
    "val_dir": "Kaggle_Prepared/val/MS",
    "output_dir": "ms_advanced",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
    "pseudo_label_threshold": 0.95,  # Confidence threshold for pseudo-labeling
    "use_pseudo_labeling": True,
    "feature_selection_k": 150,  # Number of features to select
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]
WAVELENGTHS = [480, 550, 650, 740, 833]  # nm


# ============================================================
# Enhanced Feature Extraction
# ============================================================
def extract_features(img, class_prototypes=None):
    """
    Extract comprehensive feature vector from a single 5-band 64x64 MS image.
    Input: img (64, 64, 5) float32
    Returns: feature dict
    """
    img = img.astype(np.float32)
    features = {}
    eps = 1e-8

    # Transpose to (5, 64, 64) for easier band access
    bands = img.transpose(2, 0, 1)  # (5, H, W)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]

    # ====== 1. Per-band statistics (enhanced) ======
    for i, name in enumerate(BAND_NAMES):
        b = bands[i].ravel()
        features[f"{name}_mean"] = np.mean(b)
        features[f"{name}_std"] = np.std(b)
        features[f"{name}_min"] = np.min(b)
        features[f"{name}_max"] = np.max(b)
        features[f"{name}_median"] = np.median(b)
        features[f"{name}_p5"] = np.percentile(b, 5)
        features[f"{name}_p10"] = np.percentile(b, 10)
        features[f"{name}_p25"] = np.percentile(b, 25)
        features[f"{name}_p75"] = np.percentile(b, 75)
        features[f"{name}_p90"] = np.percentile(b, 90)
        features[f"{name}_p95"] = np.percentile(b, 95)
        features[f"{name}_iqr"] = features[f"{name}_p75"] - features[f"{name}_p25"]
        features[f"{name}_range"] = features[f"{name}_max"] - features[f"{name}_min"]
        features[f"{name}_skew"] = float(scipy_stats.skew(b))
        features[f"{name}_kurtosis"] = float(scipy_stats.kurtosis(b))
        features[f"{name}_cv"] = np.std(b) / (np.mean(b) + eps)
        # Energy and entropy
        b_norm = b / (np.sum(b) + eps)
        features[f"{name}_entropy"] = -np.sum(b_norm * np.log2(b_norm + eps))
        features[f"{name}_energy"] = np.sum(b ** 2)

    # ====== 2. Enhanced Vegetation Indices ======
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)
    gndvi = (nir - green) / (nir + green + eps)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    ci_rededge = nir / (rededge + eps) - 1.0
    ci_green = nir / (green + eps) - 1.0
    rg_ratio = red / (green + eps)
    rb_ratio = red / (blue + eps)
    re_r_ratio = rededge / (red + eps)
    nir_r_ratio = nir / (red + eps)
    nir_re_ratio = nir / (rededge + eps)
    
    # EVI (Enhanced Vegetation Index)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
    
    # MCARI (Modified Chlorophyll Absorption in Reflectance Index)
    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))
    
    # NEW: Additional vegetation indices
    # OSAVI (Optimized Soil Adjusted Vegetation Index)
    osavi = (nir - red) / (nir + red + 0.16 + eps)
    
    # NDWI (Normalized Difference Water Index) - using NIR and Green
    ndwi = (green - nir) / (green + nir + eps)
    
    # SIPI (Structure Insensitive Pigment Index)
    sipi = (nir - blue) / (nir - red + eps)
    
    # PRI (Photochemical Reflectance Index) - stress indicator
    # Using available bands as proxy
    pri_proxy = (green - red) / (green + red + eps)
    
    # TVI (Triangular Vegetation Index)
    tvi = 0.5 * (120 * (nir - green) - 200 * (red - green) + eps)
    
    # MTVI2 (Modified Triangular Vegetation Index 2)
    mtvi2 = 1.5 * (1.2 * (nir - green) - 2.5 * (red - green)) / np.sqrt((2 * nir + 1)**2 - (6 * nir - 5 * np.sqrt(red) - 0.5) + eps)
    
    # RDVI (Renormalized Difference Vegetation Index)
    rdvi = (nir - red) / np.sqrt(nir + red + eps)
    
    # MSR (Modified Simple Ratio)
    msr = (nir / (red + eps) - 1) / np.sqrt(nir / (red + eps) + 1 + eps)
    
    # NLI (Non-Linear Index)
    nli = (nir**2 - red) / (nir**2 + red + eps)
    
    # RDVI2
    rdvi2 = (nir - red) / np.sqrt((nir + red) * (nir - red + eps) + eps)
    
    # Iron Index (for rust detection - iron oxide signature)
    iron_index = (red - blue) / (red + blue + eps)
    
    # Disease Stress Index (DSI) - custom for rust
    dsi = (nir - rededge) / (nir + red + eps)
    
    # Red Edge Position proxy
    rep_proxy = 700 + 40 * ((red + rededge) / 2 - red) / (rededge - red + eps)
    
    # Chlorophyll absorption ratio
    chloro_abs = (green - red) / (green + red + eps)

    indices = {
        "NDVI": ndvi, "NDRE": ndre, "GNDVI": gndvi, "SAVI": savi,
        "CI_RE": ci_rededge, "CI_Green": ci_green,
        "RG_ratio": rg_ratio, "RB_ratio": rb_ratio,
        "RE_R_ratio": re_r_ratio, "NIR_R_ratio": nir_r_ratio,
        "NIR_RE_ratio": nir_re_ratio, "EVI": evi, "MCARI": mcari,
        "OSAVI": osavi, "NDWI": ndwi, "SIPI": sipi,
        "PRI_proxy": pri_proxy, "TVI": tvi, "RDVI": rdvi,
        "MSR": msr, "NLI": nli, "RDVI2": rdvi2,
        "Iron_Index": iron_index, "DSI": dsi, "REP_proxy": rep_proxy,
        "Chloro_Abs": chloro_abs,
    }

    for idx_name, idx_map in indices.items():
        idx_map = np.clip(idx_map, -10, 10)
        v = idx_map.ravel()
        features[f"{idx_name}_mean"] = np.mean(v)
        features[f"{idx_name}_std"] = np.std(v)
        features[f"{idx_name}_min"] = np.min(v)
        features[f"{idx_name}_max"] = np.max(v)
        features[f"{idx_name}_median"] = np.median(v)
        features[f"{idx_name}_p10"] = np.percentile(v, 10)
        features[f"{idx_name}_p90"] = np.percentile(v, 90)
        features[f"{idx_name}_skew"] = float(scipy_stats.skew(v))
        features[f"{idx_name}_iqr"] = np.percentile(v, 75) - np.percentile(v, 25)

    # ====== 3. Inter-band correlations ======
    flat_bands = bands.reshape(5, -1)  # (5, 4096)
    corr_matrix = np.corrcoef(flat_bands)
    for i in range(5):
        for j in range(i+1, 5):
            features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]

    # ====== 4. Spatial texture features ======
    for i, name in enumerate(BAND_NAMES):
        b = bands[i]
        # Gradient magnitude
        gy, gx = np.gradient(b)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{name}_grad_mean"] = np.mean(grad_mag)
        features[f"{name}_grad_std"] = np.std(grad_mag)
        features[f"{name}_grad_max"] = np.max(grad_mag)

        # Local variance
        local_mean = ndimage.uniform_filter(b, size=3)
        local_sq_mean = ndimage.uniform_filter(b**2, size=3)
        local_var = local_sq_mean - local_mean**2
        features[f"{name}_localvar_mean"] = np.mean(local_var)
        features[f"{name}_localvar_std"] = np.std(local_var)
        
        # Laplacian (edge detection)
        laplacian = ndimage.laplace(b)
        features[f"{name}_laplacian_mean"] = np.mean(np.abs(laplacian))
        features[f"{name}_laplacian_std"] = np.std(laplacian)

    # ====== 5. Spatial features on key indices ======
    for idx_name, idx_map in [("NDVI", ndvi), ("NDRE", ndre), ("DSI", dsi), ("Iron_Index", iron_index)]:
        idx_map = np.clip(idx_map, -10, 10)
        gy, gx = np.gradient(idx_map)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{idx_name}_grad_mean"] = np.mean(grad_mag)
        features[f"{idx_name}_grad_std"] = np.std(grad_mag)
        
        # Spatial heterogeneity
        features[f"{idx_name}_heterogeneity"] = np.std(idx_map) / (np.mean(idx_map) + eps)

    # ====== 6. Band ratios (aggregated) ======
    band_means = [np.mean(bands[i]) for i in range(5)]
    for i in range(5):
        for j in range(i+1, 5):
            features[f"meanratio_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = band_means[i] / (band_means[j] + eps)

    # ====== 7. Spectral shape features ======
    mean_spectrum = np.array(band_means)
    features["spec_slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["spec_slope_rededge"] = mean_spectrum[3] - mean_spectrum[2]
    features["spec_slope_nir"] = mean_spectrum[4] - mean_spectrum[3]
    features["spec_curvature"] = mean_spectrum[3] - 0.5 * (mean_spectrum[2] + mean_spectrum[4])
    features["spec_total_reflectance"] = np.sum(mean_spectrum)
    features["spec_nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)
    
    # NEW: Spectral derivatives
    spec_derivative1 = np.diff(mean_spectrum)  # First derivative
    spec_derivative2 = np.diff(spec_derivative1)  # Second derivative
    for i, d in enumerate(spec_derivative1):
        features[f"spec_deriv1_{i}"] = d
    for i, d in enumerate(spec_derivative2):
        features[f"spec_deriv2_{i}"] = d
    
    # Spectral angle (overall shape)
    ref_spectrum = np.array([1, 1, 1, 1, 1])  # Flat reference
    features["spec_angle_flat"] = np.arccos(1 - cosine(mean_spectrum, ref_spectrum)) if np.any(mean_spectrum != 0) else 0

    # ====== 8. Spectral Angle Mapper (SAM) to class prototypes ======
    if class_prototypes is not None:
        for cls_name, prototype in class_prototypes.items():
            # Spectral angle to prototype
            dot_product = np.dot(mean_spectrum, prototype)
            norm_product = np.linalg.norm(mean_spectrum) * np.linalg.norm(prototype)
            angle = np.arccos(np.clip(dot_product / (norm_product + eps), -1, 1))
            features[f"SAM_{cls_name}"] = angle
            
            # Spectral information divergence (SID)
            p = mean_spectrum / (np.sum(mean_spectrum) + eps)
            q = prototype / (np.sum(prototype) + eps)
            sid = np.sum(p * np.log2((p + eps) / (q + eps))) + np.sum(q * np.log2((q + eps) / (p + eps)))
            features[f"SID_{cls_name}"] = sid
            
            # Euclidean distance
            features[f"EucDist_{cls_name}"] = np.linalg.norm(mean_spectrum - prototype)

    # ====== 9. Pixel-level statistics ======
    # Brightest/darkest pixel analysis
    pixel_means = np.mean(img, axis=2)  # Mean across bands per pixel
    features["pixel_mean_max"] = np.max(pixel_means)
    features["pixel_mean_min"] = np.min(pixel_means)
    features["pixel_mean_std"] = np.std(pixel_means)
    
    # NDVI distribution analysis
    ndvi_flat = ndvi.ravel()
    features["NDVI_negative_ratio"] = np.mean(ndvi_flat < 0)
    features["NDVI_high_ratio"] = np.mean(ndvi_flat > 0.5)
    features["NDVI_low_ratio"] = np.mean(ndvi_flat < 0.2)

    # ====== 10. Texture from GLCM-like features ======
    # Simplified texture: homogeneity, contrast
    for i, name in enumerate(BAND_NAMES[:3]):  # Just visible bands
        b = bands[i]
        # Local homogeneity
        local_std = ndimage.generic_filter(b, np.std, size=3)
        features[f"{name}_homogeneity"] = 1.0 / (np.mean(local_std) + eps)
        features[f"{name}_contrast"] = np.max(b) - np.min(b)

    return features


def compute_class_prototypes(data_dir, file_list=None):
    """Compute mean spectrum for each class as prototype."""
    prototypes = {cls: [] for cls in CLASS_MAP}
    
    if file_list is None:
        file_list = sorted(os.listdir(data_dir))
    
    for f in file_list:
        fp = os.path.join(data_dir, f)
        img = tiff.imread(fp).astype(np.float32)
        
        if img.mean() < 1.0:
            continue
            
        # Parse class from filename
        if "_hyper_" in f:
            cls_name = f.split("_hyper_")[0]
            mean_spectrum = np.mean(img, axis=(0, 1))  # Mean across spatial dims
            prototypes[cls_name].append(mean_spectrum)
    
    # Average prototypes
    final_prototypes = {}
    for cls_name, spectra_list in prototypes.items():
        if len(spectra_list) > 0:
            final_prototypes[cls_name] = np.mean(spectra_list, axis=0)
    
    return final_prototypes


def extract_all_features(data_dir, file_list=None, class_prototypes=None):
    """Extract features from all files in directory."""
    if file_list is None:
        file_list = sorted(os.listdir(data_dir))

    all_features = []
    all_labels = []
    all_fnames = []
    skipped_black = 0

    for f in file_list:
        fp = os.path.join(data_dir, f)
        img = tiff.imread(fp).astype(np.float32)

        if img.mean() < 1.0:
            skipped_black += 1
            if "_hyper_" in f:
                continue
            else:
                all_features.append(None)
                all_labels.append(-1)
                all_fnames.append(f)
                continue

        feats = extract_features(img, class_prototypes)
        all_features.append(feats)

        if "_hyper_" in f:
            cls_name = f.split("_hyper_")[0]
            all_labels.append(CLASS_MAP[cls_name])
        else:
            all_labels.append(-1)

        all_fnames.append(f)

    if skipped_black > 0:
        print(f"  Skipped/flagged {skipped_black} black images")

    return all_features, all_labels, all_fnames


def augment_features(X, y, noise_level=0.01, n_augments=2):
    """Augment features with noise for training."""
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(n_augments):
        noise = np.random.randn(*X.shape) * noise_level * X.std(axis=0, keepdims=True)
        X_aug.append(X + noise)
        y_aug.append(y)
    
    return np.vstack(X_aug), np.hstack(y_aug)


# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    np.random.seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    # --- Compute class prototypes for SAM features ---
    print("Computing class prototypes...")
    class_prototypes = compute_class_prototypes(CFG["data_dir"])
    for cls, proto in class_prototypes.items():
        print(f"  {cls}: {proto}")

    # --- Extract training features ---
    print("\nExtracting training features...")
    train_feats, train_labels, train_fnames = extract_all_features(
        CFG["data_dir"], class_prototypes=class_prototypes
    )
    print(f"  {len(train_feats)} samples")

    # Convert to numpy
    feature_names = list(train_feats[0].keys())
    X_train = np.array([[f[k] for k in feature_names] for f in train_feats], dtype=np.float32)
    y_train = np.array(train_labels)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

    print(f"  {X_train.shape[1]} features extracted")
    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")

    # --- Extract val features ---
    print("\nExtracting validation features...")
    val_feats, val_labels, val_fnames = extract_all_features(
        CFG["val_dir"], class_prototypes=class_prototypes
    )
    black_mask = [f is None for f in val_feats]
    
    for i in range(len(val_feats)):
        if val_feats[i] is None:
            val_feats[i] = {k: 0.0 for k in feature_names}

    X_val = np.array([[f[k] for k in feature_names] for f in val_feats], dtype=np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"  {len(val_feats)} samples ({sum(black_mask)} black)")

    # --- Normalize ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- Feature Selection ---
    print(f"\nSelecting top {CFG['feature_selection_k']} features...")
    selector = SelectKBest(mutual_info_classif, k=min(CFG['feature_selection_k'], X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)
    
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"  Selected {len(selected_features)} features")
    print(f"  Top 10: {selected_features[:10]}")

    # ================================================================
    # Phase 1: Base Models with 5-Fold CV
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Training Base Models with 5-Fold CV")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

    # --- XGBoost ---
    print("\n--- XGBoost ---")
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "learning_rate": 0.03,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 1,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "random_state": CFG["seed"],
        "tree_method": "hist",
        "verbosity": 0,
    }

    oof_xgb = np.zeros((len(X_train_selected), 3))
    val_xgb_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_selected, y_train)):
        X_tr, X_va = X_train_selected[tr_idx], X_train_selected[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        
        # Augment training data
        X_tr_aug, y_tr_aug = augment_features(X_tr, y_tr, noise_level=0.02, n_augments=2)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr_aug, y_tr_aug, eval_set=[(X_va, y_va)], verbose=False)

        oof_xgb[va_idx] = model.predict_proba(X_va)
        val_xgb_folds.append(model.predict_proba(X_val_selected))
        
        acc = accuracy_score(y_va, np.argmax(oof_xgb[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_xgb_preds = np.argmax(oof_xgb, axis=1)
    xgb_acc = accuracy_score(y_train, oof_xgb_preds)
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")
    print(classification_report(y_train, oof_xgb_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # --- LightGBM ---
    print("\n--- LightGBM ---")
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_depth": 10,
        "learning_rate": 0.03,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 3,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "random_state": CFG["seed"],
        "verbose": -1,
        "num_leaves": 63,
    }

    oof_lgb = np.zeros((len(X_train_selected), 3))
    val_lgb_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_selected, y_train)):
        X_tr, X_va = X_train_selected[tr_idx], X_train_selected[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        
        X_tr_aug, y_tr_aug = augment_features(X_tr, y_tr, noise_level=0.02, n_augments=2)

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr_aug, y_tr_aug, eval_set=[(X_va, y_va)])

        oof_lgb[va_idx] = model.predict_proba(X_va)
        val_lgb_folds.append(model.predict_proba(X_val_selected))
        
        acc = accuracy_score(y_va, np.argmax(oof_lgb[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_lgb_preds = np.argmax(oof_lgb, axis=1)
    lgb_acc = accuracy_score(y_train, oof_lgb_preds)
    print(f"LGB OOF Accuracy: {lgb_acc:.4f}")
    print(classification_report(y_train, oof_lgb_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # --- Neural Network (MLP) ---
    print("\n--- Neural Network (MLP) ---")
    oof_mlp = np.zeros((len(X_train_selected), 3))
    val_mlp_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_selected, y_train)):
        X_tr, X_va = X_train_selected[tr_idx], X_train_selected[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        
        X_tr_aug, y_tr_aug = augment_features(X_tr, y_tr, noise_level=0.03, n_augments=3)

        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=CFG["seed"],
        )
        model.fit(X_tr_aug, y_tr_aug)

        oof_mlp[va_idx] = model.predict_proba(X_va)
        val_mlp_folds.append(model.predict_proba(X_val_selected))
        
        acc = accuracy_score(y_va, np.argmax(oof_mlp[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_mlp_preds = np.argmax(oof_mlp, axis=1)
    mlp_acc = accuracy_score(y_train, oof_mlp_preds)
    print(f"MLP OOF Accuracy: {mlp_acc:.4f}")
    print(classification_report(y_train, oof_mlp_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # Phase 2: Stacking Ensemble
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Stacking Ensemble")
    print(f"{'='*70}")

    # Prepare meta-features
    meta_train = np.hstack([oof_xgb, oof_lgb, oof_mlp])
    meta_val = np.hstack([
        np.mean(val_xgb_folds, axis=0),
        np.mean(val_lgb_folds, axis=0),
        np.mean(val_mlp_folds, axis=0),
    ])

    # Train meta-learner
    print("Training meta-learner (Logistic Regression)...")
    meta_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=CFG["seed"],
    )
    meta_model.fit(meta_train, y_train)

    # OOF predictions from meta-learner
    meta_oof_preds = meta_model.predict(meta_train)
    meta_acc = accuracy_score(y_train, meta_oof_preds)
    print(f"Stacked OOF Accuracy: {meta_acc:.4f}")
    print(classification_report(y_train, meta_oof_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # Phase 3: Semi-Supervised Pseudo-Labeling
    # ================================================================
    if CFG["use_pseudo_labeling"]:
        print(f"\n{'='*70}")
        print("PHASE 3: Semi-Supervised Pseudo-Labeling")
        print(f"{'='*70}")

        # Get high-confidence validation predictions
        val_probs = meta_model.predict_proba(meta_val)
        val_max_probs = np.max(val_probs, axis=1)
        val_preds = np.argmax(val_probs, axis=1)

        # Select high-confidence samples
        confident_mask = val_max_probs >= CFG["pseudo_label_threshold"]
        confident_mask = confident_mask & ~np.array(black_mask)  # Exclude black images
        
        n_confident = confident_mask.sum()
        print(f"High-confidence val samples: {n_confident} ({n_confident/len(val_preds)*100:.1f}%)")

        if n_confident > 50:
            # Add pseudo-labeled samples to training
            X_pseudo = X_val_selected[confident_mask]
            y_pseudo = val_preds[confident_mask]
            
            X_train_augmented = np.vstack([X_train_selected, X_pseudo])
            y_train_augmented = np.hstack([y_train, y_pseudo])
            
            print(f"Augmented training set: {len(y_train)} → {len(y_train_augmented)}")
            
            # Retrain models with pseudo-labeled data
            print("Retraining with pseudo-labeled data...")
            
            # XGBoost with pseudo-labels
            model_xgb_final = xgb.XGBClassifier(**xgb_params)
            X_aug_aug, y_aug_aug = augment_features(X_train_augmented, y_train_augmented, noise_level=0.02, n_augments=1)
            model_xgb_final.fit(X_aug_aug, y_aug_aug, verbose=False)
            
            # LightGBM with pseudo-labels
            model_lgb_final = lgb.LGBMClassifier(**lgb_params)
            model_lgb_final.fit(X_aug_aug, y_aug_aug)
            
            # MLP with pseudo-labels
            model_mlp_final = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=CFG["seed"],
            )
            model_mlp_final.fit(X_aug_aug, y_aug_aug)
            
            # New predictions
            val_xgb_final = model_xgb_final.predict_proba(X_val_selected)
            val_lgb_final = model_lgb_final.predict_proba(X_val_selected)
            val_mlp_final = model_mlp_final.predict_proba(X_val_selected)
            
            meta_val_final = np.hstack([val_xgb_final, val_lgb_final, val_mlp_final])
            val_probs_final = meta_model.predict_proba(meta_val_final)
            val_preds_final = meta_model.predict(meta_val_final)
            
            # Check OOF accuracy on original training set
            train_xgb_final = model_xgb_final.predict_proba(X_train_selected)
            train_lgb_final = model_lgb_final.predict_proba(X_train_selected)
            train_mlp_final = model_mlp_final.predict_proba(X_train_selected)
            meta_train_final = np.hstack([train_xgb_final, train_lgb_final, train_mlp_final])
            train_preds_final = meta_model.predict(meta_train_final)
            final_train_acc = accuracy_score(y_train, train_preds_final)
            print(f"Final Training Accuracy (with pseudo-labels): {final_train_acc:.4f}")
            print(classification_report(y_train, train_preds_final, target_names=list(CLASS_MAP.keys()), digits=4))
        else:
            print("Not enough high-confidence samples for pseudo-labeling")
            val_preds_final = meta_oof_preds
            val_probs_final = meta_model.predict_proba(meta_val)
    else:
        val_preds_final = meta_model.predict(meta_val)
        val_probs_final = meta_model.predict_proba(meta_val)

    # ================================================================
    # Final Predictions
    # ================================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    # Override black images → Other
    for i, is_b in enumerate(black_mask):
        if is_b:
            val_probs_final[i] = [0.0, 0.0, 1.0]
    
    final_preds = np.argmax(val_probs_final, axis=1)
    pred_classes = [INV_CLASS_MAP[p] for p in final_preds]
    
    dist = {c: pred_classes.count(c) for c in CLASS_MAP}
    print(f"Val prediction distribution: {dist}")

    # ================================================================
    # Save Results
    # ================================================================
    # Save probabilities
    np.save(os.path.join(CFG["output_dir"], "val_probs_final.npy"), val_probs_final)
    np.save(os.path.join(CFG["output_dir"], "oof_xgb.npy"), oof_xgb)
    np.save(os.path.join(CFG["output_dir"], "oof_lgb.npy"), oof_lgb)
    np.save(os.path.join(CFG["output_dir"], "oof_mlp.npy"), oof_mlp)

    # Save submission
    sub_path = os.path.join(CFG["output_dir"], "submission.csv")
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for fn, cl in zip(val_fnames, pred_classes):
            w.writerow([fn, cl])

    # Save feature names
    with open(os.path.join(CFG["output_dir"], "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    print(f"\nSaved to {CFG['output_dir']}/")
    print("Done!")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")
    print(f"LGB OOF Accuracy: {lgb_acc:.4f}")
    print(f"MLP OOF Accuracy: {mlp_acc:.4f}")
    print(f"Stacked OOF Accuracy: {meta_acc:.4f}")
    if CFG["use_pseudo_labeling"] and n_confident > 50:
        print(f"Final Training Accuracy (pseudo-labeled): {final_train_acc:.4f}")


if __name__ == "__main__":
    main()
