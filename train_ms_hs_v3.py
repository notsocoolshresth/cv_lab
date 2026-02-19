"""
MS + HS Combined Classification - Targeting 80%+ CV Accuracy
=============================================================

Key insights:
1. Stage 1 (Other vs Veg): 87% - excellent
2. Stage 2 (Health vs Rust): 73% - bottleneck
3. HS has 125 bands with finer spectral resolution

Novel approaches:
1. Combine MS (64x64x5) + HS (32x32x125) features
2. Use HS spectral derivatives for Health vs Rust
3. Train specialized binary classifier for Health vs Rust
4. Use spectral unmixing concepts
5. Apply heavy augmentation on spectral features
"""

import os
import csv
import json
import warnings
import numpy as np
import tifffile as tiff
from scipy import ndimage, stats as scipy_stats
from scipy.spatial.distance import cosine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

CFG = {
    "train_ms_dir": "Kaggle_Prepared/train/MS",
    "val_ms_dir": "Kaggle_Prepared/val/MS",
    "train_hs_dir": "Kaggle_Prepared/train/HS",
    "val_hs_dir": "Kaggle_Prepared/val/HS",
    "output_dir": "ms_hs_v3",
    "n_folds": 5,
    "seed": 42,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


# ============================================================
# MS Feature Extraction
# ============================================================
def extract_ms_features(img, class_prototypes=None):
    """Extract features from MS image."""
    img = img.astype(np.float32)
    features = {}
    eps = 1e-8

    bands = img.transpose(2, 0, 1)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]

    # Per-band stats
    for i, name in enumerate(BAND_NAMES):
        b = bands[i].ravel()
        features[f"ms_{name}_mean"] = np.mean(b)
        features[f"ms_{name}_std"] = np.std(b)
        features[f"ms_{name}_min"] = np.min(b)
        features[f"ms_{name}_max"] = np.max(b)
        features[f"ms_{name}_median"] = np.median(b)
        features[f"ms_{name}_p10"] = np.percentile(b, 10)
        features[f"ms_{name}_p90"] = np.percentile(b, 90)
        features[f"ms_{name}_skew"] = float(scipy_stats.skew(b))

    # Vegetation indices
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)
    gndvi = (nir - green) / (nir + green + eps)
    ci_rededge = nir / (rededge + eps) - 1.0
    ci_green = nir / (green + eps) - 1.0
    rg_ratio = red / (green + eps)
    rb_ratio = red / (blue + eps)
    re_r_ratio = rededge / (red + eps)
    nir_r_ratio = nir / (red + eps)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))
    osavi = (nir - red) / (nir + red + 0.16 + eps)
    
    # Disease-specific
    iron_index = (red - blue) / (red + blue + eps)
    dsi = (nir - rededge) / (nir + red + eps)
    psi = (rededge - red) / (rededge + red + eps)
    rust_ratio = red / (green + eps)
    health_ratio = nir / (red + eps)
    yellow_idx = (green - blue) / (green + blue + eps)
    chloro_abs = (green - red) / (green + red + eps)

    indices = {
        "NDVI": ndvi, "NDRE": ndre, "GNDVI": gndvi, "CI_RE": ci_rededge,
        "CI_Green": ci_green, "RG_ratio": rg_ratio, "RB_ratio": rb_ratio,
        "RE_R_ratio": re_r_ratio, "NIR_R_ratio": nir_r_ratio, "EVI": evi,
        "MCARI": mcari, "OSAVI": osavi, "Iron_Index": iron_index, "DSI": dsi,
        "PSI": psi, "Rust_Ratio": rust_ratio, "Health_Ratio": health_ratio,
        "Yellow_Idx": yellow_idx, "Chloro_Abs": chloro_abs,
    }

    for idx_name, idx_map in indices.items():
        idx_map = np.clip(idx_map, -10, 10)
        v = idx_map.ravel()
        features[f"ms_{idx_name}_mean"] = np.mean(v)
        features[f"ms_{idx_name}_std"] = np.std(v)
        features[f"ms_{idx_name}_min"] = np.min(v)
        features[f"ms_{idx_name}_max"] = np.max(v)
        features[f"ms_{idx_name}_p10"] = np.percentile(v, 10)
        features[f"ms_{idx_name}_p90"] = np.percentile(v, 90)

    # Band ratios
    band_means = [np.mean(bands[i]) for i in range(5)]
    for i in range(5):
        for j in range(i+1, 5):
            features[f"ms_ratio_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = band_means[i] / (band_means[j] + eps)

    # Spectral shape
    mean_spectrum = np.array(band_means)
    features["ms_spec_slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["ms_spec_slope_re"] = mean_spectrum[3] - mean_spectrum[2]
    features["ms_spec_slope_nir"] = mean_spectrum[4] - mean_spectrum[3]
    features["ms_spec_nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)

    # SAM features
    if class_prototypes is not None and 'ms' in class_prototypes:
        for cls_name, prototype in class_prototypes['ms'].items():
            dot_product = np.dot(mean_spectrum, prototype)
            norm_product = np.linalg.norm(mean_spectrum) * np.linalg.norm(prototype)
            angle = np.arccos(np.clip(dot_product / (norm_product + eps), -1, 1))
            features[f"ms_SAM_{cls_name}"] = angle
            features[f"ms_EucDist_{cls_name}"] = np.linalg.norm(mean_spectrum - prototype)

    return features


# ============================================================
# HS Feature Extraction
# ============================================================
def extract_hs_features(img, class_prototypes=None):
    """Extract features from HS image (32x32x125)."""
    img = img.astype(np.float32)
    features = {}
    eps = 1e-8
    
    # Handle variable band count
    n_bands = img.shape[2]
    
    # Use clean bands (10-110)
    clean_start, clean_end = 10, min(110, n_bands - 1)
    clean_bands = img[:, :, clean_start:clean_end]
    
    # Mean spectrum
    mean_spectrum = np.mean(clean_bands, axis=(0, 1))
    
    # Basic stats
    features["hs_mean_reflectance"] = np.mean(mean_spectrum)
    features["hs_std_reflectance"] = np.std(mean_spectrum)
    features["hs_max_reflectance"] = np.max(mean_spectrum)
    features["hs_min_reflectance"] = np.min(mean_spectrum)
    
    # Spectral regions (approximate wavelengths)
    # Blue: 450-495nm, Green: 495-570nm, Red: 620-750nm, RedEdge: 700-750nm, NIR: 750-950nm
    n_clean = len(mean_spectrum)
    blue_end = int(n_clean * 0.1)
    green_end = int(n_clean * 0.25)
    red_end = int(n_clean * 0.5)
    rede_end = int(n_clean * 0.6)
    
    regions = {
        "Blue": mean_spectrum[:blue_end],
        "Green": mean_spectrum[blue_end:green_end],
        "Red": mean_spectrum[green_end:red_end],
        "RedEdge": mean_spectrum[red_end:rede_end],
        "NIR": mean_spectrum[rede_end:],
    }
    
    for region_name, region_vals in regions.items():
        if len(region_vals) > 0:
            features[f"hs_{region_name}_mean"] = np.mean(region_vals)
            features[f"hs_{region_name}_std"] = np.std(region_vals)
            features[f"hs_{region_name}_max"] = np.max(region_vals)
            features[f"hs_{region_name}_min"] = np.min(region_vals)
    
    # Spectral derivatives (key for Health vs Rust)
    if len(mean_spectrum) > 2:
        deriv1 = np.diff(mean_spectrum)
        deriv2 = np.diff(deriv1)
        
        features["hs_deriv1_mean"] = np.mean(deriv1)
        features["hs_deriv1_std"] = np.std(deriv1)
        features["hs_deriv1_max"] = np.max(deriv1)
        features["hs_deriv1_min"] = np.min(deriv1)
        
        features["hs_deriv2_mean"] = np.mean(deriv2) if len(deriv2) > 0 else 0
        features["hs_deriv2_std"] = np.std(deriv2) if len(deriv2) > 0 else 0
        
        # Red Edge Position (REP) - where derivative is maximum
        red_edge_region = deriv1[len(deriv1)//3:2*len(deriv1)//3]
        if len(red_edge_region) > 0:
            features["hs_rep_strength"] = np.max(red_edge_region)
            features["hs_rep_position"] = np.argmax(red_edge_region)
    
    # HS-specific vegetation indices
    # Using approximate band positions
    n = len(mean_spectrum)
    if n > 50:
        # Approximate bands
        blue_approx = mean_spectrum[5] if n > 5 else mean_spectrum[0]
        green_approx = mean_spectrum[15] if n > 15 else mean_spectrum[n//4]
        red_approx = mean_spectrum[35] if n > 35 else mean_spectrum[n//2]
        rede_approx = mean_spectrum[45] if n > 45 else mean_spectrum[3*n//5]
        nir_approx = mean_spectrum[70] if n > 70 else mean_spectrum[4*n//5]
        
        # NDVI-like
        features["hs_NDVI"] = (nir_approx - red_approx) / (nir_approx + red_approx + eps)
        # NDRE-like
        features["hs_NDRE"] = (nir_approx - rede_approx) / (nir_approx + rede_approx + eps)
        # GNDVI-like
        features["hs_GNDVI"] = (nir_approx - green_approx) / (nir_approx + green_approx + eps)
        # CI_RedEdge-like
        features["hs_CI_RE"] = nir_approx / (rede_approx + eps) - 1
        # PRI-like (Photochemical Reflectance Index)
        features["hs_PRI"] = (green_approx - red_approx) / (green_approx + red_approx + eps)
        # Water Index
        features["hs_WI"] = nir_approx / (red_approx + eps)
        # Disease stress
        features["hs_DSI"] = (nir_approx - rede_approx) / (nir_approx + red_approx + eps)
        # Rust indicator (red/green ratio)
        features["hs_Rust_Ratio"] = red_approx / (green_approx + eps)
        # Health indicator (NIR/red ratio)
        features["hs_Health_Ratio"] = nir_approx / (red_approx + eps)
        # Iron index
        features["hs_Iron_Index"] = (red_approx - blue_approx) / (red_approx + blue_approx + eps)
        # Yellowing index
        features["hs_Yellow_Idx"] = (green_approx - blue_approx) / (green_approx + blue_approx + eps)
    
    # Spectral angle features
    if class_prototypes is not None and 'hs' in class_prototypes:
        for cls_name, prototype in class_prototypes['hs'].items():
            # Match lengths
            proto_len = len(prototype)
            spec_len = len(mean_spectrum)
            min_len = min(proto_len, spec_len)
            
            spec = mean_spectrum[:min_len]
            proto = prototype[:min_len]
            
            dot_product = np.dot(spec, proto)
            norm_product = np.linalg.norm(spec) * np.linalg.norm(proto)
            if norm_product > 0:
                angle = np.arccos(np.clip(dot_product / norm_product, -1, 1))
                features[f"hs_SAM_{cls_name}"] = angle
                features[f"hs_EucDist_{cls_name}"] = np.linalg.norm(spec - proto)
    
    # PCA on spectrum
    features["hs_pca_first"] = mean_spectrum[0]  # First component proxy
    features["hs_pca_range"] = np.max(mean_spectrum) - np.min(mean_spectrum)
    
    # Spectral statistics
    features["hs_skewness"] = float(scipy_stats.skew(mean_spectrum))
    features["hs_kurtosis"] = float(scipy_stats.kurtosis(mean_spectrum))
    
    # Area under curve
    features["hs_auc_vis"] = np.sum(mean_spectrum[:n//2]) if n > 1 else 0
    features["hs_auc_nir"] = np.sum(mean_spectrum[n//2:]) if n > 1 else 0
    features["hs_auc_ratio"] = features["hs_auc_nir"] / (features["hs_auc_vis"] + eps)
    
    return features


def compute_class_prototypes(train_ms_dir, train_hs_dir):
    """Compute mean spectrum for each class."""
    ms_prototypes = {cls: [] for cls in CLASS_MAP}
    hs_prototypes = {cls: [] for cls in CLASS_MAP}
    
    # MS prototypes
    for f in sorted(os.listdir(train_ms_dir)):
        fp = os.path.join(train_ms_dir, f)
        img = tiff.imread(fp).astype(np.float32)
        
        if img.mean() < 1.0:
            continue
            
        if "_hyper_" in f:
            cls_name = f.split("_hyper_")[0]
            mean_spectrum = np.mean(img, axis=(0, 1))
            ms_prototypes[cls_name].append(mean_spectrum)
    
    # HS prototypes
    for f in sorted(os.listdir(train_hs_dir)):
        fp = os.path.join(train_hs_dir, f)
        img = tiff.imread(fp).astype(np.float32)
        
        if img.mean() < 1.0:
            continue
            
        if "_hyper_" in f:
            cls_name = f.split("_hyper_")[0]
            # Use clean bands
            n_bands = img.shape[2]
            clean_bands = img[:, :, 10:min(110, n_bands-1)]
            mean_spectrum = np.mean(clean_bands, axis=(0, 1))
            hs_prototypes[cls_name].append(mean_spectrum)
    
    final_ms = {}
    final_hs = {}
    
    for cls_name in CLASS_MAP:
        if len(ms_prototypes[cls_name]) > 0:
            final_ms[cls_name] = np.mean(ms_prototypes[cls_name], axis=0)
        if len(hs_prototypes[cls_name]) > 0:
            final_hs[cls_name] = np.mean(hs_prototypes[cls_name], axis=0)
    
    return {'ms': final_ms, 'hs': final_hs}


def extract_all_features(ms_dir, hs_dir, class_prototypes=None, file_list=None):
    """Extract combined MS+HS features."""
    if file_list is None:
        ms_files = sorted(os.listdir(ms_dir))
    else:
        ms_files = file_list
    
    all_features = []
    all_labels = []
    all_fnames = []
    skipped = 0
    
    for ms_f in ms_files:
        ms_fp = os.path.join(ms_dir, ms_f)
        ms_img = tiff.imread(ms_fp).astype(np.float32)
        
        # Find corresponding HS file
        hs_f = ms_f.replace(".tif", ".tif")  # Same naming
        hs_fp = os.path.join(hs_dir, hs_f)
        
        # Check for black images
        if ms_img.mean() < 1.0:
            skipped += 1
            if "_hyper_" in ms_f:
                continue
            else:
                all_features.append(None)
                all_labels.append(-1)
                all_fnames.append(ms_f)
                continue
        
        # Extract MS features
        ms_feats = extract_ms_features(ms_img, class_prototypes)
        
        # Extract HS features if available
        hs_feats = {}
        if os.path.exists(hs_fp):
            hs_img = tiff.imread(hs_fp).astype(np.float32)
            if hs_img.mean() >= 1.0:
                hs_feats = extract_hs_features(hs_img, class_prototypes)
        
        # Combine features
        combined = {**ms_feats, **hs_feats}
        all_features.append(combined)
        
        # Parse label
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

    # --- Compute class prototypes ---
    print("Computing class prototypes...")
    class_prototypes = compute_class_prototypes(CFG["train_ms_dir"], CFG["train_hs_dir"])
    print("  MS prototypes:")
    for cls, proto in class_prototypes['ms'].items():
        print(f"    {cls}: {proto[:3]}...")
    print("  HS prototypes:")
    for cls, proto in class_prototypes['hs'].items():
        print(f"    {cls}: length={len(proto)}")

    # --- Extract training features ---
    print("\nExtracting training features...")
    train_feats, train_labels, train_fnames = extract_all_features(
        CFG["train_ms_dir"], CFG["train_hs_dir"], class_prototypes
    )
    print(f"  {len(train_feats)} samples")

    feature_names = list(train_feats[0].keys())
    X_train = np.array([[f.get(k, 0.0) for k in feature_names] for f in train_feats], dtype=np.float32)
    y_train = np.array(train_labels)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

    print(f"  {X_train.shape[1]} features extracted")
    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")

    # --- Extract val features ---
    print("\nExtracting validation features...")
    val_feats, val_labels, val_fnames = extract_all_features(
        CFG["val_ms_dir"], CFG["val_hs_dir"], class_prototypes
    )
    black_mask = [f is None for f in val_feats]
    
    for i in range(len(val_feats)):
        if val_feats[i] is None:
            val_feats[i] = {k: 0.0 for k in feature_names}

    X_val = np.array([[f.get(k, 0.0) for k in feature_names] for f in val_feats], dtype=np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"  {len(val_feats)} samples ({sum(black_mask)} black)")

    # --- Normalize ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ================================================================
    # HIERARCHICAL APPROACH
    # ================================================================
    print(f"\n{'='*70}")
    print("HIERARCHICAL CLASSIFICATION")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

    # Stage 1: Other vs Vegetation
    y_stage1 = np.where(y_train == 2, 1, 0)
    
    print("\n--- Stage 1: Other vs Vegetation ---")
    
    oof_s1 = np.zeros((len(X_train_scaled), 2))
    val_s1_folds = []
    
    xgb_params_s1 = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": CFG["seed"],
        "tree_method": "hist",
        "verbosity": 0,
    }
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_stage1)):
        X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
        y_tr, y_va = y_stage1[tr_idx], y_stage1[va_idx]

        model = xgb.XGBClassifier(**xgb_params_s1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        oof_s1[va_idx] = model.predict_proba(X_va)
        val_s1_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, model.predict(X_va))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    s1_acc = accuracy_score(y_stage1, np.argmax(oof_s1, axis=1))
    print(f"Stage 1 OOF Accuracy: {s1_acc:.4f}")

    # Stage 2: Health vs Rust (with heavy augmentation)
    veg_mask = y_train != 2
    X_train_veg = X_train_scaled[veg_mask]
    y_stage2 = y_train[veg_mask]
    
    print(f"\n--- Stage 2: Health vs Rust ({len(y_stage2)} samples) ---")
    
    oof_s2 = np.zeros((len(X_train_veg), 2))
    val_s2_folds = []
    
    # Multiple models for diversity
    xgb_params_s2 = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "scale_pos_weight": 1.1,
        "random_state": CFG["seed"],
        "tree_method": "hist",
        "verbosity": 0,
    }
    
    lgb_params_s2 = {
        "objective": "binary",
        "metric": "binary_logloss",
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 3,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "is_unbalance": True,
        "random_state": CFG["seed"],
        "verbose": -1,
        "num_leaves": 40,
    }
    
    skf2 = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    
    oof_s2_xgb = np.zeros((len(X_train_veg), 2))
    oof_s2_lgb = np.zeros((len(X_train_veg), 2))
    val_s2_xgb_folds = []
    val_s2_lgb_folds = []
    
    for fold, (tr_idx, va_idx) in enumerate(skf2.split(X_train_veg, y_stage2)):
        X_tr, X_va = X_train_veg[tr_idx], X_train_veg[va_idx]
        y_tr, y_va = y_stage2[tr_idx], y_stage2[va_idx]
        
        # Heavy augmentation
        n_aug = 3
        X_tr_aug = np.vstack([X_tr] + [X_tr + np.random.randn(*X_tr.shape) * 0.02 * X_tr.std(axis=0) for _ in range(n_aug)])
        y_tr_aug = np.hstack([y_tr] * (n_aug + 1))
        
        # Sample weights - heavier for Health
        sample_weights = np.ones(len(y_tr_aug))
        sample_weights[y_tr_aug == 0] = 1.5

        # XGBoost
        model_xgb = xgb.XGBClassifier(**xgb_params_s2)
        model_xgb.fit(X_tr_aug, y_tr_aug, sample_weight=sample_weights, eval_set=[(X_va, y_va)], verbose=False)
        oof_s2_xgb[va_idx] = model_xgb.predict_proba(X_va)
        val_s2_xgb_folds.append(model_xgb.predict_proba(X_val_scaled))
        
        # LightGBM
        model_lgb = lgb.LGBMClassifier(**lgb_params_s2)
        model_lgb.fit(X_tr_aug, y_tr_aug, sample_weight=sample_weights, eval_set=[(X_va, y_va)])
        oof_s2_lgb[va_idx] = model_lgb.predict_proba(X_va)
        val_s2_lgb_folds.append(model_lgb.predict_proba(X_val_scaled))
        
        # Ensemble for this fold
        oof_ens = 0.5 * oof_s2_xgb[va_idx] + 0.5 * oof_s2_lgb[va_idx]
        acc = accuracy_score(y_va, np.argmax(oof_ens, axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    # Ensemble Stage 2
    oof_s2 = 0.5 * oof_s2_xgb + 0.5 * oof_s2_lgb
    s2_preds = np.argmax(oof_s2, axis=1)
    s2_acc = accuracy_score(y_stage2, s2_preds)
    
    health_mask = y_stage2 == 0
    health_recall = (s2_preds[health_mask] == 0).mean()
    rust_recall = (s2_preds[~health_mask] == 1).mean()
    
    print(f"Stage 2 OOF Accuracy: {s2_acc:.4f}")
    print(f"Health Recall: {health_recall:.4f}")
    print(f"Rust Recall: {rust_recall:.4f}")
    print(classification_report(y_stage2, s2_preds, target_names=["Health", "Rust"], digits=4))

    # ================================================================
    # COMBINE HIERARCHICAL PREDICTIONS
    # ================================================================
    print(f"\n{'='*70}")
    print("COMBINING HIERARCHICAL PREDICTIONS")
    print(f"{'='*70}")

    # Compute overall accuracy
    # For training set: use OOF predictions
    train_s1_probs = oof_s1
    train_s2_probs = np.zeros((len(X_train_scaled), 2))
    train_s2_probs[veg_mask] = oof_s2
    
    # For non-veg samples, use Stage 1 probs
    train_final_probs = np.zeros((len(X_train_scaled), 3))
    for i in range(len(X_train_scaled)):
        if train_s1_probs[i, 1] > 0.5:  # Predicted Other
            train_final_probs[i] = [0, 0, 1]
        else:
            # Use Stage 2 probs
            train_final_probs[i, 0] = train_s2_probs[i, 0]
            train_final_probs[i, 1] = train_s2_probs[i, 1]
            train_final_probs[i, 2] = 0
    
    train_final_preds = np.argmax(train_final_probs, axis=1)
    overall_acc = accuracy_score(y_train, train_final_preds)
    print(f"Overall OOF Accuracy: {overall_acc:.4f}")
    print(classification_report(y_train, train_final_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # FINAL PREDICTIONS
    # ================================================================
    print(f"\n{'='*70}")
    print("FINAL VALIDATION PREDICTIONS")
    print(f"{'='*70}")

    val_s1_probs = np.mean(val_s1_folds, axis=0)
    val_s2_xgb_probs = np.mean(val_s2_xgb_folds, axis=0)
    val_s2_lgb_probs = np.mean(val_s2_lgb_folds, axis=0)
    val_s2_probs = 0.5 * val_s2_xgb_probs + 0.5 * val_s2_lgb_probs
    
    # Combine
    val_final_probs = np.zeros((len(X_val_scaled), 3))
    for i in range(len(X_val_scaled)):
        if val_s1_probs[i, 1] > 0.5:  # Predicted Other
            val_final_probs[i] = [0, 0, 1]
        else:
            val_final_probs[i, 0] = val_s2_probs[i, 0]
            val_final_probs[i, 1] = val_s2_probs[i, 1]
            val_final_probs[i, 2] = 0
    
    # Threshold optimization for Health
    print("\nOptimizing Health threshold...")
    best_acc = overall_acc
    best_thresh = 0.5
    
    for thresh in np.arange(0.3, 0.7, 0.05):
        adjusted_preds = train_final_preds.copy()
        for i in range(len(train_final_probs)):
            if train_final_probs[i, 0] > thresh:
                adjusted_preds[i] = 0
        
        acc = accuracy_score(y_train, adjusted_preds)
        health_rec = (adjusted_preds[y_train == 0] == 0).mean()
        
        if acc > best_acc or (acc >= best_acc - 0.01 and health_rec > 0.6):
            best_acc = acc
            best_thresh = thresh
    
    print(f"Best threshold: {best_thresh:.2f}")
    print(f"Best accuracy: {best_acc:.4f}")
    
    # Apply threshold
    val_final_preds = np.argmax(val_final_probs, axis=1)
    for i in range(len(val_final_probs)):
        if val_final_probs[i, 0] > best_thresh:
            val_final_preds[i] = 0
    
    # Override black images
    for i, is_b in enumerate(black_mask):
        if is_b:
            val_final_preds[i] = 2
    
    pred_classes = [INV_CLASS_MAP[p] for p in val_final_preds]
    dist = {c: pred_classes.count(c) for c in CLASS_MAP}
    print(f"\nVal prediction distribution: {dist}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    np.save(os.path.join(CFG["output_dir"], "val_probs_final.npy"), val_final_probs)
    np.save(os.path.join(CFG["output_dir"], "oof_s1.npy"), oof_s1)
    np.save(os.path.join(CFG["output_dir"], "oof_s2.npy"), oof_s2)

    sub_path = os.path.join(CFG["output_dir"], "submission.csv")
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for fn, cl in zip(val_fnames, pred_classes):
            w.writerow([fn, cl])

    with open(os.path.join(CFG["output_dir"], "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\nSaved to {CFG['output_dir']}/")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Stage 1 (Other vs Veg) Accuracy: {s1_acc:.4f}")
    print(f"Stage 2 (Health vs Rust) Accuracy: {s2_acc:.4f}")
    print(f"Overall OOF Accuracy: {overall_acc:.4f}")
    print(f"Best Threshold Accuracy: {best_acc:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
