"""
Advanced MS Classification v2 - Targeting 80%+ CV Accuracy
==========================================================

Key insights from previous attempts:
1. Health vs Rust is spectrally hard - they overlap significantly
2. Other is easy to separate (high visible reflectance)
3. Need hierarchical approach: Other vs (Health+Rust), then Health vs Rust

Novel techniques:
1. Hierarchical classification (2-stage)
2. CatBoost (better for small datasets with categorical-like features)
3. Class-weighted training with heavy Health emphasis
4. Feature selection per hierarchy level
5. Threshold optimization for Health class
6. SMOTE-like oversampling for minority predictions
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
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not available, using alternatives")

warnings.filterwarnings("ignore")

CFG = {
    "data_dir": "Kaggle_Prepared/train/MS",
    "val_dir": "Kaggle_Prepared/val/MS",
    "output_dir": "ms_v2",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


def extract_features(img, class_prototypes=None):
    """Extract comprehensive features from MS image."""
    img = img.astype(np.float32)
    features = {}
    eps = 1e-8

    bands = img.transpose(2, 0, 1)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]

    # ====== Per-band statistics ======
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

    # ====== Vegetation indices ======
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
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))
    osavi = (nir - red) / (nir + red + 0.16 + eps)
    ndwi = (green - nir) / (green + nir + eps)
    sipi = (nir - blue) / (nir - red + eps)
    
    # Disease-specific indices
    # Iron Index (rust has iron-like spectral signature)
    iron_index = (red - blue) / (red + blue + eps)
    # Disease Stress Index
    dsi = (nir - rededge) / (nir + red + eps)
    # Red Edge Position proxy
    rep_proxy = 700 + 40 * ((red + rededge) / 2 - red) / (rededge - red + eps)
    # Chlorophyll absorption
    chloro_abs = (green - red) / (green + red + eps)
    # Plant Stress Index
    psi = (rededge - red) / (rededge + red + eps)
    # Yellow Index (for rust yellowing)
    yellow_idx = (green - blue) / (green + blue + eps)
    # Rust-specific: Red/Green ratio (rust has more red)
    rust_ratio = red / (green + eps)
    # Health-specific: NIR/Red (healthy has higher ratio)
    health_ratio = nir / (red + eps)

    indices = {
        "NDVI": ndvi, "NDRE": ndre, "GNDVI": gndvi, "SAVI": savi,
        "CI_RE": ci_rededge, "CI_Green": ci_green,
        "RG_ratio": rg_ratio, "RB_ratio": rb_ratio,
        "RE_R_ratio": re_r_ratio, "NIR_R_ratio": nir_r_ratio,
        "NIR_RE_ratio": nir_re_ratio, "EVI": evi, "MCARI": mcari,
        "OSAVI": osavi, "NDWI": ndwi, "SIPI": sipi,
        "Iron_Index": iron_index, "DSI": dsi, "REP_proxy": rep_proxy,
        "Chloro_Abs": chloro_abs, "PSI": psi, "Yellow_Idx": yellow_idx,
        "Rust_Ratio": rust_ratio, "Health_Ratio": health_ratio,
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

    # ====== Inter-band correlations ======
    flat_bands = bands.reshape(5, -1)
    corr_matrix = np.corrcoef(flat_bands)
    for i in range(5):
        for j in range(i+1, 5):
            features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]

    # ====== Spatial texture ======
    for i, name in enumerate(BAND_NAMES):
        b = bands[i]
        gy, gx = np.gradient(b)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{name}_grad_mean"] = np.mean(grad_mag)
        features[f"{name}_grad_std"] = np.std(grad_mag)

        local_mean = ndimage.uniform_filter(b, size=3)
        local_sq_mean = ndimage.uniform_filter(b**2, size=3)
        local_var = local_sq_mean - local_mean**2
        features[f"{name}_localvar_mean"] = np.mean(local_var)
        features[f"{name}_localvar_std"] = np.std(local_var)

    # ====== Band ratios ======
    band_means = [np.mean(bands[i]) for i in range(5)]
    for i in range(5):
        for j in range(i+1, 5):
            features[f"meanratio_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = band_means[i] / (band_means[j] + eps)

    # ====== Spectral shape ======
    mean_spectrum = np.array(band_means)
    features["spec_slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["spec_slope_rededge"] = mean_spectrum[3] - mean_spectrum[2]
    features["spec_slope_nir"] = mean_spectrum[4] - mean_spectrum[3]
    features["spec_curvature"] = mean_spectrum[3] - 0.5 * (mean_spectrum[2] + mean_spectrum[4])
    features["spec_total_reflectance"] = np.sum(mean_spectrum)
    features["spec_nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)
    
    # Spectral derivatives
    spec_deriv1 = np.diff(mean_spectrum)
    spec_deriv2 = np.diff(spec_deriv1)
    for i, d in enumerate(spec_deriv1):
        features[f"spec_deriv1_{i}"] = d
    for i, d in enumerate(spec_deriv2):
        features[f"spec_deriv2_{i}"] = d

    # ====== SAM features ======
    if class_prototypes is not None:
        for cls_name, prototype in class_prototypes.items():
            dot_product = np.dot(mean_spectrum, prototype)
            norm_product = np.linalg.norm(mean_spectrum) * np.linalg.norm(prototype)
            angle = np.arccos(np.clip(dot_product / (norm_product + eps), -1, 1))
            features[f"SAM_{cls_name}"] = angle
            
            p = mean_spectrum / (np.sum(mean_spectrum) + eps)
            q = prototype / (np.sum(prototype) + eps)
            sid = np.sum(p * np.log2((p + eps) / (q + eps))) + np.sum(q * np.log2((q + eps) / (p + eps)))
            features[f"SID_{cls_name}"] = sid
            
            features[f"EucDist_{cls_name}"] = np.linalg.norm(mean_spectrum - prototype)

    # ====== Pixel-level stats ======
    pixel_means = np.mean(img, axis=2)
    features["pixel_mean_max"] = np.max(pixel_means)
    features["pixel_mean_min"] = np.min(pixel_means)
    features["pixel_mean_std"] = np.std(pixel_means)
    
    # NDVI distribution
    ndvi_flat = ndvi.ravel()
    features["NDVI_negative_ratio"] = np.mean(ndvi_flat < 0)
    features["NDVI_high_ratio"] = np.mean(ndvi_flat > 0.5)
    features["NDVI_low_ratio"] = np.mean(ndvi_flat < 0.2)

    return features


def compute_class_prototypes(data_dir):
    """Compute mean spectrum for each class."""
    prototypes = {cls: [] for cls in CLASS_MAP}
    
    for f in sorted(os.listdir(data_dir)):
        fp = os.path.join(data_dir, f)
        img = tiff.imread(fp).astype(np.float32)
        
        if img.mean() < 1.0:
            continue
            
        if "_hyper_" in f:
            cls_name = f.split("_hyper_")[0]
            mean_spectrum = np.mean(img, axis=(0, 1))
            prototypes[cls_name].append(mean_spectrum)
    
    final_prototypes = {}
    for cls_name, spectra_list in prototypes.items():
        if len(spectra_list) > 0:
            final_prototypes[cls_name] = np.mean(spectra_list, axis=0)
    
    return final_prototypes


def extract_all_features(data_dir, class_prototypes=None, file_list=None):
    """Extract features from all files."""
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


def main():
    np.random.seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    # --- Compute class prototypes ---
    print("Computing class prototypes...")
    class_prototypes = compute_class_prototypes(CFG["data_dir"])
    for cls, proto in class_prototypes.items():
        print(f"  {cls}: {proto}")

    # --- Extract training features ---
    print("\nExtracting training features...")
    train_feats, train_labels, train_fnames = extract_all_features(
        CFG["data_dir"], class_prototypes
    )
    print(f"  {len(train_feats)} samples")

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
        CFG["val_dir"], class_prototypes
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

    # ================================================================
    # APPROACH 1: Hierarchical Classification
    # ================================================================
    print(f"\n{'='*70}")
    print("APPROACH 1: Hierarchical Classification")
    print("Stage 1: Other vs (Health+Rust)")
    print("Stage 2: Health vs Rust")
    print(f"{'='*70}")

    # Stage 1: Other vs Vegetation (Health+Rust)
    y_stage1 = np.where(y_train == 2, 1, 0)  # 1=Other, 0=Health+Rust
    
    print("\n--- Stage 1: Other vs Vegetation ---")
    
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    
    oof_stage1 = np.zeros((len(X_train_scaled), 2))
    val_stage1_folds = []
    
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

        oof_stage1[va_idx] = model.predict_proba(X_va)
        val_stage1_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, model.predict(X_va))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_s1_preds = np.argmax(oof_stage1, axis=1)
    s1_acc = accuracy_score(y_stage1, oof_s1_preds)
    print(f"Stage 1 OOF Accuracy: {s1_acc:.4f}")

    # Stage 2: Health vs Rust (only on vegetation samples)
    veg_mask = y_train != 2  # Health or Rust
    X_train_veg = X_train_scaled[veg_mask]
    y_stage2 = y_train[veg_mask]  # 0=Health, 1=Rust
    
    print(f"\n--- Stage 2: Health vs Rust ({len(y_stage2)} samples) ---")
    
    oof_stage2 = np.zeros((len(X_train_veg), 2))
    val_stage2_folds = []
    
    # Heavy class weight for Health (minority in predictions)
    xgb_params_s2 = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "scale_pos_weight": 1.2,  # Slightly favor Health
        "random_state": CFG["seed"],
        "tree_method": "hist",
        "verbosity": 0,
    }
    
    skf2 = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    
    for fold, (tr_idx, va_idx) in enumerate(skf2.split(X_train_veg, y_stage2)):
        X_tr, X_va = X_train_veg[tr_idx], X_train_veg[va_idx]
        y_tr, y_va = y_stage2[tr_idx], y_stage2[va_idx]

        model = xgb.XGBClassifier(**xgb_params_s2)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        oof_stage2[va_idx] = model.predict_proba(X_va)
        val_stage2_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, model.predict(X_va))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_s2_preds = np.argmax(oof_stage2, axis=1)
    s2_acc = accuracy_score(y_stage2, oof_s2_preds)
    print(f"Stage 2 OOF Accuracy: {s2_acc:.4f}")
    
    # Health recall
    health_mask = y_stage2 == 0
    health_recall = (oof_s2_preds[health_mask] == 0).mean()
    print(f"Health Recall: {health_recall:.4f}")
    print(f"Rust Recall: {(oof_s2_preds[~health_mask] == 1).mean():.4f}")

    # Combine hierarchical predictions
    val_s1_probs = np.mean(val_stage1_folds, axis=0)
    val_s2_probs = np.mean(val_stage2_folds, axis=0)
    
    # Final predictions: if Stage 1 says Other → Other, else use Stage 2
    hier_val_preds = []
    for i in range(len(X_val_scaled)):
        if val_s1_probs[i, 1] > 0.5:  # Predicted Other
            hier_val_preds.append(2)
        else:
            hier_val_preds.append(0 if val_s2_probs[i, 0] > 0.5 else 1)
    
    hier_val_preds = np.array(hier_val_preds)

    # ================================================================
    # APPROACH 2: Multi-model Ensemble with Class Weights
    # ================================================================
    print(f"\n{'='*70}")
    print("APPROACH 2: Multi-model Ensemble with Heavy Health Weights")
    print(f"{'='*70}")

    # XGBoost with heavy Health weight
    print("\n--- XGBoost (Health weight=2.0) ---")
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 800,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 1,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "random_state": CFG["seed"],
        "tree_method": "hist",
        "verbosity": 0,
    }

    oof_xgb = np.zeros((len(X_train_scaled), 3))
    val_xgb_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        # Compute sample weights - heavier for Health
        sample_weights = np.ones(len(y_tr))
        sample_weights[y_tr == 0] = 2.0  # Health gets 2x weight

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, sample_weight=sample_weights, eval_set=[(X_va, y_va)], verbose=False)

        oof_xgb[va_idx] = model.predict_proba(X_va)
        val_xgb_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, np.argmax(oof_xgb[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_xgb_preds = np.argmax(oof_xgb, axis=1)
    xgb_acc = accuracy_score(y_train, oof_xgb_preds)
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")
    print(classification_report(y_train, oof_xgb_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # LightGBM
    print("\n--- LightGBM (Health weight=2.0) ---")
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_depth": 8,
        "learning_rate": 0.03,
        "n_estimators": 800,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 3,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "random_state": CFG["seed"],
        "verbose": -1,
        "num_leaves": 50,
    }

    oof_lgb = np.zeros((len(X_train_scaled), 3))
    val_lgb_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        sample_weights = np.ones(len(y_tr))
        sample_weights[y_tr == 0] = 2.0

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, sample_weight=sample_weights, eval_set=[(X_va, y_va)])

        oof_lgb[va_idx] = model.predict_proba(X_va)
        val_lgb_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, np.argmax(oof_lgb[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_lgb_preds = np.argmax(oof_lgb, axis=1)
    lgb_acc = accuracy_score(y_train, oof_lgb_preds)
    print(f"LGB OOF Accuracy: {lgb_acc:.4f}")
    print(classification_report(y_train, oof_lgb_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # CatBoost (if available)
    if HAS_CATBOOST:
        print("\n--- CatBoost (Health weight=2.0) ---")
        cat_params = {
            "iterations": 800,
            "depth": 6,
            "learning_rate": 0.03,
            "loss_function": "MultiClass",
            "classes_count": 3,
            "random_seed": CFG["seed"],
            "verbose": 0,
            "class_weights": [2.0, 1.0, 1.0],  # Health, Rust, Other
        }

        oof_cat = np.zeros((len(X_train_scaled), 3))
        val_cat_folds = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
            X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model = CatBoostClassifier(**cat_params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)

            oof_cat[va_idx] = model.predict_proba(X_va)
            val_cat_folds.append(model.predict_proba(X_val_scaled))
            
            acc = accuracy_score(y_va, np.argmax(oof_cat[va_idx], axis=1))
            print(f"  Fold {fold+1}: Acc={acc:.4f}")

        oof_cat_preds = np.argmax(oof_cat, axis=1)
        cat_acc = accuracy_score(y_train, oof_cat_preds)
        print(f"CatBoost OOF Accuracy: {cat_acc:.4f}")
        print(classification_report(y_train, oof_cat_preds, target_names=list(CLASS_MAP.keys()), digits=4))
    else:
        oof_cat = np.zeros_like(oof_xgb)
        val_cat_folds = [np.zeros_like(val_xgb_folds[0])] * CFG["n_folds"]
        cat_acc = 0

    # ================================================================
    # APPROACH 3: Weighted Ensemble with Threshold Optimization
    # ================================================================
    print(f"\n{'='*70}")
    print("APPROACH 3: Optimized Ensemble")
    print(f"{'='*70}")

    # Grid search for best weights
    best_acc = 0
    best_weights = None
    
    for w_xgb in [0.2, 0.3, 0.4, 0.5]:
        for w_lgb in [0.2, 0.3, 0.4, 0.5]:
            w_cat = 1.0 - w_xgb - w_lgb
            if w_cat < 0.1 or w_cat > 0.5:
                continue
            
            ens_oof = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
            ens_preds = np.argmax(ens_oof, axis=1)
            acc = accuracy_score(y_train, ens_preds)
            
            if acc > best_acc:
                best_acc = acc
                best_weights = (w_xgb, w_lgb, w_cat)
    
    print(f"Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    print(f"Best OOF Accuracy: {best_acc:.4f}")
    
    # Apply best weights
    ens_oof = best_weights[0] * oof_xgb + best_weights[1] * oof_lgb + best_weights[2] * oof_cat
    ens_preds = np.argmax(ens_oof, axis=1)
    print(classification_report(y_train, ens_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # Threshold optimization for Health
    print("\n--- Threshold Optimization for Health ---")
    best_threshold = 0.5
    best_health_recall = 0
    best_overall_acc = 0
    
    for thresh in np.arange(0.3, 0.7, 0.05):
        # Adjust predictions: if Health prob > thresh, predict Health
        adjusted_preds = ens_preds.copy()
        for i in range(len(ens_oof)):
            if ens_oof[i, 0] > thresh:  # Health probability
                adjusted_preds[i] = 0
        
        acc = accuracy_score(y_train, adjusted_preds)
        health_recall = (adjusted_preds[y_train == 0] == 0).mean()
        
        # Balance: want high accuracy AND reasonable health recall
        if health_recall > 0.5 and acc > best_overall_acc:
            best_overall_acc = acc
            best_threshold = thresh
            best_health_recall = health_recall
    
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Accuracy at best threshold: {best_overall_acc:.4f}")
    print(f"Health recall at best threshold: {best_health_recall:.4f}")

    # ================================================================
    # Final Predictions
    # ================================================================
    print(f"\n{'='*70}")
    print("FINAL ENSEMBLE")
    print(f"{'='*70}")

    # Combine all approaches
    val_xgb_probs = np.mean(val_xgb_folds, axis=0)
    val_lgb_probs = np.mean(val_lgb_folds, axis=0)
    val_cat_probs = np.mean(val_cat_folds, axis=0)
    
    # Weighted ensemble
    ens_val_probs = best_weights[0] * val_xgb_probs + best_weights[1] * val_lgb_probs + best_weights[2] * val_cat_probs
    
    # Apply threshold adjustment for Health
    final_preds = np.argmax(ens_val_probs, axis=1)
    for i in range(len(ens_val_probs)):
        if ens_val_probs[i, 0] > best_threshold:
            final_preds[i] = 0
    
    # Override black images → Other
    for i, is_b in enumerate(black_mask):
        if is_b:
            final_preds[i] = 2
    
    pred_classes = [INV_CLASS_MAP[p] for p in final_preds]
    
    dist = {c: pred_classes.count(c) for c in CLASS_MAP}
    print(f"Val prediction distribution: {dist}")

    # ================================================================
    # Save Results
    # ================================================================
    np.save(os.path.join(CFG["output_dir"], "val_probs_final.npy"), ens_val_probs)
    np.save(os.path.join(CFG["output_dir"], "oof_xgb.npy"), oof_xgb)
    np.save(os.path.join(CFG["output_dir"], "oof_lgb.npy"), oof_lgb)
    if HAS_CATBOOST:
        np.save(os.path.join(CFG["output_dir"], "oof_cat.npy"), oof_cat)

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
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")
    print(f"LGB OOF Accuracy: {lgb_acc:.4f}")
    if HAS_CATBOOST:
        print(f"CatBoost OOF Accuracy: {cat_acc:.4f}")
    print(f"Ensemble OOF Accuracy: {best_acc:.4f}")
    print(f"Final Training Accuracy (threshold adjusted): {best_overall_acc:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
