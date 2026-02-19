"""
HS Classification via Spectral Feature Engineering + XGBoost
- Extract rich handcrafted features per image from 125-band hyperspectral data
- Train XGBoost with 5-fold stratified CV
- Trains in seconds, not minutes
- Outputs soft probabilities for late fusion with RGB ensemble
"""

import os
import csv
import json
import warnings
import numpy as np
import tifffile as tiff
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from scipy import ndimage, stats as scipy_stats

warnings.filterwarnings("ignore")

CFG = {
    "data_dir": "Kaggle_Prepared/train/HS",
    "val_dir": "Kaggle_Prepared/val/HS",
    "output_dir": "hs_xgb",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# Approximate wavelengths for HS bands (450-950nm range, 100 clean bands)
HS_WAVELENGTHS = np.linspace(450, 950, 100)


# ============================================================
# Feature extraction — this is where the magic happens
# ============================================================
def extract_features(img):
    """
    Extract rich feature vector from a single 125-band HS image.
    Input: img (64, 64, 125) float32
    Returns: feature dict
    """
    img = img.astype(np.float32)
    features = {}
    eps = 1e-8
    
    n_bands = img.shape[2]
    
    # Use clean bands (10-110) to avoid noisy edge bands
    clean_start, clean_end = 10, min(110, n_bands - 1)
    clean_hs = img[:, :, clean_start:clean_end]
    
    # Mean spectrum across spatial dimensions
    mean_spectrum = np.mean(clean_hs, axis=(0, 1))
    n = len(mean_spectrum)
    
    # ====== 1. Basic spectral statistics ======
    features["hs_mean_reflectance"] = np.mean(mean_spectrum)
    features["hs_std_reflectance"] = np.std(mean_spectrum)
    features["hs_max_reflectance"] = np.max(mean_spectrum)
    features["hs_min_reflectance"] = np.min(mean_spectrum)
    features["hs_range_reflectance"] = features["hs_max_reflectance"] - features["hs_min_reflectance"]
    features["hs_skewness"] = float(scipy_stats.skew(mean_spectrum))
    features["hs_kurtosis"] = float(scipy_stats.kurtosis(mean_spectrum))
    
    # ====== 2. Spectral Derivatives ======
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
    
    # ====== 3. Red Edge Position (REP) ======
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
        features["hs_Rust_Index"] = red_b / (green_b + eps)
        features["hs_Rust_Index2"] = (red_b - yellow_b) / (red_b + yellow_b + eps)
        
        # Health indicators
        features["hs_Health_Index"] = nir_b / (red_b + eps)
        features["hs_Chlorophyll_Index"] = (nir_b / red_b) - 1
        
        # Water index
        features["hs_Water_Index"] = nir_b / (rede_b + eps)
        
        # Plant Stress Index
        features["hs_PSI"] = (rede_b - red_b) / (rede_b + red_b + eps)
        
        # Photochemical Reflectance Index (PRI)
        features["hs_PRI"] = (green_b - red_b) / (green_b + red_b + eps)
        
        # Anthocyanin Reflectance Index (ARI)
        features["hs_ARI"] = 1 / (green_b + eps) - 1 / (rede_b + eps)
        
        # Carotenoid Reflectance Index (CRI)
        features["hs_CRI"] = 1 / (blue_b + eps) - 1 / (green_b + eps)
        
        # Iron oxide index (rust has iron signature)
        features["hs_Iron_Index"] = (red_b - blue_b) / (red_b + blue_b + eps)
        
        # Yellowing index
        features["hs_Yellowing"] = (green_b - blue_b) / (green_b + blue_b + eps)
        
        # Disease stress index
        features["hs_DSI"] = (nir_b - rede_b) / (nir_b + red_b + eps)
        
        # Additional ratios
        features["hs_NIR_Red_ratio"] = nir_b / (red_b + eps)
        features["hs_NIR_Green_ratio"] = nir_b / (green_b + eps)
        features["hs_RE_Red_ratio"] = rede_b / (red_b + eps)
    
    # ====== 6. Inter-region ratios ======
    if len(region_means) >= 4:
        features["hs_nir_red_ratio"] = region_means.get("nir", 0) / (region_means.get("red", 0) + eps)
        features["hs_nir_green_ratio"] = region_means.get("nir", 0) / (region_means.get("green", 0) + eps)
        features["hs_red_green_ratio"] = region_means.get("red", 0) / (region_means.get("green", 0) + eps)
        features["hs_rededge_red_ratio"] = region_means.get("rededge", 0) / (region_means.get("red", 0) + eps)
        features["hs_nir_vis_ratio"] = region_means.get("nir", 0) / (np.mean([region_means.get("blue", 0), region_means.get("green", 0), region_means.get("red", 0)]) + eps)
    
    # ====== 7. Spectral shape features ======
    if n > 10:
        features["hs_slope_vis"] = mean_spectrum[40] - mean_spectrum[5]  # slope visible (Blue→Red)
        features["hs_slope_rededge"] = mean_spectrum[55] - mean_spectrum[40]  # Red→RedEdge jump
        features["hs_slope_nir"] = mean_spectrum[80] - mean_spectrum[55]  # RedEdge→NIR
        features["hs_curvature"] = mean_spectrum[55] - 0.5 * (mean_spectrum[40] + mean_spectrum[80])
        features["hs_total_reflectance"] = np.sum(mean_spectrum)
    
    # ====== 8. Spatial texture features on key bands ======
    # Use specific bands for texture
    key_band_indices = [5, 17, 42, 52, 75]  # Blue, Green, Red, RedEdge, NIR approximate
    band_names = ["blue", "green", "red", "rededge", "nir"]
    
    for idx, bname in zip(key_band_indices, band_names):
        if idx < clean_hs.shape[2]:
            b = clean_hs[:, :, idx]
            # Gradient magnitude (edge strength)
            gy, gx = np.gradient(b)
            grad_mag = np.sqrt(gx**2 + gy**2)
            features[f"hs_{bname}_grad_mean"] = np.mean(grad_mag)
            features[f"hs_{bname}_grad_std"] = np.std(grad_mag)
            
            # Local variance (3x3 window)
            local_mean = ndimage.uniform_filter(b, size=3)
            local_sq_mean = ndimage.uniform_filter(b**2, size=3)
            local_var = local_sq_mean - local_mean**2
            features[f"hs_{bname}_localvar_mean"] = np.mean(local_var)
            features[f"hs_{bname}_localvar_std"] = np.std(local_var)
    
    # ====== 9. Band-to-band correlations ======
    if n > 10:
        # Sample correlations between key spectral regions
        flat_bands = clean_hs.reshape(-1, clean_hs.shape[2])  # (H*W, n_bands)
        # Sample a subset of bands for correlation
        sample_indices = [5, 17, 30, 42, 52, 65, 75, 85]
        sample_indices = [i for i in sample_indices if i < n]
        
        for i, idx_i in enumerate(sample_indices):
            for j, idx_j in enumerate(sample_indices[i+1:], i+1):
                corr = np.corrcoef(flat_bands[:, idx_i], flat_bands[:, idx_j])[0, 1]
                features[f"hs_corr_{idx_i}_{idx_j}"] = corr if not np.isnan(corr) else 0.0
    
    # ====== 10. Percentile features ======
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        features[f"hs_p{p}"] = np.percentile(mean_spectrum, p)
    
    return features


def extract_all_features(data_dir, file_list=None):
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

        # Check for black images
        if img.mean() < 1.0:
            skipped_black += 1
            # For train: skip. For val: add zero features
            if "_hyper_" in f:
                continue
            else:
                # Val black image — will predict "Other"
                all_features.append(None)
                all_labels.append(-1)
                all_fnames.append(f)
                continue

        feats = extract_features(img)
        all_features.append(feats)

        # Parse label from filename (train only)
        if "_hyper_" in f:
            cls_name = f.split("_hyper_")[0]
            all_labels.append(CLASS_MAP[cls_name])
        else:
            all_labels.append(-1)  # val set

        all_fnames.append(f)

    if skipped_black > 0:
        print(f"  Skipped/flagged {skipped_black} black images")

    return all_features, all_labels, all_fnames


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    # --- Extract training features ---
    print("Extracting training features...")
    train_feats, train_labels, train_fnames = extract_all_features(CFG["data_dir"])
    print(f"  {len(train_feats)} samples, {len(train_feats[0])} features each")

    # Convert to numpy arrays
    feature_names = list(train_feats[0].keys())
    X_train = np.array([[f[k] for k in feature_names] for f in train_feats], dtype=np.float32)
    y_train = np.array(train_labels)

    # Replace nan/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")

    # --- Extract val features ---
    print("Extracting validation features...")
    val_feats, val_labels, val_fnames = extract_all_features(CFG["val_dir"])
    black_mask = [f is None for f in val_feats]
    
    # Fill black images with zeros
    for i in range(len(val_feats)):
        if val_feats[i] is None:
            val_feats[i] = {k: 0.0 for k in feature_names}

    X_val = np.array([[f[k] for k in feature_names] for f in val_feats], dtype=np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"  {len(val_feats)} samples ({sum(black_mask)} black)")

    # --- Normalize features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ================================================================
    # XGBoost with 5-fold CV
    # ================================================================
    print(f"\n{'='*60}")
    print("XGBoost 5-Fold CV")
    print(f"{'='*60}")

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

    oof_probs = np.zeros((len(X_train), 3))
    oof_preds = np.zeros(len(X_train), dtype=int)
    val_probs_all_folds = []
    fold_results = []
    feature_importance = np.zeros(len(feature_names))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        print(f"\n--- Fold {fold+1} ---")
        X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        # OOF predictions
        va_probs = model.predict_proba(X_va)
        va_preds = np.argmax(va_probs, axis=1)
        oof_probs[va_idx] = va_probs
        oof_preds[va_idx] = va_preds

        acc = accuracy_score(y_va, va_preds)
        mf1 = f1_score(y_va, va_preds, average='macro')

        recall = {}
        for ci, cn in INV_CLASS_MAP.items():
            mask = y_va == ci
            recall[cn] = (va_preds[mask] == ci).mean() if mask.sum() > 0 else 0.0

        rc = " ".join(f"{k}:{v:.3f}" for k, v in recall.items())
        print(f"  Acc={acc:.4f} F1={mf1:.4f} | {rc}")
        print(classification_report(y_va, va_preds, target_names=list(CLASS_MAP.keys()), digits=4))

        fold_results.append({"acc": acc, "f1": mf1, "recall": recall})
        feature_importance += model.feature_importances_

        # Val predictions
        val_probs = model.predict_proba(X_val_scaled)
        val_probs_all_folds.append(val_probs)

    # --- OOF Summary ---
    print(f"\n{'='*60}")
    print("XGB CV SUMMARY")
    print(f"{'='*60}")
    oof_acc = accuracy_score(y_train, oof_preds)
    oof_f1 = f1_score(y_train, oof_preds, average='macro')
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF Macro F1: {oof_f1:.4f}")
    print(classification_report(y_train, oof_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    accs = [r["acc"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    print(f"Per-fold Acc: {[f'{a:.4f}' for a in accs]} → {np.mean(accs):.4f}±{np.std(accs):.4f}")
    print(f"Per-fold F1:  {[f'{f:.4f}' for f in f1s]} → {np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    for cn in CLASS_MAP:
        rs = [r["recall"][cn] for r in fold_results]
        print(f"  {cn} recall: {[f'{r:.3f}' for r in rs]} → {np.mean(rs):.3f}")

    # --- Top features ---
    feature_importance /= CFG["n_folds"]
    top_idx = np.argsort(feature_importance)[::-1][:25]
    print(f"\nTop 25 features:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1:2d}. {feature_names[idx]:30s} importance={feature_importance[idx]:.4f}")

    # ================================================================
    # LightGBM (diversity for ensemble)
    # ================================================================
    print(f"\n{'='*60}")
    print("LightGBM 5-Fold CV")
    print(f"{'='*60}")

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

    oof_probs_lgb = np.zeros((len(X_train), 3))
    oof_preds_lgb = np.zeros(len(X_train), dtype=int)
    val_probs_lgb_folds = []
    fold_results_lgb = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        print(f"\n--- Fold {fold+1} ---")
        X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        va_probs = model_lgb.predict_proba(X_va)
        va_preds = np.argmax(va_probs, axis=1)
        oof_probs_lgb[va_idx] = va_probs
        oof_preds_lgb[va_idx] = va_preds

        acc = accuracy_score(y_va, va_preds)
        mf1 = f1_score(y_va, va_preds, average='macro')

        recall = {}
        for ci, cn in INV_CLASS_MAP.items():
            mask = y_va == ci
            recall[cn] = (va_preds[mask] == ci).mean() if mask.sum() > 0 else 0.0

        rc = " ".join(f"{k}:{v:.3f}" for k, v in recall.items())
        print(f"  Acc={acc:.4f} F1={mf1:.4f} | {rc}")

        fold_results_lgb.append({"acc": acc, "f1": mf1, "recall": recall})
        val_probs_lgb_folds.append(model_lgb.predict_proba(X_val_scaled))

    # LGB Summary
    print(f"\nLGB CV SUMMARY")
    oof_acc_lgb = accuracy_score(y_train, oof_preds_lgb)
    oof_f1_lgb = f1_score(y_train, oof_preds_lgb, average='macro')
    print(f"OOF Acc: {oof_acc_lgb:.4f}, F1: {oof_f1_lgb:.4f}")
    print(classification_report(y_train, oof_preds_lgb, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # Ensemble XGB + LGB
    # ================================================================
    print(f"\n{'='*60}")
    print("XGB + LGB Ensemble")
    print(f"{'='*60}")

    # OOF ensemble
    ens_oof_probs = 0.5 * oof_probs + 0.5 * oof_probs_lgb
    ens_oof_preds = np.argmax(ens_oof_probs, axis=1)
    ens_acc = accuracy_score(y_train, ens_oof_preds)
    ens_f1 = f1_score(y_train, ens_oof_preds, average='macro')
    print(f"Ensemble OOF Acc: {ens_acc:.4f}, F1: {ens_f1:.4f}")
    print(classification_report(y_train, ens_oof_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # Val ensemble
    xgb_val_probs = np.mean(val_probs_all_folds, axis=0)
    lgb_val_probs = np.mean(val_probs_lgb_folds, axis=0)
    ens_val_probs = 0.5 * xgb_val_probs + 0.5 * lgb_val_probs

    # Override black images → Other
    for i, is_b in enumerate(black_mask):
        if is_b:
            ens_val_probs[i] = [0.0, 0.0, 1.0]
            xgb_val_probs[i] = [0.0, 0.0, 1.0]
            lgb_val_probs[i] = [0.0, 0.0, 1.0]

    ens_val_preds = np.argmax(ens_val_probs, axis=1)
    pred_classes = [INV_CLASS_MAP[p] for p in ens_val_preds]
    dist = {c: pred_classes.count(c) for c in CLASS_MAP}
    print(f"\nVal prediction distribution: {dist}")

    # ================================================================
    # Save everything
    # ================================================================
    # Save ensemble probs
    np.save(os.path.join(CFG["output_dir"], "hs_val_probs_xgb.npy"), xgb_val_probs)
    np.save(os.path.join(CFG["output_dir"], "hs_val_probs_lgb.npy"), lgb_val_probs)
    np.save(os.path.join(CFG["output_dir"], "hs_val_probs_ensemble.npy"), ens_val_probs)

    # Save OOF for potential stacking
    np.save(os.path.join(CFG["output_dir"], "oof_probs_xgb.npy"), oof_probs)
    np.save(os.path.join(CFG["output_dir"], "oof_probs_lgb.npy"), oof_probs_lgb)

    # Save submission
    sub_path = os.path.join(CFG["output_dir"], "hs_submission.csv")
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for fn, cl in zip(val_fnames, pred_classes):
            w.writerow([fn, cl])

    # Save feature names for reference
    with open(os.path.join(CFG["output_dir"], "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\nSaved to {CFG['output_dir']}/")
    print("Done!")


if __name__ == "__main__":
    main()
