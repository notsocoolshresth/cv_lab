"""
MS Classification via Spectral Feature Engineering + XGBoost
- Extract ~150 handcrafted features per image (band stats, vegetation indices, ratios, spatial)
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
    "data_dir": "Kaggle_Prepared/train/MS",
    "val_dir": "Kaggle_Prepared/val/MS",
    "output_dir": "ms_xgb",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


# ============================================================
# Feature extraction — this is where the magic happens
# ============================================================
def extract_features(img):
    """
    Extract rich feature vector from a single 5-band 64x64 MS image.
    Input: img (64, 64, 5) float32
    Returns: feature dict
    """
    img = img.astype(np.float32)
    features = {}

    # Transpose to (5, 64, 64) for easier band access
    bands = img.transpose(2, 0, 1)  # (5, H, W)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]
    eps = 1e-8

    # ====== 1. Per-band statistics ======
    for i, name in enumerate(BAND_NAMES):
        b = bands[i].ravel()
        features[f"{name}_mean"] = np.mean(b)
        features[f"{name}_std"] = np.std(b)
        features[f"{name}_min"] = np.min(b)
        features[f"{name}_max"] = np.max(b)
        features[f"{name}_median"] = np.median(b)
        features[f"{name}_p5"] = np.percentile(b, 5)
        features[f"{name}_p25"] = np.percentile(b, 25)
        features[f"{name}_p75"] = np.percentile(b, 75)
        features[f"{name}_p95"] = np.percentile(b, 95)
        features[f"{name}_iqr"] = features[f"{name}_p75"] - features[f"{name}_p25"]
        features[f"{name}_range"] = features[f"{name}_max"] - features[f"{name}_min"]
        features[f"{name}_skew"] = float(scipy_stats.skew(b))
        features[f"{name}_kurtosis"] = float(scipy_stats.kurtosis(b))
        # Coefficient of variation
        features[f"{name}_cv"] = np.std(b) / (np.mean(b) + eps)

    # ====== 2. Vegetation / spectral indices (per-pixel, then aggregate) ======
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

    indices = {
        "NDVI": ndvi, "NDRE": ndre, "GNDVI": gndvi, "SAVI": savi,
        "CI_RE": ci_rededge, "CI_Green": ci_green,
        "RG_ratio": rg_ratio, "RB_ratio": rb_ratio,
        "RE_R_ratio": re_r_ratio, "NIR_R_ratio": nir_r_ratio,
        "NIR_RE_ratio": nir_re_ratio, "EVI": evi, "MCARI": mcari,
    }

    for idx_name, idx_map in indices.items():
        # Clip extreme values
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

    # ====== 3. Inter-band correlations ======
    flat_bands = bands.reshape(5, -1)  # (5, 4096)
    corr_matrix = np.corrcoef(flat_bands)
    for i in range(5):
        for j in range(i+1, 5):
            features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]

    # ====== 4. Spatial texture features ======
    for i, name in enumerate(BAND_NAMES):
        b = bands[i]
        # Gradient magnitude (edge strength)
        gy, gx = np.gradient(b)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{name}_grad_mean"] = np.mean(grad_mag)
        features[f"{name}_grad_std"] = np.std(grad_mag)

        # Local variance (3x3 window)
        local_mean = ndimage.uniform_filter(b, size=3)
        local_sq_mean = ndimage.uniform_filter(b**2, size=3)
        local_var = local_sq_mean - local_mean**2
        features[f"{name}_localvar_mean"] = np.mean(local_var)
        features[f"{name}_localvar_std"] = np.std(local_var)

    # ====== 5. Spatial features on key indices ======
    for idx_name, idx_map in [("NDVI", ndvi), ("NDRE", ndre)]:
        idx_map = np.clip(idx_map, -10, 10)
        gy, gx = np.gradient(idx_map)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{idx_name}_grad_mean"] = np.mean(grad_mag)
        features[f"{idx_name}_grad_std"] = np.std(grad_mag)

    # ====== 6. Band ratios (aggregated) ======
    # Ratio of mean reflectances
    band_means = [np.mean(bands[i]) for i in range(5)]
    for i in range(5):
        for j in range(i+1, 5):
            features[f"meanratio_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = band_means[i] / (band_means[j] + eps)

    # ====== 7. Spectral shape features ======
    # Mean spectrum across image
    mean_spectrum = np.array(band_means)
    features["spec_slope_vis"] = mean_spectrum[2] - mean_spectrum[0]  # slope visible (Blue→Red)
    features["spec_slope_rededge"] = mean_spectrum[3] - mean_spectrum[2]  # Red→RedEdge jump
    features["spec_slope_nir"] = mean_spectrum[4] - mean_spectrum[3]  # RedEdge→NIR
    features["spec_curvature"] = mean_spectrum[3] - 0.5 * (mean_spectrum[2] + mean_spectrum[4])  # curvature at RedEdge
    features["spec_total_reflectance"] = np.sum(mean_spectrum)
    features["spec_nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)

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
    np.save(os.path.join(CFG["output_dir"], "ms_val_probs_xgb.npy"), xgb_val_probs)
    np.save(os.path.join(CFG["output_dir"], "ms_val_probs_lgb.npy"), lgb_val_probs)
    np.save(os.path.join(CFG["output_dir"], "ms_val_probs_ensemble.npy"), ens_val_probs)

    # Save OOF for potential stacking
    np.save(os.path.join(CFG["output_dir"], "oof_probs_xgb.npy"), oof_probs)
    np.save(os.path.join(CFG["output_dir"], "oof_probs_lgb.npy"), oof_probs_lgb)

    # Save submission
    sub_path = os.path.join(CFG["output_dir"], "ms_submission.csv")
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
