"""
Balanced Spectral Classification - Targeting 80%+ CV with Good Generalization
==============================================================================

Strategy:
1. Use proven features from train_ms_xgb.py (224 features)
2. Add semi-supervised learning with pseudo-labeling
3. Use diverse model ensemble
4. Apply moderate regularization
5. Focus on Health vs Rust separation
"""

import os
import csv
import json
import warnings
import numpy as np
import tifffile as tiff
from scipy import ndimage, stats as scipy_stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

CFG = {
    "train_ms_dir": "Kaggle_Prepared/train/MS",
    "val_ms_dir": "Kaggle_Prepared/val/MS",
    "train_hs_dir": "Kaggle_Prepared/train/HS",
    "val_hs_dir": "Kaggle_Prepared/val/HS",
    "output_dir": "balanced_model",
    "n_folds": 5,
    "seed": 42,
    "pseudo_label_threshold": 0.90,
    "use_pseudo_labels": True,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


def extract_features(ms_img, hs_img=None):
    """Extract comprehensive features (same as successful train_ms_xgb.py)."""
    ms_img = ms_img.astype(np.float32)
    features = {}
    eps = 1e-8

    bands = ms_img.transpose(2, 0, 1)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]

    # Per-band statistics
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
        features[f"{name}_cv"] = np.std(b) / (np.mean(b) + eps)

    # Vegetation indices
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

    indices = {
        "NDVI": ndvi, "NDRE": ndre, "GNDVI": gndvi, "SAVI": savi,
        "CI_RE": ci_rededge, "CI_Green": ci_green,
        "RG_ratio": rg_ratio, "RB_ratio": rb_ratio,
        "RE_R_ratio": re_r_ratio, "NIR_R_ratio": nir_r_ratio,
        "NIR_RE_ratio": nir_re_ratio, "EVI": evi, "MCARI": mcari,
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

    # Inter-band correlations
    flat_bands = bands.reshape(5, -1)
    corr_matrix = np.corrcoef(flat_bands)
    for i in range(5):
        for j in range(i+1, 5):
            features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]

    # Spatial texture
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

    # Spatial features on key indices
    for idx_name, idx_map in [("NDVI", ndvi), ("NDRE", ndre)]:
        idx_map = np.clip(idx_map, -10, 10)
        gy, gx = np.gradient(idx_map)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{idx_name}_grad_mean"] = np.mean(grad_mag)
        features[f"{idx_name}_grad_std"] = np.std(grad_mag)

    # Band ratios
    band_means = [np.mean(bands[i]) for i in range(5)]
    for i in range(5):
        for j in range(i+1, 5):
            features[f"meanratio_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = band_means[i] / (band_means[j] + eps)

    # Spectral shape
    mean_spectrum = np.array(band_means)
    features["spec_slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["spec_slope_rededge"] = mean_spectrum[3] - mean_spectrum[2]
    features["spec_slope_nir"] = mean_spectrum[4] - mean_spectrum[3]
    features["spec_curvature"] = mean_spectrum[3] - 0.5 * (mean_spectrum[2] + mean_spectrum[4])
    features["spec_total_reflectance"] = np.sum(mean_spectrum)
    features["spec_nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)

    # HS features
    if hs_img is not None:
        hs_img = hs_img.astype(np.float32)
        n_bands = hs_img.shape[2]
        clean_hs = hs_img[:, :, 10:min(110, n_bands-1)]
        hs_mean = np.mean(clean_hs, axis=(0, 1))
        
        features["hs_mean_reflectance"] = np.mean(hs_mean)
        features["hs_std_reflectance"] = np.std(hs_mean)
        
        n = len(hs_mean)
        if n > 50:
            blue_h = hs_mean[5]
            green_h = hs_mean[15]
            red_h = hs_mean[35]
            rede_h = hs_mean[45]
            nir_h = hs_mean[70]
            features["hs_NDVI"] = (nir_h - red_h) / (nir_h + red_h + eps)
            features["hs_NDRE"] = (nir_h - rede_h) / (nir_h + rede_h + eps)
            features["hs_GNDVI"] = (nir_h - green_h) / (nir_h + green_h + eps)
            features["hs_CI_RE"] = nir_h / (rede_h + eps) - 1
            features["hs_Rust_Ratio"] = red_h / (green_h + eps)
            features["hs_Health_Ratio"] = nir_h / (red_h + eps)

    return features


def extract_all(ms_dir, hs_dir=None):
    """Extract features from all files."""
    ms_files = sorted(os.listdir(ms_dir))
    all_features, all_labels, all_fnames = [], [], []
    skipped = 0
    
    for ms_f in ms_files:
        ms_fp = os.path.join(ms_dir, ms_f)
        ms_img = tiff.imread(ms_fp).astype(np.float32)
        
        hs_img = None
        if hs_dir:
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
        
        feats = extract_features(ms_img, hs_img)
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

    print("Extracting training features...")
    train_feats, train_labels, train_fnames = extract_all(CFG["train_ms_dir"], CFG["train_hs_dir"])
    print(f"  {len(train_feats)} samples")

    feature_names = list(train_feats[0].keys())
    X_train = np.array([[f.get(k, 0.0) for k in feature_names] for f in train_feats], dtype=np.float32)
    y_train = np.array(train_labels)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

    print(f"  {X_train.shape[1]} features extracted")
    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")

    print("\nExtracting validation features...")
    val_feats, val_labels, val_fnames = extract_all(CFG["val_ms_dir"], CFG["val_hs_dir"])
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
    # Phase 1: Base Models
    # ================================================================
    print(f"\n{'='*70}")
    print("Phase 1: Training Base Models")
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

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        oof_xgb[va_idx] = model.predict_proba(X_va)
        val_xgb_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, np.argmax(oof_xgb[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_xgb_preds = np.argmax(oof_xgb, axis=1)
    xgb_acc = accuracy_score(y_train, oof_xgb_preds)
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")

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

    # Initial ensemble
    ens_oof = 0.5 * oof_xgb + 0.5 * oof_lgb
    ens_preds = np.argmax(ens_oof, axis=1)
    ens_acc = accuracy_score(y_train, ens_preds)
    print(f"\nInitial Ensemble OOF Accuracy: {ens_acc:.4f}")
    print(classification_report(y_train, ens_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # Phase 2: Semi-Supervised Learning with Pseudo-Labels
    # ================================================================
    if CFG["use_pseudo_labels"]:
        print(f"\n{'='*70}")
        print("Phase 2: Semi-Supervised Learning")
        print(f"{'='*70}")

        # Get high-confidence validation predictions
        val_probs_init = 0.5 * np.mean(val_xgb_folds, axis=0) + 0.5 * np.mean(val_lgb_folds, axis=0)
        val_max_probs = np.max(val_probs_init, axis=1)
        val_preds_init = np.argmax(val_probs_init, axis=1)

        # Select high-confidence samples (excluding black images)
        confident_mask = val_max_probs >= CFG["pseudo_label_threshold"]
        confident_mask = confident_mask & ~np.array(black_mask)
        
        n_confident = confident_mask.sum()
        print(f"High-confidence val samples: {n_confident}")
        
        if n_confident >= 50:
            # Add pseudo-labeled samples
            X_pseudo = X_val_scaled[confident_mask]
            y_pseudo = val_preds_init[confident_mask]
            
            X_train_aug = np.vstack([X_train_scaled, X_pseudo])
            y_train_aug = np.hstack([y_train, y_pseudo])
            
            print(f"Augmented training set: {len(y_train)} → {len(y_train_aug)}")
            
            # Retrain with pseudo-labels
            print("\n--- Retraining XGBoost with Pseudo-Labels ---")
            oof_xgb2 = np.zeros((len(X_train_scaled), 3))
            val_xgb2_folds = []
            
            skf2 = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
            
            for fold, (tr_idx, va_idx) in enumerate(skf2.split(X_train_aug, y_train_aug)):
                X_tr, X_va = X_train_aug[tr_idx], X_train_aug[va_idx]
                y_tr, y_va = y_train_aug[tr_idx], y_train_aug[va_idx]
                
                # Only evaluate on original training data
                orig_va_mask = va_idx < len(X_train_scaled)
                if not orig_va_mask.any():
                    continue
                
                X_va_orig = X_train_scaled[va_idx[orig_va_mask]]
                y_va_orig = y_train[va_idx[orig_va_mask]]
                
                model = xgb.XGBClassifier(**xgb_params)
                model.fit(X_tr, y_tr, eval_set=[(X_va_orig, y_va_orig)], verbose=False)
                
                oof_xgb2[va_idx[orig_va_mask]] = model.predict_proba(X_va_orig)
                val_xgb2_folds.append(model.predict_proba(X_val_scaled))
                
                acc = accuracy_score(y_va_orig, np.argmax(oof_xgb2[va_idx[orig_va_mask]], axis=1))
                print(f"  Fold {fold+1}: Acc={acc:.4f}")
            
            oof_xgb2_preds = np.argmax(oof_xgb2, axis=1)
            xgb2_acc = accuracy_score(y_train, oof_xgb2_preds)
            print(f"XGB with Pseudo-Labels OOF Accuracy: {xgb2_acc:.4f}")
            
            # Use pseudo-labeled model if better
            if xgb2_acc > xgb_acc:
                print("Using pseudo-labeled model (better accuracy)")
                oof_xgb = oof_xgb2
                val_xgb_folds = val_xgb2_folds
                xgb_acc = xgb2_acc
            
            # Retrain LightGBM
            print("\n--- Retraining LightGBM with Pseudo-Labels ---")
            oof_lgb2 = np.zeros((len(X_train_scaled), 3))
            val_lgb2_folds = []
            
            for fold, (tr_idx, va_idx) in enumerate(skf2.split(X_train_aug, y_train_aug)):
                X_tr, X_va = X_train_aug[tr_idx], X_train_aug[va_idx]
                y_tr, y_va = y_train_aug[tr_idx], y_train_aug[va_idx]
                
                orig_va_mask = va_idx < len(X_train_scaled)
                if not orig_va_mask.any():
                    continue
                
                X_va_orig = X_train_scaled[va_idx[orig_va_mask]]
                y_va_orig = y_train[va_idx[orig_va_mask]]
                
                model = lgb.LGBMClassifier(**lgb_params)
                model.fit(X_tr, y_tr, eval_set=[(X_va_orig, y_va_orig)])
                
                oof_lgb2[va_idx[orig_va_mask]] = model.predict_proba(X_va_orig)
                val_lgb2_folds.append(model.predict_proba(X_val_scaled))
                
                acc = accuracy_score(y_va_orig, np.argmax(oof_lgb2[va_idx[orig_va_mask]], axis=1))
                print(f"  Fold {fold+1}: Acc={acc:.4f}")
            
            oof_lgb2_preds = np.argmax(oof_lgb2, axis=1)
            lgb2_acc = accuracy_score(y_train, oof_lgb2_preds)
            print(f"LGB with Pseudo-Labels OOF Accuracy: {lgb2_acc:.4f}")
            
            if lgb2_acc > lgb_acc:
                print("Using pseudo-labeled model (better accuracy)")
                oof_lgb = oof_lgb2
                val_lgb_folds = val_lgb2_folds
                lgb_acc = lgb2_acc

    # ================================================================
    # Final Ensemble
    # ================================================================
    print(f"\n{'='*70}")
    print("Final Ensemble")
    print(f"{'='*70}")

    ens_oof = 0.5 * oof_xgb + 0.5 * oof_lgb
    ens_preds = np.argmax(ens_oof, axis=1)
    ens_acc = accuracy_score(y_train, ens_preds)
    print(f"Final Ensemble OOF Accuracy: {ens_acc:.4f}")
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
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")
    print(f"LGB OOF Accuracy: {lgb_acc:.4f}")
    print(f"Final Ensemble OOF Accuracy: {ens_acc:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
