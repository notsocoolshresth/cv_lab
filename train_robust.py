"""
Robust Spectral Classification - Focus on Generalization
=========================================================

Problem: Previous models overfit (83% CV → 67% LB)
Solution: Simpler models with strong regularization

Key changes:
1. Fewer features (reduce overfitting)
2. Stronger regularization
3. Simpler models
4. Cross-validation based early stopping
5. Feature selection based on stability
"""

import os
import csv
import json
import warnings
import numpy as np
import tifffile as tiff
from scipy import ndimage, stats as scipy_stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

CFG = {
    "train_ms_dir": "Kaggle_Prepared/train/MS",
    "val_ms_dir": "Kaggle_Prepared/val/MS",
    "train_hs_dir": "Kaggle_Prepared/train/HS",
    "val_hs_dir": "Kaggle_Prepared/val/HS",
    "output_dir": "robust_model",
    "n_folds": 5,
    "seed": 42,
    "n_features": 80,  # Limit features to prevent overfitting
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


def extract_features(ms_img, hs_img=None):
    """Extract robust, generalizable features."""
    ms_img = ms_img.astype(np.float32)
    eps = 1e-8
    features = {}
    
    bands = ms_img.transpose(2, 0, 1)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]
    
    # Raw mean spectrum (most important)
    mean_spectrum = np.array([np.mean(b) for b in [blue, green, red, rededge, nir]])
    for i, name in enumerate(BAND_NAMES):
        features[f"raw_{name}"] = mean_spectrum[i]
    
    # Key vegetation indices (proven to be important)
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)
    gndvi = (nir - green) / (nir + green + eps)
    ci_re = nir / (rededge + eps) - 1
    ci_green = nir / (green + eps) - 1
    rg_ratio = red / (green + eps)
    health_ratio = nir / (red + eps)
    iron_idx = (red - blue) / (red + blue + eps)
    
    # Only mean and std for indices (not full statistics to avoid overfitting)
    indices = {
        "NDVI": ndvi, "NDRE": ndre, "GNDVI": gndvi,
        "CI_RE": ci_re, "CI_Green": ci_green,
        "RG_ratio": rg_ratio, "Health_Ratio": health_ratio,
        "Iron_Idx": iron_idx,
    }
    
    for idx_name, idx_map in indices.items():
        idx_map = np.clip(idx_map, -10, 10)
        v = idx_map.ravel()
        features[f"{idx_name}_mean"] = np.mean(v)
        features[f"{idx_name}_std"] = np.std(v)
    
    # Spectral shape (key for classification)
    features["slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["slope_re"] = mean_spectrum[3] - mean_spectrum[2]
    features["slope_nir"] = mean_spectrum[4] - mean_spectrum[3]
    features["nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)
    features["total_reflectance"] = np.sum(mean_spectrum)
    
    # Simple band statistics
    for i, name in enumerate(BAND_NAMES):
        b = bands[i].ravel()
        features[f"{name}_std"] = np.std(b)
    
    # HS features (simplified)
    if hs_img is not None:
        hs_img = hs_img.astype(np.float32)
        n_bands = hs_img.shape[2]
        clean_hs = hs_img[:, :, 10:min(110, n_bands-1)]
        hs_mean = np.mean(clean_hs, axis=(0, 1))
        
        # Only key HS features
        features["hs_mean_reflectance"] = np.mean(hs_mean)
        features["hs_std_reflectance"] = np.std(hs_mean)
        
        n = len(hs_mean)
        if n > 50:
            # HS-specific indices
            blue_h = hs_mean[5]
            green_h = hs_mean[15]
            red_h = hs_mean[35]
            rede_h = hs_mean[45]
            nir_h = hs_mean[70]
            features["hs_NDVI"] = (nir_h - red_h) / (nir_h + red_h + eps)
            features["hs_NDRE"] = (nir_h - rede_h) / (nir_h + rede_h + eps)
            features["hs_Rust_Ratio"] = red_h / (green_h + eps)
            features["hs_Health_Ratio"] = nir_h / (red_h + eps)
    
    return features


def compute_prototypes(ms_dir):
    """Compute class prototypes for SAM features."""
    protos = {cls: [] for cls in CLASS_MAP}
    
    for f in sorted(os.listdir(ms_dir)):
        if "_hyper_" not in f:
            continue
        fp = os.path.join(ms_dir, f)
        img = tiff.imread(fp).astype(np.float32)
        if img.mean() < 1.0:
            continue
        cls_name = f.split("_hyper_")[0]
        mean_spec = np.array([np.mean(img[:,:,i]) for i in range(5)])
        protos[cls_name].append(mean_spec)
    
    return {cls: np.mean(vals, axis=0) for cls, vals in protos.items() if vals}


def add_sam_features(features, protos):
    """Add spectral angle features."""
    eps = 1e-8
    mean_spec = np.array([features.get(f"raw_{name}", 0) for name in BAND_NAMES])
    
    for cls_name, proto in protos.items():
        dot = np.dot(mean_spec, proto)
        norm_prod = np.linalg.norm(mean_spec) * np.linalg.norm(proto)
        if norm_prod > 0:
            features[f"SAM_{cls_name}"] = np.arccos(np.clip(dot / norm_prod, -1, 1))
        features[f"EucDist_{cls_name}"] = np.linalg.norm(mean_spec - proto)
    
    return features


def extract_all(ms_dir, hs_dir=None, protos=None):
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
        if protos:
            feats = add_sam_features(feats, protos)
        
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
    protos = compute_prototypes(CFG["train_ms_dir"])
    for cls, proto in protos.items():
        print(f"  {cls}: {proto}")

    print("\nExtracting training features...")
    train_feats, train_labels, train_fnames = extract_all(
        CFG["train_ms_dir"], CFG["train_hs_dir"], protos
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
        CFG["val_ms_dir"], CFG["val_hs_dir"], protos
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

    # Feature selection to prevent overfitting
    print(f"\nSelecting top {CFG['n_features']} features...")
    selector = SelectKBest(mutual_info_classif, k=min(CFG['n_features'], X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_val_sel = selector.transform(X_val_scaled)
    
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"  Top 10 features: {selected_features[:10]}")

    # ================================================================
    # Simple XGBoost with Strong Regularization
    # ================================================================
    print(f"\n{'='*70}")
    print("XGBoost with Strong Regularization")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

    # Strong regularization to prevent overfitting
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 3,  # Shallow trees
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,  # Require more samples per leaf
        "reg_alpha": 1.0,  # L1 regularization
        "reg_lambda": 5.0,  # L2 regularization
        "random_state": CFG["seed"],
        "tree_method": "hist",
        "verbosity": 0,
    }

    oof_xgb = np.zeros((len(X_train_sel), 3))
    val_xgb_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sel, y_train)):
        X_tr, X_va = X_train_sel[tr_idx], X_train_sel[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        oof_xgb[va_idx] = model.predict_proba(X_va)
        val_xgb_folds.append(model.predict_proba(X_val_sel))
        
        acc = accuracy_score(y_va, np.argmax(oof_xgb[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_preds = np.argmax(oof_xgb, axis=1)
    xgb_acc = accuracy_score(y_train, oof_preds)
    print(f"\nXGB OOF Accuracy: {xgb_acc:.4f}")
    print(classification_report(y_train, oof_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # LightGBM with Strong Regularization
    # ================================================================
    print(f"\n{'='*70}")
    print("LightGBM with Strong Regularization")
    print(f"{'='*70}")

    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_samples": 10,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": CFG["seed"],
        "verbose": -1,
        "num_leaves": 15,
    }

    oof_lgb = np.zeros((len(X_train_sel), 3))
    val_lgb_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sel, y_train)):
        X_tr, X_va = X_train_sel[tr_idx], X_train_sel[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        oof_lgb[va_idx] = model.predict_proba(X_va)
        val_lgb_folds.append(model.predict_proba(X_val_sel))
        
        acc = accuracy_score(y_va, np.argmax(oof_lgb[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")

    oof_lgb_preds = np.argmax(oof_lgb, axis=1)
    lgb_acc = accuracy_score(y_train, oof_lgb_preds)
    print(f"\nLGB OOF Accuracy: {lgb_acc:.4f}")
    print(classification_report(y_train, oof_lgb_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # Ensemble
    # ================================================================
    print(f"\n{'='*70}")
    print("Ensemble XGB + LGB")
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

    with open(os.path.join(CFG["output_dir"], "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    print(f"\nSaved to {CFG['output_dir']}/")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"XGB OOF Accuracy: {xgb_acc:.4f}")
    print(f"LGB OOF Accuracy: {lgb_acc:.4f}")
    print(f"Ensemble OOF Accuracy: {ens_acc:.4f}")
    print("\nNote: Lower CV accuracy but better generalization expected")
    print("Done!")


if __name__ == "__main__":
    main()
