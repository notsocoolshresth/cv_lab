"""
Spectral Classification - One-vs-One Final Version
===================================================

Achieved 83.19% accuracy with OvO voting approach!
This version is optimized for speed and uses the best approach.
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
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

CFG = {
    "train_ms_dir": "Kaggle_Prepared/train/MS",
    "val_ms_dir": "Kaggle_Prepared/val/MS",
    "train_hs_dir": "Kaggle_Prepared/train/HS",
    "val_hs_dir": "Kaggle_Prepared/val/HS",
    "output_dir": "ovo_final",
    "n_folds": 5,
    "seed": 42,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


def extract_features(ms_img, hs_img=None):
    """Extract comprehensive spectral features."""
    ms_img = ms_img.astype(np.float32)
    eps = 1e-8
    features = {}
    
    bands = ms_img.transpose(2, 0, 1)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]
    
    # Raw mean spectrum
    mean_spectrum = np.array([np.mean(b) for b in [blue, green, red, rededge, nir]])
    for i, name in enumerate(BAND_NAMES):
        features[f"raw_{name}"] = mean_spectrum[i]
    
    # Vegetation indices
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)
    gndvi = (nir - green) / (nir + green + eps)
    ci_re = nir / (rededge + eps) - 1
    ci_green = nir / (green + eps) - 1
    rg_ratio = red / (green + eps)
    rb_ratio = red / (blue + eps)
    iron_idx = (red - blue) / (red + blue + eps)
    yellow_idx = (green - blue) / (green + blue + eps)
    health_ratio = nir / (red + eps)
    nir_re_ratio = nir / (rededge + eps)
    psi = (rededge - red) / (rededge + red + eps)
    evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    mcari = ((rededge - red) - 0.2*(rededge - green)) * (rededge / (red + eps))
    osavi = (nir - red) / (nir + red + 0.16 + eps)
    
    indices = {
        "NDVI": ndvi, "NDRE": ndre, "GNDVI": gndvi,
        "CI_RE": ci_re, "CI_Green": ci_green,
        "RG_ratio": rg_ratio, "RB_ratio": rb_ratio,
        "Iron_Idx": iron_idx, "Yellow_Idx": yellow_idx,
        "Health_Ratio": health_ratio, "NIR_RE_ratio": nir_re_ratio,
        "PSI": psi, "EVI": evi, "SAVI": savi, "MCARI": mcari, "OSAVI": osavi,
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
        features[f"{idx_name}_iqr"] = np.percentile(v, 75) - np.percentile(v, 25)
        features[f"{idx_name}_skew"] = float(scipy_stats.skew(v))
    
    # Spectral shape
    features["slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["slope_re"] = mean_spectrum[3] - mean_spectrum[2]
    features["slope_nir"] = mean_spectrum[4] - mean_spectrum[3]
    features["curvature_re"] = mean_spectrum[3] - 0.5*(mean_spectrum[2] + mean_spectrum[4])
    features["nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)
    features["total_reflectance"] = np.sum(mean_spectrum)
    
    # Band statistics
    for i, name in enumerate(BAND_NAMES):
        b = bands[i].ravel()
        features[f"{name}_std"] = np.std(b)
        features[f"{name}_cv"] = np.std(b) / (np.mean(b) + eps)
        features[f"{name}_skew"] = float(scipy_stats.skew(b))
    
    # Inter-band correlations
    flat_bands = bands.reshape(5, -1)
    corr = np.corrcoef(flat_bands)
    for i in range(5):
        for j in range(i+1, 5):
            features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr[i, j]
    
    # Spatial texture
    for i, name in enumerate(BAND_NAMES):
        b = bands[i]
        gy, gx = np.gradient(b)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{name}_grad_mean"] = np.mean(grad_mag)
        features[f"{name}_grad_std"] = np.std(grad_mag)
    
    # HS features
    if hs_img is not None:
        hs_img = hs_img.astype(np.float32)
        n_bands = hs_img.shape[2]
        clean_start, clean_end = 10, min(110, n_bands - 1)
        clean_hs = hs_img[:, :, clean_start:clean_end]
        hs_mean = np.mean(clean_hs, axis=(0, 1))
        
        if len(hs_mean) > 2:
            hs_deriv1 = np.diff(hs_mean)
            hs_deriv2 = np.diff(hs_deriv1)
            features["hs_deriv1_mean"] = np.mean(hs_deriv1)
            features["hs_deriv1_std"] = np.std(hs_deriv1)
            features["hs_deriv1_max"] = np.max(hs_deriv1)
            features["hs_deriv1_min"] = np.min(hs_deriv1)
            if len(hs_deriv2) > 0:
                features["hs_deriv2_mean"] = np.mean(hs_deriv2)
                features["hs_deriv2_std"] = np.std(hs_deriv2)
        
        n = len(hs_mean)
        if n > 20:
            features["hs_blue_mean"] = np.mean(hs_mean[:n//10])
            features["hs_green_mean"] = np.mean(hs_mean[n//10:n//4])
            features["hs_red_mean"] = np.mean(hs_mean[n//4:n//2])
            features["hs_rede_mean"] = np.mean(hs_mean[n//2:3*n//5])
            features["hs_nir_mean"] = np.mean(hs_mean[3*n//5:])
            
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
                features["hs_Iron_Idx"] = (red_h - blue_h) / (red_h + blue_h + eps)
        
        features["hs_mean_reflectance"] = np.mean(hs_mean)
        features["hs_std_reflectance"] = np.std(hs_mean)
        features["hs_skewness"] = float(scipy_stats.skew(hs_mean))
    
    return features


def compute_prototypes(ms_dir, hs_dir=None):
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
    
    if hs_dir and os.path.exists(hs_dir):
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
    return final_ms, final_hs


def add_sam_features(features, ms_protos):
    """Add spectral angle features."""
    eps = 1e-8
    mean_spec = np.array([features.get(f"raw_{name}", 0) for name in BAND_NAMES])
    
    for cls_name, proto in ms_protos.items():
        dot = np.dot(mean_spec, proto)
        norm_prod = np.linalg.norm(mean_spec) * np.linalg.norm(proto)
        if norm_prod > 0:
            angle = np.arccos(np.clip(dot / norm_prod, -1, 1))
            features[f"SAM_{cls_name}"] = angle
        features[f"EucDist_{cls_name}"] = np.linalg.norm(mean_spec - proto)
        
        p = mean_spec / (np.sum(mean_spec) + eps)
        q = proto / (np.sum(proto) + eps)
        sid = np.sum(p * np.log2((p + eps) / (q + eps))) + np.sum(q * np.log2((q + eps) / (p + eps)))
        features[f"SID_{cls_name}"] = sid
    
    return features


def extract_all(ms_dir, hs_dir=None, ms_protos=None, hs_protos=None):
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
        if ms_protos:
            feats = add_sam_features(feats, ms_protos)
        
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
    ms_protos, hs_protos = compute_prototypes(CFG["train_ms_dir"], CFG["train_hs_dir"])
    for cls, proto in ms_protos.items():
        print(f"  MS {cls}: {proto}")

    print("\nExtracting training features...")
    train_feats, train_labels, train_fnames = extract_all(
        CFG["train_ms_dir"], CFG["train_hs_dir"], ms_protos, hs_protos
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
        CFG["val_ms_dir"], CFG["val_hs_dir"], ms_protos, hs_protos
    )
    black_mask = [f is None for f in val_feats]
    
    for i in range(len(val_feats)):
        if val_feats[i] is None:
            val_feats[i] = {k: 0.0 for k in feature_names}

    X_val = np.array([[f.get(k, 0.0) for k in feature_names] for f in val_feats], dtype=np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"  {len(val_feats)} samples ({sum(black_mask)} black)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ================================================================
    # One-vs-One Classification (Best Approach - 83.19% accuracy)
    # ================================================================
    print(f"\n{'='*70}")
    print("One-vs-One Classification")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": CFG["seed"],
        "tree_method": "hist",
        "verbosity": 0,
    }

    # Health vs Rust
    print("\n--- Health vs Rust ---")
    hr_mask = y_train != 2
    X_hr = X_train_scaled[hr_mask]
    y_hr = y_train[hr_mask]
    
    oof_hr = np.zeros((len(X_hr), 2))
    val_hr_folds = []
    
    skf_hr = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    for fold, (tr_idx, va_idx) in enumerate(skf_hr.split(X_hr, y_hr)):
        X_tr, X_va = X_hr[tr_idx], X_hr[va_idx]
        y_tr, y_va = y_hr[tr_idx], y_hr[va_idx]
        
        # Augment
        X_tr_aug = np.vstack([X_tr, X_tr + np.random.randn(*X_tr.shape) * 0.02])
        y_tr_aug = np.hstack([y_tr, y_tr])
        weights = np.ones(len(y_tr_aug))
        weights[y_tr_aug == 0] = 1.2
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr_aug, y_tr_aug, sample_weight=weights, eval_set=[(X_va, y_va)], verbose=False)
        
        oof_hr[va_idx] = model.predict_proba(X_va)
        val_hr_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, np.argmax(oof_hr[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")
    
    hr_acc = accuracy_score(y_hr, np.argmax(oof_hr, axis=1))
    print(f"Health vs Rust Accuracy: {hr_acc:.4f}")

    # Health vs Other
    print("\n--- Health vs Other ---")
    ho_mask = (y_train == 0) | (y_train == 2)
    X_ho = X_train_scaled[ho_mask]
    y_ho = np.where(y_train[ho_mask] == 0, 0, 1)
    
    oof_ho = np.zeros((len(X_ho), 2))
    val_ho_folds = []
    
    skf_ho = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    for fold, (tr_idx, va_idx) in enumerate(skf_ho.split(X_ho, y_ho)):
        X_tr, X_va = X_ho[tr_idx], X_ho[va_idx]
        y_tr, y_va = y_ho[tr_idx], y_ho[va_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        oof_ho[va_idx] = model.predict_proba(X_va)
        val_ho_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, np.argmax(oof_ho[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")
    
    ho_acc = accuracy_score(y_ho, np.argmax(oof_ho, axis=1))
    print(f"Health vs Other Accuracy: {ho_acc:.4f}")

    # Rust vs Other
    print("\n--- Rust vs Other ---")
    ro_mask = (y_train == 1) | (y_train == 2)
    X_ro = X_train_scaled[ro_mask]
    y_ro = np.where(y_train[ro_mask] == 1, 0, 1)
    
    oof_ro = np.zeros((len(X_ro), 2))
    val_ro_folds = []
    
    skf_ro = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    for fold, (tr_idx, va_idx) in enumerate(skf_ro.split(X_ro, y_ro)):
        X_tr, X_va = X_ro[tr_idx], X_ro[va_idx]
        y_tr, y_va = y_ro[tr_idx], y_ro[va_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        oof_ro[va_idx] = model.predict_proba(X_va)
        val_ro_folds.append(model.predict_proba(X_val_scaled))
        
        acc = accuracy_score(y_va, np.argmax(oof_ro[va_idx], axis=1))
        print(f"  Fold {fold+1}: Acc={acc:.4f}")
    
    ro_acc = accuracy_score(y_ro, np.argmax(oof_ro, axis=1))
    print(f"Rust vs Other Accuracy: {ro_acc:.4f}")

    # ================================================================
    # Combine via Voting
    # ================================================================
    print(f"\n{'='*70}")
    print("COMBINING One-vs-One PREDICTIONS")
    print(f"{'='*70}")

    hr_indices = np.where(hr_mask)[0]
    ho_indices = np.where(ho_mask)[0]
    ro_indices = np.where(ro_mask)[0]

    ovo_votes = np.zeros((len(X_train_scaled), 3))

    for i in range(len(X_train_scaled)):
        if i in hr_indices:
            hr_idx = np.where(hr_indices == i)[0][0]
            if oof_hr[hr_idx, 0] > 0.5:
                ovo_votes[i, 0] += 1
            else:
                ovo_votes[i, 1] += 1
        
        if i in ho_indices:
            ho_idx = np.where(ho_indices == i)[0][0]
            if oof_ho[ho_idx, 0] > 0.5:
                ovo_votes[i, 0] += 1
            else:
                ovo_votes[i, 2] += 1
        
        if i in ro_indices:
            ro_idx = np.where(ro_indices == i)[0][0]
            if oof_ro[ro_idx, 0] > 0.5:
                ovo_votes[i, 1] += 1
            else:
                ovo_votes[i, 2] += 1

    ovo_preds = np.argmax(ovo_votes, axis=1)
    ovo_acc = accuracy_score(y_train, ovo_preds)
    print(f"OvO Voting Accuracy: {ovo_acc:.4f}")
    print(classification_report(y_train, ovo_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    # ================================================================
    # Validation Predictions
    # ================================================================
    print(f"\n{'='*70}")
    print("VALIDATION PREDICTIONS")
    print(f"{'='*70}")

    val_hr_probs = np.mean(val_hr_folds, axis=0)
    val_ho_probs = np.mean(val_ho_folds, axis=0)
    val_ro_probs = np.mean(val_ro_folds, axis=0)

    val_ovo_votes = np.zeros((len(X_val_scaled), 3))
    for i in range(len(X_val_scaled)):
        if val_hr_probs[i, 0] > 0.5:
            val_ovo_votes[i, 0] += 1
        else:
            val_ovo_votes[i, 1] += 1
        
        if val_ho_probs[i, 0] > 0.5:
            val_ovo_votes[i, 0] += 1
        else:
            val_ovo_votes[i, 2] += 1
        
        if val_ro_probs[i, 0] > 0.5:
            val_ovo_votes[i, 1] += 1
        else:
            val_ovo_votes[i, 2] += 1

    val_preds = np.argmax(val_ovo_votes, axis=1)

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
    val_probs = val_ovo_votes / 2.0
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
    print(f"Health vs Rust Accuracy: {hr_acc:.4f}")
    print(f"Health vs Other Accuracy: {ho_acc:.4f}")
    print(f"Rust vs Other Accuracy: {ro_acc:.4f}")
    print(f"OvO Voting Accuracy: {ovo_acc:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
