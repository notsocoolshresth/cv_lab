"""
MS Classification via Spectral Feature Engineering + SVM
- Reuses proven handcrafted MS features from train_ms_xgb.py
- Trains SVM variants with 5-fold stratified CV
- Uses Torch MPS acceleration for feature standardization when available
- Saves OOF/val probabilities and submission CSV for fusion
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
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    import torch
except Exception:
    torch = None


CFG = {
    "data_dir": "Kaggle_Prepared/train/MS",
    "val_dir": "Kaggle_Prepared/val/MS",
    "output_dir": "ms_svm",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
    "use_mps": True,
    "svm_rbf": {
        "C": 1.5,
        "kernel": "rbf",
        "gamma": 0.01,
        "probability": True,
        "class_weight": "balanced",
    },
    "svm_rbf_alt": {
        "C": 4.0,
        "kernel": "rbf",
        "gamma": 0.003,
        "probability": True,
        "class_weight": "balanced",
    },
}


CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


def get_torch_device(use_mps=True):
    if torch is None:
        return None
    if use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def standardize_with_torch(train_arr, val_arr, use_mps=True):
    device = get_torch_device(use_mps=use_mps)
    if torch is None or device is None:
        scaler = StandardScaler()
        return scaler.fit_transform(train_arr), scaler.transform(val_arr), "sklearn-cpu"

    train_t = torch.from_numpy(train_arr.astype(np.float32)).to(device)
    val_t = torch.from_numpy(val_arr.astype(np.float32)).to(device)

    mean = train_t.mean(dim=0, keepdim=True)
    std = train_t.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    train_scaled = (train_t - mean) / std
    val_scaled = (val_t - mean) / std

    return (
        train_scaled.cpu().numpy().astype(np.float32),
        val_scaled.cpu().numpy().astype(np.float32),
        f"torch-{device.type}",
    )


def extract_features(img):
    img = img.astype(np.float32)
    features = {}

    bands = img.transpose(2, 0, 1)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]
    eps = 1e-8

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
        "NDVI": ndvi,
        "NDRE": ndre,
        "GNDVI": gndvi,
        "SAVI": savi,
        "CI_RE": ci_rededge,
        "CI_Green": ci_green,
        "RG_ratio": rg_ratio,
        "RB_ratio": rb_ratio,
        "RE_R_ratio": re_r_ratio,
        "NIR_R_ratio": nir_r_ratio,
        "NIR_RE_ratio": nir_re_ratio,
        "EVI": evi,
        "MCARI": mcari,
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

    flat_bands = bands.reshape(5, -1)
    corr_matrix = np.corrcoef(flat_bands)
    for i in range(5):
        for j in range(i + 1, 5):
            features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]

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

    for idx_name, idx_map in [("NDVI", ndvi), ("NDRE", ndre)]:
        idx_map = np.clip(idx_map, -10, 10)
        gy, gx = np.gradient(idx_map)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{idx_name}_grad_mean"] = np.mean(grad_mag)
        features[f"{idx_name}_grad_std"] = np.std(grad_mag)

    band_means = [np.mean(bands[i]) for i in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            features[f"meanratio_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = band_means[i] / (band_means[j] + eps)

    mean_spectrum = np.array(band_means)
    features["spec_slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["spec_slope_rededge"] = mean_spectrum[3] - mean_spectrum[2]
    features["spec_slope_nir"] = mean_spectrum[4] - mean_spectrum[3]
    features["spec_curvature"] = mean_spectrum[3] - 0.5 * (mean_spectrum[2] + mean_spectrum[4])
    features["spec_total_reflectance"] = np.sum(mean_spectrum)
    features["spec_nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)

    return features


def extract_all_features(data_dir, file_list=None):
    if file_list is None:
        file_list = sorted(os.listdir(data_dir))

    all_features = []
    all_labels = []
    all_fnames = []
    skipped_black = 0

    for fname in file_list:
        fp = os.path.join(data_dir, fname)
        img = tiff.imread(fp).astype(np.float32)

        if img.mean() < 1.0:
            skipped_black += 1
            if "_hyper_" in fname:
                continue
            all_features.append(None)
            all_labels.append(-1)
            all_fnames.append(fname)
            continue

        feats = extract_features(img)
        all_features.append(feats)

        if "_hyper_" in fname:
            cls_name = fname.split("_hyper_")[0]
            all_labels.append(CLASS_MAP[cls_name])
        else:
            all_labels.append(-1)

        all_fnames.append(fname)

    if skipped_black > 0:
        print(f"  Skipped/flagged {skipped_black} black images")

    return all_features, all_labels, all_fnames


def fold_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    recall = {}
    for ci, cn in INV_CLASS_MAP.items():
        mask = y_true == ci
        recall[cn] = (y_pred[mask] == ci).mean() if mask.sum() > 0 else 0.0
    return acc, mf1, recall


def run_cv_model(model_name, model_params, X_train, y_train, X_val, black_mask):
    print(f"\n{'='*60}")
    print(f"{model_name} 5-Fold CV")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    oof_probs = np.zeros((len(X_train), CFG["num_classes"]), dtype=np.float32)
    oof_preds = np.zeros(len(X_train), dtype=int)
    val_probs_folds = []
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold + 1} ---")
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        X_tr_scaled, X_va_scaled, scaling_backend = standardize_with_torch(
            X_tr, X_va, use_mps=CFG["use_mps"]
        )
        _, X_val_scaled, _ = standardize_with_torch(X_tr, X_val, use_mps=CFG["use_mps"])

        if fold == 0:
            print(f"  Scaling backend: {scaling_backend}")

        model = SVC(**model_params, random_state=CFG["seed"])
        model.fit(X_tr_scaled, y_tr)

        va_probs = model.predict_proba(X_va_scaled)
        va_preds = np.argmax(va_probs, axis=1)
        oof_probs[va_idx] = va_probs
        oof_preds[va_idx] = va_preds

        acc, mf1, recall = fold_metrics(y_va, va_preds)
        rec_str = " ".join([f"{k}:{v:.3f}" for k, v in recall.items()])
        print(f"  Acc={acc:.4f} F1={mf1:.4f} | {rec_str}")

        fold_results.append({"acc": acc, "f1": mf1, "recall": recall})
        val_probs_folds.append(model.predict_proba(X_val_scaled))

    print(f"\n{model_name} CV SUMMARY")
    oof_acc = accuracy_score(y_train, oof_preds)
    oof_f1 = f1_score(y_train, oof_preds, average="macro")
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF Macro F1: {oof_f1:.4f}")
    print(classification_report(y_train, oof_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    val_probs = np.mean(val_probs_folds, axis=0)
    for i, is_black in enumerate(black_mask):
        if is_black:
            val_probs[i] = [0.0, 0.0, 1.0]

    return {
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "val_probs": val_probs,
        "fold_results": fold_results,
        "oof_acc": oof_acc,
        "oof_f1": oof_f1,
    }


def main():
    np.random.seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    print("Extracting training features...")
    train_feats, train_labels, train_fnames = extract_all_features(CFG["data_dir"])
    print(f"  {len(train_feats)} samples, {len(train_feats[0])} features each")

    feature_names = list(train_feats[0].keys())
    X_train = np.array([[f[k] for k in feature_names] for f in train_feats], dtype=np.float32)
    y_train = np.array(train_labels)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")

    print("Extracting validation features...")
    val_feats, _, val_fnames = extract_all_features(CFG["val_dir"])
    black_mask = [f is None for f in val_feats]
    for i in range(len(val_feats)):
        if val_feats[i] is None:
            val_feats[i] = {k: 0.0 for k in feature_names}

    X_val = np.array([[f[k] for k in feature_names] for f in val_feats], dtype=np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"  {len(val_feats)} samples ({sum(black_mask)} black)")

    backend_device = get_torch_device(use_mps=CFG["use_mps"])
    backend_name = "sklearn-cpu" if backend_device is None else f"torch-{backend_device.type}"
    print(f"Preprocessing backend preference: {backend_name}")

    rbf_result = run_cv_model("SVM-RBF", CFG["svm_rbf"], X_train, y_train, X_val, black_mask)
    rbf_alt_result = run_cv_model("SVM-RBF-ALT", CFG["svm_rbf_alt"], X_train, y_train, X_val, black_mask)

    print(f"\n{'='*60}")
    print("SVM Ensemble (RBF + RBF-ALT)")
    print(f"{'='*60}")
    ens_oof_probs = 0.5 * rbf_result["oof_probs"] + 0.5 * rbf_alt_result["oof_probs"]
    ens_oof_preds = np.argmax(ens_oof_probs, axis=1)
    ens_acc = accuracy_score(y_train, ens_oof_preds)
    ens_f1 = f1_score(y_train, ens_oof_preds, average="macro")
    print(f"Ensemble OOF Acc: {ens_acc:.4f}, F1: {ens_f1:.4f}")
    print(classification_report(y_train, ens_oof_preds, target_names=list(CLASS_MAP.keys()), digits=4))

    rbf_val_probs = rbf_result["val_probs"]
    rbf_alt_val_probs = rbf_alt_result["val_probs"]
    ens_val_probs = 0.5 * rbf_val_probs + 0.5 * rbf_alt_val_probs

    candidate_models = [
        ("svm_rbf", rbf_result["oof_acc"], rbf_result["oof_f1"], rbf_val_probs),
        ("svm_rbf_alt", rbf_alt_result["oof_acc"], rbf_alt_result["oof_f1"], rbf_alt_val_probs),
        ("svm_ensemble", ens_acc, ens_f1, ens_val_probs),
    ]
    best_name, best_acc, best_f1, best_val_probs = max(candidate_models, key=lambda x: (x[1], x[2]))
    print(f"\nBest model for submission: {best_name} (OOF Acc={best_acc:.4f}, F1={best_f1:.4f})")

    best_val_preds = np.argmax(best_val_probs, axis=1)
    pred_classes = [INV_CLASS_MAP[p] for p in best_val_preds]
    dist = {c: pred_classes.count(c) for c in CLASS_MAP}
    print(f"\nVal prediction distribution: {dist}")

    np.save(os.path.join(CFG["output_dir"], "ms_val_probs_svm_rbf.npy"), rbf_val_probs)
    np.save(os.path.join(CFG["output_dir"], "ms_val_probs_svm_rbf_alt.npy"), rbf_alt_val_probs)
    np.save(os.path.join(CFG["output_dir"], "ms_val_probs_svm_ensemble.npy"), ens_val_probs)
    np.save(os.path.join(CFG["output_dir"], "ms_val_probs_svm_best.npy"), best_val_probs)

    np.save(os.path.join(CFG["output_dir"], "oof_probs_svm_rbf.npy"), rbf_result["oof_probs"])
    np.save(os.path.join(CFG["output_dir"], "oof_probs_svm_rbf_alt.npy"), rbf_alt_result["oof_probs"])

    sub_path = os.path.join(CFG["output_dir"], "ms_submission_svm.csv")
    with open(sub_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        for fname, cls in zip(val_fnames, pred_classes):
            writer.writerow([fname, cls])

    with open(os.path.join(CFG["output_dir"], "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    summary = {
        "svm_rbf_oof_acc": float(rbf_result["oof_acc"]),
        "svm_rbf_oof_f1": float(rbf_result["oof_f1"]),
        "svm_rbf_alt_oof_acc": float(rbf_alt_result["oof_acc"]),
        "svm_rbf_alt_oof_f1": float(rbf_alt_result["oof_f1"]),
        "svm_ensemble_oof_acc": float(ens_acc),
        "svm_ensemble_oof_f1": float(ens_f1),
        "best_model": best_name,
        "best_model_oof_acc": float(best_acc),
        "best_model_oof_f1": float(best_f1),
        "preprocess_backend": backend_name,
    }
    with open(os.path.join(CFG["output_dir"], "svm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs to: {CFG['output_dir']}/")
    print("Done!")


if __name__ == "__main__":
    main()
