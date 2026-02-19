"""
Spectral Classification v4 - Targeting 80%+ CV Accuracy
========================================================

Key insight: The bottleneck is Health vs Rust separation (73%).
Previous approaches failed because:
1. Hierarchical approach has error propagation
2. Standard features don't capture subtle spectral differences

New approach:
1. Use RAW spectral signatures as features (not just statistics)
2. Train specialized models for each class pair
3. Use One-vs-One classification with voting
4. Apply spectral matching techniques
5. Use all available spectral information

Uses CatBoost for excellent GPU support on Kaggle.
"""

import os
import csv
import json
import warnings
import numpy as np
import tifffile as tiff
from scipy import ndimage, stats as scipy_stats
from scipy.spatial.distance import cosine, euclidean
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# Check if GPU is available for CatBoost
# On Kaggle, GPU must be enabled in notebook settings (Settings > Accelerator > GPU T4 x2)
def check_gpu_available():
    """Check if GPU is available for CatBoost training."""
    gpu_available = False
    
    # First check if CUDA/nvidia-smi is available
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("NVIDIA GPU detected via nvidia-smi:")
            lines = result.stdout.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"  {line}")
            gpu_available = True
    except Exception as e:
        print(f"nvidia-smi check failed: {type(e).__name__}")
    
    # If nvidia-smi found GPU, try CatBoost GPU
    if gpu_available:
        try:
            test_clf = CatBoostClassifier(
                iterations=2,
                task_type="GPU",
                devices="0",
                verbose=0
            )
            test_clf.fit(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([0, 1]))
            print("CatBoost GPU test: SUCCESS")
            return True
        except Exception as e:
            print(f"CatBoost GPU test failed: {type(e).__name__}: {str(e)[:200]}")
            print("Note: On Kaggle, ensure GPU is enabled in Settings > Accelerator")
    
    return False

if check_gpu_available():
    TASK_TYPE = "GPU"
    DEVICES = "0"
    print("✓ Using GPU for CatBoost")
else:
    TASK_TYPE = "CPU"
    DEVICES = None
    print("✓ Using CPU for CatBoost")
    print("  Tip: Enable GPU in Kaggle notebook settings for faster training")

CFG = {
    "train_ms_dir": "Kaggle_Prepared/train/MS",
    "val_ms_dir": "Kaggle_Prepared/val/MS",
    "train_hs_dir": "Kaggle_Prepared/train/HS",
    "val_hs_dir": "Kaggle_Prepared/val/HS",
    "output_dir": "spectral_v4",
    "n_folds": 5,
    "seed": 42,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]


def extract_raw_spectral_features(ms_img, hs_img=None):
    """
    Extract raw spectral features + derived features.
    Key: Use the actual spectral values, not just statistics.
    """
    ms_img = ms_img.astype(np.float32)
    eps = 1e-8
    features = {}

    # ====== MS RAW SPECTRAL VALUES ======
    bands = ms_img.transpose(2, 0, 1)
    blue, green, red, rededge, nir = bands[0], bands[1], bands[2], bands[3], bands[4]

    # Mean spectrum (5 values) - THE KEY FEATURES
    mean_spectrum = np.array([np.mean(b) for b in [blue, green, red, rededge, nir]])

    # Raw mean values as features
    for i, name in enumerate(BAND_NAMES):
        features[f"raw_{name}"] = mean_spectrum[i]

    # ====== CRITICAL: SPECTRAL INDICES (per-pixel then aggregated) ======
    # These are the most discriminative features for vegetation health

    # NDVI: Key for vegetation vigor
    ndvi = (nir - red) / (nir + red + eps)
    # NDRE: Key for chlorophyll content (Health vs Rust differentiator!)
    ndre = (nir - rededge) / (nir + rededge + eps)
    # GNDVI: Green NDVI
    gndvi = (nir - green) / (nir + green + eps)
    # CI_RedEdge: Chlorophyll Index
    ci_re = nir / (rededge + eps) - 1
    # CI_Green
    ci_green = nir / (green + eps) - 1

    # RUST-SPECIFIC INDICES
    # Red/Green ratio - rust has more red (iron signature)
    rg_ratio = red / (green + eps)
    # Red/Blue ratio
    rb_ratio = red / (blue + eps)
    # Iron index
    iron_idx = (red - blue) / (red + blue + eps)
    # Yellowing index (rust causes yellowing)
    yellow_idx = (green - blue) / (green + blue + eps)

    # HEALTH-SPECIFIC INDICES
    # NIR/Red ratio - healthy has higher ratio
    health_ratio = nir / (red + eps)
    # NIR/RedEdge ratio
    nir_re_ratio = nir / (rededge + eps)
    # Plant stress index
    psi = (rededge - red) / (rededge + red + eps)

    # EVI
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    # SAVI
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    # MCARI
    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))
    # OSAVI
    osavi = (nir - red) / (nir + red + 0.16 + eps)

    # Aggregate indices
    indices = {
        "NDVI": ndvi,
        "NDRE": ndre,
        "GNDVI": gndvi,
        "CI_RE": ci_re,
        "CI_Green": ci_green,
        "RG_ratio": rg_ratio,
        "RB_ratio": rb_ratio,
        "Iron_Idx": iron_idx,
        "Yellow_Idx": yellow_idx,
        "Health_Ratio": health_ratio,
        "NIR_RE_ratio": nir_re_ratio,
        "PSI": psi,
        "EVI": evi,
        "SAVI": savi,
        "MCARI": mcari,
        "OSAVI": osavi,
    }

    for idx_name, idx_map in indices.items():
        idx_map = np.clip(idx_map, -10, 10)
        v = idx_map.ravel()
        # Full statistics
        features[f"{idx_name}_mean"] = np.mean(v)
        features[f"{idx_name}_std"] = np.std(v)
        features[f"{idx_name}_min"] = np.min(v)
        features[f"{idx_name}_max"] = np.max(v)
        features[f"{idx_name}_median"] = np.median(v)
        features[f"{idx_name}_p10"] = np.percentile(v, 10)
        features[f"{idx_name}_p25"] = np.percentile(v, 25)
        features[f"{idx_name}_p75"] = np.percentile(v, 75)
        features[f"{idx_name}_p90"] = np.percentile(v, 90)
        features[f"{idx_name}_iqr"] = (
            features[f"{idx_name}_p75"] - features[f"{idx_name}_p25"]
        )
        features[f"{idx_name}_skew"] = float(scipy_stats.skew(v))

    # ====== SPECTRAL SHAPE FEATURES ======
    # These capture the "shape" of the spectral curve

    # Slopes between bands
    features["slope_vis"] = mean_spectrum[2] - mean_spectrum[0]  # Red - Blue
    features["slope_re"] = mean_spectrum[3] - mean_spectrum[2]  # RedEdge - Red
    features["slope_nir"] = mean_spectrum[4] - mean_spectrum[3]  # NIR - RedEdge

    # Curvature at Red Edge (key for vegetation health)
    features["curvature_re"] = mean_spectrum[3] - 0.5 * (
        mean_spectrum[2] + mean_spectrum[4]
    )

    # NIR/VIS ratio (overall vegetation vigor)
    features["nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)

    # Total reflectance
    features["total_reflectance"] = np.sum(mean_spectrum)

    # ====== BAND STATISTICS ======
    for i, name in enumerate(BAND_NAMES):
        b = bands[i].ravel()
        features[f"{name}_std"] = np.std(b)
        features[f"{name}_cv"] = np.std(b) / (np.mean(b) + eps)
        features[f"{name}_skew"] = float(scipy_stats.skew(b))

    # ====== INTER-BAND CORRELATIONS ======
    flat_bands = bands.reshape(5, -1)
    corr = np.corrcoef(flat_bands)
    for i in range(5):
        for j in range(i + 1, 5):
            features[f"corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr[i, j]

    # ====== SPATIAL TEXTURE ======
    for i, name in enumerate(BAND_NAMES):
        b = bands[i]
        gy, gx = np.gradient(b)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"{name}_grad_mean"] = np.mean(grad_mag)
        features[f"{name}_grad_std"] = np.std(grad_mag)

    # ====== HS FEATURES (if available) ======
    if hs_img is not None:
        hs_img = hs_img.astype(np.float32)
        n_bands = hs_img.shape[2]

        # Use clean bands
        clean_start, clean_end = 10, min(110, n_bands - 1)
        clean_hs = hs_img[:, :, clean_start:clean_end]

        # Mean HS spectrum
        hs_mean = np.mean(clean_hs, axis=(0, 1))

        # HS spectral derivatives (key for subtle differences)
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

        # HS regions
        n = len(hs_mean)
        if n > 20:
            # Approximate spectral regions
            features["hs_blue_mean"] = np.mean(hs_mean[: n // 10])
            features["hs_green_mean"] = np.mean(hs_mean[n // 10 : n // 4])
            features["hs_red_mean"] = np.mean(hs_mean[n // 4 : n // 2])
            features["hs_rede_mean"] = np.mean(hs_mean[n // 2 : 3 * n // 5])
            features["hs_nir_mean"] = np.mean(hs_mean[3 * n // 5 :])

            # HS-specific indices
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

        # HS statistics
        features["hs_mean_reflectance"] = np.mean(hs_mean)
        features["hs_std_reflectance"] = np.std(hs_mean)
        features["hs_skewness"] = float(scipy_stats.skew(hs_mean))

    return features


def compute_class_prototypes(ms_dir, hs_dir=None):
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
        mean_spec = np.array([np.mean(img[:, :, i]) for i in range(5)])
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
            clean = img[:, :, 10 : min(110, n_bands - 1)]
            mean_spec = np.mean(clean, axis=(0, 1))
            hs_protos[cls_name].append(mean_spec)

    final_ms = {cls: np.mean(vals, axis=0) for cls, vals in ms_protos.items() if vals}
    final_hs = {cls: np.mean(vals, axis=0) for cls, vals in hs_protos.items() if vals}

    return final_ms, final_hs


def add_spectral_matching_features(features, ms_protos, hs_protos=None):
    """Add spectral angle and distance features to class prototypes."""
    eps = 1e-8

    # Get mean spectrum from features
    mean_spec = np.array([features.get(f"raw_{name}", 0) for name in BAND_NAMES])

    for cls_name, proto in ms_protos.items():
        # Spectral Angle Mapper
        dot = np.dot(mean_spec, proto)
        norm_prod = np.linalg.norm(mean_spec) * np.linalg.norm(proto)
        if norm_prod > 0:
            angle = np.arccos(np.clip(dot / norm_prod, -1, 1))
            features[f"SAM_{cls_name}"] = angle

        # Euclidean distance
        features[f"EucDist_{cls_name}"] = np.linalg.norm(mean_spec - proto)

        # Spectral Information Divergence
        p = mean_spec / (np.sum(mean_spec) + eps)
        q = proto / (np.sum(proto) + eps)
        sid = np.sum(p * np.log2((p + eps) / (q + eps))) + np.sum(
            q * np.log2((q + eps) / (p + eps))
        )
        features[f"SID_{cls_name}"] = sid

    return features


def extract_all_features(ms_dir, hs_dir=None, ms_protos=None, hs_protos=None):
    """Extract features from all files."""
    ms_files = sorted(os.listdir(ms_dir))

    all_features = []
    all_labels = []
    all_fnames = []
    skipped = 0

    for ms_f in ms_files:
        ms_fp = os.path.join(ms_dir, ms_f)
        ms_img = tiff.imread(ms_fp).astype(np.float32)

        # Find HS file
        hs_img = None
        if hs_dir:
            hs_fp = os.path.join(hs_dir, ms_f)
            if os.path.exists(hs_fp):
                hs_img = tiff.imread(hs_fp).astype(np.float32)
                if hs_img.mean() < 1.0:
                    hs_img = None

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

        # Extract features
        feats = extract_raw_spectral_features(ms_img, hs_img)

        # Add spectral matching features
        if ms_protos:
            feats = add_spectral_matching_features(feats, ms_protos, hs_protos)

        all_features.append(feats)

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

    # --- Compute prototypes ---
    print("Computing class prototypes...")
    ms_protos, hs_protos = compute_class_prototypes(
        CFG["train_ms_dir"], CFG["train_hs_dir"]
    )
    for cls, proto in ms_protos.items():
        print(f"  MS {cls}: {proto}")

    # --- Extract training features ---
    print("\nExtracting training features...")
    train_feats, train_labels, train_fnames = extract_all_features(
        CFG["train_ms_dir"], CFG["train_hs_dir"], ms_protos, hs_protos
    )
    print(f"  {len(train_feats)} samples")

    feature_names = list(train_feats[0].keys())
    X_train = np.array(
        [[f.get(k, 0.0) for k in feature_names] for f in train_feats], dtype=np.float32
    )
    y_train = np.array(train_labels)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

    print(f"  {X_train.shape[1]} features extracted")
    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")

    # --- Extract val features ---
    print("\nExtracting validation features...")
    val_feats, val_labels, val_fnames = extract_all_features(
        CFG["val_ms_dir"], CFG["val_hs_dir"], ms_protos, hs_protos
    )
    black_mask = [f is None for f in val_feats]

    for i in range(len(val_feats)):
        if val_feats[i] is None:
            val_feats[i] = {k: 0.0 for k in feature_names}

    X_val = np.array(
        [[f.get(k, 0.0) for k in feature_names] for f in val_feats], dtype=np.float32
    )
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"  {len(val_feats)} samples ({sum(black_mask)} black)")

    # --- Normalize ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ================================================================
    # APPROACH 1: One-vs-One Classification with Voting
    # ================================================================
    print(f"\n{'=' * 70}")
    print("APPROACH 1: One-vs-One Classification")
    print("Train 3 binary classifiers: Health vs Rust, Health vs Other, Rust vs Other")
    print(f"{'=' * 70}")

    skf = StratifiedKFold(
        n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"]
    )

    # Classifier 1: Health vs Rust
    print("\n--- Health vs Rust ---")
    hr_mask = y_train != 2  # Not Other
    X_hr = X_train_scaled[hr_mask]
    y_hr = y_train[hr_mask]  # 0=Health, 1=Rust

    oof_hr = np.zeros((len(X_hr), 2))
    val_hr_folds = []

    cat_params_hr = {
        "iterations": 600,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "random_seed": CFG["seed"],
        "task_type": TASK_TYPE,
        "verbose": 0,
    }
    if DEVICES:
        cat_params_hr["devices"] = DEVICES

    skf_hr = StratifiedKFold(
        n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"]
    )

    for fold, (tr_idx, va_idx) in enumerate(skf_hr.split(X_hr, y_hr)):
        X_tr, X_va = X_hr[tr_idx], X_hr[va_idx]
        y_tr, y_va = y_hr[tr_idx], y_hr[va_idx]

        # Augment
        n_aug = 2
        X_tr_aug = np.vstack(
            [X_tr] + [X_tr + np.random.randn(*X_tr.shape) * 0.02 for _ in range(n_aug)]
        )
        y_tr_aug = np.hstack([y_tr] * (n_aug + 1))

        # Sample weights
        weights = np.ones(len(y_tr_aug))
        weights[y_tr_aug == 0] = 1.2  # Slightly favor Health

        model = CatBoostClassifier(**cat_params_hr)
        model.fit(
            X_tr_aug,
            y_tr_aug,
            sample_weight=weights,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )

        oof_hr[va_idx] = model.predict_proba(X_va)
        val_hr_folds.append(model.predict_proba(X_val_scaled))

        acc = accuracy_score(y_va, np.argmax(oof_hr[va_idx], axis=1))
        print(f"  Fold {fold + 1}: Acc={acc:.4f}")

    hr_acc = accuracy_score(y_hr, np.argmax(oof_hr, axis=1))
    hr_health_recall = (np.argmax(oof_hr, axis=1)[y_hr == 0] == 0).mean()
    print(f"Health vs Rust Accuracy: {hr_acc:.4f}")
    print(f"Health Recall: {hr_health_recall:.4f}")

    # Classifier 2: Health vs Other
    print("\n--- Health vs Other ---")
    ho_mask = (y_train == 0) | (y_train == 2)
    X_ho = X_train_scaled[ho_mask]
    y_ho = np.where(y_train[ho_mask] == 0, 0, 1)  # 0=Health, 1=Other

    oof_ho = np.zeros((len(X_ho), 2))
    val_ho_folds = []

    skf_ho = StratifiedKFold(
        n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"]
    )

    for fold, (tr_idx, va_idx) in enumerate(skf_ho.split(X_ho, y_ho)):
        X_tr, X_va = X_ho[tr_idx], X_ho[va_idx]
        y_tr, y_va = y_ho[tr_idx], y_ho[va_idx]

        model = CatBoostClassifier(**cat_params_hr)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)

        oof_ho[va_idx] = model.predict_proba(X_va)
        val_ho_folds.append(model.predict_proba(X_val_scaled))

        acc = accuracy_score(y_va, np.argmax(oof_ho[va_idx], axis=1))
        print(f"  Fold {fold + 1}: Acc={acc:.4f}")

    ho_acc = accuracy_score(y_ho, np.argmax(oof_ho, axis=1))
    print(f"Health vs Other Accuracy: {ho_acc:.4f}")

    # Classifier 3: Rust vs Other
    print("\n--- Rust vs Other ---")
    ro_mask = (y_train == 1) | (y_train == 2)
    X_ro = X_train_scaled[ro_mask]
    y_ro = np.where(y_train[ro_mask] == 1, 0, 1)  # 0=Rust, 1=Other

    oof_ro = np.zeros((len(X_ro), 2))
    val_ro_folds = []

    skf_ro = StratifiedKFold(
        n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"]
    )

    for fold, (tr_idx, va_idx) in enumerate(skf_ro.split(X_ro, y_ro)):
        X_tr, X_va = X_ro[tr_idx], X_ro[va_idx]
        y_tr, y_va = y_ro[tr_idx], y_ro[va_idx]

        model = CatBoostClassifier(**cat_params_hr)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)

        oof_ro[va_idx] = model.predict_proba(X_va)
        val_ro_folds.append(model.predict_proba(X_val_scaled))

        acc = accuracy_score(y_va, np.argmax(oof_ro[va_idx], axis=1))
        print(f"  Fold {fold + 1}: Acc={acc:.4f}")

    ro_acc = accuracy_score(y_ro, np.argmax(oof_ro, axis=1))
    print(f"Rust vs Other Accuracy: {ro_acc:.4f}")

    # ================================================================
    # Combine OvO predictions via voting
    # ================================================================
    print(f"\n{'=' * 70}")
    print("COMBINING One-vs-One PREDICTIONS")
    print(f"{'=' * 70}")

    # For each training sample, count votes
    oof_hr_full = np.zeros((len(X_train_scaled), 2))
    oof_ho_full = np.zeros((len(X_train_scaled), 2))
    oof_ro_full = np.zeros((len(X_train_scaled), 2))

    # Map back to full indices
    hr_indices = np.where(hr_mask)[0]
    ho_indices = np.where(ho_mask)[0]
    ro_indices = np.where(ro_mask)[0]

    oof_hr_full[hr_indices] = oof_hr
    oof_ho_full[ho_indices] = oof_ho
    oof_ro_full[ro_indices] = oof_ro

    # Voting
    ovo_votes = np.zeros((len(X_train_scaled), 3))  # Votes for Health, Rust, Other

    for i in range(len(X_train_scaled)):
        # Health vs Rust
        if i in hr_indices:
            hr_idx = np.where(hr_indices == i)[0][0]
            if oof_hr[hr_idx, 0] > 0.5:
                ovo_votes[i, 0] += 1  # Vote for Health
            else:
                ovo_votes[i, 1] += 1  # Vote for Rust

        # Health vs Other
        if i in ho_indices:
            ho_idx = np.where(ho_indices == i)[0][0]
            if oof_ho[ho_idx, 0] > 0.5:
                ovo_votes[i, 0] += 1  # Vote for Health
            else:
                ovo_votes[i, 2] += 1  # Vote for Other

        # Rust vs Other
        if i in ro_indices:
            ro_idx = np.where(ro_indices == i)[0][0]
            if oof_ro[ro_idx, 0] > 0.5:
                ovo_votes[i, 1] += 1  # Vote for Rust
            else:
                ovo_votes[i, 2] += 1  # Vote for Other

    ovo_preds = np.argmax(ovo_votes, axis=1)
    ovo_acc = accuracy_score(y_train, ovo_preds)
    print(f"OvO Voting Accuracy: {ovo_acc:.4f}")
    print(
        classification_report(
            y_train, ovo_preds, target_names=list(CLASS_MAP.keys()), digits=4
        )
    )

    # ================================================================
    # APPROACH 2: Direct Multi-class with Heavy Class Weights
    # ================================================================
    print(f"\n{'=' * 70}")
    print("APPROACH 2: Direct Multi-class with Optimized Weights")
    print(f"{'=' * 70}")

    # Try different class weight combinations
    best_acc = 0
    best_weights = (1.0, 1.0, 1.0)  # Default weights
    best_oof = np.zeros((len(X_train_scaled), 3))  # Default OOF

    for w_health in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for w_rust in [1.0, 1.2, 1.5]:
            for w_other in [1.0, 0.8, 1.2]:
                class_weights = {0: w_health, 1: w_rust, 2: w_other}

                oof_mc = np.zeros((len(X_train_scaled), 3))

                cat_params_mc = {
                    "iterations": 600,
                    "depth": 6,
                    "learning_rate": 0.03,
                    "l2_leaf_reg": 3,
                    "random_seed": CFG["seed"],
                    "task_type": TASK_TYPE,
                    "verbose": 0,
                }
                if DEVICES:
                    cat_params_mc["devices"] = DEVICES

                for fold, (tr_idx, va_idx) in enumerate(
                    skf.split(X_train_scaled, y_train)
                ):
                    X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                    sample_weights = np.array([class_weights[y] for y in y_tr])

                    model = CatBoostClassifier(**cat_params_mc)
                    model.fit(
                        X_tr,
                        y_tr,
                        sample_weight=sample_weights,
                        eval_set=(X_va, y_va),
                        early_stopping_rounds=50,
                        verbose=False,
                    )

                    oof_mc[va_idx] = model.predict_proba(X_va)

                mc_preds = np.argmax(oof_mc, axis=1)
                acc = accuracy_score(y_train, mc_preds)
                health_rec = (mc_preds[y_train == 0] == 0).mean()

                # Prefer models with good health recall
                if health_rec > 0.5 and acc > best_acc:
                    best_acc = acc
                    best_weights = (w_health, w_rust, w_other)
                    best_oof = oof_mc.copy()

    print(
        f"Best weights: Health={best_weights[0]}, Rust={best_weights[1]}, Other={best_weights[2]}"
    )
    print(f"Best Multi-class Accuracy: {best_acc:.4f}")
    mc_preds = np.argmax(best_oof, axis=1)
    print(
        classification_report(
            y_train, mc_preds, target_names=list(CLASS_MAP.keys()), digits=4
        )
    )

    # ================================================================
    # APPROACH 3: Ensemble of OvO and Multi-class
    # ================================================================
    print(f"\n{'=' * 70}")
    print("APPROACH 3: Ensemble OvO + Multi-class")
    print(f"{'=' * 70}")

    # Convert OvO votes to probabilities
    ovo_probs = ovo_votes / 2.0  # Normalize to [0, 1]

    # Ensemble
    ens_oof = 0.5 * ovo_probs + 0.5 * best_oof
    ens_preds = np.argmax(ens_oof, axis=1)
    ens_acc = accuracy_score(y_train, ens_preds)
    print(f"Ensemble Accuracy: {ens_acc:.4f}")
    print(
        classification_report(
            y_train, ens_preds, target_names=list(CLASS_MAP.keys()), digits=4
        )
    )

    # ================================================================
    # FINAL PREDICTIONS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("FINAL VALIDATION PREDICTIONS")
    print(f"{'=' * 70}")

    # Get validation predictions from all approaches
    val_hr_probs = np.mean(val_hr_folds, axis=0)
    val_ho_probs = np.mean(val_ho_folds, axis=0)
    val_ro_probs = np.mean(val_ro_folds, axis=0)

    # OvO voting for validation
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

    val_ovo_probs = val_ovo_votes / 2.0

    # Multi-class predictions
    val_mc_folds = []
    cat_params_mc = {
        "iterations": 600,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "random_seed": CFG["seed"],
        "task_type": TASK_TYPE,
        "verbose": 0,
    }
    if DEVICES:
        cat_params_mc["devices"] = DEVICES

    class_weights = {0: best_weights[0], 1: best_weights[1], 2: best_weights[2]}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        sample_weights = np.array([class_weights[y] for y in y_tr])

        model = CatBoostClassifier(**cat_params_mc)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weights,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )

        val_mc_folds.append(model.predict_proba(X_val_scaled))

    val_mc_probs = np.mean(val_mc_folds, axis=0)

    # Ensemble
    val_ens_probs = 0.5 * val_ovo_probs + 0.5 * val_mc_probs
    val_preds = np.argmax(val_ens_probs, axis=1)

    # Override black images
    for i, is_b in enumerate(black_mask):
        if is_b:
            val_preds[i] = 2

    pred_classes = [INV_CLASS_MAP[p] for p in val_preds]
    dist = {c: pred_classes.count(c) for c in CLASS_MAP}
    print(f"Val prediction distribution: {dist}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    np.save(os.path.join(CFG["output_dir"], "val_probs_final.npy"), val_ens_probs)
    np.save(os.path.join(CFG["output_dir"], "oof_ovo.npy"), ovo_probs)
    np.save(os.path.join(CFG["output_dir"], "oof_mc.npy"), best_oof)

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
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Health vs Rust Accuracy: {hr_acc:.4f}")
    print(f"Health vs Other Accuracy: {ho_acc:.4f}")
    print(f"Rust vs Other Accuracy: {ro_acc:.4f}")
    print(f"OvO Voting Accuracy: {ovo_acc:.4f}")
    print(f"Multi-class Accuracy: {best_acc:.4f}")
    print(f"Ensemble Accuracy: {ens_acc:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
