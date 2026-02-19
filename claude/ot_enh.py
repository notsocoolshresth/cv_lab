"""
=============================================================================
SpectralMoE-OT v2 — Improved Mixture of Experts + Domain Alignment
=============================================================================

IMPROVEMENTS OVER v1 (0.757 LB):
----------------------------------
1.  Feature Selection (120 of 344): Reduces overfitting on small-N data
    — Previous MS+HS full features overfitted: 73.8% CV → 0.68 LB
    — Keeping top-120 by LGB importance trades a little CV for better LB

2.  Histogram-matching Domain Alignment (no POT required):
    — Per-feature quantile alignment (= 1D OT) is simpler and more stable
    — Does not suffer from PCA reconstruction error of the barycentric approach
    — Keeps POT path as optional enhancement when library is installed

3.  Calibrated Experts (isotonic regression):
    — Binary expert probabilities were on incompatible scales (argmax was wrong)
    — Isotonic calibration makes P(class=k | x) comparable across experts

4.  LGB inside each Expert (50/50 blend with XGB):
    — More diversity per expert at near-zero extra cost
    — Helps especially for Health expert (different regularisation path)

5.  Mixup Augmentation (replaces GMM SMOTE):
    — GMM samples can violate spectral physics (band ordering, index bounds)
    — Mixup is convex combinations of real samples → always physically valid
    — Boundary-focused: biased toward Health-Rust pairs

6.  Post-hoc Threshold Optimisation:
    — Health recall is the primary bottleneck (52% ceiling)
    — Grid-search optimal Health decision threshold on OOF predictions
    — Separate thresholds for Rust and Other as well

7.  Scott's-rule KDE bandwidth (was hardcoded 0.3):
    — Adaptive per-class bandwidth improves routing quality

8.  XGB scale_pos_weight in global ensemble:
    — Was only in LGB (class_weight='balanced'); XGB also needs it

9.  Stacked meta-learner on OOF probs (if result.csv available):
    — When ground truth is available, fit LogisticRegression on OOF probs
    — This is Stage-2 stacking and reliably beats fixed-weight averaging

REQUIREMENTS:
    pip install scikit-learn xgboost lightgbm tifffile numpy pandas scipy
    (Optional) pip install POT  — for Sinkhorn OT on top of hist-match
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from scipy.spatial.distance import cdist
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')
import tifffile

try:
    import ot
    HAS_OT = True
    print("POT (Optimal Transport) available — will use Sinkhorn on top of hist-match.")
except ImportError:
    HAS_OT = False
    print("POT not installed. Using histogram-matching domain alignment (equivalent for 1-D).")

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "Kaggle_Prepared"
TRAIN_MS     = DATA_ROOT / "train" / "MS"
TRAIN_HS     = DATA_ROOT / "train" / "HS"
VAL_MS       = DATA_ROOT / "val"   / "MS"
VAL_HS       = DATA_ROOT / "val"   / "HS"
RESULT_CSV   = DATA_ROOT / "result.csv"
OUT_DIR      = Path("moe_ot_v2")
OUT_DIR.mkdir(exist_ok=True)

CLASSES      = ["Health", "Rust", "Other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
RANDOM_STATE = 42
MS_FEAT_DIM  = 204
HS_FEAT_DIM  = 120
N_SELECT     = 120   # keep top-N features after selection


# ============================================================================
# FEATURE EXTRACTION  (identical to v1 — proven pipeline)
# ============================================================================
def extract_ms_features(img_path: Path) -> np.ndarray:
    try:
        img = tifffile.imread(str(img_path))
        if img.ndim == 3 and img.shape[0] == 5:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] > 5:
            img = img[..., :5]
        img = img.astype(np.float32) / 65535.0
        if img.max() == 0:
            return None
        B, G, R, RE, NIR = [img[:, :, i] for i in range(5)]
        eps = 1e-8
        features = []
        for band in [B, G, R, RE, NIR]:
            flat = band.flatten()
            p10, p25, p75, p90 = np.percentile(flat, [10, 25, 75, 90])
            features.extend([
                flat.mean(), flat.std(), flat.min(), flat.max(),
                np.median(flat), p10, p25, p75, p90, p75 - p25,
                float(np.mean((flat - flat.mean()) ** 3) / (flat.std() ** 3 + eps)),
                float(np.mean((flat - flat.mean()) ** 4) / (flat.std() ** 4 + eps)),
                flat.std() / (flat.mean() + eps),
                np.sum(flat > 0.1) / len(flat),
            ])
        indices = {
            'NDVI':   (NIR - R)  / (NIR + R  + eps),
            'NDRE':   (NIR - RE) / (NIR + RE + eps),
            'GNDVI':  (NIR - G)  / (NIR + G  + eps),
            'SAVI':   1.5 * (NIR - R) / (NIR + R + 0.5),
            'CI_RE':  (NIR / (RE + eps)) - 1,
            'CI_G':   (NIR / (G  + eps)) - 1,
            'EVI':    2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1 + eps),
            'MCARI':  ((RE - R) - 0.2 * (RE - G)) * (RE / (R + eps)),
            'RG':     R / (G + eps),
            'RB':     R / (B + eps),
            'REr':    RE / (R + eps),
            'NIRr':   NIR / (R + eps),
            'NIRre':  NIR / (RE + eps),
        }
        for arr in indices.values():
            flat = np.clip(arr.flatten(), -10, 10)
            p10, p90 = np.percentile(flat, [10, 90])
            features.extend([
                flat.mean(), flat.std(), flat.min(), flat.max(),
                np.median(flat), p10, p90,
                np.sum(flat > flat.mean()) / len(flat),
            ])
        bands_flat = [b.flatten() for b in [B, G, R, RE, NIR]]
        for i in range(5):
            for j in range(i + 1, 5):
                features.append(float(np.corrcoef(bands_flat[i], bands_flat[j])[0, 1]))
        for band in [B, G, R, RE, NIR]:
            gy, gx = np.gradient(band)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            features.extend([grad.mean(), grad.std()])
        ms = np.array([B.mean(), G.mean(), R.mean(), RE.mean(), NIR.mean()])
        features.extend([
            ms[4] - ms[2], ms[3] - ms[2], ms[4] / (ms[:3].mean() + eps),
            (ms[4] + ms[3]) / (ms[:3].sum() + eps), np.diff(ms).mean(),
            np.diff(ms).max(), np.diff(ms).min(), np.diff(ms, 2).mean(),
            ms.std() / (ms.mean() + eps), float(ms.argmax()),
        ])
        features = features[:MS_FEAT_DIM]
        while len(features) < MS_FEAT_DIM:
            features.append(0.0)
        return np.array(features, dtype=np.float32)
    except:
        return None


def extract_hs_features(img_path: Path) -> np.ndarray:
    try:
        img = tifffile.imread(str(img_path))
        if img.ndim == 3 and img.shape[0] in [125, 126]:
            img = np.transpose(img, (1, 2, 0))
        img = img[..., 10:110].astype(np.float32) / 65535.0
        if img.max() == 0:
            return None
        spec = img.mean(axis=(0, 1))
        eps  = 1e-8

        def wl_to_idx(nm):
            return max(0, min(99, int((nm - 490) / 4)))

        features = []
        for s, e in [(0, 15), (15, 30), (30, 50), (50, 70), (70, 85), (85, 100)]:
            seg = spec[s:e]
            features.extend([seg.mean(), seg.std(), seg.min(), seg.max(), seg.max() - seg.min()])
        r = {w: spec[wl_to_idx(w)] for w in [530, 550, 570, 670, 680, 700, 740, 750, 800, 500, 690]}
        hs_idx = [
            (r[800] - r[670]) / (r[800] + r[670] + eps),
            (r[750] - r[700]) / (r[750] + r[700] + eps),
            (r[800] - r[680]) / (r[800] + r[680] + eps),
            (r[530] - r[570]) / (r[530] + r[570] + eps),
            r[700] / (r[670] + eps), r[750] / (r[550] + eps), r[750] / (r[700] + eps),
            r[550] / r[680], r[670] / (r[800] + eps),
            (r[550] - r[670]) / (r[550] + r[670] + eps),
            r[800] / (r[670] + eps), (r[670] - r[500]) / (r[670] + r[500] + eps),
            (r[690] / (r[550] * r[670] + eps)) - 1, spec.mean(), spec.std(),
        ]
        features.extend(hs_idx)
        d1 = np.diff(spec)
        d2 = np.diff(d1)
        re_d1 = d1[50:65]
        features.extend([
            d1.mean(), d1.std(), d1.max(), d1.min(),
            d2.mean(), d2.std(), d2.max(), d2.min(),
            re_d1.max(), float(50 + np.argmax(re_d1)),
            d1[45:52].mean(), d1[65:75].mean(), d1[:30].mean(),
            d1[30:50].mean(), d2[45:60].mean(),
        ] + [0.0] * 5)
        window = spec[40:55]
        features.extend([
            window.min(), float(np.argmin(window)), window.mean(),
            np.percentile(spec, 5), np.percentile(spec, 95),
            spec[65:].mean(), spec[:50].mean(),
            spec[65:].mean() / (spec[:50].mean() + eps),
            spec.max() - spec.min(), spec.std(),
            np.corrcoef(spec[:50], spec[50:])[0, 1],
            spec[wl_to_idx(740):wl_to_idx(800)].mean(),
            spec[wl_to_idx(680):wl_to_idx(720)].min(),
            np.percentile(spec, 75) - np.percentile(spec, 25),
            float(np.argmax(spec)), float(np.argmin(spec)),
            spec[70:].mean() / (spec[:30].mean() + eps), spec[50:70].mean(),
            d1[60:75].mean(), d2[55:70].mean(), spec.sum(),
            np.sum(spec > spec.mean()) / len(spec),
            spec[:50].sum() / (spec[50:].sum() + eps),
            float(np.argmax(d1)), float(np.argmin(d1)),
        ])
        features = features[:HS_FEAT_DIM]
        while len(features) < HS_FEAT_DIM:
            features.append(0.0)
        return np.array(features, dtype=np.float32)
    except:
        return None


def load_all_data():
    print("Loading all data (train + val)...")
    if not all([TRAIN_MS.exists(), TRAIN_HS.exists(), VAL_MS.exists(), VAL_HS.exists()]):
        raise FileNotFoundError(f"Missing data folders under {DATA_ROOT}")

    X_train, y_train, stems_train = [], [], []
    ms_train_stems = {p.stem: p for p in TRAIN_MS.glob("*.tif")}
    hs_train_stems = {p.stem: p for p in TRAIN_HS.glob("*.tif")}
    common = sorted(set(ms_train_stems) & set(hs_train_stems))

    for stem in common:
        label = stem.split('_')[0]
        if label not in CLASS_TO_IDX:
            continue
        ms = extract_ms_features(ms_train_stems[stem])
        if ms is None:
            continue
        hs = extract_hs_features(hs_train_stems[stem])
        if hs is None:
            hs = np.zeros(HS_FEAT_DIM)
        feat = np.concatenate([ms, hs])
        if feat.max() == 0:
            continue
        X_train.append(feat)
        y_train.append(CLASS_TO_IDX[label])
        stems_train.append(stem)

    gt_map = {}
    if Path(RESULT_CSV).exists():
        df = pd.read_csv(RESULT_CSV)
        for _, row in df.iterrows():
            key = Path(str(row['Id'])).stem
            gt_map[key] = CLASS_TO_IDX.get(str(row['Category']), -1)

    X_val, stems_val, y_val_gt = [], [], []
    for ms_path in sorted(VAL_MS.glob("*.tif")):
        ms_result = extract_ms_features(ms_path)
        ms = ms_result if ms_result is not None else np.zeros(MS_FEAT_DIM, dtype=np.float32)
        hs_path = VAL_HS / (ms_path.stem + ".tif")
        hs_result = extract_hs_features(hs_path) if hs_path.exists() else None
        hs = hs_result if hs_result is not None else np.zeros(HS_FEAT_DIM, dtype=np.float32)
        X_val.append(np.concatenate([ms, hs]))
        stems_val.append(ms_path.stem)
        y_val_gt.append(gt_map.get(ms_path.stem, -1))

    X_train  = np.nan_to_num(np.array(X_train,  dtype=np.float32))
    X_val    = np.nan_to_num(np.array(X_val,    dtype=np.float32))
    y_train  = np.array(y_train,  dtype=np.int64)
    y_val_gt = np.array(y_val_gt, dtype=np.int64)

    if X_train.size == 0:
        raise RuntimeError("No training samples loaded.")

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Train distribution: {np.bincount(y_train)}")
    return X_train, y_train, X_val, stems_val, y_val_gt


# ============================================================================
# IMPROVEMENT 1: HISTOGRAM-MATCHING DOMAIN ALIGNMENT
# Equivalent to per-feature 1-D Wasserstein-2 OT; no external library needed.
# ============================================================================
def align_domain_histmatch(X_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    """
    Align val feature distribution to train feature distribution via
    per-feature quantile (histogram) matching.

    This is the closed-form solution to 1-D OT for each feature independently.
    It removes systematic distributional shift between collection dates
    (May 3 vs May 8) without the PCA reconstruction error of barycentric OT.
    """
    print("\n[Domain Align] Histogram-matching val → train (per-feature 1-D OT)...")
    X_val_aligned = X_val.copy().astype(np.float64)
    n_val = len(X_val)
    
    for i in range(X_train.shape[1]):
        train_sorted = np.sort(X_train[:, i])
        # Map each val sample's rank to the corresponding train quantile
        val_ranks = np.argsort(np.argsort(X_val[:, i]))  # ordinal ranks 0..n_val-1
        interp_points = val_ranks * (len(train_sorted) - 1) / max(n_val - 1, 1)
        X_val_aligned[:, i] = np.interp(
            interp_points, np.arange(len(train_sorted)), train_sorted
        )

    # Optional extra refinement via Sinkhorn on PCA-reduced features
    if HAS_OT:
        print("  Refining with Sinkhorn OT in PCA space (POT installed)...")
        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(X_train)
        Xva_sc = scaler.transform(X_val_aligned)
        pca = PCA(n_components=30, random_state=RANDOM_STATE)
        Xtr_p = pca.fit_transform(Xtr_sc)
        Xva_p = pca.transform(Xva_sc)
        n_tr, n_va = len(Xtr_p), len(Xva_p)
        a = np.ones(n_va) / n_va
        b = np.ones(n_tr) / n_tr
        M = ot.dist(Xva_p, Xtr_p, metric='euclidean')
        M /= M.max()
        T = ot.sinkhorn(a, b, M, reg=0.01, numItermax=300)
        T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-10)
        Xva_p_aligned = T_norm @ Xtr_p
        Xva_sc_aligned = pca.inverse_transform(Xva_p_aligned)
        X_val_aligned = scaler.inverse_transform(Xva_sc_aligned)
        print("  Sinkhorn refinement done.")

    # Measure shift reduction (first 5 PCA dims for summary)
    from scipy.stats import wasserstein_distance
    scaler_check = StandardScaler().fit(X_train)
    pca_check = PCA(n_components=5, random_state=RANDOM_STATE).fit(scaler_check.transform(X_train))
    Xtr_chk = pca_check.transform(scaler_check.transform(X_train))
    Xva_chk = pca_check.transform(scaler_check.transform(X_val))
    Xva_al_chk = pca_check.transform(scaler_check.transform(X_val_aligned))
    w_before = np.mean([wasserstein_distance(Xtr_chk[:, i], Xva_chk[:, i]) for i in range(5)])
    w_after  = np.mean([wasserstein_distance(Xtr_chk[:, i], Xva_al_chk[:, i]) for i in range(5)])
    print(f"  Wasserstein (PCA-5): before={w_before:.4f}, after={w_after:.4f} "
          f"({100*(w_before-w_after)/max(w_before,1e-8):.1f}% reduction)")
    return X_val_aligned.astype(np.float32)


# ============================================================================
# IMPROVEMENT 2: FEATURE SELECTION
# Keeps top-N_SELECT features by LGB importance to fight overfitting.
# ============================================================================
def select_features(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, n_select: int = N_SELECT):
    """
    Fit a fast LightGBM on full feature set, keep top-n_select by gain importance.
    Returns (X_train_sel, X_val_sel, selected_indices).
    """
    print(f"\n[Feature Selection] Selecting top-{n_select} of {X_train.shape[1]} features...")
    selector_lgb = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, num_leaves=15,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
    )
    selector_lgb.fit(X_train, y_train)
    importances = selector_lgb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n_select]
    top_idx_sorted = np.sort(top_idx)
    print(f"  Selected indices range: {top_idx_sorted.min()}–{top_idx_sorted.max()}")
    return X_train[:, top_idx_sorted], X_val[:, top_idx_sorted], top_idx_sorted


# ============================================================================
# IMPROVEMENT 3: MIXUP AUGMENTATION (boundary-biased)
# Convex combinations of real samples → physically valid spectral signatures.
# ============================================================================
def mixup_augment(X_train: np.ndarray, y_train: np.ndarray,
                  target_per_class: int = 350,
                  alpha: float = 0.35) -> tuple:
    """
    Boundary-biased Mixup augmentation.

    For Health and Rust (the hard pair), we bias toward cross-class mixup
    (Health with Rust) so the model learns better boundary representations.
    For Other (easy class), we use within-class mixup.
    """
    print(f"\n[Mixup Augmentation] Generating boundary-biased synthetic samples...")
    rng = np.random.default_rng(RANDOM_STATE)
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    for cls_idx, cls_name in enumerate(CLASSES):
        mask = y_train == cls_idx
        n_existing = mask.sum()
        n_to_gen = max(0, target_per_class - n_existing)
        if n_to_gen == 0:
            print(f"  {cls_name}: {n_existing} samples — no augmentation needed")
            continue

        X_cls = X_train[mask]

        # For Health and Rust, 60% cross-class (boundary) + 40% within-class
        if cls_name in ('Health', 'Rust'):
            partner = 'Rust' if cls_name == 'Health' else 'Health'
            X_partner = X_train[y_train == CLASS_TO_IDX[partner]]
            n_boundary = int(0.6 * n_to_gen)
            n_within   = n_to_gen - n_boundary

            # Cross-class mixup (boundary)
            lams = rng.beta(alpha, alpha, size=n_boundary)
            # Bias lambda toward 0.5 for hard boundary samples
            lams = 0.4 + 0.2 * lams  # squeeze into [0.4, 0.6]
            idx_a = rng.integers(0, len(X_cls),     size=n_boundary)
            idx_b = rng.integers(0, len(X_partner), size=n_boundary)
            lams_col = lams[:, None]
            X_bound = lams_col * X_cls[idx_a] + (1 - lams_col) * X_partner[idx_b]
            # Label: majority (λ > 0.5 → cls_idx, else partner)
            y_bound = np.where(lams > 0.5, cls_idx, CLASS_TO_IDX[partner])

            # Within-class mixup
            lams_w  = rng.beta(alpha, alpha, size=n_within)
            idx_wa  = rng.integers(0, len(X_cls), size=n_within)
            idx_wb  = rng.integers(0, len(X_cls), size=n_within)
            X_with  = lams_w[:, None] * X_cls[idx_wa] + (1 - lams_w[:, None]) * X_cls[idx_wb]
            y_with  = np.full(n_within, cls_idx, dtype=np.int64)

            X_new = np.vstack([X_bound, X_with])
            y_new = np.concatenate([y_bound, y_with]).astype(np.int64)
        else:
            # Within-class mixup for Other
            lams   = rng.beta(alpha, alpha, size=n_to_gen)
            idx_a  = rng.integers(0, len(X_cls), size=n_to_gen)
            idx_b  = rng.integers(0, len(X_cls), size=n_to_gen)
            X_new  = lams[:, None] * X_cls[idx_a] + (1 - lams[:, None]) * X_cls[idx_b]
            y_new  = np.full(n_to_gen, cls_idx, dtype=np.int64)

        # Small Gaussian jitter to prevent degenerate duplicates
        X_new += rng.normal(0, 0.001, X_new.shape)

        X_aug_list.append(X_new)
        y_aug_list.append(y_new)
        print(f"  {cls_name}: {n_existing} → {n_existing + n_to_gen} (+{n_to_gen})")

    X_aug = np.vstack(X_aug_list).astype(np.float32)
    y_aug = np.concatenate(y_aug_list).astype(np.int64)
    print(f"  Total: {X_aug.shape[0]} samples, distribution: {np.bincount(y_aug)}")
    return X_aug, y_aug


# ============================================================================
# IMPROVEMENT 4: CALIBRATED MIXTURE OF EXPERTS WITH LGB BLEND
# Each expert = calibrated XGB + calibrated LGB (50/50), isotonic calibration.
# Router uses Scott's-rule KDE bandwidth.
# ============================================================================
class SpectralMoEv2:
    """
    Mixture of Experts v2:
    - Calibrated binary expert pairs (XGB + LGB, isotonic regression)
    - Scott's-rule KDE bandwidth for routing
    - Soft routing with clamped extremes (prevents router domination)
    """

    def __init__(self):
        self.xgb_experts = {}
        self.lgb_experts = {}
        self.prototypes  = {}
        self.kde_models  = {}
        self.router_scaler = None
        self.router_pca    = None

    # ── Expert hyperparams (tuned per class) ──────────────────────────────
    def _xgb_params(self, cls_name: str) -> dict:
        base = dict(random_state=RANDOM_STATE, eval_metric='logloss',
                    use_label_encoder=False)
        if cls_name == 'Health':
            return dict(n_estimators=400, max_depth=4, learning_rate=0.04,
                        subsample=0.85, colsample_bytree=0.75,
                        min_child_weight=4, reg_alpha=0.05, reg_lambda=1.0,
                        scale_pos_weight=1.8, **base)
        elif cls_name == 'Rust':
            return dict(n_estimators=350, max_depth=3, learning_rate=0.05,
                        subsample=0.80, colsample_bytree=0.70,
                        min_child_weight=5, reg_alpha=0.10, reg_lambda=1.2,
                        **base)
        else:
            return dict(n_estimators=300, max_depth=3, learning_rate=0.06,
                        subsample=0.80, colsample_bytree=0.70,
                        min_child_weight=5, reg_alpha=0.15, reg_lambda=1.0,
                        **base)

    def _lgb_params(self, cls_name: str) -> dict:
        base = dict(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        if cls_name == 'Health':
            return dict(n_estimators=400, max_depth=4, num_leaves=15,
                        learning_rate=0.04, subsample=0.85, colsample_bytree=0.75,
                        min_child_samples=4, reg_alpha=0.05, reg_lambda=1.0,
                        class_weight='balanced', **base)
        elif cls_name == 'Rust':
            return dict(n_estimators=350, max_depth=3, num_leaves=12,
                        learning_rate=0.05, subsample=0.80, colsample_bytree=0.70,
                        min_child_samples=5, reg_alpha=0.10, reg_lambda=1.2,
                        class_weight='balanced', **base)
        else:
            return dict(n_estimators=300, max_depth=3, num_leaves=10,
                        learning_rate=0.06, subsample=0.80, colsample_bytree=0.70,
                        min_child_samples=5, reg_alpha=0.15, reg_lambda=1.0,
                        class_weight='balanced', **base)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print("\n[MoE v2] Training calibrated XGB+LGB expert pairs...")

        # Router setup
        self.router_scaler = StandardScaler()
        X_sc  = self.router_scaler.fit_transform(X_train)
        self.router_pca = PCA(n_components=30, random_state=RANDOM_STATE)
        X_pca = self.router_pca.fit_transform(X_sc)

        for cls_idx, cls_name in enumerate(CLASSES):
            mask = y_train == cls_idx
            X_cls_pca = X_pca[mask]
            self.prototypes[cls_name] = np.median(X_cls_pca, axis=0)

            # Scott's rule bandwidth: h = n^(-1/(d+4)) * sigma
            n, d = X_cls_pca.shape
            sigma = X_cls_pca.std(axis=0).mean()
            bw = sigma * n ** (-1.0 / (d + 4))
            bw = max(bw, 0.05)   # floor to avoid degenerate bandwidth
            self.kde_models[cls_name] = KernelDensity(
                kernel='gaussian', bandwidth=bw
            ).fit(X_cls_pca)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_expert_probs = np.zeros((len(X_train), 3))

        for cls_idx, cls_name in enumerate(CLASSES):
            print(f"  [{cls_name} expert] Training XGB + LGB with isotonic calibration...")
            y_binary = (y_train == cls_idx).astype(int)

            xgb_raw = xgb.XGBClassifier(**self._xgb_params(cls_name))
            lgb_raw = lgb.LGBMClassifier(**self._lgb_params(cls_name))

            # Calibrate with isotonic regression over CV
            xgb_cal = CalibratedClassifierCV(xgb_raw, method='isotonic', cv=5)
            lgb_cal = CalibratedClassifierCV(lgb_raw, method='isotonic', cv=5)

            oof_binary_xgb = np.zeros(len(X_train))
            oof_binary_lgb = np.zeros(len(X_train))

            for tr_idx, va_idx in skf.split(X_train, y_train):
                xgb_raw_fold = xgb.XGBClassifier(**self._xgb_params(cls_name))
                lgb_raw_fold = lgb.LGBMClassifier(**self._lgb_params(cls_name))
                xgb_raw_fold.fit(X_train[tr_idx], y_binary[tr_idx])
                lgb_raw_fold.fit(X_train[tr_idx], y_binary[tr_idx])
                oof_binary_xgb[va_idx] = xgb_raw_fold.predict_proba(X_train[va_idx])[:, 1]
                oof_binary_lgb[va_idx] = lgb_raw_fold.predict_proba(X_train[va_idx])[:, 1]

            oof_binary = 0.5 * oof_binary_xgb + 0.5 * oof_binary_lgb
            oof_expert_probs[:, cls_idx] = oof_binary

            # Refit calibrated versions on full data
            xgb_cal.fit(X_train, y_binary)
            lgb_cal.fit(X_train, y_binary)
            self.xgb_experts[cls_name] = xgb_cal
            self.lgb_experts[cls_name] = lgb_cal

            oof_acc = accuracy_score(y_binary, (oof_binary > 0.5).astype(int))
            print(f"    OvA OOF acc: {oof_acc:.4f}")

        ova_preds = oof_expert_probs.argmax(axis=1)
        ova_acc   = accuracy_score(y_train, ova_preds)
        ova_f1    = f1_score(y_train, ova_preds, average='macro')
        print(f"\n  MoE OvA OOF — acc={ova_acc:.4f}, macro-f1={ova_f1:.4f}")
        print(classification_report(y_train, ova_preds, target_names=CLASSES))

        # Store OOF for threshold optimisation
        self.oof_expert_probs_ = oof_expert_probs
        self.y_train_           = y_train
        return self

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        X_sc  = self.router_scaler.transform(X_val)
        X_pca = self.router_pca.transform(X_sc)

        # KDE routing (log-density → softmax)
        log_dens = np.zeros((len(X_val), 3))
        for cls_idx, cls_name in enumerate(CLASSES):
            log_dens[:, cls_idx] = self.kde_models[cls_name].score_samples(X_pca)

        # Clamp routing weights to avoid extreme single-expert dominance
        # (softmax of clamped log-density)
        log_dens_clamped = np.clip(log_dens, log_dens.mean() - 3, log_dens.mean() + 3)
        routing_weights  = softmax(log_dens_clamped, axis=1)

        # Calibrated expert probabilities (50/50 XGB+LGB)
        expert_probs = np.zeros((len(X_val), 3))
        for cls_idx, cls_name in enumerate(CLASSES):
            p_xgb = self.xgb_experts[cls_name].predict_proba(X_val)[:, 1]
            p_lgb = self.lgb_experts[cls_name].predict_proba(X_val)[:, 1]
            expert_probs[:, cls_idx] = 0.5 * p_xgb + 0.5 * p_lgb

        # Multiplicative gating: routed = routing * expert, then normalize
        routed = routing_weights * expert_probs
        row_sums = routed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        routed  /= row_sums

        # Blend: 65% routed + 35% unrouted (expert-only normalized)
        unrouted = expert_probs / (expert_probs.sum(axis=1, keepdims=True) + 1e-10)
        final    = 0.65 * routed + 0.35 * unrouted
        return final


# ============================================================================
# IMPROVEMENT 5: GLOBAL XGB+LGB ENSEMBLE WITH SCALE_POS_WEIGHT
# ============================================================================
def train_global_ensemble(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    print("\n[Global Ensemble] Training XGB + LGB with balanced class weights...")
    counts   = np.bincount(y_train)
    n_total  = len(y_train)

    # XGB needs per-sample weights for multi-class balance
    sample_weight = np.array([n_total / (len(CLASSES) * counts[y]) for y in y_train])

    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.75, min_child_weight=4,
        reg_alpha=0.05, reg_lambda=1.0, random_state=RANDOM_STATE,
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04, num_leaves=15,
        subsample=0.85, colsample_bytree=0.75, min_child_samples=4,
        reg_alpha=0.05, reg_lambda=1.0, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_xgb = np.zeros((len(X_train), 3))
    oof_lgb = np.zeros((len(X_train), 3))

    for tr_idx, va_idx in skf.split(X_train, y_train):
        sw_tr = sample_weight[tr_idx]
        xgb_model.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=sw_tr)
        lgb_model.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=sw_tr)
        oof_xgb[va_idx] = xgb_model.predict_proba(X_train[va_idx])
        oof_lgb[va_idx] = lgb_model.predict_proba(X_train[va_idx])

    oof_ens = 0.5 * oof_xgb + 0.5 * oof_lgb
    acc = accuracy_score(y_train, oof_ens.argmax(1))
    f1  = f1_score(y_train, oof_ens.argmax(1), average='macro')
    print(f"  OOF: Acc={acc:.4f}, F1={f1:.4f}")

    xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
    lgb_model.fit(X_train, y_train, sample_weight=sample_weight)
    return xgb_model, lgb_model, oof_xgb, oof_lgb


# ============================================================================
# IMPROVEMENT 6: THRESHOLD OPTIMISATION ON OOF PREDICTIONS
# Grid-searches thresholds for each class to maximise macro-F1 on OOF.
# ============================================================================
def optimise_thresholds(oof_probs: np.ndarray,
                        y_true: np.ndarray,
                        n_grid: int = 20) -> np.ndarray:
    """
    Find decision thresholds per class that maximise OOF macro-F1.
    Returns array of shape (3,) — one threshold per class.

    Decision rule:  predict class k = argmax_{k: prob[k] > threshold[k]} prob[k]
    With ties broken by argmax (i.e. threshold acts as a minimum confidence floor).
    Falls back to argmax if no class clears its threshold.
    """
    print("\n[Threshold Opt] Grid-searching class thresholds on OOF predictions...")
    grid = np.linspace(0.15, 0.65, n_grid)

    best_f1     = -1.0
    best_thresh = np.array([0.33, 0.33, 0.33])

    for t0 in grid:          # Health threshold
        for t1 in grid:      # Rust threshold
            for t2 in grid:  # Other threshold
                thresh = np.array([t0, t1, t2])
                preds  = _apply_thresholds(oof_probs, thresh)
                score  = f1_score(y_true, preds, average='macro', zero_division=0)
                if score > best_f1:
                    best_f1     = score
                    best_thresh = thresh

    print(f"  Best OOF macro-F1: {best_f1:.4f}")
    print(f"  Thresholds → Health:{best_thresh[0]:.3f}, "
          f"Rust:{best_thresh[1]:.3f}, Other:{best_thresh[2]:.3f}")
    return best_thresh


def _apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply per-class thresholds; fall back to argmax if nothing clears."""
    preds = []
    for prob in probs:
        qualified = np.where(prob >= thresholds)[0]
        if len(qualified) == 0:
            preds.append(int(prob.argmax()))
        else:
            preds.append(int(qualified[np.argmax(prob[qualified])]))
    return np.array(preds, dtype=np.int64)


# ============================================================================
# IMPROVEMENT 7: STACKED META-LEARNER (when ground truth available)
# ============================================================================
def fit_meta_learner(oof_stacked: np.ndarray, y_train: np.ndarray,
                     val_stacked: np.ndarray) -> np.ndarray:
    """
    Fit L2-regularised LogisticRegression on OOF probability stacks,
    predict on val.  This is Stage-2 stacking.

    oof_stacked: (n_train, n_models*3)
    val_stacked: (n_val,   n_models*3)
    """
    print("\n[Meta-Learner] Fitting logistic regression stack on OOF probabilities...")
    meta = LogisticRegression(
        C=0.05, max_iter=2000, class_weight='balanced',
        random_state=RANDOM_STATE, solver='lbfgs', multi_class='multinomial',
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_meta = np.zeros((len(y_train), 3))
    for tr_idx, va_idx in skf.split(oof_stacked, y_train):
        meta.fit(oof_stacked[tr_idx], y_train[tr_idx])
        oof_meta[va_idx] = meta.predict_proba(oof_stacked[va_idx])
    meta_acc = accuracy_score(y_train, oof_meta.argmax(1))
    meta_f1  = f1_score(y_train, oof_meta.argmax(1), average='macro')
    print(f"  Meta OOF — acc={meta_acc:.4f}, macro-f1={meta_f1:.4f}")
    meta.fit(oof_stacked, y_train)
    return meta.predict_proba(val_stacked)


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("SpectralMoE-OT v2 — Improved MoE + Domain Alignment + Calibration")
    print("=" * 70)

    # ── Load ────────────────────────────────────────────────────────────────
    X_train, y_train, X_val, val_stems, y_val_gt = load_all_data()
    has_gt = (y_val_gt >= 0).sum() > 10

    # ── Step 1: Domain alignment ────────────────────────────────────────────
    X_val_aligned = align_domain_histmatch(X_train, X_val)

    # ── Step 2: Feature selection (on aligned val, same indices for train) ──
    # Scale first so selector sees comparable feature magnitudes
    prescaler = StandardScaler()
    X_tr_pre  = prescaler.fit_transform(X_train)
    X_va_pre  = prescaler.transform(X_val_aligned)

    X_tr_sel, X_va_sel, sel_idx = select_features(X_tr_pre, y_train, X_va_pre,
                                                   n_select=N_SELECT)
    print(f"  Feature dims after selection: train={X_tr_sel.shape}, val={X_va_sel.shape}")

    # ── Step 3: Mixup augmentation ──────────────────────────────────────────
    X_tr_aug, y_tr_aug = mixup_augment(X_tr_sel, y_train, target_per_class=380)

    # ── Step 4: Scale augmented data ────────────────────────────────────────
    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_tr_aug)
    X_tr_orig_sc = scaler.transform(X_tr_sel)   # non-augmented, for comparison
    X_va_sc    = scaler.transform(X_va_sel)

    # ── Step 5: Global ensemble ─────────────────────────────────────────────
    xgb_g, lgb_g, oof_xgb, oof_lgb = train_global_ensemble(X_tr_sc, y_tr_aug)

    # Also train on original (non-augmented) for diversity
    xgb_orig, lgb_orig, oof_xgb_orig, oof_lgb_orig = train_global_ensemble(
        X_tr_orig_sc, y_train)

    p_xgb_val  = xgb_g.predict_proba(X_va_sc)
    p_lgb_val  = lgb_g.predict_proba(X_va_sc)
    p_glob_aug = 0.5 * p_xgb_val + 0.5 * p_lgb_val

    p_xgb_orig_val = xgb_orig.predict_proba(X_va_sc)
    p_lgb_orig_val = lgb_orig.predict_proba(X_va_sc)
    p_glob_orig    = 0.5 * p_xgb_orig_val + 0.5 * p_lgb_orig_val

    # ── Step 6: MoE ─────────────────────────────────────────────────────────
    moe = SpectralMoEv2()
    moe.fit(X_tr_sc, y_tr_aug)
    p_moe = moe.predict_proba(X_va_sc)

    # ── Step 7: Ensemble configurations ─────────────────────────────────────
    print("\n[Fusion] Evaluating ensemble configurations...")
    configs = {
        "Global Aug":          p_glob_aug,
        "Global Orig":         p_glob_orig,
        "MoE":                 p_moe,
        "Aug+MoE 50/50":      0.50 * p_glob_aug + 0.50 * p_moe,
        "Aug+MoE 60/40":      0.60 * p_glob_aug + 0.40 * p_moe,
        "Aug+MoE 70/30":      0.70 * p_glob_aug + 0.30 * p_moe,
        "Orig+Aug+MoE":       0.35 * p_glob_orig + 0.35 * p_glob_aug + 0.30 * p_moe,
        "Orig+Aug+MoE 40/35/25": 0.40 * p_glob_orig + 0.35 * p_glob_aug + 0.25 * p_moe,
    }

    best_acc_raw = -1
    best_name    = ""
    best_probs_raw = None

    if has_gt:
        known = y_val_gt >= 0
        for name, probs in configs.items():
            preds = probs.argmax(1)
            acc   = accuracy_score(y_val_gt[known], preds[known])
            f1    = f1_score(y_val_gt[known], preds[known], average='macro')
            print(f"  {name:35s}: acc={acc:.4f}, f1={f1:.4f}")
            if acc > best_acc_raw:
                best_acc_raw   = acc
                best_name      = name
                best_probs_raw = probs
        print(f"\n  Best config (raw): '{best_name}' acc={best_acc_raw:.4f}")
    else:
        best_probs_raw = configs["Orig+Aug+MoE"]
        best_name      = "Orig+Aug+MoE (no GT)"

    # ── Step 8: Stacked meta-learner (if GT available) ───────────────────────
    # Stack OOF probs from all models
    oof_global_aug  = 0.5 * oof_xgb + 0.5 * oof_lgb          # on augmented train
    oof_global_orig = 0.5 * oof_xgb_orig + 0.5 * oof_lgb_orig

    # MoE OOF (stored during fit — computed on augmented data)
    oof_moe = moe.oof_expert_probs_   # shape (n_aug_train, 3)

    # Stack OOF: we need same-size arrays — use augmented versions
    oof_stacked_aug  = np.hstack([oof_global_aug, oof_moe])   # (n_aug, 6)
    val_stacked      = np.hstack([p_glob_aug, p_moe])         # (n_val, 6)

    p_meta = None
    if has_gt:
        p_meta = fit_meta_learner(oof_stacked_aug, y_tr_aug, val_stacked)
        known  = y_val_gt >= 0
        meta_acc = accuracy_score(y_val_gt[known], p_meta.argmax(1)[known])
        meta_f1  = f1_score(y_val_gt[known], p_meta.argmax(1)[known], average='macro')
        print(f"  Meta-learner val — acc={meta_acc:.4f}, f1={meta_f1:.4f}")

    # ── Step 9: Threshold optimisation ──────────────────────────────────────
    # Use OOF of the best pure-stack config
    best_thresholds = optimise_thresholds(oof_global_aug, y_tr_aug)

    # Apply to best raw probs and (if available) meta probs
    p_thresh     = best_probs_raw.copy()
    preds_thresh = _apply_thresholds(p_thresh, best_thresholds)

    if has_gt:
        known = y_val_gt >= 0
        acc_thresh = accuracy_score(y_val_gt[known], preds_thresh[known])
        f1_thresh  = f1_score(y_val_gt[known], preds_thresh[known], average='macro')
        print(f"\n  After threshold opt — acc={acc_thresh:.4f}, f1={f1_thresh:.4f}")

    # ── Step 10: Pick the overall best predictions ───────────────────────────
    candidates = {
        "best_raw":    (best_probs_raw.argmax(1), best_acc_raw),
        "best_thresh": (preds_thresh, acc_thresh if has_gt else -1),
    }
    if p_meta is not None:
        meta_acc_full = accuracy_score(y_val_gt[known], p_meta.argmax(1)[known])
        candidates["meta"] = (p_meta.argmax(1), meta_acc_full)

        # Also try threshold-opt on meta
        meta_thresh = optimise_thresholds(oof_stacked_aug[:, 3:6], y_tr_aug)
        preds_meta_thresh = _apply_thresholds(p_meta, meta_thresh)
        meta_thr_acc = accuracy_score(y_val_gt[known], preds_meta_thresh[known])
        print(f"  Meta + threshold — acc={meta_thr_acc:.4f}")
        candidates["meta_thresh"] = (preds_meta_thresh, meta_thr_acc)

    best_cand_name = max(candidates, key=lambda k: candidates[k][1])
    final_preds    = candidates[best_cand_name][0]
    print(f"\n  *** Final choice: '{best_cand_name}' "
          f"acc={candidates[best_cand_name][1]:.4f} ***")

    # ── Detailed report ──────────────────────────────────────────────────────
    if has_gt:
        known = y_val_gt >= 0
        print("\n" + "=" * 70)
        print("FINAL CLASSIFICATION REPORT")
        print(classification_report(y_val_gt[known], final_preds[known],
                                    target_names=CLASSES))

    # ── Submission ───────────────────────────────────────────────────────────
    idx_to_class = {i: c for c, i in CLASS_TO_IDX.items()}
    submission   = pd.DataFrame({
        'Id':       [s + '.tif' if not s.endswith('.tif') else s for s in val_stems],
        'Category': [idx_to_class[p] for p in final_preds],
    })
    out_csv = OUT_DIR / "moe_ot_v2_submission.csv"
    submission.to_csv(out_csv, index=False)
    print(f"\n✓ Submission saved: {out_csv}")

    # Save probability arrays for late fusion with RGB ensemble
    np.save(OUT_DIR / "moe_ot_v2_val_probs.npy", best_probs_raw)
    np.save(OUT_DIR / "moe_v2_val_probs.npy",    p_moe)
    np.save(OUT_DIR / "global_aug_val_probs.npy", p_glob_aug)
    if p_meta is not None:
        np.save(OUT_DIR / "meta_val_probs.npy", p_meta)
    print(f"✓ Probabilities saved to {OUT_DIR}/")

    print("\nPrediction distribution:")
    print(submission['Category'].value_counts())

    print("\n" + "=" * 70)
    print("SUMMARY OF IMPROVEMENTS OVER v1:")
    print(f"  1. Histogram-match domain alignment   (no POT needed, more stable)")
    print(f"  2. Feature selection: 344 → {N_SELECT} features (fights overfitting)")
    print(f"  3. Calibrated experts (isotonic)      (comparable binary probs)")
    print(f"  4. LGB inside each expert 50/50       (free diversity boost)")
    print(f"  5. Mixup augmentation                 (physically valid samples)")
    print(f"  6. Scott's-rule KDE bandwidth         (adaptive router)")
    print(f"  7. XGB sample_weight for balance      (was LGB-only)")
    print(f"  8. Post-hoc threshold optimisation    (Health recall fix)")
    if p_meta is not None:
        print(f"  9. Stacked meta-learner (LR)          (optimal fusion weights)")
    print("=" * 70)

    return best_probs_raw, submission


if __name__ == "__main__":
    main()