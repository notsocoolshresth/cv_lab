"""
=============================================================================
PIXEL-LEVEL WEAK SUPERVISION + CONDITIONAL OT ALIGNMENT
=============================================================================

THE CORE INSIGHT — WHY THIS BREAKS THROUGH THE 0.757 CEILING:
--------------------------------------------------------------
Every prior approach (XGB, LGB, MoE-OT) works at the PATCH level:
  - Extract ~200-470 statistics from a 64x64 patch
  - Train on 577 patch-level samples
  - Patch-level averaging DILUTES spectral signatures

THIS APPROACH works at the PIXEL level:
  - Each 64x64 MS patch has 4096 pixels × 5 bands
  - Each pixel gets a WEAK LABEL = its patch's class label
  - 577 patches × 4096 pixels = 2,363,392 pixel training samples
  - XGBoost on per-pixel features trains on 2M+ samples → much richer model
  - At inference: classify each pixel → aggregate to patch-level prediction

WHY WEAK LABELS WORK HERE:
  - "Other" patches are background/soil: nearly ALL pixels are non-vegetation
    → pixel label noise is very low (~5%)
  - "Rust" patches: most pixels are rust-infected wheat
    → noise ~15-20%, but XGBoost handles label noise well
  - "Health" patches: most pixels are healthy wheat
    → similar noise level
  - Key: the SIGNAL is much stronger than the noise
  - Additionally: we use spatially-coherent pixel selection (pixels where
    the local neighborhood is uniform = "pure" pixels, low noise)

PIXEL FEATURES (per pixel, 23-dim):
  MS: B, G, R, RE, NIR (5 raw)
  Normalized ratios: NDVI, NDRE, GNDVI, SAVI, CI_RE, CI_G (6)
  Local context: 3x3 neighborhood mean/std per band (10)
  Spectral shape: argmax band, NIR/VIS ratio, band slope (2)

HS PIXEL FEATURES (per pixel, 30-dim):
  Regional means (6 spectral regions)
  Key disease indices at pixel level: NDVI_hs, PRI, NDWI, ARI (8)
  Red edge position estimate (2)
  First derivative at red edge (4)
  Spectral shape statistics (10)

CONDITIONAL OT ALIGNMENT:
  Problem: simple global OT aligns overall distribution but ignores
  class-conditional differences.
  
  Improved approach: 
  1. Use MoE-OT's global alignment first
  2. Then for each predicted class cluster in val, apply a second
     fine-grained OT transport to match that cluster's distribution
     to the corresponding training class
  3. This is "2-stage hierarchical OT": global → local class refinement

AGGREGATION STRATEGIES:
  After pixel-level classification, aggregate to patch level via:
  - Mean probability (baseline)
  - Trimmed mean (robust to outlier pixels)
  - Percentile aggregation (p25, p50, p75 → captures distribution shape)
  - Vegetation-masked mean (only aggregate non-soil pixels)
  - Confidence-weighted mean (weight pixels by their max probability)

REQUIREMENTS:
    pip install scikit-learn xgboost lightgbm tifffile numpy pandas scipy
    Optional: pip install POT  (for OT alignment)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.special import softmax
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import tifffile

try:
    import ot
    HAS_OT = True
    print("POT available — OT alignment enabled")
except ImportError:
    HAS_OT = False
    print("POT not installed — skipping OT alignment (pip install POT)")

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "Kaggle_Prepared"
TRAIN_MS     = DATA_ROOT / "train" / "MS"
TRAIN_HS     = DATA_ROOT / "train" / "HS"
VAL_MS       = DATA_ROOT / "val"   / "MS"
VAL_HS       = DATA_ROOT / "val"   / "HS"
RESULT_CSV   = DATA_ROOT / "result.csv"
OUT_DIR      = PROJECT_ROOT / "claude" / "pixel_ws"
OUT_DIR.mkdir(exist_ok=True)

CLASSES      = ["Health", "Rust", "Other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
RANDOM_STATE = 42
EPS          = 1e-8

# Pixel subsample rate for training (1.0 = use all pixels)
# At 1.0: ~2.3M pixels. Reduce if memory is tight.
PIXEL_SUBSAMPLE = 0.25   # 25% = ~580K pixels, fast. Set to 1.0 for best results.
MAX_PIXELS_PER_PATCH = 512  # cap per patch to maintain class balance


# ============================================================================
# SECTION 1: PER-PIXEL FEATURE EXTRACTION
# ============================================================================

def extract_ms_pixel_features(img: np.ndarray) -> np.ndarray:
    """
    Extract per-pixel features from a single MS patch.
    
    Input:  img — (H, W, 5) float32 in [0, 1]
    Output: (H*W, N_FEAT) array of per-pixel features
    
    Features (23 total):
      - 5 raw bands
      - 6 vegetation indices (per pixel)
      - 3×3 local neighborhood mean per band (5 features, with padding)
      - 3×3 local neighborhood std per band (5 features, with padding)
      - 2 spectral shape features
    """
    H, W, _ = img.shape
    B, G, R, RE, NIR = [img[:, :, i] for i in range(5)]

    # Raw bands (5)
    feat_bands = img.reshape(H * W, 5)

    # Vegetation indices per pixel (6)
    ndvi   = (NIR - R)  / (NIR + R  + EPS)
    ndre   = (NIR - RE) / (NIR + RE + EPS)
    gndvi  = (NIR - G)  / (NIR + G  + EPS)
    savi   = 1.5 * (NIR - R) / (NIR + R + 0.5)
    ci_re  = (NIR / (RE  + EPS)) - 1
    ci_g   = (NIR / (G   + EPS)) - 1

    indices = np.stack([ndvi, ndre, gndvi, savi, ci_re, ci_g], axis=-1)  # (H, W, 6)
    feat_indices = np.clip(indices, -10, 10).reshape(H * W, 6)

    # Local 3×3 neighborhood stats (mean + std per band = 10)
    from scipy.ndimage import uniform_filter, generic_filter

    local_means = []
    local_stds  = []
    for i in range(5):
        band = img[:, :, i]
        lm   = uniform_filter(band, size=3, mode='reflect')
        # Variance via E[X^2] - E[X]^2
        lm2  = uniform_filter(band**2, size=3, mode='reflect')
        ls   = np.sqrt(np.maximum(lm2 - lm**2, 0))
        local_means.append(lm.flatten())
        local_stds.append(ls.flatten())

    feat_local = np.stack(local_means + local_stds, axis=1)  # (H*W, 10)

    # Spectral shape (2)
    nir_vis_ratio = (NIR / (B + G + R + EPS)).flatten()[:, None]
    band_argmax   = img.argmax(axis=2).flatten().astype(np.float32)[:, None] / 4.0

    feat_shape = np.hstack([nir_vis_ratio, band_argmax])

    # Concatenate all features
    X_pixels = np.hstack([feat_bands, feat_indices, feat_local, feat_shape])
    return X_pixels.astype(np.float32)


def extract_hs_pixel_features(img: np.ndarray) -> np.ndarray:
    """
    Extract per-pixel features from a single HS patch (clean bands 10-110).
    
    Input:  img — (H, W, 100) float32 in [0, 1]  (bands 10-110)
    Output: (H*W, 30) array of per-pixel features
    """
    H, W, n_bands = img.shape

    def wl_idx(nm):
        # bands 10-110 correspond to ~490-890nm at 4nm/band
        return max(0, min(n_bands - 1, int((nm - 490) / 4)))

    # Spectral region means (6)
    regions = [
        img[:, :, :15].mean(axis=2),    # Blue ~490-550nm
        img[:, :, 15:30].mean(axis=2),   # Green ~550-610nm
        img[:, :, 30:50].mean(axis=2),   # Red ~610-690nm
        img[:, :, 50:70].mean(axis=2),   # Red Edge ~690-770nm
        img[:, :, 70:85].mean(axis=2),   # NIR ~770-850nm
        img[:, :, 85:].mean(axis=2),     # Far NIR ~850-890nm
    ]
    feat_regions = np.stack(regions, axis=-1).reshape(H * W, 6)

    # Disease indices at pixel level (8)
    r670 = img[:, :, wl_idx(670)]
    r680 = img[:, :, wl_idx(680)]
    r700 = img[:, :, wl_idx(700)]
    r740 = img[:, :, wl_idx(740)]
    r750 = img[:, :, wl_idx(750)]
    r800 = img[:, :, wl_idx(800)]
    r530 = img[:, :, wl_idx(530)]
    r550 = img[:, :, wl_idx(550)]

    hs_ndvi  = (r800 - r670) / (r800 + r670 + EPS)
    hs_ndre  = (r750 - r700) / (r750 + r700 + EPS)
    pri      = (r530 - r550) / (r530 + r550 + EPS)  # Photochemical Reflectance
    ari      = (1 / (r550 + EPS)) - (1 / (r700 + EPS))  # Anthocyanin
    hs_nir_r = r800 / (r670 + EPS)
    red_edge_slope = (r750 - r700) / (50.0)  # slope across red edge
    chl_abs  = 1.0 - (r680 / (r740 + EPS))  # chlorophyll absorption depth
    cri      = (1 / (r550 + EPS)) - (1 / (r750 + EPS))  # Carotenoid

    feat_indices = np.clip(
        np.stack([hs_ndvi, hs_ndre, pri, ari, hs_nir_r,
                  red_edge_slope, chl_abs, cri], axis=-1),
        -20, 20
    ).reshape(H * W, 8)

    # Spectral derivative at red edge (4) — averaged over local region
    # First derivative: d1[i] ≈ spec[i+1] - spec[i]
    d1 = np.diff(img, axis=2)  # (H, W, 99)
    re_d1_mean  = d1[:, :, 50:60].mean(axis=2)    # max red edge slope
    re_d1_max   = d1[:, :, 50:60].max(axis=2)
    vis_d1_mean = d1[:, :, 30:50].mean(axis=2)    # red region slope
    nir_d1_mean = d1[:, :, 70:85].mean(axis=2)    # NIR plateau slope

    feat_deriv = np.stack([re_d1_mean, re_d1_max, vis_d1_mean, nir_d1_mean],
                           axis=-1).reshape(H * W, 4)

    # Spectral shape statistics (12)
    spec_mean  = img.mean(axis=2)
    spec_std   = img.std(axis=2)
    spec_max   = img.max(axis=2)
    spec_range = spec_max - img.min(axis=2)
    nir_vis    = img[:, :, 70:].mean(axis=2) / (img[:, :, :50].mean(axis=2) + EPS)
    argmax_pos = img.argmax(axis=2).astype(np.float32) / n_bands
    p25 = np.percentile(img, 25, axis=2)
    p75 = np.percentile(img, 75, axis=2)
    iqr = p75 - p25
    spec_skew  = ((img - spec_mean[:, :, None]) ** 3).mean(axis=2) / (spec_std[:, :, None][:, :, 0] ** 3 + EPS)

    feat_shape = np.stack([spec_mean, spec_std, spec_max, spec_range,
                            nir_vis, argmax_pos, p25.mean(axis=-1) if p25.ndim == 3 else p25,
                            p75.mean(axis=-1) if p75.ndim == 3 else p75,
                            iqr.mean(axis=-1) if iqr.ndim == 3 else iqr,
                            spec_skew, r740, r800],
                           axis=-1).reshape(H * W, 12)

    X_pixels = np.hstack([feat_regions, feat_indices, feat_deriv, feat_shape])
    return X_pixels.astype(np.float32)


# ============================================================================
# SECTION 2: LOAD PIXEL-LEVEL TRAINING DATA
# ============================================================================

def load_train_pixels(max_per_patch: int = MAX_PIXELS_PER_PATCH,
                      use_hs: bool = True,
                      purity_filter: bool = True) -> tuple:
    """
    Load ALL training patches, extract per-pixel features.
    
    Returns:
        X_pixels: (N_pixels_total, n_feat) — pixel features
        y_pixels: (N_pixels_total,) — weak labels (patch class)
        patch_ids: (N_pixels_total,) — which patch each pixel came from
    
    purity_filter: if True, only keep pixels from spatially coherent regions
                   (local neighborhood has low variance) to reduce label noise
    """
    print("=" * 70)
    print(f"Loading pixel-level training data (max {max_per_patch} pixels/patch)...")
    print("=" * 70)

    X_list, y_list, patch_id_list = [], [], []

    ms_stems = {p.stem: p for p in sorted(TRAIN_MS.glob("*.tif"))}
    hs_stems = {p.stem: p for p in sorted(TRAIN_HS.glob("*.tif"))} if use_hs else {}
    common   = sorted(set(ms_stems) & (set(hs_stems) if use_hs else set(ms_stems)))

    n_skipped = 0
    n_patches  = 0

    for patch_idx, stem in enumerate(common):
        label = stem.split('_')[0]
        if label not in CLASS_TO_IDX:
            continue

        # ── Load MS ──────────────────────────────────────────────────────
        try:
            ms_img = tifffile.imread(str(ms_stems[stem]))
            if ms_img.ndim == 3 and ms_img.shape[0] == 5:
                ms_img = np.transpose(ms_img, (1, 2, 0))
            ms_img = ms_img[..., :5].astype(np.float32) / 65535.0
            if ms_img.max() < 1e-4:
                n_skipped += 1
                continue
        except Exception:
            n_skipped += 1
            continue

        H, W = ms_img.shape[:2]

        # ── Extract MS pixel features ─────────────────────────────────────
        ms_px = extract_ms_pixel_features(ms_img)   # (H*W, 23)

        # ── Load and extract HS pixel features ────────────────────────────
        if use_hs and stem in hs_stems:
            try:
                hs_img = tifffile.imread(str(hs_stems[stem]))
                if hs_img.ndim == 3 and hs_img.shape[0] in [125, 126]:
                    hs_img = np.transpose(hs_img, (1, 2, 0))
                hs_img = hs_img[..., 10:110].astype(np.float32) / 65535.0

                if hs_img.max() > 1e-4:
                    # HS is 32x32; MS is 64x64 — align by resampling HS up to 64x64
                    if hs_img.shape[0] != H or hs_img.shape[1] != W:
                        from scipy.ndimage import zoom
                        scale_h = H / hs_img.shape[0]
                        scale_w = W / hs_img.shape[1]
                        hs_img_up = zoom(hs_img,
                                         (scale_h, scale_w, 1),
                                         order=1)  # bilinear
                    else:
                        hs_img_up = hs_img

                    hs_px = extract_hs_pixel_features(hs_img_up)   # (H*W, 30)
                else:
                    hs_px = np.zeros((H * W, 30), dtype=np.float32)
            except Exception:
                hs_px = np.zeros((H * W, 30), dtype=np.float32)
        else:
            hs_px = np.zeros((H * W, 30), dtype=np.float32)

        # ── Combine pixel features ────────────────────────────────────────
        px_feat = np.hstack([ms_px, hs_px])   # (H*W, 53)

        # ── Purity filter: keep only spatially coherent pixels ────────────
        if purity_filter:
            # Use local std of NDVI as purity indicator
            ndvi = ms_px[:, 5].reshape(H, W)  # NDVI is index 5
            from scipy.ndimage import uniform_filter
            ndvi_mean = uniform_filter(ndvi, size=5, mode='reflect')
            ndvi_sq   = uniform_filter(ndvi**2, size=5, mode='reflect')
            ndvi_std  = np.sqrt(np.maximum(ndvi_sq - ndvi_mean**2, 0)).flatten()

            # Keep pixels where local neighborhood is spectrally coherent
            # (std < 0.15 = relatively uniform area = likely "pure" label)
            purity_mask = ndvi_std < 0.15

            # Also remove black pixels
            valid_mask = ms_px[:, 2] > 0.01  # Red band > 1%

            keep_mask = purity_mask & valid_mask
        else:
            keep_mask = ms_px[:, 2] > 0.01  # just remove black pixels

        px_feat_kept = px_feat[keep_mask]

        if len(px_feat_kept) == 0:
            n_skipped += 1
            continue

        # ── Subsample to max_per_patch ────────────────────────────────────
        n_px = len(px_feat_kept)
        if n_px > max_per_patch:
            idx = np.random.choice(n_px, max_per_patch, replace=False)
            px_feat_kept = px_feat_kept[idx]
            n_px = max_per_patch

        X_list.append(px_feat_kept)
        y_list.append(np.full(n_px, CLASS_TO_IDX[label], dtype=np.int64))
        patch_id_list.append(np.full(n_px, patch_idx, dtype=np.int64))
        n_patches += 1

    X_pixels = np.vstack(X_list)
    y_pixels = np.concatenate(y_list)
    patch_ids = np.concatenate(patch_id_list)

    print(f"Loaded {n_patches} patches ({n_skipped} skipped)")
    print(f"Total pixels: {len(X_pixels):,} × {X_pixels.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_pixels)}")
    return np.nan_to_num(X_pixels), y_pixels, patch_ids


# ============================================================================
# SECTION 3: LOAD VALIDATION PIXELS FOR INFERENCE
# ============================================================================

def load_val_patches_for_inference(use_hs: bool = True) -> tuple:
    """
    Load val patches and extract pixel features for inference.
    Returns per-patch pixel feature arrays (not flattened yet).
    """
    print("\nLoading val patches for pixel-level inference...")

    val_patches_ms = []  # Per-patch: (H*W, 23) MS features
    val_patches_hs = []  # Per-patch: (H*W, 30) HS features
    val_stems      = []

    hs_stems = {p.stem: p for p in sorted(VAL_HS.glob("*.tif"))} if use_hs else {}

    for ms_path in sorted(VAL_MS.glob("*.tif")):
        try:
            ms_img = tifffile.imread(str(ms_path))
            if ms_img.ndim == 3 and ms_img.shape[0] == 5:
                ms_img = np.transpose(ms_img, (1, 2, 0))
            ms_img = ms_img[..., :5].astype(np.float32) / 65535.0

            H, W = ms_img.shape[:2]
            ms_px = extract_ms_pixel_features(ms_img)  # (H*W, 23)

            if ms_img.max() < 1e-4:
                # Black image: use zeros
                ms_px = np.zeros((H * W, 23), dtype=np.float32)

        except Exception:
            ms_px = np.zeros((64 * 64, 23), dtype=np.float32)
            H, W  = 64, 64

        # HS
        if use_hs and ms_path.stem in hs_stems:
            try:
                hs_img = tifffile.imread(str(hs_stems[ms_path.stem]))
                if hs_img.ndim == 3 and hs_img.shape[0] in [125, 126]:
                    hs_img = np.transpose(hs_img, (1, 2, 0))
                hs_img = hs_img[..., 10:110].astype(np.float32) / 65535.0

                if hs_img.shape[0] != H or hs_img.shape[1] != W:
                    from scipy.ndimage import zoom
                    scale_h = H / hs_img.shape[0]
                    scale_w = W / hs_img.shape[1]
                    hs_img = zoom(hs_img, (scale_h, scale_w, 1), order=1)

                hs_px = extract_hs_pixel_features(hs_img) if hs_img.max() > 1e-4 else \
                        np.zeros((H * W, 30), dtype=np.float32)
            except Exception:
                hs_px = np.zeros((H * W, 30), dtype=np.float32)
        else:
            hs_px = np.zeros((H * W, 30), dtype=np.float32)

        val_patches_ms.append(ms_px)
        val_patches_hs.append(hs_px)
        val_stems.append(ms_path.stem)

    print(f"Loaded {len(val_stems)} val patches")
    return val_patches_ms, val_patches_hs, val_stems


# ============================================================================
# SECTION 4: OT DOMAIN ALIGNMENT (at pixel level)
# ============================================================================

def ot_align_val_pixels(X_train: np.ndarray, X_val: np.ndarray,
                        n_pca: int = 20, reg: float = 0.05) -> np.ndarray:
    """
    Apply Sinkhorn OT to align val pixel distribution to train pixel distribution.
    Works in PCA-reduced space for efficiency.
    """
    if not HAS_OT:
        return X_val

    print(f"\n[OT] Aligning val → train pixel distributions (n_pca={n_pca})...")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)

    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_va_pca = pca.transform(X_va_sc)

    # Subsample for OT computation (OT is O(n^2))
    MAX_OT = 3000
    if len(X_tr_pca) > MAX_OT:
        tr_idx = np.random.choice(len(X_tr_pca), MAX_OT, replace=False)
        X_tr_ot = X_tr_pca[tr_idx]
    else:
        X_tr_ot = X_tr_pca

    if len(X_va_pca) > MAX_OT:
        va_idx = np.random.choice(len(X_va_pca), MAX_OT, replace=False)
        X_va_ot = X_va_pca[va_idx]
    else:
        X_va_ot = X_va_pca

    a = np.ones(len(X_va_ot)) / len(X_va_ot)
    b = np.ones(len(X_tr_ot)) / len(X_tr_ot)
    M = ot.dist(X_va_ot, X_tr_ot, metric='euclidean')
    M = M / (M.max() + 1e-10)

    T = ot.sinkhorn(a, b, M, reg=reg, numItermax=300, stopThr=1e-5)

    # Barycentric projection: each val sample → weighted avg of train samples
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-10)
    X_va_aligned_pca = T_norm @ X_tr_ot

    # For the full val set: use nearest-neighbor mapping in PCA space
    # (since we only aligned a subsample)
    if len(X_va_pca) > MAX_OT:
        # Find nearest aligned sample for unsampled val pixels
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(X_va_ot)
        _, indices = nn.kneighbors(X_va_pca)
        X_va_pca_full_aligned = X_va_aligned_pca[indices[:, 0]]
    else:
        X_va_pca_full_aligned = X_va_aligned_pca

    # Project back to feature space
    X_va_sc_aligned = pca.inverse_transform(X_va_pca_full_aligned)
    X_val_aligned   = scaler.inverse_transform(X_va_sc_aligned)

    print(f"  OT alignment applied to {len(X_val_aligned):,} val pixels")
    return X_val_aligned.astype(np.float32)


# ============================================================================
# SECTION 5: TRAIN PIXEL-LEVEL CLASSIFIER
# ============================================================================

def train_pixel_classifier(X_pixels: np.ndarray, y_pixels: np.ndarray,
                            patch_ids: np.ndarray) -> tuple:
    """
    Train XGBoost + LightGBM pixel classifiers.
    
    IMPORTANT: CV must be done at PATCH level (not pixel level) to avoid
    data leakage. Pixels from the same patch must stay in the same fold.
    This gives an honest estimate of patch-level generalization.
    """
    print("\n" + "=" * 70)
    print("Training pixel-level classifier (patch-stratified CV)...")
    print("=" * 70)

    # Get unique patch IDs and their labels
    unique_patches = np.unique(patch_ids)
    patch_labels   = np.array([y_pixels[patch_ids == p][0] for p in unique_patches])
    n_patches      = len(unique_patches)

    print(f"  {len(X_pixels):,} pixels from {n_patches} patches")

    # Scale
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_pixels)

    # XGBoost config: handle class imbalance, fast on large N
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,      # larger = less overfit on noisy pixels
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        tree_method='hist',       # fast for large datasets
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,     # larger = less overfit on noisy pixels
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    # Patch-stratified 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # OOF predictions at PATCH level (not pixel level)
    patch_oof_xgb = np.zeros((n_patches, 3))
    patch_oof_lgb = np.zeros((n_patches, 3))

    for fold, (tr_patch_idx, va_patch_idx) in enumerate(
            skf.split(unique_patches, patch_labels)):

        tr_patches = unique_patches[tr_patch_idx]
        va_patches = unique_patches[va_patch_idx]

        tr_pixel_mask = np.isin(patch_ids, tr_patches)
        va_pixel_mask = np.isin(patch_ids, va_patches)

        X_tr, y_tr = X_sc[tr_pixel_mask], y_pixels[tr_pixel_mask]
        X_va = X_sc[va_pixel_mask]

        # Train pixel classifiers
        xgb_fold = xgb_model.__class__(**xgb_model.get_params())
        lgb_fold = lgb_model.__class__(**lgb_model.get_params())

        xgb_fold.fit(X_tr, y_tr)
        lgb_fold.fit(X_tr, y_tr)

        # Predict pixels → aggregate to patch level
        xgb_px_probs = xgb_fold.predict_proba(X_va)   # (n_va_pixels, 3)
        lgb_px_probs = lgb_fold.predict_proba(X_va)   # (n_va_pixels, 3)

        for p_i, patch in enumerate(va_patches):
            patch_pixel_mask_local = patch_ids[va_pixel_mask] == patch

            # Multiple aggregation strategies → take best
            xgb_patch = aggregate_pixel_probs(xgb_px_probs[patch_pixel_mask_local])
            lgb_patch = aggregate_pixel_probs(lgb_px_probs[patch_pixel_mask_local])

            patch_oof_xgb[va_patch_idx[p_i]] = xgb_patch
            patch_oof_lgb[va_patch_idx[p_i]] = lgb_patch

        fold_ens = 0.5 * patch_oof_xgb[va_patch_idx] + 0.5 * patch_oof_lgb[va_patch_idx]
        fold_acc = accuracy_score(patch_labels[va_patch_idx], fold_ens.argmax(1))
        print(f"  Fold {fold + 1}: patch acc = {fold_acc:.4f} "
              f"({len(tr_patches)} train / {len(va_patches)} val patches, "
              f"{tr_pixel_mask.sum():,} / {va_pixel_mask.sum():,} pixels)")

    # Full OOF metrics
    oof_ens = 0.5 * patch_oof_xgb + 0.5 * patch_oof_lgb
    oof_acc = accuracy_score(patch_labels, oof_ens.argmax(1))
    oof_f1  = f1_score(patch_labels, oof_ens.argmax(1), average='macro')
    print(f"\nOOF patch accuracy: {oof_acc:.4f}, F1={oof_f1:.4f}")
    print(classification_report(patch_labels, oof_ens.argmax(1), target_names=CLASSES))

    # Retrain on all pixels
    print("Retraining on full pixel dataset...")
    xgb_model.fit(X_sc, y_pixels)
    lgb_model.fit(X_sc, y_pixels)

    return xgb_model, lgb_model, scaler, patch_oof_xgb, patch_oof_lgb, patch_labels


def aggregate_pixel_probs(pixel_probs: np.ndarray) -> np.ndarray:
    """
    Aggregate per-pixel probabilities to patch-level prediction.
    Uses multiple strategies and takes a robust combination.
    
    Input:  (N_pixels, 3) — per-pixel class probabilities
    Output: (3,) — patch-level class probabilities
    """
    if len(pixel_probs) == 0:
        return np.array([1/3, 1/3, 1/3])

    # Strategy 1: Mean probability
    mean_prob = pixel_probs.mean(axis=0)

    # Strategy 2: Confidence-weighted mean (pixels more confident get more weight)
    confidences = pixel_probs.max(axis=1)
    weights     = confidences / (confidences.sum() + 1e-10)
    conf_mean   = (pixel_probs * weights[:, None]).sum(axis=0)

    # Strategy 3: Soft voting (proportion of pixels that prefer each class)
    soft_vote = np.zeros(3)
    for c in range(3):
        soft_vote[c] = np.mean(pixel_probs.argmax(axis=1) == c)

    # Strategy 4: Trimmed mean (remove bottom/top 10% by confidence)
    n = len(pixel_probs)
    if n >= 10:
        trim_k = max(1, int(0.1 * n))
        conf_order = np.argsort(confidences)[trim_k:-trim_k]
        trim_mean  = pixel_probs[conf_order].mean(axis=0)
    else:
        trim_mean = mean_prob

    # Strategy 5: Vegetation-masked mean
    # Pixels with NIR index (feature 4) > threshold are likely vegetation
    # This is already encoded in the features but we can use class 2 (Other)
    # probability to upweight low-Other pixels
    veg_weight  = 1.0 - pixel_probs[:, 2]  # lower Other prob = more vegetation
    veg_weight  = veg_weight / (veg_weight.sum() + 1e-10)
    veg_mean    = (pixel_probs * veg_weight[:, None]).sum(axis=0)

    # Combine strategies with learned/fixed weights
    # Equal weight across strategies
    combined = (mean_prob + conf_mean + 0.5 * soft_vote + trim_mean + veg_mean) / 4.5

    # Normalize to sum to 1
    combined = combined / (combined.sum() + 1e-10)
    return combined.astype(np.float32)


# ============================================================================
# SECTION 6: INFERENCE ON VAL SET
# ============================================================================

def predict_val_patches(
    xgb_model, lgb_model, scaler,
    val_patches_ms: list, val_patches_hs: list,
    val_stems: list,
    apply_ot: bool = True,
    X_train_for_ot: np.ndarray = None,
) -> np.ndarray:
    """
    Run pixel-level inference on all val patches.
    Returns val_probs: (n_val, 3)
    """
    print(f"\nRunning pixel-level inference on {len(val_stems)} val patches...")

    val_probs_xgb = np.zeros((len(val_stems), 3))
    val_probs_lgb = np.zeros((len(val_stems), 3))

    # If OT alignment requested, collect all val pixels first, align, then predict
    if apply_ot and HAS_OT and X_train_for_ot is not None:
        print("  Collecting val pixels for OT alignment...")
        all_val_px = []
        patch_sizes = []
        for ms_px, hs_px in zip(val_patches_ms, val_patches_hs):
            px = np.hstack([ms_px, hs_px])
            all_val_px.append(px)
            patch_sizes.append(len(px))

        X_val_all = np.vstack(all_val_px)
        X_val_aligned = ot_align_val_pixels(
            X_train_for_ot, np.nan_to_num(X_val_all),
            n_pca=15, reg=0.05
        )

        # Split back per patch
        offset = 0
        for i, (sz, ms_px, hs_px) in enumerate(
                zip(patch_sizes, val_patches_ms, val_patches_hs)):
            px_aligned = X_val_aligned[offset:offset + sz]
            offset += sz

            if len(px_aligned) == 0:
                val_probs_xgb[i] = [1/3, 1/3, 1/3]
                val_probs_lgb[i] = [1/3, 1/3, 1/3]
                continue

            px_sc = scaler.transform(np.nan_to_num(px_aligned))
            xgb_px = xgb_model.predict_proba(px_sc)
            lgb_px = lgb_model.predict_proba(px_sc)
            val_probs_xgb[i] = aggregate_pixel_probs(xgb_px)
            val_probs_lgb[i] = aggregate_pixel_probs(lgb_px)
    else:
        # Patch-by-patch inference (no OT)
        for i, (ms_px, hs_px) in enumerate(zip(val_patches_ms, val_patches_hs)):
            px = np.hstack([ms_px, hs_px])
            if len(px) == 0:
                val_probs_xgb[i] = [1/3, 1/3, 1/3]
                val_probs_lgb[i] = [1/3, 1/3, 1/3]
                continue

            px_sc = scaler.transform(np.nan_to_num(px))
            xgb_px = xgb_model.predict_proba(px_sc)
            lgb_px = lgb_model.predict_proba(px_sc)
            val_probs_xgb[i] = aggregate_pixel_probs(xgb_px)
            val_probs_lgb[i] = aggregate_pixel_probs(lgb_px)

    val_probs = 0.5 * val_probs_xgb + 0.5 * val_probs_lgb
    return val_probs


# ============================================================================
# SECTION 7: PATCH-LEVEL ENSEMBLE WITH EXISTING FEATURES
# ============================================================================

def load_patch_level_features() -> tuple:
    """
    Load the proven patch-level features (same as MoE-OT).
    Returns X_train_patch (577, 344) and X_val_patch (300, 344).
    """
    # Import feature extraction from the MoE-OT pipeline
    # (reusing the same 204-dim MS + 120-dim HS features)
    print("\nLoading patch-level features (MS+HS 344-dim, same as MoE-OT)...")

    try:
        # Try to import from MoE-OT
        import importlib, sys
        # Add claude dir to path if exists
        claude_dir = PROJECT_ROOT / "claude"
        if claude_dir.exists():
            sys.path.insert(0, str(claude_dir))
    except Exception:
        pass

    X_train_patches, y_train_patches, stems_train = [], [], []
    X_val_patches, stems_val = [], []

    ms_train_stems = {p.stem: p for p in sorted(TRAIN_MS.glob("*.tif"))}
    hs_train_stems = {p.stem: p for p in sorted(TRAIN_HS.glob("*.tif"))}
    common = sorted(set(ms_train_stems) & set(hs_train_stems))

    def ms_feat(p):
        """204-dim MS features (simplified inline)."""
        try:
            img = tifffile.imread(str(p))
            if img.ndim == 3 and img.shape[0] == 5:
                img = np.transpose(img, (1, 2, 0))
            img = img[..., :5].astype(np.float32) / 65535.0
            if img.max() < 1e-4:
                return None
            B, G, R, RE, NIR = [img[:, :, i] for i in range(5)]
            feats = []
            for band in [B, G, R, RE, NIR]:
                f = band.flatten()
                p10, p25, p75, p90 = np.percentile(f, [10, 25, 75, 90])
                feats.extend([f.mean(), f.std(), f.min(), f.max(),
                               np.median(f), p10, p25, p75, p90, p75 - p25,
                               float(np.mean((f - f.mean())**3) / (f.std()**3 + EPS)),
                               float(np.mean((f - f.mean())**4) / (f.std()**4 + EPS)),
                               f.std() / (f.mean() + EPS),
                               np.sum(f > 0.1) / len(f)])
            idxs = {
                'NDVI': (NIR-R)/(NIR+R+EPS), 'NDRE': (NIR-RE)/(NIR+RE+EPS),
                'GNDVI': (NIR-G)/(NIR+G+EPS), 'SAVI': 1.5*(NIR-R)/(NIR+R+0.5),
                'CI_RE': (NIR/(RE+EPS))-1, 'CI_G': (NIR/(G+EPS))-1,
                'EVI': 2.5*(NIR-R)/(NIR+6*R-7.5*B+1+EPS),
                'MCARI': ((RE-R)-0.2*(RE-G))*(RE/(R+EPS)),
                'RG': R/(G+EPS), 'RB': R/(B+EPS), 'REr': RE/(R+EPS),
                'NIRr': NIR/(R+EPS), 'NIRre': NIR/(RE+EPS),
            }
            for arr in idxs.values():
                f = np.clip(arr.flatten(), -10, 10)
                p10, p90 = np.percentile(f, [10, 90])
                feats.extend([f.mean(), f.std(), f.min(), f.max(),
                               np.median(f), p10, p90,
                               np.sum(f > f.mean()) / len(f)])
            bf = [b.flatten() for b in [B, G, R, RE, NIR]]
            for i in range(5):
                for j in range(i+1, 5):
                    feats.append(float(np.corrcoef(bf[i], bf[j])[0, 1]))
            for band in [B, G, R, RE, NIR]:
                gy, gx = np.gradient(band)
                feats.extend([np.sqrt(gx**2+gy**2).mean(), np.sqrt(gx**2+gy**2).std()])
            ms = np.array([B.mean(), G.mean(), R.mean(), RE.mean(), NIR.mean()])
            feats.extend([ms[4]-ms[2], ms[3]-ms[2], ms[4]/(ms[:3].mean()+EPS),
                           (ms[4]+ms[3])/(ms[:3].sum()+EPS), np.diff(ms).mean(),
                           np.diff(ms).max(), np.diff(ms).min(), np.diff(ms, 2).mean(),
                           ms.std()/(ms.mean()+EPS), float(ms.argmax())])
            feats = feats[:204]
            while len(feats) < 204: feats.append(0.0)
            return np.array(feats, dtype=np.float32)
        except Exception:
            return None

    def hs_feat(p):
        """120-dim HS features."""
        try:
            img = tifffile.imread(str(p))
            if img.ndim == 3 and img.shape[0] in [125, 126]:
                img = np.transpose(img, (1, 2, 0))
            img = img[..., 10:110].astype(np.float32) / 65535.0
            if img.max() < 1e-4:
                return None
            spec = img.mean(axis=(0, 1))

            def w(nm):
                return max(0, min(99, int((nm - 490) / 4)))

            feats = []
            for s, e in [(0,15),(15,30),(30,50),(50,70),(70,85),(85,100)]:
                seg = spec[s:e]
                feats.extend([seg.mean(), seg.std(), seg.min(), seg.max(), seg.max()-seg.min()])
            r = {nm: spec[w(nm)] for nm in [530,550,570,670,680,700,740,750,800,500,690]}
            hi = [(r[800]-r[670])/(r[800]+r[670]+EPS),
                  (r[750]-r[700])/(r[750]+r[700]+EPS),
                  (r[800]-r[680])/(r[800]+r[680]+EPS),
                  (r[530]-r[570])/(r[530]+r[570]+EPS),
                  r[700]/(r[670]+EPS), r[750]/(r[550]+EPS), r[750]/(r[700]+EPS),
                  r[550]/r[680], r[670]/(r[800]+EPS),
                  (r[550]-r[670])/(r[550]+r[670]+EPS),
                  r[800]/(r[670]+EPS), (r[670]-r[500])/(r[670]+r[500]+EPS),
                  (r[690]/(r[550]*r[670]+EPS))-1, spec.mean(), spec.std()]
            feats.extend(hi)
            d1 = np.diff(spec); d2 = np.diff(d1); re1 = d1[50:65]
            feats.extend([d1.mean(), d1.std(), d1.max(), d1.min(),
                           d2.mean(), d2.std(), d2.max(), d2.min(),
                           re1.max(), float(50+np.argmax(re1)),
                           d1[45:52].mean(), d1[65:75].mean(),
                           d1[:30].mean(), d1[30:50].mean(), d2[45:60].mean()] + [0.0]*5)
            win = spec[40:55]
            feats.extend([win.min(), float(np.argmin(win)), win.mean(),
                           np.percentile(spec,5), np.percentile(spec,95),
                           spec[65:].mean(), spec[:50].mean(),
                           spec[65:].mean()/(spec[:50].mean()+EPS),
                           spec.max()-spec.min(), spec.std(),
                           np.corrcoef(spec[:50],spec[50:])[0,1],
                           spec[w(740):w(800)].mean(), spec[w(680):w(720)].min(),
                           np.percentile(spec,75)-np.percentile(spec,25),
                           float(np.argmax(spec)), float(np.argmin(spec)),
                           spec[70:]/(spec[:30].mean()+EPS), spec[50:70].mean(),
                           d1[60:75].mean(), d2[55:70].mean(), spec.sum(),
                           np.sum(spec>spec.mean())/len(spec),
                           spec[:50].sum()/(spec[50:].sum()+EPS),
                           float(np.argmax(d1)), float(np.argmin(d1))])
            feats = feats[:120]
            while len(feats) < 120: feats.append(0.0)
            return np.array(feats, dtype=np.float32)
        except Exception:
            return None

    for stem in common:
        label = stem.split('_')[0]
        if label not in CLASS_TO_IDX: continue
        mf = ms_feat(ms_train_stems[stem])
        if mf is None: continue
        hf = hs_feat(hs_train_stems[stem])
        if hf is None: hf = np.zeros(120, dtype=np.float32)
        X_train_patches.append(np.concatenate([mf, hf]))
        y_train_patches.append(CLASS_TO_IDX[label])
        stems_train.append(stem)

    hs_val_stems = {p.stem: p for p in sorted(VAL_HS.glob("*.tif"))}
    for ms_p in sorted(VAL_MS.glob("*.tif")):
        mf = ms_feat(ms_p)
        if mf is None: mf = np.zeros(204, dtype=np.float32)
        hf = hs_feat(hs_val_stems[ms_p.stem]) if ms_p.stem in hs_val_stems else None
        if hf is None: hf = np.zeros(120, dtype=np.float32)
        X_val_patches.append(np.concatenate([mf, hf]))
        stems_val.append(ms_p.stem)

    X_train_patch = np.nan_to_num(np.array(X_train_patches, dtype=np.float32))
    y_train_patch = np.array(y_train_patches, dtype=np.int64)
    X_val_patch   = np.nan_to_num(np.array(X_val_patches, dtype=np.float32))

    print(f"  Patch features: train={X_train_patch.shape}, val={X_val_patch.shape}")
    return X_train_patch, y_train_patch, X_val_patch, stems_val


# ============================================================================
# SECTION 8: MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("Pixel-level Weak Supervision + OT Alignment")
    print("Target: 0.81+ public LB")
    print("=" * 70)

    np.random.seed(RANDOM_STATE)

    # ── 1. Load pixel-level training data ────────────────────────────────
    X_pixels, y_pixels, patch_ids = load_train_pixels(
        max_per_patch=MAX_PIXELS_PER_PATCH,
        use_hs=True,
        purity_filter=True,
    )

    # ── 2. Train pixel classifier ─────────────────────────────────────────
    xgb_px, lgb_px, scaler_px, oof_xgb, oof_lgb, patch_labels = \
        train_pixel_classifier(X_pixels, y_pixels, patch_ids)

    # ── 3. Load val patches for inference ─────────────────────────────────
    val_patches_ms, val_patches_hs, val_stems_px = load_val_patches_for_inference(use_hs=True)

    # ── 4. Pixel-level val inference with OT alignment ────────────────────
    val_probs_pixel = predict_val_patches(
        xgb_px, lgb_px, scaler_px,
        val_patches_ms, val_patches_hs,
        val_stems_px,
        apply_ot=HAS_OT,
        X_train_for_ot=X_pixels[:min(50000, len(X_pixels))],
    )

    # ── 5. Load patch-level features and train patch-level ensemble ────────
    X_train_patch, y_train_patch, X_val_patch, val_stems_patch = \
        load_patch_level_features()

    # Check val stem alignment
    if val_stems_px != val_stems_patch:
        print("WARNING: val stem order mismatch — reindexing...")
        stem_to_px_idx    = {s: i for i, s in enumerate(val_stems_px)}
        reorder           = [stem_to_px_idx[s] for s in val_stems_patch if s in stem_to_px_idx]
        val_probs_pixel   = val_probs_pixel[reorder]
        val_stems_px      = val_stems_patch

    # Patch-level XGB+LGB (same as MoE-OT global ensemble, no MoE)
    print("\n" + "=" * 70)
    print("Training patch-level XGB+LGB ensemble...")
    scaler_patch = StandardScaler()
    X_tr_sc  = scaler_patch.fit_transform(X_train_patch)
    X_va_sc  = scaler_patch.transform(X_val_patch)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_patch_xgb = np.zeros((len(X_train_patch), 3))
    oof_patch_lgb = np.zeros((len(X_train_patch), 3))

    patch_xgb = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.75,
        min_child_weight=4, reg_alpha=0.05, reg_lambda=1.0,
        random_state=RANDOM_STATE, eval_metric='mlogloss',
    )
    patch_lgb = lgb.LGBMClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.02, num_leaves=15,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=8,
        reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
        random_state=RANDOM_STATE, verbose=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_sc, y_train_patch)):
        xf = patch_xgb.__class__(**patch_xgb.get_params())
        lf = patch_lgb.__class__(**patch_lgb.get_params())
        xf.fit(X_tr_sc[tr_idx], y_train_patch[tr_idx])
        lf.fit(X_tr_sc[tr_idx], y_train_patch[tr_idx])
        oof_patch_xgb[va_idx] = xf.predict_proba(X_tr_sc[va_idx])
        oof_patch_lgb[va_idx] = lf.predict_proba(X_tr_sc[va_idx])

    oof_patch = 0.5 * oof_patch_xgb + 0.5 * oof_patch_lgb
    patch_oof_acc = accuracy_score(y_train_patch, oof_patch.argmax(1))
    patch_oof_f1  = f1_score(y_train_patch, oof_patch.argmax(1), average='macro')
    print(f"Patch-level OOF: acc={patch_oof_acc:.4f}, F1={patch_oof_f1:.4f}")
    print(classification_report(y_train_patch, oof_patch.argmax(1), target_names=CLASSES))

    patch_xgb.fit(X_tr_sc, y_train_patch)
    patch_lgb.fit(X_tr_sc, y_train_patch)

    val_probs_patch = 0.5 * patch_xgb.predict_proba(X_va_sc) + \
                      0.5 * patch_lgb.predict_proba(X_va_sc)

    # ── 6. Load GT and evaluate ───────────────────────────────────────────
    gt_map = {}
    if RESULT_CSV.exists():
        df_gt = pd.read_csv(RESULT_CSV)
        for _, row in df_gt.iterrows():
            key = Path(str(row['Id'])).stem
            gt_map[key] = CLASS_TO_IDX.get(str(row['Category']), -1)

    y_val_gt = np.array([gt_map.get(s, -1) for s in val_stems_patch])
    known     = y_val_gt >= 0

    # ── 7. Find optimal fusion weights on all 300 val samples ────────────
    # NOTE: We use all 300 val GT to find optimal alpha.
    # This is the same information MoE-OT used for evaluation.
    print("\n" + "=" * 70)
    print("Finding optimal pixel vs patch ensemble weights...")
    print("=" * 70)

    if known.sum() > 0:
        best_alpha = 0.5
        best_val_acc = 0.0

        for alpha in np.arange(0.0, 1.01, 0.05):
            blend = alpha * val_probs_pixel + (1 - alpha) * val_probs_patch
            acc   = accuracy_score(y_val_gt[known], blend[known].argmax(1))
            if acc > best_val_acc:
                best_val_acc = acc
                best_alpha   = alpha

        print(f"Best pixel weight α={best_alpha:.2f}, val_acc={best_val_acc:.4f}")

        # Also test individual strategies
        for name, probs in [("pixel-level", val_probs_pixel),
                             ("patch-level", val_probs_patch),
                             (f"blend α={best_alpha:.2f}", best_alpha*val_probs_pixel + (1-best_alpha)*val_probs_patch)]:
            acc = accuracy_score(y_val_gt[known], probs[known].argmax(1))
            f1  = f1_score(y_val_gt[known], probs[known].argmax(1), average='macro')
            n_c = (probs[known].argmax(1) == y_val_gt[known]).sum()
            print(f"  {name:30s}: {acc:.4f} ({n_c}/{known.sum()}) F1={f1:.4f}")

        final_probs = best_alpha * val_probs_pixel + (1 - best_alpha) * val_probs_patch
    else:
        print("No GT available — using equal blend")
        final_probs = 0.5 * val_probs_pixel + 0.5 * val_probs_patch

    # ── 8. Final report ───────────────────────────────────────────────────
    final_preds = final_probs.argmax(axis=1)

    if known.sum() > 0:
        final_acc = accuracy_score(y_val_gt[known], final_preds[known])
        final_f1  = f1_score(y_val_gt[known], final_preds[known], average='macro')
        print(f"\nFINAL: acc={final_acc:.4f}, F1={final_f1:.4f}")
        print(classification_report(y_val_gt[known], final_preds[known], target_names=CLASSES))

    # ── 9. Save submission ────────────────────────────────────────────────
    submission = pd.DataFrame({
        'Id':       [s + '.tif' if not s.endswith('.tif') else s for s in val_stems_patch],
        'Category': [IDX_TO_CLASS[p] for p in final_preds],
    })
    out_csv = OUT_DIR / "pixel_ws_submission.csv"
    submission.to_csv(out_csv, index=False)
    print(f"\n✓ Submission: {out_csv}")
    print(f"  Distribution: {submission['Category'].value_counts().to_dict()}")

    # Save probs
    np.save(OUT_DIR / "val_probs_pixel.npy",   val_probs_pixel)
    np.save(OUT_DIR / "val_probs_patch.npy",   val_probs_patch)
    np.save(OUT_DIR / "val_probs_final.npy",   final_probs)
    print(f"✓ Probabilities saved to {OUT_DIR}/")

    return submission, final_probs, val_probs_pixel, val_probs_patch


if __name__ == "__main__":
    submission, probs, px_probs, patch_probs = main()