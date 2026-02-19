"""
MS + HS Classification via Spectral Feature Engineering + XGBoost
- Extract ~224 handcrafted features per image from MS (5 bands, 64x64)
- Extract ~250+ additional features from HS (125 bands, 32x32)
- Combine MS+HS features → XGBoost + LightGBM ensemble with 5-fold CV
- Outputs soft probabilities for late fusion with RGB ensemble
"""

import os
import csv
import json
import warnings
import numpy as np
import tifffile as tiff
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
from scipy import ndimage, stats as scipy_stats

warnings.filterwarnings("ignore")

CFG = {
    "ms_train_dir": "Kaggle_Prepared/train/MS",
    "ms_val_dir": "Kaggle_Prepared/val/MS",
    "hs_train_dir": "Kaggle_Prepared/train/HS",
    "hs_val_dir": "Kaggle_Prepared/val/HS",
    "output_dir": "ms_hs_xgb",
    "n_folds": 5,
    "seed": 42,
    "num_classes": 3,
    # HS band configuration
    "hs_clean_start": 10,   # Skip first ~10 noisy bands
    "hs_clean_end": 111,    # Skip last ~14 noisy bands (for 125-band images)
    "hs_pca_components": 20, # Number of PCA components from HS
    # HS spectral range: 450-950nm, 125 bands → ~4nm per band
    "hs_wavelength_start_nm": 450,
    "hs_wavelength_step_nm": 4.0,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR"]

# HS spectral region definitions (approximate band indices for 125-band, 450-950nm, 4nm step)
# These are used for region-wise statistics
HS_REGIONS = {
    "Blue":     (8, 20),     # ~480-530nm
    "Green":    (20, 35),    # ~530-590nm
    "Red":      (45, 58),    # ~630-680nm
    "RedEdge":  (58, 78),    # ~680-760nm
    "NIR":      (78, 111),   # ~760-900nm
    "VIS":      (8, 58),     # ~480-680nm (all visible)
}


# ============================================================
# MS Feature extraction (same as train_ms_xgb.py)
# ============================================================
def extract_ms_features(img):
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
        features[f"ms_{name}_mean"] = np.mean(b)
        features[f"ms_{name}_std"] = np.std(b)
        features[f"ms_{name}_min"] = np.min(b)
        features[f"ms_{name}_max"] = np.max(b)
        features[f"ms_{name}_median"] = np.median(b)
        features[f"ms_{name}_p5"] = np.percentile(b, 5)
        features[f"ms_{name}_p25"] = np.percentile(b, 25)
        features[f"ms_{name}_p75"] = np.percentile(b, 75)
        features[f"ms_{name}_p95"] = np.percentile(b, 95)
        features[f"ms_{name}_iqr"] = features[f"ms_{name}_p75"] - features[f"ms_{name}_p25"]
        features[f"ms_{name}_range"] = features[f"ms_{name}_max"] - features[f"ms_{name}_min"]
        features[f"ms_{name}_skew"] = float(scipy_stats.skew(b))
        features[f"ms_{name}_kurtosis"] = float(scipy_stats.kurtosis(b))
        features[f"ms_{name}_cv"] = np.std(b) / (np.mean(b) + eps)

    # ====== 2. Vegetation / spectral indices ======
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
        features[f"ms_{idx_name}_mean"] = np.mean(v)
        features[f"ms_{idx_name}_std"] = np.std(v)
        features[f"ms_{idx_name}_min"] = np.min(v)
        features[f"ms_{idx_name}_max"] = np.max(v)
        features[f"ms_{idx_name}_median"] = np.median(v)
        features[f"ms_{idx_name}_p10"] = np.percentile(v, 10)
        features[f"ms_{idx_name}_p90"] = np.percentile(v, 90)
        features[f"ms_{idx_name}_skew"] = float(scipy_stats.skew(v))

    # ====== 3. Inter-band correlations ======
    flat_bands = bands.reshape(5, -1)
    corr_matrix = np.corrcoef(flat_bands)
    for i in range(5):
        for j in range(i+1, 5):
            features[f"ms_corr_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = corr_matrix[i, j]

    # ====== 4. Spatial texture features ======
    for i, name in enumerate(BAND_NAMES):
        b = bands[i]
        gy, gx = np.gradient(b)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"ms_{name}_grad_mean"] = np.mean(grad_mag)
        features[f"ms_{name}_grad_std"] = np.std(grad_mag)

        local_mean = ndimage.uniform_filter(b, size=3)
        local_sq_mean = ndimage.uniform_filter(b**2, size=3)
        local_var = local_sq_mean - local_mean**2
        features[f"ms_{name}_localvar_mean"] = np.mean(local_var)
        features[f"ms_{name}_localvar_std"] = np.std(local_var)

    # ====== 5. Spatial features on key indices ======
    for idx_name, idx_map in [("NDVI", ndvi), ("NDRE", ndre)]:
        idx_map = np.clip(idx_map, -10, 10)
        gy, gx = np.gradient(idx_map)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"ms_{idx_name}_grad_mean"] = np.mean(grad_mag)
        features[f"ms_{idx_name}_grad_std"] = np.std(grad_mag)

    # ====== 6. Band ratios (aggregated) ======
    band_means = [np.mean(bands[i]) for i in range(5)]
    for i in range(5):
        for j in range(i+1, 5):
            features[f"ms_meanratio_{BAND_NAMES[i]}_{BAND_NAMES[j]}"] = band_means[i] / (band_means[j] + eps)

    # ====== 7. Spectral shape features ======
    mean_spectrum = np.array(band_means)
    features["ms_spec_slope_vis"] = mean_spectrum[2] - mean_spectrum[0]
    features["ms_spec_slope_rededge"] = mean_spectrum[3] - mean_spectrum[2]
    features["ms_spec_slope_nir"] = mean_spectrum[4] - mean_spectrum[3]
    features["ms_spec_curvature"] = mean_spectrum[3] - 0.5 * (mean_spectrum[2] + mean_spectrum[4])
    features["ms_spec_total_reflectance"] = np.sum(mean_spectrum)
    features["ms_spec_nir_vis_ratio"] = mean_spectrum[4] / (np.mean(mean_spectrum[:3]) + eps)

    return features


# ============================================================
# HS Feature extraction
# ============================================================
def get_clean_hs_bands(img):
    """
    Get clean HS bands, handling variable band counts (125 or 126).
    Input: img (32, 32, N_bands) float32
    Returns: clean_bands (32, 32, ~100) float32, band_wavelengths array
    """
    n_bands = img.shape[2]
    
    # Adjust clean range for different band counts
    if n_bands >= 126:
        clean_start = 10
        clean_end = 112  # one extra band
    else:
        clean_start = CFG["hs_clean_start"]
        clean_end = CFG["hs_clean_end"]
    
    clean_end = min(clean_end, n_bands)
    clean_bands = img[:, :, clean_start:clean_end]
    
    # Approximate wavelengths for clean bands
    wavelengths = np.array([
        CFG["hs_wavelength_start_nm"] + (clean_start + i) * CFG["hs_wavelength_step_nm"]
        for i in range(clean_bands.shape[2])
    ])
    
    return clean_bands, wavelengths


def find_nearest_band(wavelengths, target_nm):
    """Find the index of the band closest to target wavelength."""
    return np.argmin(np.abs(wavelengths - target_nm))


def extract_hs_features(img):
    """
    Extract rich feature vector from a single HS image (32, 32, 125-126 bands).
    Input: img (32, 32, N_bands) float32
    Returns: feature dict
    """
    img = img.astype(np.float32)
    features = {}
    eps = 1e-8
    
    n_bands_total = img.shape[2]
    features["hs_n_bands"] = n_bands_total
    
    # Get clean bands
    clean_img, wavelengths = get_clean_hs_bands(img)
    n_clean = clean_img.shape[2]
    
    # ====== 1. Region-wise statistics ======
    # Aggregate stats per spectral region
    for region_name, (start_raw, end_raw) in HS_REGIONS.items():
        # Map raw band indices to clean band indices
        clean_start_idx = max(0, start_raw - CFG["hs_clean_start"])
        clean_end_idx = min(n_clean, end_raw - CFG["hs_clean_start"])
        
        if clean_end_idx <= clean_start_idx:
            continue
            
        region = clean_img[:, :, clean_start_idx:clean_end_idx]
        region_flat = region.ravel()
        
        features[f"hs_{region_name}_mean"] = np.mean(region_flat)
        features[f"hs_{region_name}_std"] = np.std(region_flat)
        features[f"hs_{region_name}_median"] = np.median(region_flat)
        features[f"hs_{region_name}_p5"] = np.percentile(region_flat, 5)
        features[f"hs_{region_name}_p95"] = np.percentile(region_flat, 95)
        features[f"hs_{region_name}_skew"] = float(scipy_stats.skew(region_flat))
        features[f"hs_{region_name}_kurtosis"] = float(scipy_stats.kurtosis(region_flat))
        
        # Mean spectrum within region
        region_mean_spectrum = np.mean(region, axis=(0, 1))  # (n_bands_in_region,)
        features[f"hs_{region_name}_spec_slope"] = region_mean_spectrum[-1] - region_mean_spectrum[0]
        features[f"hs_{region_name}_spec_range"] = np.ptp(region_mean_spectrum)
        features[f"hs_{region_name}_spec_std"] = np.std(region_mean_spectrum)
    
    # ====== 2. Mean spectrum statistics ======
    # Compute mean spectrum across all pixels (spatial mean)
    mean_spectrum = np.mean(clean_img, axis=(0, 1))  # (n_clean,)
    
    features["hs_total_reflectance"] = np.sum(mean_spectrum)
    features["hs_mean_reflectance"] = np.mean(mean_spectrum)
    features["hs_spec_std"] = np.std(mean_spectrum)
    features["hs_spec_range"] = np.ptp(mean_spectrum)
    features["hs_spec_skew"] = float(scipy_stats.skew(mean_spectrum))
    features["hs_spec_kurtosis"] = float(scipy_stats.kurtosis(mean_spectrum))
    
    # Spectral percentiles
    for p in [10, 25, 50, 75, 90]:
        features[f"hs_spec_p{p}"] = np.percentile(mean_spectrum, p)
    
    # ====== 3. First and second derivative features ======
    # First derivative (spectral slope)
    first_deriv = np.gradient(mean_spectrum)
    features["hs_deriv1_max"] = np.max(first_deriv)
    features["hs_deriv1_min"] = np.min(first_deriv)
    features["hs_deriv1_mean"] = np.mean(first_deriv)
    features["hs_deriv1_std"] = np.std(first_deriv)
    features["hs_deriv1_max_pos"] = float(np.argmax(first_deriv)) / n_clean  # Normalized position
    features["hs_deriv1_min_pos"] = float(np.argmin(first_deriv)) / n_clean
    
    # Red edge position (max of first derivative in red edge region)
    re_start = max(0, 58 - CFG["hs_clean_start"])
    re_end = min(n_clean, 78 - CFG["hs_clean_start"])
    if re_end > re_start:
        re_deriv = first_deriv[re_start:re_end]
        features["hs_red_edge_position"] = float(re_start + np.argmax(re_deriv)) / n_clean
        features["hs_red_edge_slope_max"] = np.max(re_deriv)
        features["hs_red_edge_slope_mean"] = np.mean(re_deriv)
    
    # Second derivative (curvature)
    second_deriv = np.gradient(first_deriv)
    features["hs_deriv2_max"] = np.max(second_deriv)
    features["hs_deriv2_min"] = np.min(second_deriv)
    features["hs_deriv2_std"] = np.std(second_deriv)
    
    # ====== 4. HS-specific vegetation indices ======
    # Find bands closest to specific wavelengths
    # These are computed pixel-wise then aggregated
    b_531 = clean_img[:, :, find_nearest_band(wavelengths, 531)]
    b_570 = clean_img[:, :, find_nearest_band(wavelengths, 570)]
    b_550 = clean_img[:, :, find_nearest_band(wavelengths, 550)]
    b_700 = clean_img[:, :, find_nearest_band(wavelengths, 700)]
    b_670 = clean_img[:, :, find_nearest_band(wavelengths, 670)]
    b_800 = clean_img[:, :, find_nearest_band(wavelengths, 800)]
    b_900 = clean_img[:, :, find_nearest_band(wavelengths, 900)]
    b_970 = clean_img[:, :, find_nearest_band(wavelengths, 950)]  # Closest to 970
    b_680 = clean_img[:, :, find_nearest_band(wavelengths, 680)]
    b_500 = clean_img[:, :, find_nearest_band(wavelengths, 500)]
    b_750 = clean_img[:, :, find_nearest_band(wavelengths, 750)]
    b_710 = clean_img[:, :, find_nearest_band(wavelengths, 710)]
    b_720 = clean_img[:, :, find_nearest_band(wavelengths, 720)]
    b_740 = clean_img[:, :, find_nearest_band(wavelengths, 740)]
    b_760 = clean_img[:, :, find_nearest_band(wavelengths, 760)]
    b_650 = clean_img[:, :, find_nearest_band(wavelengths, 650)]
    b_480 = clean_img[:, :, find_nearest_band(wavelengths, 480)]
    b_510 = clean_img[:, :, find_nearest_band(wavelengths, 510)]
    
    # PRI (Photochemical Reflectance Index) — stress indicator
    pri = (b_531 - b_570) / (b_531 + b_570 + eps)
    
    # ARI (Anthocyanin Reflectance Index) — disease pigmentation
    ari = (1.0 / (b_550 + eps)) - (1.0 / (b_700 + eps))
    
    # WBI (Water Band Index) — water content
    wbi = b_900 / (b_970 + eps)
    
    # PSRI (Plant Senescence Reflectance Index) — senescence
    psri = (b_680 - b_500) / (b_750 + eps)
    
    # SIPI (Structure Insensitive Pigment Index) — carotenoid:chlorophyll ratio
    sipi = (b_800 - b_480) / (b_800 - b_680 + eps)
    
    # MCARI/OSAVI — chlorophyll content
    mcari_hs = ((b_700 - b_670) - 0.2 * (b_700 - b_550)) * (b_700 / (b_670 + eps))
    osavi = 1.16 * (b_800 - b_670) / (b_800 + b_670 + 0.16 + eps)
    tcari_osavi = mcari_hs / (osavi + eps)
    
    # NDVI from HS bands (higher spectral precision than MS)
    hs_ndvi = (b_800 - b_670) / (b_800 + b_670 + eps)
    
    # RENDVI (Red Edge NDVI)
    rendvi = (b_750 - b_710) / (b_750 + b_710 + eps)
    
    # VOG indices (Vogelmann Red Edge Indices) — chlorophyll
    vog1 = b_740 / (b_720 + eps)
    
    # mSR (Modified Simple Ratio)
    msr = (b_800 / (b_670 + eps) - 1) / (np.sqrt(b_800 / (b_670 + eps)) + 1 + eps)
    
    # CRI (Carotenoid Reflectance Index)
    cri1 = (1.0 / (b_510 + eps)) - (1.0 / (b_550 + eps))
    
    hs_indices = {
        "PRI": pri, "ARI": ari, "WBI": wbi, "PSRI": psri, "SIPI": sipi,
        "MCARI_HS": mcari_hs, "OSAVI": osavi, "TCARI_OSAVI": tcari_osavi,
        "HS_NDVI": hs_ndvi, "RENDVI": rendvi, "VOG1": vog1, "mSR": msr,
        "CRI1": cri1,
    }
    
    for idx_name, idx_map in hs_indices.items():
        idx_map = np.clip(idx_map, -50, 50)
        v = idx_map.ravel()
        features[f"hs_{idx_name}_mean"] = np.mean(v)
        features[f"hs_{idx_name}_std"] = np.std(v)
        features[f"hs_{idx_name}_min"] = np.min(v)
        features[f"hs_{idx_name}_max"] = np.max(v)
        features[f"hs_{idx_name}_median"] = np.median(v)
        features[f"hs_{idx_name}_p10"] = np.percentile(v, 10)
        features[f"hs_{idx_name}_p90"] = np.percentile(v, 90)
        features[f"hs_{idx_name}_skew"] = float(scipy_stats.skew(v))
    
    # ====== 5. Absorption band features ======
    # Chlorophyll absorption around 680nm
    chlor_center = find_nearest_band(wavelengths, 680)
    chlor_left = find_nearest_band(wavelengths, 630)
    chlor_right = find_nearest_band(wavelengths, 750)
    
    if chlor_left < n_clean and chlor_right < n_clean and chlor_center < n_clean:
        # Continuum line between shoulders
        continuum_680 = mean_spectrum[chlor_left] + (mean_spectrum[chlor_right] - mean_spectrum[chlor_left]) * \
                        (chlor_center - chlor_left) / (chlor_right - chlor_left + eps)
        absorption_depth_680 = 1.0 - mean_spectrum[chlor_center] / (continuum_680 + eps)
        features["hs_absorption_680_depth"] = absorption_depth_680
        
        # Absorption width (FWHM proxy)
        half_depth = 0.5 * (continuum_680 - mean_spectrum[chlor_center])
        threshold = continuum_680 - half_depth
        absorption_width = 0
        for bi in range(chlor_left, chlor_right + 1):
            if bi < n_clean and mean_spectrum[bi] < threshold:
                absorption_width += 1
        features["hs_absorption_680_width"] = absorption_width
    
    # Water absorption around 950nm (partial, since sensor caps at 950nm)
    water_band = find_nearest_band(wavelengths, 940)
    water_shoulder = find_nearest_band(wavelengths, 850)
    if water_band < n_clean and water_shoulder < n_clean:
        features["hs_water_absorption_ratio"] = mean_spectrum[water_band] / (mean_spectrum[water_shoulder] + eps)
    
    # ====== 6. Spectral area features ======
    # Area under curve for different regions
    if n_clean > 0:
        features["hs_area_vis"] = np.trapezoid(mean_spectrum[:max(1, 48 - CFG["hs_clean_start"])])
        features["hs_area_nir"] = np.trapezoid(mean_spectrum[max(0, 68 - CFG["hs_clean_start"]):])
        vis_area = features["hs_area_vis"]
        nir_area = features["hs_area_nir"]
        features["hs_nir_vis_area_ratio"] = nir_area / (vis_area + eps)
    
    # ====== 7. Inter-region ratios ======
    # Ratio of region means (HS-specific, finer than MS)
    for rn1, (s1, e1) in HS_REGIONS.items():
        for rn2, (s2, e2) in HS_REGIONS.items():
            if rn1 >= rn2:
                continue
            cs1, ce1 = max(0, s1 - CFG["hs_clean_start"]), min(n_clean, e1 - CFG["hs_clean_start"])
            cs2, ce2 = max(0, s2 - CFG["hs_clean_start"]), min(n_clean, e2 - CFG["hs_clean_start"])
            if ce1 > cs1 and ce2 > cs2:
                m1 = np.mean(mean_spectrum[cs1:ce1])
                m2 = np.mean(mean_spectrum[cs2:ce2])
                features[f"hs_ratio_{rn1}_{rn2}"] = m1 / (m2 + eps)
    
    # ====== 8. Spatial texture features on key HS bands ======
    key_hs_bands = {
        "b670": b_670, "b700": b_700, "b750": b_750, "b800": b_800,
    }
    for bname, bdata in key_hs_bands.items():
        gy, gx = np.gradient(bdata)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features[f"hs_{bname}_grad_mean"] = np.mean(grad_mag)
        features[f"hs_{bname}_grad_std"] = np.std(grad_mag)
        
        local_mean = ndimage.uniform_filter(bdata, size=3)
        local_sq_mean = ndimage.uniform_filter(bdata**2, size=3)
        local_var = local_sq_mean - local_mean**2
        features[f"hs_{bname}_localvar_mean"] = np.mean(local_var)
    
    # ====== 9. Spectral texture — pixel-wise spectral variability ======
    # Standard deviation of spectrum per pixel, then spatial statistics
    pixel_spec_std = np.std(clean_img, axis=2)  # (32, 32) — how diverse is each pixel's spectrum
    features["hs_pixel_spec_std_mean"] = np.mean(pixel_spec_std)
    features["hs_pixel_spec_std_std"] = np.std(pixel_spec_std)
    features["hs_pixel_spec_std_median"] = np.median(pixel_spec_std)
    
    # Spectral angle diversity across image
    center_spectrum = mean_spectrum / (np.linalg.norm(mean_spectrum) + eps)
    pixel_spectra = clean_img.reshape(-1, n_clean).astype(np.float64)
    pixel_norms = np.linalg.norm(pixel_spectra, axis=1, keepdims=True)
    pixel_unit = pixel_spectra / (pixel_norms + eps)
    cos_angles = np.dot(pixel_unit, center_spectrum)
    cos_angles = np.clip(cos_angles, -1, 1)
    features["hs_spectral_angle_mean"] = np.mean(cos_angles)
    features["hs_spectral_angle_std"] = np.std(cos_angles)
    
    return features


def extract_hs_pca_features(hs_images_flat, n_components=20, fit=True, pca_model=None):
    """
    Apply PCA to HS band data and return PCA components as features.
    Input: hs_images_flat — list of (n_clean_bands,) mean spectra
    Returns: PCA features array and fitted PCA model
    """
    X_hs = np.array(hs_images_flat, dtype=np.float32)
    X_hs = np.nan_to_num(X_hs, nan=0.0, posinf=10.0, neginf=-10.0)
    
    if fit:
        n_components = min(n_components, X_hs.shape[0], X_hs.shape[1])
        pca_model = PCA(n_components=n_components, random_state=CFG["seed"])
        X_pca = pca_model.fit_transform(X_hs)
    else:
        # Handle dimension mismatch for val set
        if X_hs.shape[1] != pca_model.n_features_in_:
            # Pad or truncate to match training dimensions
            target_dim = pca_model.n_features_in_
            if X_hs.shape[1] < target_dim:
                X_hs = np.pad(X_hs, ((0, 0), (0, target_dim - X_hs.shape[1])))
            else:
                X_hs = X_hs[:, :target_dim]
        X_pca = pca_model.transform(X_hs)
    
    return X_pca, pca_model


# ============================================================
# Combined feature extraction
# ============================================================
def extract_all_features(ms_dir, hs_dir, file_list=None):
    """Extract MS + HS features from all files."""
    if file_list is None:
        file_list = sorted(os.listdir(ms_dir))
    
    all_ms_features = []
    all_hs_features = []
    all_hs_mean_spectra = []  # For PCA
    all_labels = []
    all_fnames = []
    skipped_black = 0
    
    for f in file_list:
        ms_fp = os.path.join(ms_dir, f)
        hs_fp = os.path.join(hs_dir, f)
        
        ms_img = tiff.imread(ms_fp).astype(np.float32)
        hs_img = tiff.imread(hs_fp).astype(np.float32)
        
        # Check for black images (using MS as reference)
        if ms_img.mean() < 1.0:
            skipped_black += 1
            if "_hyper_" in f:
                continue  # Skip black training images
            else:
                # Val black image — will predict "Other"
                all_ms_features.append(None)
                all_hs_features.append(None)
                all_hs_mean_spectra.append(None)
                all_labels.append(-1)
                all_fnames.append(f)
                continue
        
        ms_feats = extract_ms_features(ms_img)
        hs_feats = extract_hs_features(hs_img)
        
        # Store mean spectrum for PCA
        clean_hs, _ = get_clean_hs_bands(hs_img)
        hs_mean_spec = np.mean(clean_hs, axis=(0, 1))
        
        all_ms_features.append(ms_feats)
        all_hs_features.append(hs_feats)
        all_hs_mean_spectra.append(hs_mean_spec)
        
        # Parse label from filename
        if "_hyper_" in f:
            cls_name = f.split("_hyper_")[0]
            all_labels.append(CLASS_MAP[cls_name])
        else:
            all_labels.append(-1)
        
        all_fnames.append(f)
    
    if skipped_black > 0:
        print(f"  Skipped/flagged {skipped_black} black images")
    
    return all_ms_features, all_hs_features, all_hs_mean_spectra, all_labels, all_fnames


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)
    
    # --- Extract training features ---
    print("=" * 70)
    print("MS + HS XGBoost Pipeline")
    print("=" * 70)
    
    print("\nExtracting training features (MS + HS)...")
    train_ms, train_hs, train_hs_spectra, train_labels, train_fnames = \
        extract_all_features(CFG["ms_train_dir"], CFG["hs_train_dir"])
    print(f"  {len(train_ms)} samples")
    print(f"  MS features: {len(train_ms[0])} per sample")
    print(f"  HS features: {len(train_hs[0])} per sample")
    
    # --- Extract val features ---
    print("\nExtracting validation features (MS + HS)...")
    val_ms, val_hs, val_hs_spectra, val_labels, val_fnames = \
        extract_all_features(CFG["ms_val_dir"], CFG["hs_val_dir"])
    black_mask = [f is None for f in val_ms]
    print(f"  {len(val_ms)} samples ({sum(black_mask)} black)")
    
    # --- Build feature matrices ---
    # Get feature names from first valid sample
    ms_feature_names = list(train_ms[0].keys())
    hs_feature_names = list(train_hs[0].keys())
    
    # PCA on HS mean spectra
    # Normalize spectrum lengths: find the minimum clean band count
    train_spec_lengths = [len(s) for s in train_hs_spectra]
    val_spec_lengths = [len(s) for s in val_hs_spectra if s is not None]
    min_spec_len = min(min(train_spec_lengths), min(val_spec_lengths) if val_spec_lengths else min(train_spec_lengths))
    print(f"\n  HS spectrum lengths: train min={min(train_spec_lengths)}, max={max(train_spec_lengths)}")
    print(f"  Truncating all to {min_spec_len} bands for PCA")
    
    # Truncate all spectra to min length
    train_hs_spectra_trunc = [s[:min_spec_len] for s in train_hs_spectra]
    
    # Fit PCA on training data
    n_pca = CFG["hs_pca_components"]
    pca_X_train, pca_model = extract_hs_pca_features(
        train_hs_spectra_trunc, n_components=n_pca, fit=True
    )
    print(f"  PCA: {n_pca} components, explained variance: {sum(pca_model.explained_variance_ratio_):.4f}")
    
    pca_feature_names = [f"hs_pca_{i}" for i in range(n_pca)]
    
    # Build training feature matrix
    y_train = np.array(train_labels)
    
    X_ms_train = np.array([[f[k] for k in ms_feature_names] for f in train_ms], dtype=np.float32)
    X_hs_train = np.array([[f[k] for k in hs_feature_names] for f in train_hs], dtype=np.float32)
    X_pca_train = pca_X_train.astype(np.float32)
    
    X_train = np.hstack([X_ms_train, X_hs_train, X_pca_train])
    all_feature_names = ms_feature_names + hs_feature_names + pca_feature_names
    
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
    
    print(f"\n  Total features: {len(all_feature_names)} ({len(ms_feature_names)} MS + {len(hs_feature_names)} HS + {n_pca} PCA)")
    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(y_train == ci).sum()}")
    
    # Build val feature matrix
    # Fill black images with zeros
    for i in range(len(val_ms)):
        if val_ms[i] is None:
            val_ms[i] = {k: 0.0 for k in ms_feature_names}
        if val_hs[i] is None:
            val_hs[i] = {k: 0.0 for k in hs_feature_names}
        if val_hs_spectra[i] is None:
            val_hs_spectra[i] = np.zeros(min_spec_len)
    
    X_ms_val = np.array([[f[k] for k in ms_feature_names] for f in val_ms], dtype=np.float32)
    X_hs_val = np.array([[f[k] for k in hs_feature_names] for f in val_hs], dtype=np.float32)
    
    val_hs_spectra_trunc = [s[:min_spec_len] for s in val_hs_spectra]
    pca_X_val, _ = extract_hs_pca_features(
        val_hs_spectra_trunc, n_components=n_pca, fit=False, pca_model=pca_model
    )
    X_pca_val = pca_X_val.astype(np.float32)
    
    X_val = np.hstack([X_ms_val, X_hs_val, X_pca_val])
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # --- Normalize features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # ================================================================
    # XGBoost with 5-fold CV
    # ================================================================
    print(f"\n{'='*70}")
    print("XGBoost 5-Fold CV (MS + HS features)")
    print(f"{'='*70}")
    
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.6,  # Slightly lower due to more features
        "min_child_weight": 3,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": CFG["seed"],
        "tree_method": "hist",
        "verbosity": 0,
    }
    
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    
    oof_probs = np.zeros((len(X_train), 3))
    oof_preds = np.zeros(len(X_train), dtype=int)
    val_probs_all_folds = []
    fold_results = []
    feature_importance = np.zeros(len(all_feature_names))
    
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
        
        val_probs = model.predict_proba(X_val_scaled)
        val_probs_all_folds.append(val_probs)
    
    # --- OOF Summary ---
    print(f"\n{'='*70}")
    print("XGB CV SUMMARY (MS + HS)")
    print(f"{'='*70}")
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
    top_idx = np.argsort(feature_importance)[::-1][:30]
    print(f"\nTop 30 features:")
    for i, idx in enumerate(top_idx):
        prefix = "MS" if all_feature_names[idx].startswith("ms_") else "HS"
        print(f"  {i+1:2d}. [{prefix}] {all_feature_names[idx]:40s} importance={feature_importance[idx]:.4f}")
    
    # Count MS vs HS features in top 30
    ms_in_top = sum(1 for idx in top_idx if all_feature_names[idx].startswith("ms_"))
    hs_in_top = len(top_idx) - ms_in_top
    print(f"\n  Top 30 breakdown: {ms_in_top} MS features, {hs_in_top} HS features")
    
    # ================================================================
    # LightGBM (diversity for ensemble)
    # ================================================================
    print(f"\n{'='*70}")
    print("LightGBM 5-Fold CV (MS + HS features)")
    print(f"{'='*70}")
    
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_samples": 5,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
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
    print(f"\nLGB CV SUMMARY (MS + HS)")
    oof_acc_lgb = accuracy_score(y_train, oof_preds_lgb)
    oof_f1_lgb = f1_score(y_train, oof_preds_lgb, average='macro')
    print(f"OOF Acc: {oof_acc_lgb:.4f}, F1: {oof_f1_lgb:.4f}")
    print(classification_report(y_train, oof_preds_lgb, target_names=list(CLASS_MAP.keys()), digits=4))
    
    # ================================================================
    # Ensemble XGB + LGB
    # ================================================================
    print(f"\n{'='*70}")
    print("XGB + LGB Ensemble (MS + HS)")
    print(f"{'='*70}")
    
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
    # Compare with MS-only baseline
    # ================================================================
    print(f"\n{'='*70}")
    print("COMPARISON: MS-only vs MS+HS")
    print(f"{'='*70}")
    
    # MS-only XGBoost (quick version using only MS features)
    X_ms_only_train = X_train_scaled[:, :len(ms_feature_names)]
    X_ms_only_val = X_val_scaled[:, :len(ms_feature_names)]
    
    oof_probs_ms_only = np.zeros((len(X_train), 3))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_ms_only_train, y_train)):
        X_tr, X_va = X_ms_only_train[tr_idx], X_ms_only_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        model_ms = xgb.XGBClassifier(**xgb_params)
        model_ms.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_probs_ms_only[va_idx] = model_ms.predict_proba(X_va)
    
    ms_only_preds = np.argmax(oof_probs_ms_only, axis=1)
    ms_only_acc = accuracy_score(y_train, ms_only_preds)
    ms_only_f1 = f1_score(y_train, ms_only_preds, average='macro')
    
    print(f"  MS-only XGB:  Acc={ms_only_acc:.4f}, F1={ms_only_f1:.4f}")
    print(f"  MS+HS XGB:    Acc={oof_acc:.4f}, F1={oof_f1:.4f}")
    print(f"  MS+HS Ens:    Acc={ens_acc:.4f}, F1={ens_f1:.4f}")
    print(f"  Improvement:  Acc +{ens_acc - ms_only_acc:.4f}, F1 +{ens_f1 - ms_only_f1:.4f}")
    
    # ================================================================
    # Save everything
    # ================================================================
    np.save(os.path.join(CFG["output_dir"], "ms_hs_val_probs_xgb.npy"), xgb_val_probs)
    np.save(os.path.join(CFG["output_dir"], "ms_hs_val_probs_lgb.npy"), lgb_val_probs)
    np.save(os.path.join(CFG["output_dir"], "ms_hs_val_probs_ensemble.npy"), ens_val_probs)
    
    # Save OOF for potential stacking
    np.save(os.path.join(CFG["output_dir"], "oof_probs_xgb.npy"), oof_probs)
    np.save(os.path.join(CFG["output_dir"], "oof_probs_lgb.npy"), oof_probs_lgb)
    
    # Save ensemble submission
    sub_path = os.path.join(CFG["output_dir"], "ms_hs_submission.csv")
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for fn, cl in zip(val_fnames, pred_classes):
            w.writerow([fn, cl])
    
    # Save XGBoost-only (best model) submission
    xgb_val_preds = np.argmax(xgb_val_probs, axis=1)
    xgb_pred_classes = [INV_CLASS_MAP[p] for p in xgb_val_preds]
    sub_xgb_path = os.path.join(CFG["output_dir"], "ms_hs_submission_xgb.csv")
    with open(sub_xgb_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for fn, cl in zip(val_fnames, xgb_pred_classes):
            w.writerow([fn, cl])
    
    print(f"\n  Saved ensemble submission → {sub_path}")
    print(f"  Saved XGBoost-only submission → {sub_xgb_path}")
    
    # Save feature names for reference
    with open(os.path.join(CFG["output_dir"], "feature_names.json"), "w") as f:
        json.dump(all_feature_names, f, indent=2)
    
    print(f"\nSaved to {CFG['output_dir']}/")
    print("Done!")


if __name__ == "__main__":
    main()
