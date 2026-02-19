"""Quick exploration of MS (5-band multispectral) data."""
import os
import numpy as np
import tifffile as tiff
from collections import defaultdict

DATA_DIR = "Kaggle_Prepared/train/MS"
VAL_DIR = "Kaggle_Prepared/val/MS"

# --- 1. Load a few samples, check shape/dtype/range ---
files = sorted(os.listdir(DATA_DIR))
print(f"Total train MS files: {len(files)}")
print(f"Sample filenames: {files[:3]} ... {files[-3:]}")

# Load first file to check format
sample = tiff.imread(os.path.join(DATA_DIR, files[0]))
print(f"\nSample shape: {sample.shape}, dtype: {sample.dtype}")
print(f"Min: {sample.min()}, Max: {sample.max()}, Mean: {sample.mean():.2f}")

# --- 2. Check all shapes across dataset ---
shapes = defaultdict(int)
class_counts = defaultdict(int)
black_images = []

all_means = []
all_stds = []
band_stats = {i: {"mins": [], "maxs": [], "means": [], "stds": []} for i in range(sample.shape[0] if sample.ndim == 3 else 1)}

for f in files:
    cls = f.split("_hyper_")[0]
    class_counts[cls] += 1
    
    img = tiff.imread(os.path.join(DATA_DIR, f))
    shapes[img.shape] += 1
    
    mean_val = img.mean()
    all_means.append(mean_val)
    
    # Check for black/corrupt images
    if mean_val < 1.0:
        black_images.append((f, mean_val))
    
    # Per-band stats
    if img.ndim == 3:
        for b in range(img.shape[0]):
            band_stats[b]["mins"].append(img[b].min())
            band_stats[b]["maxs"].append(img[b].max())
            band_stats[b]["means"].append(img[b].mean())
            band_stats[b]["stds"].append(img[b].std())

print(f"\n--- Shapes ---")
for s, c in shapes.items():
    print(f"  {s}: {c} files")

print(f"\n--- Class counts ---")
for cls, c in sorted(class_counts.items()):
    print(f"  {cls}: {c}")

print(f"\n--- Per-band statistics (across all train) ---")
for b in sorted(band_stats.keys()):
    s = band_stats[b]
    print(f"  Band {b}: min={np.mean(s['mins']):.1f}, max={np.mean(s['maxs']):.1f}, "
          f"mean={np.mean(s['means']):.1f}, std={np.mean(s['stds']):.1f}")

print(f"\n--- Black/corrupt images ({len(black_images)}) ---")
for f, m in black_images:
    print(f"  {f}: mean={m:.4f}")

# Also check a near-zero threshold
low_mean = [(f, m) for f, m in zip(files, all_means) if m < 50]
print(f"\n--- Low-mean images (mean < 50): {len(low_mean)} ---")
for f, m in low_mean[:10]:
    print(f"  {f}: mean={m:.4f}")

# --- 3. Check val set ---
val_files = sorted(os.listdir(VAL_DIR))
print(f"\n--- Val MS files: {len(val_files)} ---")
print(f"Sample: {val_files[:3]}")

# Check val for black images
val_black = []
for f in val_files:
    img = tiff.imread(os.path.join(VAL_DIR, f))
    if img.mean() < 1.0:
        val_black.append((f, img.mean()))
print(f"Val black images: {len(val_black)}")
for f, m in val_black:
    print(f"  {f}: mean={m:.4f}")

# Check val shape consistency
val_shape = tiff.imread(os.path.join(VAL_DIR, val_files[0])).shape
print(f"Val sample shape: {val_shape}")
