"""
MS CNN v3 — GPU-optimized, fast convergence
- ALL data preloaded into tensors (no per-sample disk I/O)
- Spectral indices precomputed once
- Augmentation on GPU tensors (no numpy in training loop)
- Weighted CE + macro F1 model selection
- Simple proven CNN architecture with SE blocks
"""

import os
import json
import csv
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
import tifffile as tiff

warnings.filterwarnings("ignore")

CFG = {
    "data_dir": "Kaggle_Prepared/train/MS",
    "val_dir": "Kaggle_Prepared/val/MS",
    "output_dir": "ms_models_v3",
    "n_folds": 5,
    "epochs": 100,
    "batch_size": 64,  # bigger batch — everything is in memory
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "patience": 20,
    "seed": 42,
    "num_classes": 3,
    "tta_flips": True,
    "class_weights": [1.3, 0.9, 1.0],
    "label_smoothing": 0.05,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Preload + precompute everything
# ============================================================
def compute_indices_tensor(x):
    """
    x: (N, 5, H, W) tensor — z-score normalized MS bands
    Returns: (N, 11, H, W) — original 5 + 6 vegetation indices
    """
    eps = 1e-8
    blue, green, red, re, nir = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - re) / (nir + re + eps)
    gndvi = (nir - green) / (nir + green + eps)
    ci_re = (nir / (re + eps) - 1.0).clamp(-5, 5)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    rg = (red / (green + eps)).clamp(-5, 5)

    indices = torch.stack([ndvi, ndre, gndvi, ci_re, savi, rg], dim=1)  # (N, 6, H, W)
    return torch.cat([x, indices], dim=1)  # (N, 11, H, W)


def load_all_train(data_dir):
    """Load entire training set into a single tensor. Returns (images, labels, band_mean, band_std)."""
    all_files = sorted(os.listdir(data_dir))
    imgs_list = []
    labels_list = []
    skipped = 0

    for f in all_files:
        fp = os.path.join(data_dir, f)
        cls_name = f.split("_hyper_")[0]
        label = CLASS_MAP[cls_name]

        img = tiff.imread(fp).astype(np.float32)  # (64, 64, 5)
        if img.mean() < 1.0:
            skipped += 1
            continue

        img = img.transpose(2, 0, 1)  # (5, 64, 64)
        imgs_list.append(img)
        labels_list.append(label)

    print(f"Loaded {len(imgs_list)} training samples (skipped {skipped} black images)")

    images = torch.from_numpy(np.stack(imgs_list))  # (N, 5, 64, 64)
    labels = torch.tensor(labels_list, dtype=torch.long)  # (N,)

    # Per-band normalization
    band_mean = images.mean(dim=(0, 2, 3))  # (5,)
    band_std = images.std(dim=(0, 2, 3))    # (5,)

    images = (images - band_mean[None, :, None, None]) / (band_std[None, :, None, None] + 1e-8)

    # Precompute spectral indices
    images = compute_indices_tensor(images)  # (N, 11, 64, 64)

    return images, labels, band_mean, band_std


def load_all_val(val_dir, band_mean, band_std):
    """Load entire val set into tensors."""
    val_files = sorted(os.listdir(val_dir))
    imgs_list = []
    fnames = []
    is_black = []

    for f in val_files:
        fp = os.path.join(val_dir, f)
        img = tiff.imread(fp).astype(np.float32)  # (64, 64, 5)

        if img.mean() < 1.0:
            imgs_list.append(np.zeros((5, 64, 64), dtype=np.float32))
            is_black.append(True)
        else:
            img = img.transpose(2, 0, 1)  # (5, 64, 64)
            imgs_list.append(img)
            is_black.append(False)

        fnames.append(f)

    images = torch.from_numpy(np.stack(imgs_list))  # (N, 5, 64, 64)
    images = (images - band_mean[None, :, None, None]) / (band_std[None, :, None, None] + 1e-8)
    images = compute_indices_tensor(images)  # (N, 11, 64, 64)

    return images, fnames, is_black


# ============================================================
# GPU augmentation (no CPU numpy in training loop)
# ============================================================
def augment_batch(imgs):
    """Random augmentation on GPU tensors. imgs: (B, C, H, W)."""
    B = imgs.size(0)

    # Random horizontal flip
    mask = torch.rand(B, device=imgs.device) > 0.5
    if mask.any():
        imgs[mask] = imgs[mask].flip(3)

    # Random vertical flip
    mask = torch.rand(B, device=imgs.device) > 0.5
    if mask.any():
        imgs[mask] = imgs[mask].flip(2)

    # Random 90° rotation (per sample)
    for k in [1, 2, 3]:
        mask = torch.rand(B, device=imgs.device) > 0.75  # ~25% chance each rotation
        if mask.any():
            imgs[mask] = torch.rot90(imgs[mask], k, [2, 3])

    # Gaussian noise
    noise_mask = torch.rand(B, device=imgs.device) > 0.5
    if noise_mask.any():
        noise = torch.randn_like(imgs[noise_mask]) * 0.03
        imgs[noise_mask] = imgs[noise_mask] + noise

    # Band dropout (zero one of first 5 bands)
    drop_mask = torch.rand(B, device=imgs.device) > 0.85
    if drop_mask.any():
        drop_bands = torch.randint(0, 5, (drop_mask.sum().item(),), device=imgs.device)
        for i, (idx, band) in enumerate(zip(drop_mask.nonzero(as_tuple=True)[0], drop_bands)):
            imgs[idx, band] = 0.0

    return imgs


# ============================================================
# Model
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.squeeze(x).flatten(1)
        w = self.excite(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class MSCNNv3(nn.Module):
    def __init__(self, in_ch=11, num_classes=3):
        super().__init__()
        # 1×1 spectral mixing
        self.spec = nn.Sequential(nn.Conv2d(in_ch, 32, 1), nn.BatchNorm2d(32), nn.ReLU(True))

        # Spatial blocks
        self.b1 = self._block(32, 48)    # 64 → 32
        self.b2 = self._block(48, 96)    # 32 → 16
        self.b3 = self._block(96, 192)   # 16 → 8

        self.se = SEBlock(192)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(192, 64),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )

    def forward(self, x):
        x = self.spec(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.se(x)
        x = self.pool(x)
        return self.head(x)


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, images, labels, indices, criterion, optimizer, device, batch_size):
    model.train()
    perm = torch.randperm(len(indices), device=device)
    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, len(perm), batch_size):
        idx = indices[perm[start:start + batch_size]]
        imgs = images[idx].clone()  # (B, 11, 64, 64) — already on device
        labs = labels[idx]

        imgs = augment_batch(imgs)

        optimizer.zero_grad(set_to_none=True)
        out = model(imgs)
        loss = criterion(out, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labs).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, images, labels, indices, device, batch_size=128):
    model.eval()
    all_preds = []
    all_probs = []

    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        imgs = images[idx]
        out = model(imgs)
        all_probs.append(F.softmax(out, dim=1).cpu())
        all_preds.append(out.argmax(1).cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    true = labels[indices].cpu().numpy()

    acc = accuracy_score(true, all_preds)
    mf1 = f1_score(true, all_preds, average='macro')

    recall = {}
    for ci, cn in INV_CLASS_MAP.items():
        mask = true == ci
        recall[cn] = (all_preds[mask] == ci).mean() if mask.sum() > 0 else 0.0

    return acc, mf1, recall, all_preds, true


# ============================================================
# Main
# ============================================================
def main():
    seed_everything(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(CFG["output_dir"], exist_ok=True)

    # --- Preload everything ---
    print("Loading all training data into memory...")
    images, labels, band_mean, band_std = load_all_train(CFG["data_dir"])
    print(f"  Shape: {images.shape}, dtype: {images.dtype}")
    for ci in range(3):
        print(f"  {INV_CLASS_MAP[ci]}: {(labels == ci).sum().item()}")

    # Save band stats
    stats = {"band_mean": band_mean.tolist(), "band_std": band_std.tolist()}
    with open(os.path.join(CFG["output_dir"], "band_stats.json"), "w") as f:
        json.dump(stats, f)

    # Move everything to device
    images = images.to(device)
    labels = labels.to(device)

    print(f"  Data on {device}, total GPU memory: ~{images.nelement() * 4 / 1e6:.1f} MB")

    # --- Loss ---
    class_w = torch.tensor(CFG["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=CFG["label_smoothing"])

    # --- K-Fold ---
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    fold_results = []
    labels_np = labels.cpu().numpy()
    indices_all = np.arange(len(labels_np))

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices_all, labels_np)):
        print(f"\n{'='*60}\nFOLD {fold+1}/{CFG['n_folds']}\n{'='*60}")

        train_idx_t = torch.tensor(train_idx, device=device, dtype=torch.long)
        val_idx_t = torch.tensor(val_idx, device=device, dtype=torch.long)

        for ci in range(3):
            nt = (labels[train_idx_t] == ci).sum().item()
            nv = (labels[val_idx_t] == ci).sum().item()
            print(f"  {INV_CLASS_MAP[ci]}: train={nt}, val={nv}")

        model = MSCNNv3(in_ch=11, num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-6)

        best_f1 = 0
        best_ep = 0
        patience = 0

        for ep in range(CFG["epochs"]):
            t_loss, t_acc = train_one_epoch(model, images, labels, train_idx_t,
                                            criterion, optimizer, device, CFG["batch_size"])
            v_acc, v_f1, recall, _, _ = validate(model, images, labels, val_idx_t, device)
            scheduler.step()

            if v_f1 > best_f1:
                best_f1 = v_f1
                best_ep = ep + 1
                patience = 0
                torch.save(model.state_dict(), os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"))
            else:
                patience += 1

            if (ep + 1) % 5 == 0 or ep == 0 or v_f1 >= best_f1 - 0.01:
                rc = " ".join(f"{k}:{v:.2f}" for k, v in recall.items())
                print(f"  Ep {ep+1:3d} | TL:{t_loss:.3f} TA:{t_acc:.3f} | "
                      f"VA:{v_acc:.3f} F1:{v_f1:.3f} | {rc} | Best:{best_f1:.3f}(ep{best_ep})")

            if patience >= CFG["patience"]:
                print(f"  Early stop ep {ep+1}")
                break

        # Final eval
        model.load_state_dict(torch.load(os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"),
                                         weights_only=True))
        fa, ff, fr, fp, ft = validate(model, images, labels, val_idx_t, device)
        fold_results.append({"fold": fold, "ep": best_ep, "acc": fa, "f1": ff, "recall": fr})
        print(f"\n  Fold {fold+1} Best: Acc={fa:.4f} F1={ff:.4f} (ep {best_ep})")
        print(f"  Recall: {fr}")
        print(classification_report(ft, fp, target_names=list(CLASS_MAP.keys()), digits=4))

    # --- Summary ---
    print(f"\n{'='*60}\nCV SUMMARY\n{'='*60}")
    accs = [r["acc"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    print(f"Acc:  {[f'{a:.4f}' for a in accs]}  mean={np.mean(accs):.4f}±{np.std(accs):.4f}")
    print(f"F1:   {[f'{f:.4f}' for f in f1s]}  mean={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    for cn in CLASS_MAP:
        rs = [r["recall"][cn] for r in fold_results]
        print(f"  {cn}: {[f'{r:.3f}' for r in rs]} → {np.mean(rs):.3f}")

    # --- Val inference ---
    print(f"\n{'='*60}\nVAL PREDICTIONS\n{'='*60}")
    print("Loading val data...")
    val_images, val_fnames, val_is_black = load_all_val(CFG["val_dir"], band_mean, band_std)
    val_images = val_images.to(device)

    all_fold_probs = []
    for fold in range(CFG["n_folds"]):
        model = MSCNNv3(in_ch=11, num_classes=3).to(device)
        model.load_state_dict(torch.load(os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"),
                                         weights_only=True, map_location=device))
        model.eval()

        with torch.no_grad():
            if CFG["tta_flips"]:
                tta_versions = [
                    val_images,
                    val_images.flip(3),
                    val_images.flip(2),
                    torch.rot90(val_images, 1, [2, 3]),
                    torch.rot90(val_images, 2, [2, 3]),
                    torch.rot90(val_images, 3, [2, 3]),
                    val_images.flip(3).flip(2),
                    torch.rot90(val_images, 1, [2, 3]).flip(3),
                ]
                all_tta = []
                for v in tta_versions:
                    probs = F.softmax(model(v), dim=1)
                    all_tta.append(probs)
                fold_probs = torch.stack(all_tta).mean(0).cpu().numpy()
            else:
                fold_probs = F.softmax(model(val_images), dim=1).cpu().numpy()

        # Override black images
        for i, ib in enumerate(val_is_black):
            if ib:
                fold_probs[i] = [0.0, 0.0, 1.0]

        all_fold_probs.append(fold_probs)
        print(f"  Fold {fold} done")

    ens_probs = np.mean(all_fold_probs, axis=0)
    ens_preds = np.argmax(ens_probs, axis=1)

    np.save(os.path.join(CFG["output_dir"], "ms_val_probs.npy"), ens_probs)

    pred_classes = [INV_CLASS_MAP[p] for p in ens_preds]
    print(f"Distribution: { {c: pred_classes.count(c) for c in CLASS_MAP} }")

    sub_path = os.path.join(CFG["output_dir"], "ms_submission.csv")
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for fn, cl in zip(val_fnames, pred_classes):
            w.writerow([fn, cl])

    print(f"Saved: {sub_path}")
    print("Done!")


if __name__ == "__main__":
    main()
