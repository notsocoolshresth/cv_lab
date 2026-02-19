"""
MS (5-band Multispectral) CNN Classifier v2 — Wheat Disease Classification
Fixes Health recall collapse via:
  - Focal Loss with per-class weights (Health upweighted)
  - Additional spectral indices: GNDVI, CI_RedEdge, SAVI, Red/Green ratio
  - Squeeze-and-Excitation channel attention
  - Mixup augmentation for better decision boundaries
  - Per-class recall tracking + best model selection on macro F1
"""

import os
import sys
import json
import csv
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
import tifffile as tiff

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
CFG = {
    "data_dir": "Kaggle_Prepared/train/MS",
    "val_dir": "Kaggle_Prepared/val/MS",
    "output_dir": "ms_models_v2",
    "n_folds": 5,
    "epochs": 100,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 20,
    "seed": 42,
    "n_bands": 5,
    "img_size": 64,
    "num_classes": 3,
    "tta_flips": True,
    # Class weights: boost Health (loses 9 samples to black images + spectrally ambiguous with Rust)
    "class_weights": [1.3, 0.9, 1.0],
    # Mixup (off by default)
    "use_mixup": False,
    "mixup_alpha": 0.2,
    "mixup_prob": 0.3,
    # Label smoothing
    "label_smoothing": 0.05,
}

CLASS_MAP = {"Health": 0, "Rust": 1, "Other": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Spectral Feature Engineering
# ============================================================
def compute_spectral_indices(img_normed):
    """
    Compute vegetation/spectral indices from normalized 5-band MS data.
    Input: img_normed (5, H, W) — z-score normalized
    Returns: (11, H, W) — 5 original + 6 indices
    
    Indices designed to separate Health vs Rust:
    - NDVI:      (NIR - Red) / (NIR + Red)           — general vegetation vigor
    - NDRE:      (NIR - RE) / (NIR + RE)              — chlorophyll/canopy
    - GNDVI:     (NIR - Green) / (NIR + Green)        — green chlorophyll
    - CI_RE:     NIR / RedEdge - 1                    — chlorophyll index red edge
    - SAVI:      1.5*(NIR - Red) / (NIR + Red + 0.5)  — soil-adjusted
    - RG_ratio:  Red / Green                          — disease reddening indicator
    """
    blue, green, red, red_edge, nir = img_normed[0], img_normed[1], img_normed[2], img_normed[3], img_normed[4]
    eps = 1e-8

    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - red_edge) / (nir + red_edge + eps)
    gndvi = (nir - green) / (nir + green + eps)
    ci_re = nir / (red_edge + eps) - 1.0
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    rg_ratio = red / (green + eps)

    # Clip extreme values from ratio indices
    ci_re = np.clip(ci_re, -5.0, 5.0)
    rg_ratio = np.clip(rg_ratio, -5.0, 5.0)

    indices = np.stack([ndvi, ndre, gndvi, ci_re, savi, rg_ratio], axis=0)  # (6, H, W)
    return np.concatenate([img_normed, indices], axis=0)  # (11, H, W)


# ============================================================
# Losses
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss with per-class weights and optional label smoothing."""

    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits, targets):
        num_classes = logits.size(1)

        # Label smoothing: convert hard labels to soft
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Per-class weighting
        if self.weight is not None:
            w = self.weight.to(logits.device)
            class_weight = w[targets].unsqueeze(1)  # (B, 1)
            focal_weight = focal_weight * class_weight

        loss = -focal_weight * smooth_targets * log_probs
        return loss.sum(dim=1).mean()


# ============================================================
# Dataset
# ============================================================
class MSDataset(Dataset):
    """Loads 5-band MS .tif files with extended spectral indices."""

    def __init__(self, file_paths, labels, band_mean, band_std, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.band_mean = band_mean
        self.band_std = band_std
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = tiff.imread(self.file_paths[idx]).astype(np.float32)  # (64, 64, 5)
        img = img.transpose(2, 0, 1)  # (5, 64, 64)

        # Per-band z-score normalization
        for b in range(5):
            img[b] = (img[b] - self.band_mean[b]) / (self.band_std[b] + 1e-8)

        # Compute spectral indices → (11, 64, 64)
        img = compute_spectral_indices(img)

        img = torch.from_numpy(img)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img = torch.flip(img, [2])
            if random.random() > 0.5:
                img = torch.flip(img, [1])
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, [1, 2])
            # Gaussian noise
            if random.random() > 0.5:
                noise = torch.randn_like(img) * 0.03
                img = img + noise
            # Band dropout: randomly zero one original band (not indices)
            if random.random() > 0.85:
                drop_band = random.randint(0, 4)
                img[drop_band] = 0.0

        return img, self.labels[idx]


class MSValDataset(Dataset):
    """Loads val MS .tif files for inference."""

    def __init__(self, file_paths, filenames, band_mean, band_std):
        self.file_paths = file_paths
        self.filenames = filenames
        self.band_mean = band_mean
        self.band_std = band_std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = tiff.imread(self.file_paths[idx]).astype(np.float32)

        if img.mean() < 1.0:
            return torch.zeros(11, 64, 64), self.filenames[idx], True

        img = img.transpose(2, 0, 1)
        for b in range(5):
            img[b] = (img[b] - self.band_mean[b]) / (self.band_std[b] + 1e-8)

        img = compute_spectral_indices(img)
        return torch.from_numpy(img), self.filenames[idx], False


# ============================================================
# Model: CNN with Squeeze-and-Excitation Attention
# ============================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns per-channel importance weights."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU → SE → Pool → Dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.1, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.convs = nn.Sequential(*layers)
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.convs(x)
        x = self.se(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class MSCNNv2(nn.Module):
    """CNN with SE attention for 11-channel 64×64 multispectral input."""

    def __init__(self, in_channels=11, num_classes=3):
        super().__init__()

        # Spectral embedding: 1×1 conv to mix bands before spatial processing
        self.spectral_mix = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = ConvBlock(32, 48, dropout=0.1)    # 64→32
        self.block2 = ConvBlock(48, 96, dropout=0.15)    # 32→16
        self.block3 = ConvBlock(96, 192, dropout=0.2)    # 16→8
        self.block4 = ConvBlock(192, 256, dropout=0.2, pool=False)  # 8→8

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.spectral_mix(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ============================================================
# Mixup
# ============================================================
def mixup_data(x, y, alpha=0.3):
    """Mixup: creates convex combinations of training pairs."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# Training utilities
# ============================================================
def compute_band_stats(file_paths):
    all_bands = []
    for fp in file_paths:
        img = tiff.imread(fp).astype(np.float32)
        if img.mean() < 1.0:
            continue
        img = img.transpose(2, 0, 1)
        all_bands.append(img.reshape(5, -1))
    all_bands = np.concatenate(all_bands, axis=1)
    return all_bands.mean(axis=1), all_bands.std(axis=1)


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True, alpha=0.3, mixup_prob=0.5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup and random.random() < mixup_prob:
            mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha)
            outputs = model(mixed_imgs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            # For accuracy tracking, use the dominant label
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(y_a).sum().item() +
                        (1 - lam) * predicted.eq(y_b).sum().item())
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Per-class recall
    per_class_recall = {}
    for cls_idx, cls_name in INV_CLASS_MAP.items():
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            per_class_recall[cls_name] = (all_preds[mask] == cls_idx).mean()
        else:
            per_class_recall[cls_name] = 0.0

    return total_loss / len(all_labels), acc, macro_f1, per_class_recall, all_preds, all_labels


# ============================================================
# Main
# ============================================================
def main():
    seed_everything(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CFG["output_dir"], exist_ok=True)

    # --- Load training data ---
    data_dir = CFG["data_dir"]
    all_files = sorted(os.listdir(data_dir))

    file_paths = []
    labels = []
    skipped_black = []

    for f in all_files:
        fp = os.path.join(data_dir, f)
        cls_name = f.split("_hyper_")[0]
        label = CLASS_MAP[cls_name]

        img = tiff.imread(fp)
        if img.mean() < 1.0:
            skipped_black.append(f)
            continue

        file_paths.append(fp)
        labels.append(label)

    labels = np.array(labels)
    print(f"Loaded {len(file_paths)} training samples (skipped {len(skipped_black)} black images)")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {INV_CLASS_MAP[u]}: {c}")

    # --- Compute per-band stats ---
    print("Computing per-band statistics...")
    band_mean, band_std = compute_band_stats(file_paths)
    print(f"Band means: {band_mean}")
    print(f"Band stds:  {band_std}")

    stats = {"band_mean": band_mean.tolist(), "band_std": band_std.tolist()}
    with open(os.path.join(CFG["output_dir"], "band_stats.json"), "w") as f_out:
        json.dump(stats, f_out)

    in_channels = 11  # 5 bands + 6 spectral indices

    # --- Loss: weighted CE with label smoothing ---
    class_w = torch.tensor(CFG["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_w,
        label_smoothing=CFG["label_smoothing"],
    )
    val_criterion = nn.CrossEntropyLoss()

    # --- K-Fold CV ---
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{CFG['n_folds']}")
        print(f"{'='*60}")

        train_fps = [file_paths[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_fps = [file_paths[i] for i in val_idx]
        val_labels = labels[val_idx]

        # Print fold class distribution
        for cls_idx in range(3):
            n_train = (train_labels == cls_idx).sum()
            n_val = (val_labels == cls_idx).sum()
            print(f"  {INV_CLASS_MAP[cls_idx]}: train={n_train}, val={n_val}")

        train_ds = MSDataset(train_fps, train_labels, band_mean, band_std, augment=True)
        val_ds = MSDataset(val_fps, val_labels, band_mean, band_std, augment=False)

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"],
                                shuffle=False, num_workers=2, pin_memory=True)

        model = MSCNNv2(in_channels=in_channels, num_classes=CFG["num_classes"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                      weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["epochs"], eta_min=1e-6
        )

        best_macro_f1 = 0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(CFG["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                use_mixup=CFG["use_mixup"], alpha=CFG["mixup_alpha"], mixup_prob=CFG["mixup_prob"]
            )
            val_loss, val_acc, macro_f1, recall, val_preds, val_true = validate(
                model, val_loader, val_criterion, device
            )
            scheduler.step()

            # Select on macro F1 (prevents Health recall collapse)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(),
                           os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"))
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0 or macro_f1 >= best_macro_f1 - 0.01:
                recall_str = " | ".join(f"{k}:{v:.3f}" for k, v in recall.items())
                print(f"  Ep {epoch+1:3d} | TrL: {train_loss:.4f} TrA: {train_acc:.4f} | "
                      f"VL: {val_loss:.4f} VA: {val_acc:.4f} F1: {macro_f1:.4f} | "
                      f"R: {recall_str} | Best F1: {best_macro_f1:.4f} (ep {best_epoch})")

            if patience_counter >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Reload best model
        model.load_state_dict(torch.load(os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"),
                                         weights_only=True))
        _, final_acc, final_f1, final_recall, final_preds, final_true = validate(
            model, val_loader, val_criterion, device
        )

        fold_results.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "val_acc": final_acc,
            "macro_f1": final_f1,
            "recall": final_recall,
        })

        print(f"\n  Fold {fold+1} Best: Acc={final_acc:.4f} F1={final_f1:.4f} (epoch {best_epoch})")
        print(f"  Per-class recall: {final_recall}")
        print(classification_report(final_true, final_preds,
                                    target_names=list(CLASS_MAP.keys()), digits=4))

    # --- Summary ---
    print("\n" + "="*60)
    print("CV RESULTS SUMMARY")
    print("="*60)
    accs = [r["val_acc"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    print(f"Per-fold accuracy: {[f'{a:.4f}' for a in accs]}")
    print(f"Per-fold macro F1: {[f'{f:.4f}' for f in f1s]}")
    print(f"Mean CV Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean CV Macro F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Per-class recall across folds
    for cls_name in CLASS_MAP:
        recalls = [r["recall"][cls_name] for r in fold_results]
        print(f"  {cls_name} recall: {[f'{r:.3f}' for r in recalls]} → mean {np.mean(recalls):.3f}")

    # --- Inference on validation set ---
    print("\n" + "="*60)
    print("GENERATING VALIDATION PREDICTIONS")
    print("="*60)

    val_dir = CFG["val_dir"]
    val_files = sorted(os.listdir(val_dir))
    val_fps = [os.path.join(val_dir, f) for f in val_files]

    val_dataset = MSValDataset(val_fps, val_files, band_mean, band_std)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_fold_probs = []

    for fold in range(CFG["n_folds"]):
        model = MSCNNv2(in_channels=in_channels, num_classes=CFG["num_classes"]).to(device)
        model.load_state_dict(torch.load(os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"),
                                         weights_only=True, map_location=device))
        model.eval()

        fold_probs = []

        with torch.no_grad():
            for imgs, fnames, is_black in val_dataloader:
                if is_black.item():
                    probs = torch.tensor([[0.0, 0.0, 1.0]])
                else:
                    imgs = imgs.to(device)

                    if CFG["tta_flips"]:
                        tta_imgs = [
                            imgs,
                            torch.flip(imgs, [3]),
                            torch.flip(imgs, [2]),
                            torch.rot90(imgs, 1, [2, 3]),
                            torch.rot90(imgs, 2, [2, 3]),
                            torch.rot90(imgs, 3, [2, 3]),
                            torch.flip(torch.rot90(imgs, 1, [2, 3]), [3]),
                            torch.flip(torch.rot90(imgs, 1, [2, 3]), [2]),
                        ]
                        tta_probs = []
                        for t_img in tta_imgs:
                            out = model(t_img)
                            tta_probs.append(F.softmax(out, dim=1))
                        probs = torch.stack(tta_probs).mean(0).cpu()
                    else:
                        out = model(imgs)
                        probs = F.softmax(out, dim=1).cpu()

                fold_probs.append(probs.numpy())

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_fold_probs.append(fold_probs)
        print(f"  Fold {fold} predictions: shape {fold_probs.shape}")

    ensemble_probs = np.mean(all_fold_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    np.save(os.path.join(CFG["output_dir"], "ms_val_probs.npy"), ensemble_probs)

    pred_classes = [INV_CLASS_MAP[p] for p in ensemble_preds]
    dist = {c: pred_classes.count(c) for c in CLASS_MAP}
    print(f"\nPrediction distribution: {dist}")

    submission_path = os.path.join(CFG["output_dir"], "ms_submission.csv")
    with open(submission_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Id", "Category"])
        for fname, cls in zip(val_files, pred_classes):
            writer.writerow([fname, cls])

    print(f"Submission saved to {submission_path}")
    print(f"Probabilities saved to {os.path.join(CFG['output_dir'], 'ms_val_probs.npy')}")
    print("\nDone!")


if __name__ == "__main__":
    main()
