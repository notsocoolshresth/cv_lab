"""
MS (5-band Multispectral) CNN Classifier for Wheat Disease Classification
- 5-fold stratified CV with lightweight CNN
- Per-band z-score normalization
- Augmentation: flips, rotations, noise, band dropout
- TTA at inference
- Outputs soft probabilities for late fusion with RGB ensemble
"""

import os
import sys
import json
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import tifffile as tiff
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
CFG = {
    "data_dir": "Kaggle_Prepared/train/MS",
    "val_dir": "Kaggle_Prepared/val/MS",
    "output_dir": "ms_models",
    "n_folds": 5,
    "epochs": 80,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 15,
    "seed": 42,
    "n_bands": 5,
    "img_size": 64,
    "num_classes": 3,
    "use_veg_indices": True,  # append NDVI, NDRE as extra channels
    "tta_flips": True,
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
# Dataset
# ============================================================
class MSDataset(Dataset):
    """Loads 5-band MS .tif files, optionally appends vegetation indices."""

    def __init__(self, file_paths, labels, band_mean, band_std,
                 augment=False, use_veg_indices=True):
        self.file_paths = file_paths
        self.labels = labels
        self.band_mean = band_mean  # (C,)
        self.band_std = band_std    # (C,)
        self.augment = augment
        self.use_veg_indices = use_veg_indices

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = tiff.imread(self.file_paths[idx]).astype(np.float32)  # (64, 64, 5)
        img = img.transpose(2, 0, 1)  # (5, 64, 64) — C, H, W

        # Per-band z-score normalization
        for b in range(img.shape[0]):
            img[b] = (img[b] - self.band_mean[b]) / (self.band_std[b] + 1e-8)

        # Compute vegetation indices before augmentation
        if self.use_veg_indices:
            # MS bands: 0=Blue, 1=Green, 2=Red, 3=RedEdge, 4=NIR
            red = img[2]
            red_edge = img[3]
            nir = img[4]

            ndvi = (nir - red) / (nir + red + 1e-8)
            ndre = (nir - red_edge) / (nir + red_edge + 1e-8)

            img = np.concatenate([img, ndvi[None], ndre[None]], axis=0)  # (7, 64, 64)

        img = torch.from_numpy(img)

        # Augmentation (spatial only — applied to all channels uniformly)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, [2])
            # Random vertical flip
            if random.random() > 0.5:
                img = torch.flip(img, [1])
            # Random 90-degree rotation
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, [1, 2])
            # Gaussian noise (small)
            if random.random() > 0.5:
                noise = torch.randn_like(img) * 0.05
                img = img + noise
            # Band dropout: randomly zero out one of the original 5 bands
            if random.random() > 0.85:
                drop_band = random.randint(0, 4)
                img[drop_band] = 0.0

        label = self.labels[idx]
        return img, label


class MSValDataset(Dataset):
    """Loads val MS .tif files for inference."""

    def __init__(self, file_paths, filenames, band_mean, band_std, use_veg_indices=True):
        self.file_paths = file_paths
        self.filenames = filenames
        self.band_mean = band_mean
        self.band_std = band_std
        self.use_veg_indices = use_veg_indices

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = tiff.imread(self.file_paths[idx]).astype(np.float32)  # (64, 64, 5)

        # Handle black images
        if img.mean() < 1.0:
            n_channels = 7 if self.use_veg_indices else 5
            return torch.zeros(n_channels, 64, 64), self.filenames[idx], True

        img = img.transpose(2, 0, 1)  # (5, 64, 64)

        for b in range(img.shape[0]):
            img[b] = (img[b] - self.band_mean[b]) / (self.band_std[b] + 1e-8)

        if self.use_veg_indices:
            red = img[2]
            red_edge = img[3]
            nir = img[4]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndre = (nir - red_edge) / (nir + red_edge + 1e-8)
            img = np.concatenate([img, ndvi[None], ndre[None]], axis=0)

        return torch.from_numpy(img), self.filenames[idx], False


# ============================================================
# Model
# ============================================================
class MSCNN(nn.Module):
    """Lightweight CNN for 5/7-channel 64x64 multispectral classification."""

    def __init__(self, in_channels=7, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global avg pool -> (256, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Training utilities
# ============================================================
def compute_band_stats(file_paths):
    """Compute per-band mean and std from training files (excluding black images)."""
    all_bands = []
    for fp in file_paths:
        img = tiff.imread(fp).astype(np.float32)  # (64, 64, 5)
        if img.mean() < 1.0:
            continue
        img = img.transpose(2, 0, 1)  # (5, 64, 64)
        all_bands.append(img.reshape(5, -1))  # (5, 4096)

    all_bands = np.concatenate(all_bands, axis=1)  # (5, N*4096)
    band_mean = all_bands.mean(axis=1)  # (5,)
    band_std = all_bands.std(axis=1)    # (5,)
    return band_mean, band_std


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(all_labels), acc, np.array(all_preds), np.array(all_labels)


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

        # Filter black images
        img = tiff.imread(fp)
        if img.mean() < 1.0:
            skipped_black.append(f)
            continue

        file_paths.append(fp)
        labels.append(label)

    labels = np.array(labels)
    print(f"Loaded {len(file_paths)} training samples (skipped {len(skipped_black)} black images)")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # --- Compute per-band normalization stats ---
    print("Computing per-band statistics...")
    band_mean, band_std = compute_band_stats(file_paths)
    print(f"Band means: {band_mean}")
    print(f"Band stds:  {band_std}")

    # Save stats for inference
    stats = {"band_mean": band_mean.tolist(), "band_std": band_std.tolist()}
    with open(os.path.join(CFG["output_dir"], "band_stats.json"), "w") as f:
        json.dump(stats, f)

    in_channels = 7 if CFG["use_veg_indices"] else 5

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

        train_ds = MSDataset(train_fps, train_labels, band_mean, band_std,
                             augment=True, use_veg_indices=CFG["use_veg_indices"])
        val_ds = MSDataset(val_fps, val_labels, band_mean, band_std,
                           augment=False, use_veg_indices=CFG["use_veg_indices"])

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"],
                                shuffle=False, num_workers=2, pin_memory=True)

        model = MSCNN(in_channels=in_channels, num_classes=CFG["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                      weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(CFG["epochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_true = validate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(),
                           os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"))
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0 or val_acc > best_val_acc - 0.001:
                print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Best: {best_val_acc:.4f} (ep {best_epoch})")

            if patience_counter >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Reload best model and get final val predictions
        model.load_state_dict(torch.load(os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"),
                                         weights_only=True))
        _, final_acc, final_preds, final_true = validate(model, val_loader, criterion, device)

        fold_results.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "val_acc": final_acc,
        })

        print(f"\n  Fold {fold+1} Best Acc: {final_acc:.4f} (epoch {best_epoch})")
        print(classification_report(final_true, final_preds,
                                    target_names=list(CLASS_MAP.keys()), digits=4))

    # --- Summary ---
    print("\n" + "="*60)
    print("CV RESULTS SUMMARY")
    print("="*60)
    accs = [r["val_acc"] for r in fold_results]
    print(f"Per-fold accuracy: {[f'{a:.4f}' for a in accs]}")
    print(f"Mean CV Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # --- Inference on validation set ---
    print("\n" + "="*60)
    print("GENERATING VALIDATION PREDICTIONS")
    print("="*60)

    val_dir = CFG["val_dir"]
    val_files = sorted(os.listdir(val_dir))
    val_fps = [os.path.join(val_dir, f) for f in val_files]

    val_dataset = MSValDataset(val_fps, val_files, band_mean, band_std,
                               use_veg_indices=CFG["use_veg_indices"])
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Collect predictions from all fold models
    all_fold_probs = []

    for fold in range(CFG["n_folds"]):
        model = MSCNN(in_channels=in_channels, num_classes=CFG["num_classes"]).to(device)
        model.load_state_dict(torch.load(os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"),
                                         weights_only=True, map_location=device))
        model.eval()

        fold_probs = []

        with torch.no_grad():
            for imgs, fnames, is_black in val_dataloader:
                # is_black is a tensor of booleans for the batch
                if is_black.item():
                    # Black image → predict "Other" with high confidence
                    probs = torch.tensor([[0.0, 0.0, 1.0]])  # Other = class 2
                else:
                    imgs = imgs.to(device)

                    if CFG["tta_flips"]:
                        # TTA: original + hflip + vflip + rot90
                        tta_imgs = [
                            imgs,
                            torch.flip(imgs, [3]),       # hflip
                            torch.flip(imgs, [2]),       # vflip
                            torch.rot90(imgs, 1, [2, 3]),  # 90 deg
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

        fold_probs = np.concatenate(fold_probs, axis=0)  # (300, 3)
        all_fold_probs.append(fold_probs)
        print(f"  Fold {fold} predictions done: shape {fold_probs.shape}")

    # Average across folds
    ensemble_probs = np.mean(all_fold_probs, axis=0)  # (300, 3)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    # Save soft probabilities for late fusion
    np.save(os.path.join(CFG["output_dir"], "ms_val_probs.npy"), ensemble_probs)

    # Save submission CSV
    pred_classes = [INV_CLASS_MAP[p] for p in ensemble_preds]
    print(f"\nPrediction distribution: { {c: pred_classes.count(c) for c in CLASS_MAP} }")

    import csv
    submission_path = os.path.join(CFG["output_dir"], "ms_submission.csv")
    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        for fname, cls in zip(val_files, pred_classes):
            writer.writerow([fname, cls])

    print(f"Submission saved to {submission_path}")
    print(f"Probabilities saved to {os.path.join(CFG['output_dir'], 'ms_val_probs.npy')}")
    print("\nDone!")


if __name__ == "__main__":
    main()
