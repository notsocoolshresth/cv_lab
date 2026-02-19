from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from data import CLASS_MAP, MultiModalAugmentation, WheatMultiModalDataset, hs_mixup
from losses import SupConLoss
from models import HSBranch3D, MSBranch


INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


CFG = {
    "root": "Kaggle_Prepared",
    "output_dir": "sst_cmf_output",
    "n_folds": 5,
    "epochs": 20,
    "batch_size": 16,
    "num_workers": 2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "seed": 42,
    "num_classes": 3,
    "ce_weight": 0.9,
    "supcon_weight": 0.1,
    "mixup_alpha": 0.2,
    "class_weights": [1.4, 1.0, 1.0],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Batch:
    ms: torch.Tensor
    hs: torch.Tensor
    labels: torch.Tensor


class SSTCMFNet(nn.Module):
    def __init__(self, num_classes: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        self.hs_branch = HSBranch3D(in_bands=100, embed_dim=embed_dim)
        self.ms_branch = MSBranch(in_channels=5, embed_dim=embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, ms: torch.Tensor, hs: torch.Tensor) -> Dict[str, torch.Tensor]:
        hs_out = self.hs_branch(hs)
        ms_out = self.ms_branch(ms)

        fused = self.fusion(torch.cat([hs_out["embedding"], ms_out["embedding"]], dim=1))
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "embedding": fused,
            "hs_embedding": hs_out["embedding"],
            "ms_embedding": ms_out["embedding"],
        }


def make_loader(dataset, indices, batch_size, num_workers, shuffle):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def to_device(batch, device) -> Batch:
    return Batch(
        ms=batch["ms"].to(device, non_blocking=True),
        hs=batch["hs"].to(device, non_blocking=True),
        labels=batch["label"].to(device, non_blocking=True),
    )


def train_one_epoch(model, loader, optimizer, ce_loss_fn, supcon_loss_fn, device):
    model.train()
    running_loss = 0.0
    all_true, all_pred = [], []

    for raw_batch in loader:
        b = to_device(raw_batch, device)

        mixed_hs, y_a, y_b, lam = hs_mixup(b.hs, b.labels, alpha=CFG["mixup_alpha"])
        out = model(b.ms, mixed_hs)

        ce_a = ce_loss_fn(out["logits"], y_a)
        ce_b = ce_loss_fn(out["logits"], y_b)
        ce_loss = lam * ce_a + (1 - lam) * ce_b
        supcon_loss = supcon_loss_fn(out["embedding"], b.labels)
        loss = CFG["ce_weight"] * ce_loss + CFG["supcon_weight"] * supcon_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * b.labels.size(0)
        preds = out["logits"].argmax(dim=1)
        all_true.extend(b.labels.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())

    n = max(1, len(all_true))
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    return running_loss / n, acc, f1


@torch.no_grad()
def evaluate(model, loader, ce_loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_true, all_pred = [], []
    all_probs = []

    for raw_batch in loader:
        b = to_device(raw_batch, device)
        out = model(b.ms, b.hs)
        loss = ce_loss_fn(out["logits"], b.labels)

        running_loss += loss.item() * b.labels.size(0)
        probs = out["logits"].softmax(dim=1)
        preds = probs.argmax(dim=1)

        all_true.extend(b.labels.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    n = max(1, len(all_true))
    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3), dtype=np.float32)
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro")
    recall = recall_score(all_true, all_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return running_loss / n, acc, f1, recall, all_probs


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_probs = []
    for raw_batch in loader:
        ms = raw_batch["ms"].to(device, non_blocking=True)
        hs = raw_batch["hs"].to(device, non_blocking=True)
        probs = model(ms, hs)["logits"].softmax(dim=1)
        all_probs.append(probs.detach().cpu().numpy())
    if not all_probs:
        return np.zeros((0, CFG["num_classes"]), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


def build_datasets():
    train_root = os.path.join(CFG["root"], "train")
    val_root = os.path.join(CFG["root"], "val")

    stats = WheatMultiModalDataset.compute_train_stats(
        ms_dir=os.path.join(train_root, "MS"),
        hs_dir=os.path.join(train_root, "HS"),
        rgb_dir=os.path.join(train_root, "RGB"),
        hs_band_range=(10, 110),
        hs_resize_to=(64, 64),
    )

    train_aug = MultiModalAugmentation()

    train_ds = WheatMultiModalDataset.from_split_root(
        root=CFG["root"],
        split="train",
        stats=stats,
        is_train=True,
        transform=train_aug,
        hs_band_range=(10, 110),
        hs_resize_to=(64, 64),
    )
    train_eval_ds = WheatMultiModalDataset.from_split_root(
        root=CFG["root"],
        split="train",
        stats=stats,
        is_train=True,
        transform=None,
        hs_band_range=(10, 110),
        hs_resize_to=(64, 64),
    )
    val_ds = WheatMultiModalDataset.from_split_root(
        root=CFG["root"],
        split="val",
        stats=stats,
        is_train=False,
        transform=None,
        hs_band_range=(10, 110),
        hs_resize_to=(64, 64),
    )

    with open(os.path.join(CFG["output_dir"], "band_stats.json"), "w") as f:
        json.dump(
            {
                "ms_mean": stats["ms"].mean.tolist(),
                "ms_std": stats["ms"].std.tolist(),
                "hs_mean": stats["hs"].mean.tolist(),
                "hs_std": stats["hs"].std.tolist(),
                "rgb_mean": stats["rgb"].mean.tolist(),
                "rgb_std": stats["rgb"].std.tolist(),
            },
            f,
            indent=2,
        )

    return train_ds, train_eval_ds, val_ds


def run_training():
    seed_everything(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    train_ds, train_eval_ds, val_ds = build_datasets()

    labels = np.array([train_eval_ds[i]["label"] for i in range(len(train_eval_ds))], dtype=np.int64)
    indices = np.arange(len(train_eval_ds))

    device = CFG["device"]
    class_weights = torch.tensor(CFG["class_weights"], dtype=torch.float32, device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    oof_probs = np.zeros((len(train_eval_ds), CFG["num_classes"]), dtype=np.float32)
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

    for fold, (train_idx, valid_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n=== Fold {fold + 1}/{CFG['n_folds']} ===")
        model = SSTCMFNet(num_classes=CFG["num_classes"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        supcon_loss_fn = SupConLoss(temperature=0.1, queue_size=512)

        train_loader = make_loader(train_ds, train_idx, CFG["batch_size"], CFG["num_workers"], shuffle=True)
        valid_loader = make_loader(train_eval_ds, valid_idx, CFG["batch_size"], CFG["num_workers"], shuffle=False)

        best_f1 = -1.0
        best_path = os.path.join(CFG["output_dir"], f"fold{fold}_best.pt")

        for epoch in range(1, CFG["epochs"] + 1):
            tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, supcon_loss_fn, device)
            va_loss, va_acc, va_f1, va_recall, va_probs = evaluate(model, valid_loader, ce_loss_fn, device)

            recall_str = " ".join([f"{INV_CLASS_MAP[i]}:{va_recall[i]:.3f}" for i in range(3)])
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f} | {recall_str}"
            )

            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path, map_location=device))
        _, _, final_f1, _, final_probs = evaluate(model, valid_loader, ce_loss_fn, device)
        oof_probs[valid_idx] = final_probs
        print(f"Fold {fold} best macro-F1: {final_f1:.4f}")

    np.save(os.path.join(CFG["output_dir"], "oof_probs.npy"), oof_probs)

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    fold_val_probs: List[np.ndarray] = []
    for fold in range(CFG["n_folds"]):
        model = SSTCMFNet(num_classes=CFG["num_classes"]).to(device)
        model.load_state_dict(torch.load(os.path.join(CFG["output_dir"], f"fold{fold}_best.pt"), map_location=device))
        fold_probs = predict_loader(model, val_loader, device)
        fold_val_probs.append(fold_probs)

    val_probs_final = np.mean(fold_val_probs, axis=0)
    np.save(os.path.join(CFG["output_dir"], "val_probs_final.npy"), val_probs_final)

    submission_path = os.path.join(CFG["output_dir"], "submission.csv")
    preds = val_probs_final.argmax(axis=1)
    with open(submission_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Category"])
        for i, sample in enumerate(val_ds.samples):
            w.writerow([sample["fname"], INV_CLASS_MAP[int(preds[i])]])

    cfg_path = os.path.join(CFG["output_dir"], "sst_cmf_config.json")
    with open(cfg_path, "w") as f:
        json.dump(CFG, f, indent=2)

    print("\nSaved outputs:")
    print(f"- {os.path.join(CFG['output_dir'], 'oof_probs.npy')}")
    print(f"- {os.path.join(CFG['output_dir'], 'val_probs_final.npy')}")
    print(f"- {submission_path}")


if __name__ == "__main__":
    run_training()
