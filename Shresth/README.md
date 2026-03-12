# Wheat Disease Classification

A deep learning pipeline for classifying wheat diseases into 3 classes — **Healthy**, **Other**, and **Rust** — using multimodal imagery (RGB, Multispectral, and Hyperspectral).

---

## Classes

| Label | Description |
|-------|-------------|
| Health | Healthy wheat |
| Rust | Wheat with rust disease |
| Other | Other conditions |

---

## Dataset Structure

```
Kaggle_prepared/
├── train/
│   ├── RGB/          # PNG images (3 channels)
│   ├── MS/           # Multispectral TIF images (5 channels)
│   └── HS/           # Hyperspectral TIF images
└── val/
    ├── RGB/
    ├── MS/
    └── HS/
```

Filenames follow the pattern: `{ClassName}_{type}_{index}.{ext}`
e.g., `Health_hyper_1.png`, `Rust_multi_3.tif`

To split raw training data into train/val:

```bash
python split_dataset.py
```

---

## Models

| Model | File | Description |
|-------|------|-------------|
| `WheatCNN` | `model.py` | Custom 4-block CNN baseline |
| `WheatResNet` | `model.py` | Transfer learning with ResNet-18/34/50 |
| Multimodal ConvNeXtV2 | `train_convnextv2_cv.py` | Production-grade multimodal model using `facebook/convnextv2-tiny-1k-224` |

### Multimodal ConvNeXtV2 Architecture

- Channel adapters (1×1 conv) project MS and HS bands to 3 channels
- Shared ConvNeXtV2 backbone extracts features from all three modalities
- Feature fusion layer combines RGB + MS + HS representations
- 3-class classifier head

**Training Phases:**
1. Train adapters + fusion + classifier only (backbone frozen) — 5 epochs
2. Unfreeze last ConvNeXt stage — 10 epochs
3. Full fine-tuning with differential learning rates — 25 epochs

---

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies:**

- `torch >= 2.0.0`, `torchvision >= 0.15.0`
- `rasterio >= 1.3.0` — for loading multispectral/hyperspectral TIF files
- `transformers`, `timm` — for ConvNeXtV2 backbone
- `albumentations` — augmentation in the advanced pipeline
- `xgboost`, `scikit-learn` — for ensemble/feature-based experiments
- `tensorboard` — training visualization

---

## Usage

### 1. Prepare the dataset

```bash
python split_dataset.py
```

### 2. Train (simple pipeline)

```bash
python train.py
```

### 3. Train (multimodal ConvNeXtV2 with cross-validation)

```bash
python train_convnextv2_cv.py
```

### 4. Run usage examples

```bash
python example.py
```

---

## File Overview

| File | Purpose |
|------|---------|
| `dataset.py` | `WheatDataset` class and `get_data_loaders()` helper |
| `model.py` | `WheatCNN`, `WheatResNet`, and multimodal model definitions |
| `train.py` | General trainer with TensorBoard logging and checkpoint saving |
| `train_convnextv2_cv.py` | Full production pipeline with stratified K-fold CV |
| `split_dataset.py` | Utility to split train data into train/val |
| `example.py` | Usage walkthrough for dataset loading and inference |
| `code.ipynb` / `newcode.ipynb` | Experimental notebooks |
| `vit_xgboost_notebook.ipynb` | ViT + XGBoost ensemble experiments |

---

## Checkpoints

Pretrained weights are saved under `checkpoints/` and the root directory:

| File | Description |
|------|-------------|
| `best_multimodal_model.pth` | Best multimodal ConvNeXtV2 model |
| `best_clean_model.pth` | Best clean single-modality model |
| `best_simple_model.pth` | Best simple CNN model |
| `best_health_binary.pth` | Binary classifier (Healthy vs. Not) |
| `feature_extractor.pth` | Pretrained feature extractor backbone |
| `checkpoints/best_model.pth` | Latest best checkpoint |

---

## Logging

TensorBoard logs are written to the `logs/` directory.

```bash
tensorboard --logdir logs/
```
