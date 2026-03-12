"""
Production-Grade Multimodal ConvNeXtV2 Training Pipeline for 3-Class Image Classification
Optimized for small datasets (480 images) with aggressive augmentation and CV strategy

Multimodal Architecture:
- Uses RGB, Multispectral (MS), and Hyperspectral (HS) images
- Channel adapters (1x1 convolutions) project MS and HS to 3 channels
- Shared ConvNeXtV2 backbone extracts features from all modalities
- Feature fusion layer combines all three modality features
- Final classifier head for 3-class prediction

Training Strategy:
- Phase 1: Train adapters + fusion + classifier (backbone frozen) - 5 epochs
- Phase 2: Unfreeze last ConvNeXt stage + continue training - 10 epochs
- Phase 3: Full fine-tuning with differential learning rates - 25 epochs
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import cv2

from transformers import AutoImageProcessor, AutoModelForImageClassification
from timm.data.mixup import Mixup

warnings.filterwarnings('ignore')

# ======================== CONFIG ========================
class CFG:
    # Paths
    train_rgb_dir = "Kaggle_prepared/train/RGB"
    train_ms_dir = "Kaggle_prepared/train/MS"
    train_hs_dir = "Kaggle_prepared/train/HS"
    output_dir = "convnextv2_multimodal_outputs"
    
    # Model
    model_name = "facebook/convnextv2-tiny-1k-224"
    num_classes = 3
    image_size = 224
    
    # Modality channels (will be auto-detected but can set defaults)
    rgb_channels = 3
    ms_channels = 5  # Auto-detect
    hs_channels = 125  # Auto-detect
    
    # Training
    n_folds = 5
    epochs = 40
    batch_size = 16
    num_workers = 0  # Set to 0 for Windows compatibility
    
    # Optimization
    head_lr = 1e-3
    last_stage_lr = 1e-4
    backbone_lr = 1e-5
    weight_decay = 5e-4
    warmup_epochs = 5
    
    # Regularization
    label_smoothing = 0.1
    dropout = 0.3
    grad_clip = 1.0
    
    # MixUp/CutMix
    mixup_alpha = 0.4
    cutmix_alpha = 1.0
    mixup_prob = 0.5
    
    # Early Stopping
    patience = 8
    
    # TTA
    tta_transforms = 4  # Number of TTA iterations
    
    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = True  # Automatic Mixed Precision
    
    # Reproducibility
    seed = 42


# ======================== UTILITIES ========================
def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(phase='train'):
    """
    Get augmentation pipeline
    Training: Aggressive augmentation for small dataset
    Validation: Simple resize + center crop
    """
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(height=CFG.image_size, width=CFG.image_size, scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=CFG.image_size, width=CFG.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_tta_transforms():
    """Get TTA augmentation variants"""
    tta_list = []
    
    # Original
    tta_list.append(A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=CFG.image_size, width=CFG.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    # Horizontal Flip
    tta_list.append(A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=CFG.image_size, width=CFG.image_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    # Vertical Flip
    tta_list.append(A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=CFG.image_size, width=CFG.image_size),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    # Slight crop variation
    tta_list.append(A.Compose([
        A.Resize(height=256, width=256),
        A.RandomCrop(height=CFG.image_size, width=CFG.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]))
    
    return tta_list


# ======================== DATASET ========================
class MultiModalImageDataset(Dataset):
    """Custom Dataset for multimodal image classification (RGB + MS + HS)"""
    
    def __init__(self, rgb_paths, ms_paths, hs_paths, labels, transforms=None):
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.hs_paths = hs_paths
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        # Load RGB
        rgb_img = cv2.imread(self.rgb_paths[idx])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Load MS (multispectral) - could be TIFF or multi-channel
        ms_img = cv2.imread(self.ms_paths[idx], cv2.IMREAD_UNCHANGED)
        if len(ms_img.shape) == 2:  # Grayscale
            ms_img = np.stack([ms_img] * 3, axis=-1)
        elif ms_img.shape[-1] > 3:  # Already multi-channel
            pass
        else:  # RGB MS image
            ms_img = cv2.cvtColor(ms_img, cv2.COLOR_BGR2RGB) if ms_img.shape[-1] == 3 else ms_img
        
        # Load HS (hyperspectral) - could be TIFF or multi-channel
        hs_img = cv2.imread(self.hs_paths[idx], cv2.IMREAD_UNCHANGED)
        if len(hs_img.shape) == 2:  # Grayscale
            hs_img = np.stack([hs_img] * 3, axis=-1)
        elif hs_img.shape[-1] > 3:  # Already multi-channel
            pass
        else:  # RGB HS image
            hs_img = cv2.cvtColor(hs_img, cv2.COLOR_BGR2RGB) if hs_img.shape[-1] == 3 else hs_img
        
        # Apply transforms
        if self.transforms:
            augmented_rgb = self.transforms(image=rgb_img)
            augmented_ms = self.transforms(image=ms_img)
            augmented_hs = self.transforms(image=hs_img)
            
            rgb_img = augmented_rgb['image']
            ms_img = augmented_ms['image']
            hs_img = augmented_hs['image']
        
        label = self.labels[idx]
        
        return rgb_img, ms_img, hs_img, label


def prepare_multimodal_data(rgb_dir: str, ms_dir: str, hs_dir: str) -> Tuple[List[str], List[str], List[str], List[int], Dict[int, str], Tuple[int, int, int]]:
    """
    Prepare multimodal dataset by scanning directory structure
    Assumes structure: data_dir/class_name/images
    Returns paths for RGB, MS, HS and detects channel counts
    """
    rgb_paths = []
    ms_paths = []
    hs_paths = []
    labels = []
    
    class_names = sorted(os.listdir(rgb_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    for class_name in class_names:
        rgb_class_dir = os.path.join(rgb_dir, class_name)
        ms_class_dir = os.path.join(ms_dir, class_name)
        hs_class_dir = os.path.join(hs_dir, class_name)
        
        if not os.path.isdir(rgb_class_dir):
            continue
        
        rgb_files = sorted([f for f in os.listdir(rgb_class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        for img_name in rgb_files:
            rgb_path = os.path.join(rgb_class_dir, img_name)
            ms_path = os.path.join(ms_class_dir, img_name)
            hs_path = os.path.join(hs_class_dir, img_name)
            
            # Check if corresponding MS and HS files exist
            if os.path.exists(ms_path) and os.path.exists(hs_path):
                rgb_paths.append(rgb_path)
                ms_paths.append(ms_path)
                hs_paths.append(hs_path)
                labels.append(class_to_idx[class_name])
    
    print(f"Found {len(rgb_paths)} images across {len(class_names)} classes")
    print(f"Class distribution: {dict(pd.Series(labels).value_counts().sort_index())}")
    
    # Detect channel counts from first image of each modality
    if len(rgb_paths) > 0:
        sample_rgb = cv2.imread(rgb_paths[0], cv2.IMREAD_UNCHANGED)
        sample_ms = cv2.imread(ms_paths[0], cv2.IMREAD_UNCHANGED)
        sample_hs = cv2.imread(hs_paths[0], cv2.IMREAD_UNCHANGED)
        
        rgb_channels = 3  # Always 3 for RGB
        ms_channels = sample_ms.shape[-1] if len(sample_ms.shape) == 3 else 1
        hs_channels = sample_hs.shape[-1] if len(sample_hs.shape) == 3 else 1
        
        print(f"\nDetected channels - RGB: {rgb_channels}, MS: {ms_channels}, HS: {hs_channels}")
    else:
        rgb_channels, ms_channels, hs_channels = 3, 3, 3
    
    return rgb_paths, ms_paths, hs_paths, labels, idx_to_class, (rgb_channels, ms_channels, hs_channels)


# ======================== MODEL ========================
class MultiModalConvNeXtV2(nn.Module):
    """
    Multimodal ConvNeXtV2 with separate channel adapters for RGB, MS, HS
    Uses convolutions to project all modalities to compatible dimensions
    """
    
    def __init__(self, model_name: str, num_classes: int, 
                 rgb_channels: int = 3, ms_channels: int = 4, hs_channels: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        
        # Channel adapters - project to 3 channels each
        self.rgb_adapter = nn.Identity() if rgb_channels == 3 else nn.Conv2d(rgb_channels, 3, 1)
        self.ms_adapter = nn.Conv2d(ms_channels, 3, kernel_size=1, bias=False) if ms_channels != 3 else nn.Identity()
        self.hs_adapter = nn.Conv2d(hs_channels, 3, kernel_size=1, bias=False) if hs_channels != 3 else nn.Identity()
        
        # Load pretrained ConvNeXtV2
        self.backbone = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Get feature dimension
        backbone_features = self.backbone.classifier.in_features
        
        # Fusion layer - merge 3 modality features
        self.fusion = nn.Sequential(
            nn.Linear(backbone_features * 3, backbone_features * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(backbone_features * 2, backbone_features),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Replace classifier
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # New classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_features, num_classes)
        )
        
        # Store reference for freezing
        self.convnextv2_backbone = self.backbone.convnextv2
    
    def forward(self, rgb, ms, hs):
        # Adapt channels
        rgb_adapted = self.rgb_adapter(rgb)
        ms_adapted = self.ms_adapter(ms)
        hs_adapted = self.hs_adapter(hs)
        
        # Extract features from each modality
        rgb_features = self.backbone.convnextv2(rgb_adapted).pooler_output
        ms_features = self.backbone.convnextv2(ms_adapted).pooler_output
        hs_features = self.backbone.convnextv2(hs_adapted).pooler_output
        
        # Concatenate features
        fused = torch.cat([rgb_features, ms_features, hs_features], dim=1)
        
        # Fusion
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.convnextv2_backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_last_stage(self):
        """Unfreeze only the last ConvNeXt stage"""
        if hasattr(self.convnextv2_backbone.encoder, 'stages'):
            for param in self.convnextv2_backbone.encoder.stages[-1].parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze entire backbone"""
        for param in self.convnextv2_backbone.parameters():
            param.requires_grad = True
    
    def get_param_groups(self, head_lr, last_stage_lr, backbone_lr):
        """Get parameter groups with differential learning rates"""
        # Adapters and fusion
        adapter_params = list(self.rgb_adapter.parameters()) + \
                        list(self.ms_adapter.parameters()) + \
                        list(self.hs_adapter.parameters()) + \
                        list(self.fusion.parameters())
        
        head_params = list(self.classifier.parameters())
        
        last_stage_params = []
        if hasattr(self.convnextv2_backbone.encoder, 'stages'):
            last_stage_params = list(self.convnextv2_backbone.encoder.stages[-1].parameters())
        
        # All other backbone params
        backbone_param_ids = set(id(p) for p in self.convnextv2_backbone.parameters())
        last_stage_ids = set(id(p) for p in last_stage_params)
        adapter_ids = set(id(p) for p in adapter_params)
        head_ids = set(id(p) for p in head_params)
        
        other_backbone_params = [
            p for p in self.convnextv2_backbone.parameters()
            if id(p) not in last_stage_ids and id(p) not in head_ids and id(p) not in adapter_ids
        ]
        
        return [
            {'params': adapter_params + head_params, 'lr': head_lr},
            {'params': last_stage_params, 'lr': last_stage_lr},
            {'params': other_backbone_params, 'lr': backbone_lr}
        ]


# ======================== MIXUP / CUTMIX ========================
class MultiModalMixupCutmixCollate:
    """Collate function with MixUp and CutMix for multimodal data"""
    
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5, num_classes=3):
        self.mixup = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=prob,
            num_classes=num_classes,
            label_smoothing=CFG.label_smoothing
        )
    
    def __call__(self, batch):
        rgb_images = torch.stack([item[0] for item in batch])
        ms_images = torch.stack([item[1] for item in batch])
        hs_images = torch.stack([item[2] for item in batch])
        labels = torch.tensor([item[3] for item in batch])
        
        # Apply same mixup/cutmix transformation to all modalities
        rgb_mixed, labels_mixed = self.mixup(rgb_images, labels)
        ms_mixed, _ = self.mixup(ms_images, labels)
        hs_mixed, _ = self.mixup(hs_images, labels)
        
        return rgb_mixed, ms_mixed, hs_mixed, labels_mixed


# ======================== TRAINER ========================
class Trainer:
    """Training orchestration class"""
    
    def __init__(self, model, fold: int):
        self.model = model.to(CFG.device)
        self.fold = fold
        self.scaler = GradScaler() if CFG.amp else None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
        
        self.best_score = 0
        self.patience_counter = 0
        
        # Create output directory
        self.output_dir = Path(CFG.output_dir) / f"fold_{fold}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_optimizer_and_scheduler(self, phase: int, num_training_steps: int):
        """Create optimizer and scheduler based on training phase"""
        if phase == 1:
            # Phase 1: Only head parameters (backbone frozen) + adapters
            adapter_params = list(self.model.rgb_adapter.parameters()) + \
                           list(self.model.ms_adapter.parameters()) + \
                           list(self.model.hs_adapter.parameters()) + \
                           list(self.model.fusion.parameters())
            head_params = list(self.model.classifier.parameters())
            params = [{'params': adapter_params + head_params, 'lr': CFG.head_lr}]
        elif phase == 2:
            # Phase 2: Head + last stage
            params = self.model.get_param_groups(CFG.head_lr, CFG.last_stage_lr, 0)
            params = [p for p in params if len(list(p['params'])) > 0]
        else:
            # Phase 3: Full model with differential LR
            params = self.model.get_param_groups(CFG.head_lr, CFG.last_stage_lr, CFG.backbone_lr)
        
        optimizer = torch.optim.AdamW(params, weight_decay=CFG.weight_decay)
        
        # Cosine annealing with warmup
        warmup_steps = CFG.warmup_epochs * num_training_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[p['lr'] for p in params],
            total_steps=num_training_steps,
            pct_start=warmup_steps / num_training_steps,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Single training epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for rgb, ms, hs, labels in pbar:
            rgb = rgb.to(CFG.device)
            ms = ms.to(CFG.device)
            hs = hs.to(CFG.device)
            labels = labels.to(CFG.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if CFG.amp:
                with autocast():
                    logits = self.model(rgb, ms, hs)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = self.model(rgb, ms, hs)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.grad_clip)
                optimizer.step()
            
            scheduler.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
        
        return running_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validation with metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        running_loss = 0.0
        
        pbar = tqdm(val_loader, desc="Validation")
        for rgb, ms, hs, labels in pbar:
            rgb = rgb.to(CFG.device)
            ms = ms.to(CFG.device)
            hs = hs.to(CFG.device)
            labels = labels.to(CFG.device)
            
            if CFG.amp:
                with autocast():
                    logits = self.model(rgb, ms, hs)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(rgb, ms, hs)
                loss = self.criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()
        
        avg_loss = running_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, f1, all_preds, all_labels
    
    @torch.no_grad()
    def predict_tta(self, val_loader, tta_transforms):
        """Test-Time Augmentation predictions"""
        self.model.eval()
        all_logits = []
        all_labels = []
        
        # Get original dataset
        original_dataset = val_loader.dataset
        
        for tta_idx, tta_transform in enumerate(tta_transforms):
            print(f"TTA {tta_idx + 1}/{len(tta_transforms)}")
            
            # Create new dataset with TTA transform
            tta_dataset = MultiModalImageDataset(
                original_dataset.rgb_paths,
                original_dataset.ms_paths,
                original_dataset.hs_paths,
                original_dataset.labels,
                tta_transform
            )
            tta_loader = DataLoader(
                tta_dataset,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
                pin_memory=True
            )
            
            tta_logits = []
            labels_list = []
            
            for rgb, ms, hs, labels in tqdm(tta_loader, desc=f"TTA {tta_idx+1}"):
                rgb = rgb.to(CFG.device)
                ms = ms.to(CFG.device)
                hs = hs.to(CFG.device)
                
                if CFG.amp:
                    with autocast():
                        logits = self.model(rgb, ms, hs)
                else:
                    logits = self.model(rgb, ms, hs)
                
                tta_logits.append(logits.cpu())
                labels_list.extend(labels.numpy())
            
            all_logits.append(torch.cat(tta_logits, dim=0))
            if tta_idx == 0:
                all_labels = labels_list
        
        # Average logits across TTA
        avg_logits = torch.stack(all_logits).mean(dim=0)
        preds = torch.argmax(avg_logits, dim=1).numpy()
        
        accuracy = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, average='macro')
        
        return preds, all_labels, accuracy, f1
    
    def train_phase(self, train_loader, val_loader, phase: int, epochs: int):
        """Train a specific phase"""
        print(f"\n{'='*50}")
        print(f"Phase {phase} Training - {epochs} epochs")
        print(f"{'='*50}")
        
        # Configure model freezing
        if phase == 1:
            self.model.freeze_backbone()
            print("✓ Backbone frozen, training adapters + fusion + head only")
        elif phase == 2:
            self.model.unfreeze_last_stage()
            print("✓ Last ConvNeXt stage unfrozen")
        else:
            self.model.unfreeze_all()
            print("✓ Full backbone unfrozen")
        
        # Setup optimizer and scheduler
        num_training_steps = len(train_loader) * epochs
        optimizer, scheduler = self._get_optimizer_and_scheduler(phase, num_training_steps)
        
        best_phase_score = 0
        
        for epoch in range(epochs):
            print(f"\nPhase {phase} - Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_loss, accuracy, f1, preds, labels = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
            
            # Save best model
            if accuracy > best_phase_score:
                best_phase_score = accuracy
                self.save_checkpoint(f'best_phase{phase}.pth', accuracy, f1)
                print(f"✓ Model saved (Phase {phase} best)")
        
        print(f"\nPhase {phase} Best Accuracy: {best_phase_score:.4f}")
    
    def save_checkpoint(self, filename: str, accuracy: float, f1: float):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'f1': f1,
            'fold': self.fold
        }
        torch.save(checkpoint, self.output_dir / filename)


# ======================== CROSS-VALIDATION ========================
def run_kfold_training(rgb_paths, ms_paths, hs_paths, labels, idx_to_class, channel_info):
    """Execute 5-fold cross-validation training"""
    
    rgb_channels, ms_channels, hs_channels = channel_info
    
    # Prepare K-Fold splits
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    
    oof_predictions = np.zeros(len(labels))
    oof_labels = np.array(labels)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(rgb_paths, labels)):
        print(f"\n{'#'*60}")
        print(f"FOLD {fold + 1}/{CFG.n_folds}")
        print(f"{'#'*60}")
        
        # Split data
        train_rgb = [rgb_paths[i] for i in train_idx]
        train_ms = [ms_paths[i] for i in train_idx]
        train_hs = [hs_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        
        val_rgb = [rgb_paths[i] for i in val_idx]
        val_ms = [ms_paths[i] for i in val_idx]
        val_hs = [hs_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Create datasets
        train_dataset = MultiModalImageDataset(train_rgb, train_ms, train_hs, train_labels, get_transforms('train'))
        val_dataset = MultiModalImageDataset(val_rgb, val_ms, val_hs, val_labels, get_transforms('val'))
        
        # Create dataloaders with MixUp/CutMix for training
        mixup_collate = MultiModalMixupCutmixCollate(
            mixup_alpha=CFG.mixup_alpha,
            cutmix_alpha=CFG.cutmix_alpha,
            prob=CFG.mixup_prob,
            num_classes=CFG.num_classes
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            collate_fn=mixup_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True
        )
        
        # Initialize model and trainer
        model = MultiModalConvNeXtV2(
            CFG.model_name, 
            CFG.num_classes, 
            rgb_channels=rgb_channels,
            ms_channels=ms_channels,
            hs_channels=hs_channels,
            dropout=CFG.dropout
        )
        trainer = Trainer(model, fold)
        
        # Three-phase training
        trainer.train_phase(train_loader, val_loader, phase=1, epochs=5)
        trainer.train_phase(train_loader, val_loader, phase=2, epochs=10)
        trainer.train_phase(train_loader, val_loader, phase=3, epochs=25)
        
        # Load best model and evaluate with TTA
        print("\n" + "="*50)
        print("Final Evaluation with TTA")
        print("="*50)
        
        best_checkpoint = torch.load(trainer.output_dir / 'best_phase3.pth')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        tta_transforms = get_tta_transforms()
        preds, val_labels_check, tta_accuracy, tta_f1 = trainer.predict_tta(val_loader, tta_transforms)
        
        # Store OOF predictions
        oof_predictions[val_idx] = preds
        
        print(f"\nFold {fold + 1} Results:")
        print(f"TTA Accuracy: {tta_accuracy:.4f}")
        print(f"TTA Macro F1: {tta_f1:.4f}")
        
        fold_scores.append({
            'fold': fold + 1,
            'accuracy': tta_accuracy,
            'f1': tta_f1
        })
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
    
    # Overall OOF scores
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    for score in fold_scores:
        print(f"Fold {score['fold']}: Accuracy={score['accuracy']:.4f}, F1={score['f1']:.4f}")
    
    avg_accuracy = np.mean([s['accuracy'] for s in fold_scores])
    avg_f1 = np.mean([s['f1'] for s in fold_scores])
    
    print(f"\nAverage Accuracy: {avg_accuracy:.4f} ± {np.std([s['accuracy'] for s in fold_scores]):.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {np.std([s['f1'] for s in fold_scores]):.4f}")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'rgb_path': rgb_paths,
        'ms_path': ms_paths,
        'hs_path': hs_paths,
        'true_label': oof_labels,
        'pred_label': oof_predictions.astype(int),
        'true_class': [idx_to_class[label] for label in oof_labels],
        'pred_class': [idx_to_class[int(pred)] for pred in oof_predictions]
    })
    oof_df.to_csv(Path(CFG.output_dir) / 'oof_predictions.csv', index=False)
    print(f"\nOOF predictions saved to {CFG.output_dir}/oof_predictions.csv")
    
    return fold_scores, oof_df


# ======================== MAIN ========================
def main():
    """Main execution function"""
    
    print("="*60)
    print("Multimodal ConvNeXtV2 Training Pipeline")
    print("="*60)
    print(f"Device: {CFG.device}")
    print(f"Model: {CFG.model_name}")
    print(f"Image Size: {CFG.image_size}")
    print(f"Batch Size: {CFG.batch_size}")
    print(f"Folds: {CFG.n_folds}")
    print(f"Epochs: {CFG.epochs}")
    print(f"Modalities: RGB + MS + HS")
    print("="*60)
    
    # Set seed
    set_seed(CFG.seed)
    
    # Prepare multimodal data
    print("\nPreparing multimodal data...")
    rgb_paths, ms_paths, hs_paths, labels, idx_to_class, channel_info = prepare_multimodal_data(
        CFG.train_rgb_dir,
        CFG.train_ms_dir,
        CFG.train_hs_dir
    )
    
    # Run K-Fold training
    fold_scores, oof_df = run_kfold_training(rgb_paths, ms_paths, hs_paths, labels, idx_to_class, channel_info)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Models saved in: {CFG.output_dir}")
    print(f"OOF predictions: {CFG.output_dir}/oof_predictions.csv")


if __name__ == "__main__":
    main()
