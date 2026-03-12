import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
import rasterio


class WheatDataset(Dataset):
    """
    PyTorch Dataset for Wheat Disease Classification
    
    Dataset structure:
        - train/val folders
        - RGB (PNG), HS (TIF), MS (TIF) subfolders
        - Classes: Health, Other, Rust
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train',
        modality: str = 'RGB',
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            root_dir: Root directory containing train/val folders
            split: 'train' or 'val'
            modality: 'RGB', 'HS', or 'MS'
            transform: Optional transform to be applied on images
            image_size: Target size for images (height, width)
        """
        self.root_dir = root_dir
        self.split = split
        self.modality = modality
        self.transform = transform
        self.image_size = image_size
        
        # Class mapping
        self.classes = ['Health', 'Other', 'Rust']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Build dataset
        self.data_dir = os.path.join(root_dir, split, modality)
        self.samples = self._load_samples()
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and labels"""
        samples = []
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")
        
        # Get all files in the modality directory
        files = os.listdir(self.data_dir)
        
        for filename in files:
            # Extract class from filename (e.g., "Health_hyper_1.png" -> "Health")
            class_name = filename.split('_')[0]
            
            if class_name in self.classes:
                filepath = os.path.join(self.data_dir, filename)
                label = self.class_to_idx[class_name]
                samples.append((filepath, label))
        
        return samples
    
    def _get_default_transforms(self):
        """Get default transforms based on split"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_rgb_image(self, filepath: str) -> Image.Image:
        """Load RGB image (PNG)"""
        return Image.open(filepath).convert('RGB')
    
    def _load_multispectral_image(self, filepath: str) -> np.ndarray:
        """Load multispectral/hyperspectral image (TIF)"""
        with rasterio.open(filepath) as src:
            # Read all bands
            image = src.read()  # Shape: (bands, height, width)
            # Convert to (height, width, bands)
            image = np.transpose(image, (1, 2, 0))
        return image
    
    def _normalize_multispectral(self, image: np.ndarray) -> np.ndarray:
        """Normalize multispectral image to 0-1 range"""
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        return image
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        
        if self.modality == 'RGB':
            # Load RGB image
            image = self._load_rgb_image(filepath)
            if self.transform:
                image = self.transform(image)
        else:
            # Load multispectral/hyperspectral image
            image = self._load_multispectral_image(filepath)
            image = self._normalize_multispectral(image)
            
            # Resize
            from PIL import Image as PILImage
            # For visualization, take first 3 bands
            if image.shape[2] >= 3:
                rgb_approx = image[:, :, :3]
            else:
                rgb_approx = np.repeat(image[:, :, 0:1], 3, axis=2)
            
            # Convert to PIL for transforms
            rgb_approx = (rgb_approx * 255).astype(np.uint8)
            image_pil = PILImage.fromarray(rgb_approx)
            
            if self.transform:
                image = self.transform(image_pil)
        
        return image, label
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset"""
        labels = [label for _, label in self.samples]
        distribution = {cls: labels.count(idx) for cls, idx in self.class_to_idx.items()}
        return distribution


def get_data_loaders(
    root_dir: str,
    modality: str = 'RGB',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        root_dir: Root directory containing train/val folders
        modality: 'RGB', 'HS', or 'MS'
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        image_size: Target size for images
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = WheatDataset(
        root_dir=root_dir,
        split='train',
        modality=modality,
        image_size=image_size
    )
    
    val_dataset = WheatDataset(
        root_dir=root_dir,
        split='val',
        modality=modality,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Train class distribution: {train_dataset.get_class_distribution()}")
    print(f"Val class distribution: {val_dataset.get_class_distribution()}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    root_dir = r"c:\Users\Shresth\vscode2\pyfiles\deepLearning\wheat\Kaggle_Prepared"
    
    # Test RGB modality
    print("Testing RGB Dataset:")
    train_loader, val_loader = get_data_loaders(
        root_dir=root_dir,
        modality='RGB',
        batch_size=16,
        num_workers=0  # Use 0 for testing on Windows
    )
    
    # Get one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")
