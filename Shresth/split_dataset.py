"""
Utility script to split train data into train/val sets
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_dataset(root_dir, test_size=0.2, random_state=42):
    """
    Split training data into train/val sets
    
    Args:
        root_dir: Root directory containing train folder
        test_size: Fraction of data to use for validation (default 0.2 = 20%)
        random_state: Random seed for reproducibility
    """
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    
    modalities = ['RGB', 'HS', 'MS']
    
    # Create val directory structure
    for modality in modalities:
        os.makedirs(os.path.join(val_dir, modality), exist_ok=True)
    
    # Process each modality
    for modality in modalities:
        modality_dir = os.path.join(train_dir, modality)
        
        if not os.path.exists(modality_dir):
            print(f"Warning: {modality_dir} not found, skipping...")
            continue
        
        # Get all files
        files = [f for f in os.listdir(modality_dir) if os.path.isfile(os.path.join(modality_dir, f))]
        
        # Split into train/val
        train_files, val_files = train_test_split(files, test_size=test_size, random_state=random_state)
        
        # Move val files
        print(f"\n{modality} Images:")
        print(f"  Total: {len(files)}")
        print(f"  Train: {len(train_files)}")
        print(f"  Val: {len(val_files)}")
        
        for val_file in val_files:
            src = os.path.join(modality_dir, val_file)
            dst = os.path.join(val_dir, modality, val_file)
            shutil.move(src, dst)
        
        print(f"  ✓ Moved {len(val_files)} files to val/{modality}/")


def reset_dataset(root_dir):
    """
    Reset dataset by moving all val files back to train
    (in case you want to start over)
    """
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    
    modalities = ['RGB', 'HS', 'MS']
    
    for modality in modalities:
        val_modality_dir = os.path.join(val_dir, modality)
        
        if not os.path.exists(val_modality_dir):
            continue
        
        # Get all files from val
        files = os.listdir(val_modality_dir)
        
        for val_file in files:
            src = os.path.join(val_modality_dir, val_file)
            dst = os.path.join(train_dir, modality, val_file)
            if os.path.isfile(src):
                shutil.move(src, dst)
        
        print(f"✓ Moved {len(files)} files from val/{modality}/ back to train/{modality}/")


if __name__ == "__main__":
    root_dir = r"c:\Users\Shresth\vscode2\pyfiles\deepLearning\wheat\Kaggle_Prepared"
    
    print("=" * 60)
    print("Dataset Split Utility")
    print("=" * 60)
    
    # Split with 80% train, 20% val
    split_dataset(root_dir, test_size=0.2, random_state=42)
    
    print("\n" + "=" * 60)
    print("✓ Dataset split complete!")
    print("=" * 60)
