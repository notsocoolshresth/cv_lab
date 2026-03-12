"""
Simple examples demonstrating how to use the wheat disease classification dataset and models
"""

import torch
from dataset import get_data_loaders, WheatDataset
from model import get_model, count_parameters


def example_1_load_dataset():
    """Example 1: Load and inspect dataset"""
    print("=" * 60)
    print("EXAMPLE 1: Load and Inspect Dataset")
    print("=" * 60)
    
    root_dir = r"c:\Users\Shresth\vscode2\pyfiles\deepLearning\wheat\Kaggle_Prepared"
    
    # Create dataset
    dataset = WheatDataset(
        root_dir=root_dir,
        split='train',
        modality='RGB'
    )
    
    print(f"\nDataset Information:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Classes: {dataset.classes}")
    print(f"  - Class distribution: {dataset.get_class_distribution()}")
    
    # Get a sample
    image, label = dataset[0]
    print(f"\nSample Information:")
    print(f"  - Image shape: {image.shape}")
    print(f"  - Label: {label} ({dataset.classes[label]})")
    print(f"  - Image dtype: {image.dtype}")
    print(f"  - Value range: [{image.min():.3f}, {image.max():.3f}]")


def example_2_data_loaders():
    """Example 2: Create data loaders and iterate"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Create Data Loaders")
    print("=" * 60)
    
    root_dir = r"c:\Users\Shresth\vscode2\pyfiles\deepLearning\wheat\Kaggle_Prepared"
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        root_dir=root_dir,
        modality='RGB',
        batch_size=8,
        num_workers=0
    )
    
    # Get one batch
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch Information:")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Batch labels: {labels.tolist()}")
    
    # Count batches
    print(f"\nDataLoader Statistics:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")


def example_3_models():
    """Example 3: Create and test different models"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Create and Test Models")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    models_to_test = [
        ('cnn', {}),
        ('resnet', {'model_name': 'resnet18', 'pretrained': False}),
        ('efficientnet', {'model_name': 'efficientnet_b0', 'pretrained': False}),
    ]
    
    for model_type, kwargs in models_to_test:
        print(f"\n{model_type.upper()} Model:")
        print("-" * 40)
        
        try:
            model = get_model(model_type, num_classes=3, **kwargs).to(device)
            output = model(x)
            
            print(f"  - Input shape: {x.shape}")
            print(f"  - Output shape: {output.shape}")
            print(f"  - Parameters: {count_parameters(model):,}")
            print(f"  - Model size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB (float32)")
            print(f"  ✓ Model works correctly")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def example_4_inference():
    """Example 4: Simple inference example"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Model Inference")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model('resnet', num_classes=3, pretrained=False, model_name='resnet18').to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    class_names = ['Health', 'Other', 'Rust']
    
    print(f"\nInference Results:")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Raw output: {output.cpu().numpy()}")
    print(f"  - Probabilities: {probabilities.cpu().numpy()}")
    print(f"  - Predicted class: {predicted.item()} ({class_names[predicted.item()]})")
    
    print(f"\nClass Probabilities:")
    for i, class_name in enumerate(class_names):
        prob = probabilities[0, i].item() * 100
        print(f"  - {class_name}: {prob:.2f}%")


def example_5_batch_processing():
    """Example 5: Process a batch of images"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Batch Processing")
    print("=" * 60)
    
    root_dir = r"c:\Users\Shresth\vscode2\pyfiles\deepLearning\wheat\Kaggle_Prepared"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, _ = get_data_loaders(
        root_dir=root_dir,
        modality='RGB',
        batch_size=8,
        num_workers=0
    )
    
    # Create model
    model = get_model('resnet', num_classes=3, pretrained=False, model_name='resnet18').to(device)
    model.eval()
    
    # Get one batch
    images, labels = next(iter(train_loader))
    images = images.to(device)
    
    # Process batch
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    class_names = ['Health', 'Other', 'Rust']
    
    print(f"\nBatch Processing Results:")
    print(f"  - Batch size: {images.shape[0]}")
    print(f"  - True labels: {[class_names[l.item()] for l in labels]}")
    print(f"  - Predictions: {[class_names[p.item()] for p in predicted.cpu()]}")
    
    # Calculate accuracy
    correct = (predicted.cpu() == labels).sum().item()
    accuracy = correct / len(labels) * 100
    print(f"  - Accuracy: {accuracy:.2f}% ({correct}/{len(labels)})")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("WHEAT DISEASE CLASSIFICATION - EXAMPLES")
    print("=" * 60)
    
    try:
        example_1_load_dataset()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_2_data_loaders()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_3_models()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_4_inference()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    try:
        example_5_batch_processing()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
