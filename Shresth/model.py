import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class WheatCNN(nn.Module):
    """
    Basic CNN model for wheat disease classification
    """
    
    def __init__(self, num_classes: int = 3, input_channels: int = 3):
        """
        Args:
            num_classes: Number of output classes (Health, Other, Rust)
            input_channels: Number of input channels (3 for RGB)
        """
        super(WheatCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class WheatResNet(nn.Module):
    """
    Transfer learning model using ResNet for wheat disease classification
    """
    
    def __init__(
        self, 
        num_classes: int = 3,
        pretrained: bool = True,
        model_name: str = 'resnet18',
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            model_name: 'resnet18', 'resnet34', 'resnet50'
            freeze_backbone: Whether to freeze backbone weights
        """
        super(WheatResNet, self).__init__()
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class WheatViT(nn.Module):
    """
    Vision Transformer model for wheat disease classification
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        model_name: str = 'vit_b_16'
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            model_name: 'vit_b_16', 'vit_b_32', 'vit_l_16'
        """
        super(WheatViT, self).__init__()
        
        # Load pretrained ViT
        if model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            num_features = 768
        elif model_name == 'vit_b_32':
            self.backbone = models.vit_b_32(pretrained=pretrained)
            num_features = 768
        elif model_name == 'vit_l_16':
            self.backbone = models.vit_l_16(pretrained=pretrained)
            num_features = 1024
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace classification head
        self.backbone.heads.head = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class WheatEfficientNet(nn.Module):
    """
    EfficientNet model for wheat disease classification
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        model_name: str = 'efficientnet_b0'
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            model_name: 'efficientnet_b0' to 'efficientnet_b7'
        """
        super(WheatEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = 1280
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            num_features = 1280
        elif model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            num_features = 1408
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_model(
    model_type: str = 'resnet',
    num_classes: int = 3,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to get model
    
    Args:
        model_type: 'cnn', 'resnet', 'vit', 'efficientnet'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for specific models
    
    Returns:
        PyTorch model
    """
    if model_type.lower() == 'cnn':
        return WheatCNN(num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'resnet':
        return WheatResNet(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_type.lower() == 'vit':
        return WheatViT(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_type.lower() == 'efficientnet':
        return WheatEfficientNet(num_classes=num_classes, pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Test CNN
    print("=" * 50)
    print("Testing WheatCNN:")
    model_cnn = WheatCNN(num_classes=3).to(device)
    output = model_cnn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(model_cnn):,}")
    
    # Test ResNet
    print("\n" + "=" * 50)
    print("Testing WheatResNet:")
    model_resnet = WheatResNet(num_classes=3, pretrained=False).to(device)
    output = model_resnet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(model_resnet):,}")
    
    # Test factory function
    print("\n" + "=" * 50)
    print("Testing factory function:")
    model = get_model('resnet', num_classes=3, pretrained=False, model_name='resnet18')
    print(f"Model created: {type(model).__name__}")
    print(f"Parameters: {count_parameters(model):,}")
