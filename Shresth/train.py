import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib

from dataset import get_data_loaders
from model import get_model, count_parameters


class Trainer:
    """
    Trainer class for wheat disease classification
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs: int = 50,
        save_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self) -> tuple:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> tuple:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, precision, recall, f1, _, _ = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Metrics/precision', precision, epoch)
            self.writer.add_scalar('Metrics/recall', recall, epoch)
            self.writer.add_scalar('Metrics/f1', f1, epoch)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch, val_acc, val_loss)
                print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_acc, val_loss)
        
        # Save final model
        self.save_checkpoint('final_model.pth', self.num_epochs-1, val_acc, val_loss)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Final evaluation
        self.final_evaluation()
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str, epoch: int, val_acc: float, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300)
        plt.close()
    
class ViTWithXGBoost:
    """
    ViT feature extractor + XGBoost classifier
    Uses ViT to extract logits and trains XGBoost on top
    """
    
    def __init__(
        self,
        vit_model: nn.Module,
        device,
        vit_epochs: int = 30,
        xgb_params: dict = None,
        save_dir: str = './checkpoints'
    ):
        self.vit_model = vit_model
        self.device = device
        self.vit_epochs = vit_epochs
        self.save_dir = save_dir
        
        # XGBoost parameters
        self.xgb_params = xgb_params or {
            'objective': 'multi:softmax',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.xgb_model = None
        self.vit_losses = []
        self.vit_accs = []
    
    def extract_logits(self, data_loader, phase='train'):
        """Extract logits from ViT model"""
        self.vit_model.eval()
        all_logits = []
        all_labels = []
        
        pbar = tqdm(data_loader, desc=f'Extracting {phase} logits')
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                logits = self.vit_model(images)
                
                all_logits.append(logits.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        logits = np.vstack(all_logits)
        labels = np.array(all_labels)
        
        print(f"{phase.capitalize()} logits shape: {logits.shape}")
        print(f"{phase.capitalize()} labels shape: {labels.shape}")
        
        return logits, labels
    
    def train_vit(self, train_loader, val_loader, criterion, optimizer, lr_scheduler=None):
        """Train ViT for a few epochs to generate good logits"""
        print("\n" + "=" * 60)
        print("PHASE 1: Training ViT Feature Extractor")
        print("=" * 60)
        
        self.vit_model.train()
        best_val_acc = 0.0
        
        for epoch in range(self.vit_epochs):
            # Training
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            pbar = tqdm(train_loader, desc=f'ViT Epoch {epoch+1}/{self.vit_epochs}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.vit_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = accuracy_score(all_labels, all_preds)
            self.vit_losses.append(epoch_loss)
            self.vit_accs.append(epoch_acc)
            
            # Validation
            self.vit_model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    outputs = self.vit_model(images)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.numpy())
            
            val_acc = accuracy_score(val_labels, val_preds)
            
            print(f"Epoch {epoch+1}/{self.vit_epochs} - Train Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.vit_model.state_dict(), 
                          os.path.join(self.save_dir, 'best_vit_model.pth'))
                print(f"✓ Saved best ViT model (Val Acc: {val_acc:.4f})")
            
            self.vit_model.train()
        
        print(f"\n✓ ViT training completed. Best Val Acc: {best_val_acc:.4f}")
    
    def train_xgboost(self, train_logits, train_labels, val_logits, val_labels):
        """Train XGBoost on ViT logits"""
        print("\n" + "=" * 60)
        print("PHASE 2: Training XGBoost on ViT Logits")
        print("=" * 60)
        
        # Create XGBoost dataset
        dtrain = xgb.DMatrix(train_logits, label=train_labels)
        dval = xgb.DMatrix(val_logits, label=val_labels)
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'eval')]
        evals_result = {}
        
        print("\nTraining XGBoost...")
        self.xgb_model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        # Save XGBoost model
        self.xgb_model.save_model(os.path.join(self.save_dir, 'xgb_model.json'))
        
        print(f"\n✓ XGBoost training completed")
        print(f"Best iteration: {self.xgb_model.best_iteration}")
        
        return evals_result
    
    def evaluate(self, val_logits, val_labels):
        """Evaluate XGBoost predictions"""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained yet")
        
        dval = xgb.DMatrix(val_logits)
        predictions = self.xgb_model.predict(dval)
        predictions = predictions.astype(int)
        
        accuracy = accuracy_score(val_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, predictions, average='weighted', zero_division=0
        )
        
        return accuracy, precision, recall, f1, predictions, val_labels
    
    def plot_results(self, evals_result):
        """Plot training results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ViT training curves
        axes[0, 0].plot(self.vit_losses)
        axes[0, 0].set_title('ViT Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.vit_accs)
        axes[0, 1].set_title('ViT Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
        
        # XGBoost training curves
        axes[1, 0].plot(evals_result.get('train', {}).get('mlogloss', []), label='Train')
        axes[1, 0].plot(evals_result.get('eval', {}).get('mlogloss', []), label='Val')
        axes[1, 0].set_title('XGBoost Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Feature importance
        importance = self.xgb_model.get_score(importance_type='weight')
        if importance:
            features = list(importance.keys())
            scores = list(importance.values())
            axes[1, 1].barh(features[:10], scores[:10])
            axes[1, 1].set_title('Top 10 Feature Importance (XGBoost)')
            axes[1, 1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'vit_xgboost_results.png'), dpi=300)
        plt.close()
        print(f"\n✓ Results saved to {os.path.join(self.save_dir, 'vit_xgboost_results.png')}")


def main():
    # Configuration
    ROOT_DIR = r"c:\Users\Shresth\vscode2\pyfiles\deepLearning\wheat\Kaggle_Prepared"
    MODALITY = 'RGB'  # 'RGB', 'HS', or 'MS'
    BATCH_SIZE = 32
    VIT_EPOCHS = 30  # Epochs for ViT training
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0  # Use 0 for Windows
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Data loaders
    print("\nLoading datasets...")
    train_loader, val_loader = get_data_loaders(
        root_dir=ROOT_DIR,
        modality=MODALITY,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # ViT Model
    print(f"\nCreating Vision Transformer model...")
    vit_model = get_model(
        model_type='vit',
        num_classes=3,
        pretrained=True,
        model_name='vit_b_16'
    ).to(device)
    
    print(f"ViT Model parameters: {count_parameters(vit_model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit_model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Initialize ViT + XGBoost pipeline
    pipeline = ViTWithXGBoost(
        vit_model=vit_model,
        device=device,
        vit_epochs=VIT_EPOCHS,
        save_dir='./checkpoints'
    )
    
    # PHASE 1: Train ViT
    pipeline.train_vit(train_loader, val_loader, criterion, optimizer, lr_scheduler)
    
    # PHASE 2: Extract logits
    print("\n" + "=" * 60)
    print("Extracting Logits from ViT")
    print("=" * 60)
    
    train_logits, train_labels = pipeline.extract_logits(train_loader, phase='train')
    val_logits, val_labels = pipeline.extract_logits(val_loader, phase='val')
    
    # PHASE 3: Train XGBoost
    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    evals_result = pipeline.train_xgboost(train_logits, train_labels, val_logits, val_labels)
    
    # PHASE 4: Evaluate
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    accuracy, precision, recall, f1, predictions, labels = pipeline.evaluate(val_logits, val_labels)
    
    print(f"\nMetrics on Validation Set:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    class_names = ['Health', 'Other', 'Rust']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ViT + XGBoost')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join('./checkpoints', 'confusion_matrix_vit_xgb.png'), dpi=300)
    plt.close()
    
    # Plot results
    pipeline.plot_results(evals_result)
    
    print("\n" + "=" * 60)
    print("✓ ViT + XGBoost Training Pipeline Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
