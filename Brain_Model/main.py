import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 1. CLAHE Preprocessing ====================
class CLAHETransform:
    """Apply CLAHE for contrast enhancement on grayscale medical images"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        # Create CLAHE object in __call__ to avoid pickling issues
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        enhanced = clahe.apply(img_np)
        return transforms.functional.to_pil_image(enhanced)

# ==================== 2. Advanced Data Augmentation ====================
# Calculate dataset-specific mean and std (run once)
def calculate_dataset_stats(data_dir):
    """Calculate mean and std for grayscale medical images"""
    temp_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    temp_dataset = datasets.ImageFolder(data_dir, transform=temp_transform)
    loader = DataLoader(temp_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    mean = 0.
    std = 0.
    total = 0
    
    for images, _ in tqdm(loader, desc="Calculating stats"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += batch_samples
    
    mean /= total
    std /= total
    return mean.item(), std.item()

# Training augmentation (strong)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((256, 256)),
    transforms.RandomRotation(10),  # ±10 degrees
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels for pretrained model
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # Adjust based on your data
])

# Validation/Test augmentation (minimal)
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

# ==================== 3. Load and Split Dataset ====================
train_dir = 'D:/ml/brain/dataset/train'
test_dir = 'D:/ml/brain/dataset/test'

# Load training data
full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset_raw = datasets.ImageFolder(train_dir, transform=val_transform)

# 80-20 train-val split with stratification
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# Use same indices for both to maintain stratification
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = random_split(range(len(full_train_dataset)), 
                                          [train_size, val_size], 
                                          generator=generator)

train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
val_dataset = torch.utils.data.Subset(val_dataset_raw, val_indices.indices)

# Test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

class_names = full_train_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ==================== 4. Model with Dropout ====================
class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.4):
        super(ResNetWithDropout, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        # Remove original FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)

model = ResNetWithDropout(num_classes=num_classes, dropout_rate=0.4).to(device)

# ==================== 5. Progressive Unfreezing ====================
def freeze_layers(model, unfreeze_from=None):
    """
    Freeze ResNet layers progressively
    unfreeze_from: None (freeze all), 'fc', 'layer4', 'layer3', etc.
    """
    # Freeze all ResNet layers
    for param in model.resnet.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier (always trainable)
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Progressive unfreezing
    if unfreeze_from == 'layer4':
        for param in model.resnet.layer4.parameters():
            param.requires_grad = True
    elif unfreeze_from == 'layer3':
        for param in model.resnet.layer4.parameters():
            param.requires_grad = True
        for param in model.resnet.layer3.parameters():
            param.requires_grad = True

# ==================== 6. Loss with Label Smoothing ====================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -log_preds.sum(dim=1).mean()
        nll = torch.nn.functional.nll_loss(log_preds, target)
        return (1 - self.smoothing) * nll + self.smoothing * loss / n_classes

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# ==================== 7. Training Function ====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss/len(loader), 100.*correct/total, all_preds, all_labels

# ==================== 8. Training Loop with Progressive Unfreezing ====================
def train_model(model, train_loader, val_loader, num_epochs=30):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Stage 1: Train only classifier (epochs 0-5)
    print("\n=== Stage 1: Training classifier only ===")
    freeze_layers(model, unfreeze_from=None)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5 [Stage 1]")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Stage 2: Unfreeze layer4 (epochs 6-15)
    print("\n=== Stage 2: Unfreezing layer4 ===")
    freeze_layers(model, unfreeze_from='layer4')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    for epoch in range(10):
        print(f"\nEpoch {epoch+6}/15 [Stage 2]")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Stage 3: Unfreeze layer3 (epochs 16-30) - optional
    print("\n=== Stage 3: Unfreezing layer3 (fine-tuning) ===")
    freeze_layers(model, unfreeze_from='layer3')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=5e-5, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    for epoch in range(15):
        print(f"\nEpoch {epoch+16}/30 [Stage 3]")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return history

# ==================== 9. Evaluation & Confusion Matrix ====================
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    
    # Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Per-class metrics
    print("\n" + "="*60)
    print("CONFUSION MATRIX VALUES")
    print("="*60)
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  True Positives: {cm[i, i]}")
        print(f"  False Positives: {cm[:, i].sum() - cm[i, i]}")
        print(f"  False Negatives: {cm[i, :].sum() - cm[i, i]}")
        print(f"  True Negatives: {cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]}")
    
    # Overall metrics
    accuracy = 100. * np.trace(cm) / np.sum(cm)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n{'='*60}")
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(f"{'='*60}")
    
    return cm, accuracy, f1

# ==================== 10. Plot Training History ====================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()

# ==================== 11. Main Execution ====================
if __name__ == '__main__':
    # Train model
    print("Starting training with progressive unfreezing...")
    history = train_model(model, train_loader, val_loader, num_epochs=30)
    
    # Plot training history
    plot_history(history)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    print("\n" + "="*60)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*60)
    cm, test_acc, test_f1 = evaluate_model(model, test_loader, class_names)
    
    print("\n✅ Training complete!")
    print(f"Best model saved as 'best_model.pth'")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")