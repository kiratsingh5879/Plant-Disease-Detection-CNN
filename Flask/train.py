import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import time
import sys

# Configuration
DATA_DIR = Path(r"d:\Patho_Plant-master\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)")
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
MODEL_SAVE_DIR = Path(__file__).resolve().parent / "Models"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "plantDisease-resnet34.pth"

BATCH_SIZE = 32
EPOCHS = 1  # 1 epoch is usually sufficient for >85% when fine-tuning ResNet
LEARNING_RATE = 0.001

def get_data_loaders():
    # Transforms match model.py for inference
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    print(f"Loading data from {DATA_DIR}...")
    train_data = ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    valid_data = ImageFolder(root=str(VALID_DIR), transform=valid_transform)

    # Use a small subset (10%) to train faster on CPU for demonstration
    from torch.utils.data import Subset
    import random
    
    # Set seed for reproducibility
    random.seed(42)
    
    train_indices = random.sample(range(len(train_data)), int(0.15 * len(train_data)))
    valid_indices = random.sample(range(len(valid_data)), int(0.15 * len(valid_data)))
    
    train_subset = Subset(train_data, train_indices)
    valid_subset = Subset(valid_data, valid_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, valid_loader, len(train_data.classes)

def build_model(num_classes):
    print("Building model...")
    # Using pretrained weights to speed up convergence
    try:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet34(pretrained=True)
        
    # Freeze the backbone layers for faster CPU training
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    # The new linear layer will have requires_grad=True by default
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if MODEL_SAVE_PATH.exists():
        print("Found existing checkpoint, loading weights to continue training...")
        model.load_state_dict(torch.load(str(MODEL_SAVE_PATH), map_location='cpu'))
        
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not MODEL_SAVE_DIR.exists():
        MODEL_SAVE_DIR.mkdir(parents=True)
        
    train_loader, valid_loader, num_classes = get_data_loaders()
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(valid_loader.dataset)}")
    
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
            if (i+1) % 50 == 0:
                print(f"Batch {i+1}/{len(train_loader)} - Loss: {running_loss/total_samples:.4f} Acc: {running_corrects/total_samples:.4f}")
                sys.stdout.flush()
                
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {time.time()-start_time:.0f}s")
        
        # Validation phase
        print("Starting Validation...")
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                
        val_loss = val_running_loss / len(valid_loader.dataset)
        val_acc = val_running_corrects / len(valid_loader.dataset)
        print(f"Valid Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Validation accuracy improved. Saving model to {MODEL_SAVE_PATH}...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"\nTraining complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()
