import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# Import the exact same preprocessing used in inference
from preprocess import preprocess_image

class OralCancerDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data_list = data_list
        self.augment = augment
        
        # Data augmentation on PIL Image BEFORE standard preprocessing
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = item['filename']
        label = 1 if item['label'] == 'malignant' else 0
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.aug_transform:
                img = self.aug_transform(img)
                
            # Use the exact same preprocessing function from inference (Resize + CLAHE + ToTensor + Normalize)
            tensor = preprocess_image(img)
            
            if tensor is None:
                # Fallback to random tensor if rejected by min_resolution (should not happen for valid datasets)
                tensor = torch.zeros(3, 224, 224)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            tensor = torch.zeros(3, 224, 224)
            
        return tensor, torch.tensor(label, dtype=torch.long)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=20, patience=5, min_delta=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Early stopping check on validation loss
            if phase == 'val':
                if epoch_loss < best_loss - min_delta:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs (Patience: {patience}).")
            break
            
        print()
        
    print(f"Best val Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Load dataset splits
    index_path = "../data/index_pool.json"
    val_path = "../data/validation_pool.json"
    
    with open(index_path, 'r') as f:
        index_pool = json.load(f)
    with open(val_path, 'r') as f:
        val_pool = json.load(f)
        
    print(f"Training on {len(index_pool)} images, Validating on {len(val_pool)} images.")
    
    train_dataset = OralCancerDataset(index_pool, augment=True)
    val_dataset = OralCancerDataset(val_pool, augment=False)
    
    # Due to small dataset size, we can use a small batch size
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    }
    
    # Initialize MobileNetV2 with ImageNet weights
    print("Loading pre-trained MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Create classification head (will be stripped later for embeddings)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # ── Phase 1: Train Classification Head Only ──
    print("\nStarting Phase 1: Training classification head (frozen backbone)...")
    for param in model.features.parameters():
        param.requires_grad = False
        
    optimizer_phase1 = optim.Adam(model.classifier.parameters(), lr=1e-3)
    model = train_model(model, dataloaders, criterion, optimizer_phase1, num_epochs=20, patience=5, min_delta=0.001)
    
    # ── Phase 2: Gradual Unfreezing ──
    print("\nStarting Phase 2: Fine-tuning last 2 blocks of the backbone...")
    # MobileNetV2's features module has 19 layers. We unfreeze the last 3 layers (which corresponds to the last 2 inverted residual blocks and the final 1x1 conv layer)
    for param in model.features[-3:].parameters():
        param.requires_grad = True
        
    optimizer_phase2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    model = train_model(model, dataloaders, criterion, optimizer_phase2, num_epochs=20, patience=5, min_delta=0.001)
    
    # Save the fine-tuned model
    os.makedirs("../models", exist_ok=True)
    save_path = "../models/finetuned_mobilenetv2.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nFine-tuning complete. Model saved to {save_path}")
    print("NOTE: To extract embeddings for CBIR inference, load this model and execute: model.classifier = nn.Identity() to strip the classification head.")

if __name__ == "__main__":
    main()
