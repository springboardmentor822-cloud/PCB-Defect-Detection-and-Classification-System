import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os
import json
from pathlib import Path

def train_model(data_dir, model_save_path, history_path, num_epochs=15, batch_size=32):
    """
    Trains an optimized EfficientNet-B0 model on the PCB ROI dataset.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # Advanced Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Optimized Training on device: {device}")
    print(f"Classes: {class_names}")

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze higher-level blocks for fine-tuning
    # EfficientNet-B0 has features indexed 0 to 8
    # We'll unfreeze blocks 7 and 8
    for param in model.features[7:].parameters():
        param.requires_grad = True

    # Replace the classifier head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW for better weight decay
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)
    
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            batch_idx = 0
            for inputs, labels in dataloaders[phase]:
                batch_idx += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        # Gradient Clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(dataloaders['train'])} Loss: {loss.item():.4f}")

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved with accuracy: {best_acc:.4f}")

    # Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f)

    print(f'Best Test Acc: {best_acc:4f}')
    return model

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_PATH = MODEL_DIR / "classifier_best.pt"
    
    RESULTS_DIR = BASE_DIR / "results" / "milestone2"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH = RESULTS_DIR / "training_history.json"

    print("--- Starting Optimized Training ---")
    train_model(DATA_DIR, MODEL_PATH, HISTORY_PATH, num_epochs=12)
    print(f"--- Training Complete. History saved to {HISTORY_PATH} ---")
