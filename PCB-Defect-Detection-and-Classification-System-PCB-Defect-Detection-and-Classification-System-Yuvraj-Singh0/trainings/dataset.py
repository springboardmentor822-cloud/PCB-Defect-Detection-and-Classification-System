from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("F:/PCB_Defect_Detection_System/data/processed/rois")
IMAGE_SIZE = 128
BATCH_SIZE = 16
TRAIN_SPLIT = 0.8
SEED = 42

# --------------------------------------------------
# Transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------
# Dataset Loader Function
# --------------------------------------------------

def get_dataloaders():
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=None
    )

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, full_dataset.classes
