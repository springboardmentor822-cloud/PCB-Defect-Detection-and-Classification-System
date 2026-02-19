
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from pathlib import Path

# ---------------- CONFIG ---------------- #

DATA_DIR = Path("data/rois")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

MODEL_OUT = Path("models/efficientnet_b0_pcb_v2.pth")
MODEL_OUT.parent.mkdir(exist_ok=True)

CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORMS ---------------- #

train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATA ---------------- #

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

print("[INFO] Classes:", train_ds.classes)

# ---------------- MODEL ---------------- #

model = efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, len(CLASSES)
)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN ---------------- #

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------- VALIDATION -------- #
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)
        print("✔ Saved best model")

print("\nTraining complete.")
print("Best validation accuracy:", best_val_acc)
