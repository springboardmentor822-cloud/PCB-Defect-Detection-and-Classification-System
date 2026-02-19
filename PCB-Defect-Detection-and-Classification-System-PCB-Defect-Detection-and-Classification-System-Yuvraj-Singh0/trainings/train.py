import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_dataloaders

# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/efficientnet_b0_pcb.pth"

# --------------------------------------------------
# Load data
# --------------------------------------------------

train_loader, val_loader, class_names = get_dataloaders()
num_classes = len(class_names)

# --------------------------------------------------
# Load EfficientNet-B0
# --------------------------------------------------

model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Replace classifier head
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)

# --------------------------------------------------
# Loss & Optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------------------------------
# Training Loop
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total

    # ---------------- Validation ----------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

# --------------------------------------------------
# Save model
# --------------------------------------------------

import os
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print("\nTraining completed. Model saved.")
