import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders

# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/custom_cnn_pcb.pth"

# --------------------------------------------------
# Custom CNN Model
# --------------------------------------------------

class CustomPCBNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --------------------------------------------------
# Load data
# --------------------------------------------------

train_loader, val_loader, class_names = get_dataloaders()
num_classes = len(class_names)

model = CustomPCBNet(num_classes).to(DEVICE)

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
    correct, total = 0, 0
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    train_acc = 100 * correct / total

    # ---------------- Validation ----------------
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss: {running_loss:.2f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

# --------------------------------------------------
# Save model
# --------------------------------------------------

import os
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print("\nCustom CNN training completed. Model saved.")
