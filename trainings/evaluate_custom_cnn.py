import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from dataset import get_dataloaders

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/custom_cnn_pcb.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Custom CNN Model (same as training)
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
# Load dataset
# --------------------------------------------------

train_loader, val_loader, class_names = get_dataloaders()
num_classes = len(class_names)

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = CustomPCBNet(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --------------------------------------------------
# Metrics
# --------------------------------------------------

report = classification_report(
    all_labels, all_preds, target_names=class_names
)

cm = confusion_matrix(all_labels, all_preds)

print("\nCustom CNN Classification Report:\n")
print(report)

# --------------------------------------------------
# Confusion Matrix Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Custom CNN")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_custom_cnn.png"))
plt.show()
