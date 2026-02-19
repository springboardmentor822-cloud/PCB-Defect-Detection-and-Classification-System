import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from dataset import get_dataloaders

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/efficientnet_b0_pcb.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

train_loader, val_loader, class_names = get_dataloaders()
num_classes = len(class_names)

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

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

cm = confusion_matrix(all_labels, all_preds)
report = classification_report(
    all_labels, all_preds, target_names=class_names
)

print("\nClassification Report:\n")
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
plt.title("Confusion Matrix - PCB Defect Classification")
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

plt.savefig("outputs/confusion_matrix.png")
plt.show()
