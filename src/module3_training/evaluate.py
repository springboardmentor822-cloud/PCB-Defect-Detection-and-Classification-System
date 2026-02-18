import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from dataset import PCBDataset
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = Path("data/processed/rois_split/test")
MODEL_PATH = Path("models/efficientnet_pcb.pth")
OUT_DIR = Path("outputs/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = PCBDataset(TEST_DIR)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = build_model(len(dataset.classes)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\n Test Accuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=dataset.classes,
    yticklabels=dataset.classes,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("PCB Defect Confusion Matrix")

out_path = OUT_DIR / "confusion_matrix.png"
plt.savefig(out_path)
plt.close()

print(f" Confusion matrix saved at: {out_path}")
