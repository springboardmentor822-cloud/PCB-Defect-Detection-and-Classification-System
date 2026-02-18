import torch
from torch.utils.data import DataLoader
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import PCBDataset
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = Path("data/processed/rois_split/train")
MODEL_OUT = Path("models/efficientnet_pcb.pth")
PLOT_OUT = Path("outputs/training")
PLOT_OUT.mkdir(parents=True, exist_ok=True)


dataset = PCBDataset(TRAIN_DIR)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


model = build_model(num_classes=len(dataset.classes)).to(DEVICE)


labels = [label for _, label in dataset.samples]
counts = Counter(labels)

weights = torch.tensor(
    [1.0 / counts[i] for i in range(len(dataset.classes))],
    dtype=torch.float
).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

EPOCHS = 30
loss_history = []

print(" Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), MODEL_OUT)
print(" Model training completed and saved.")

#  LOSS CURVE 
plt.figure()
plt.plot(range(1, EPOCHS + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)

loss_plot_path = PLOT_OUT / "training_loss_curve.png"
plt.savefig(loss_plot_path)
plt.close()

print(f" Training loss curve saved at: {loss_plot_path}")
