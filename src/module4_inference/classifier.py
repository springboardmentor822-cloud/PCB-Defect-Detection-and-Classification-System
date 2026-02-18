import torch
import cv2
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

def load_model(model_path):
    model = EfficientNet.from_pretrained(
        "efficientnet-b0",
        num_classes=len(CLASSES)
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def classify_roi(model, roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = transform(roi).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(roi)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return CLASSES[pred.item()], conf.item()
