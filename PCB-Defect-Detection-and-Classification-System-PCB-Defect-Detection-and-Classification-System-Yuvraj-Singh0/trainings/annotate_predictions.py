
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.models import efficientnet_b0
import torch.nn as nn

# ---------------- CONFIG ---------------- #

# RAW_IMAGE_ROOT = Path("PCB_DATASET/PCB_DATASET/images")
RAW_IMAGE_ROOT = Path("sample_input")
DIFF_MASK_DIR = Path("data/processed/diff_masks")
# MODEL_PATH = Path("models/efficientnet_pcb.pth")
MODEL_PATH = Path("models/efficientnet_b0_pcb.pth")
# OUTPUT_DIR = Path("outputs/annotated_images")
OUTPUT_DIR = Path("sample_output")

CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- MODEL ---------------- #

model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- TRANSFORM ---------------- #

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- PROCESS ---------------- #

print("\nStarting annotation using diff_masks...\n")

for class_name in CLASSES:
    raw_dir = RAW_IMAGE_ROOT / class_name
    if not raw_dir.exists():
        continue

    for img_path in raw_dir.glob("*.jpg"):
        mask_path = DIFF_MASK_DIR / img_path.name

        if not mask_path.exists():
            continue

        raw_img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if raw_img is None or mask is None:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < 15 or h < 15:
                continue

            roi = raw_img[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_tensor = transform(roi_rgb).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(roi_tensor)
                pred_class = CLASSES[torch.argmax(outputs, dim=1).item()]

            cv2.rectangle(raw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                raw_img,
                pred_class,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        output_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(output_path), raw_img)

print("\nAnnotation completed successfully.")
print(f"Annotated images saved in: {OUTPUT_DIR.resolve()}")
