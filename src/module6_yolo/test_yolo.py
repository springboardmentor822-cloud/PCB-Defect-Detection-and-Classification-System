from ultralytics import YOLO
from pathlib import Path

model = YOLO("models/yolo_pcb.pt")

image_path = "data/raw/test_images/Missing_hole/01_missing_hole_01.jpg"

results = model(image_path, save=True, conf=0.25)

print("Inference completed.")
