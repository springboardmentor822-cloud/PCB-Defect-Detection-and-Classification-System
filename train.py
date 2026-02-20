#!/usr/bin/env python3
"""
Train YOLOv5 model for PCB Defect Detection
300 epochs for best performance
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"

print("🚀 Starting YOLOv5 Training for PCB Defect Detection")
print("=" * 70)
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 YOLOv5 Dir: {YOLOV5_DIR}")
print(f"📊 Dataset YAML: {DATASET_YAML}")
print("=" * 70)

# Verify dataset.yaml exists
if not DATASET_YAML.exists():
    print(f"❌ Error: dataset.yaml not found at {DATASET_YAML}")
    sys.exit(1)

print("✅ Dataset configuration found")

# Change to yolov5 directory
os.chdir(YOLOV5_DIR)
print(f"✅ Changed to {YOLOV5_DIR}")

# Training command with 300 epochs
train_cmd = [
    sys.executable,
    "train.py",
    "--img", "416",
    "--batch", "16",
    "--epochs", "300",
    "--data", str(DATASET_YAML),
    "--weights", "yolov5s.pt",
    "--cache",
    "--name", "pcb_1st",
    "--patience", "50",  # Early stopping
    "--save-period", "10",  # Save checkpoint every 10 epochs
    "--workers", "0",  # Disable multiprocessing for macOS compatibility
    "--device", "cpu",  # Use CPU explicitly if GPU has issues
]

print("\n📋 Training Command:")
print(" ".join(train_cmd))
print("\n" + "=" * 70)
print("🔥 Starting training... This will take a while!\n")

# Run training
result = subprocess.run(train_cmd)

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print(f"📦 Model saved at: {YOLOV5_DIR}/runs/train/pcb_1st/weights/best.pt")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("❌ Training failed!")
    print("=" * 70)
    sys.exit(1)
