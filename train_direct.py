#!/usr/bin/env python3
"""
Direct training script without subprocess
Trains YOLOv5 for PCB Defect Detection - 300 epochs
"""

import os
import sys
from pathlib import Path

# Add yolov5 directory to path
PROJECT_ROOT = Path(__file__).parent
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
sys.path.insert(0, str(YOLOV5_DIR))

os.chdir(str(YOLOV5_DIR))

print("🚀 Starting YOLOv5 Training for PCB Defect Detection")
print("=" * 70)
print(f"📁 Working Directory: {os.getcwd()}")
print("=" * 70)

# Import after adding to path
from train import main
from utils.general import init_seeds
import argparse

# Initialize seeds
init_seeds(seed=0)

# Create arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='yolov5s.pt')
parser.add_argument('--cfg', type=str, default='')
parser.add_argument('--data', type=str, default='../dataset.yaml')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=416)
parser.add_argument('--rect', action='store_true')
parser.add_argument('--resume', nargs='?', const=True, default=False)
parser.add_argument('--nosave', action='store_true')
parser.add_argument('--noval', action='store_true')
parser.add_argument('--noautoanchor', action='store_true')
parser.add_argument('--noplots', action='store_true')
parser.add_argument('--evolve', type=int, nargs='?', const=300)
parser.add_argument('--bucket', type=str, default='')
parser.add_argument('--cache', type=str, nargs='?', const='ram')
parser.add_argument('--image-weights', action='store_true')
parser.add_argument('--device', default='')
parser.add_argument('--multi-scale', action='store_true')
parser.add_argument('--single-cls', action='store_true')
parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD')
parser.add_argument('--sync-bn', action='store_true')
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--project', type=str, default='runs/train')
parser.add_argument('--name', type=str, default='pcb_1st')
parser.add_argument('--exist-ok', action='store_true')
parser.add_argument('--quad', action='store_true')
parser.add_argument('--cos-lr', action='store_true')
parser.add_argument('--label-smoothing', type=float, default=0.0)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--freeze', nargs='+', type=int, default=[0])
parser.add_argument('--save-period', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--local-rank', type=int, default=-1)
parser.add_argument('--entity', default=None)
parser.add_argument('--upload-dataset', action='store_true')
parser.add_argument('--bbox-interval', type=int, default=-1)
parser.add_argument('--artifact-alias', type=str, default='latest')

opt = parser.parse_args()

print("\n📋 Training Configuration:")
print(f"  Epochs: {opt.epochs}")
print(f"  Batch Size: {opt.batch_size}")
print(f"  Image Size: {opt.imgsz}")
print(f"  Workers: {opt.workers}")
print(f"  Device: {opt.device if opt.device else 'auto'}")
print("=" * 70)
print("\n🔥 Starting training...\n")

try:
    main(opt)
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
