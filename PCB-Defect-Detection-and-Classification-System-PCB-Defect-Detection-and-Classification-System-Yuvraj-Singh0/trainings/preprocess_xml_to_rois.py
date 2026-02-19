
# src/preprocess_xml_to_rois.py

import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import random

# ================= CONFIG ================= #

DATASET_ROOT = Path("PCB_DATASET/PCB_DATASET")
ANNOTATIONS_DIR = DATASET_ROOT / "Annotations"
IMAGES_DIR = DATASET_ROOT / "images"

OUTPUT_ROOT = Path("data/rois")
TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"

VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Defect classes (folder names)
CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

# ================= HELPERS ================= #

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    objects = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bnd = obj.find("bndbox")

        xmin = int(bnd.find("xmin").text)
        ymin = int(bnd.find("ymin").text)
        xmax = int(bnd.find("xmax").text)
        ymax = int(bnd.find("ymax").text)

        objects.append((label, xmin, ymin, xmax, ymax))

    return filename, objects

# ================= MAIN ================= #

def main():
    print("[INFO] Starting XML → ROI preprocessing")

    samples = []

    for cls in CLASSES:
        xml_dir = ANNOTATIONS_DIR / cls
        img_dir = IMAGES_DIR / cls

        if not xml_dir.exists():
            print(f"[WARN] Missing XML dir: {xml_dir}")
            continue

        for xml_file in xml_dir.glob("*.xml"):
            filename, objects = parse_xml(xml_file)
            img_path = img_dir / filename

            if not img_path.exists():
                print(f"[WARN] Image not found: {img_path}")
                continue

            samples.append((img_path, objects))

    print(f"[INFO] Found {len(samples)} annotated images")

    if len(samples) == 0:
        raise RuntimeError("No annotated samples found. Check paths.")

    # Split
    train_samples, val_samples = train_test_split(
        samples, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    # Clean output
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split_name, split_data in [("train", train_samples), ("val", val_samples)]:
        for cls in CLASSES:
            (OUTPUT_ROOT / split_name / cls).mkdir(parents=True, exist_ok=True)

        for img_path, objects in split_data:
            img = cv2.imread(str(img_path))

            for idx, (label, xmin, ymin, xmax, ymax) in enumerate(objects):
                roi = img[ymin:ymax, xmin:xmax]

                out_path = (
                    OUTPUT_ROOT / split_name / label /
                    f"{img_path.stem}_{idx}.jpg"
                )

                cv2.imwrite(str(out_path), roi)

    print("[INFO] ROI preprocessing completed")
    print(f"[INFO] Output saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
