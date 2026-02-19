import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path("F:/PCB_Defect_Detection_System")
DATASET_DIR = BASE_DIR / "PCB_DATASET" / "PCB_DATASET"

TEMPLATE_DIR = DATASET_DIR / "PCB_USED"
IMAGE_BASE_DIR = DATASET_DIR / "images"

OUTPUT_DIR = BASE_DIR / "data/processed/diff_masks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Defect classes
DEFECT_CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

# --------------------------------------------------
# Image subtraction for all classes
# --------------------------------------------------

for defect_class in DEFECT_CLASSES:

    print(f"\nProcessing class: {defect_class}")

    class_image_dir = IMAGE_BASE_DIR / defect_class
    image_files = list(class_image_dir.glob("*.jpg")) + list(class_image_dir.glob("*.JPG"))

    if not image_files:
        print(f"No images found for class: {defect_class}")
        continue

    for img_path in image_files:

        # Extract template ID (e.g., 01 from 01_missing_hole_01.jpg)
        template_id = img_path.name.split("_")[0]
        template_path = TEMPLATE_DIR / f"{template_id}.JPG"

        if not template_path.exists():
            print(f"Template not found for {img_path.name}")
            continue

        template_img = cv2.imread(str(template_path))
        defect_img = cv2.imread(str(img_path))

        if template_img is None or defect_img is None:
            print(f"Failed to read image: {img_path.name}")
            continue

        # Convert to grayscale
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        defect_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)

        # Resize to match template
        defect_gray = cv2.resize(defect_gray, template_gray.shape[::-1])

        # Absolute difference
        diff = cv2.absdiff(template_gray, defect_gray)

        # Otsu thresholding
        _, thresh = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Morphological noise removal + slight expansion
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.dilate(opened, kernel, iterations=1)

        # Save mask
        output_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(output_path), cleaned)

        print(f"Processed: {img_path.name}")

print("\nImage subtraction completed for all classes.")
