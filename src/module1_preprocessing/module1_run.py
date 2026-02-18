import cv2
from pathlib import Path

from src.subtraction import align_images, subtract_and_threshold
from src.morphology import clean_mask
from src.config import TEMPLATE_DIR, TEST_IMG_DIR, DIFF_MASK_DIR

print("Starting Module 1: Template subtraction pipeline")

template_files = list(TEMPLATE_DIR.glob("*.jpg"))

if len(template_files) == 0:
    raise RuntimeError(" No template images found in data/raw/templates")

template_path = template_files[0]
template = cv2.imread(str(template_path))

print(f"Using template: {template_path.name}")

for defect_path in TEST_IMG_DIR.glob("*.jpg"):
    defect_img = cv2.imread(str(defect_path))
    if defect_img is None:
        continue

    aligned = align_images(template, defect_img)
    raw_mask = subtract_and_threshold(template, aligned)
    final_mask = clean_mask(raw_mask)

    out_path = DIFF_MASK_DIR / defect_path.name
    cv2.imwrite(str(out_path), final_mask)

    print(f"Processed: {defect_path.name}")

print(" Module 1 completed")
