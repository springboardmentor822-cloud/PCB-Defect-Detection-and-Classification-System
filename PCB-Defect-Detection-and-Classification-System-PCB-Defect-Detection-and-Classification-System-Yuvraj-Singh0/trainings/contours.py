import cv2
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path("F:/PCB_Defect_Detection_System")
DATASET_DIR = BASE_DIR / "PCB_DATASET" / "PCB_DATASET"

MASK_DIR = BASE_DIR / "data/processed/diff_masks"
IMAGE_BASE_DIR = DATASET_DIR / "images"

CONTOUR_OUTPUT_DIR = BASE_DIR / "data/processed/contours"
ROI_BASE_DIR = BASE_DIR / "data/processed/rois"

CONTOUR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROI_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Defect classes (from dataset)
DEFECT_CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

MIN_CONTOUR_AREA = 100  # noise filter

# --------------------------------------------------
# Process each defect class
# --------------------------------------------------

for defect_class in DEFECT_CLASSES:

    print(f"\nProcessing class: {defect_class}")

    class_image_dir = IMAGE_BASE_DIR / defect_class
    class_roi_dir = ROI_BASE_DIR / defect_class
    class_roi_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(class_image_dir.glob("*.jpg")) + list(class_image_dir.glob("*.JPG"))

    if not image_files:
        print(f"No images found for class: {defect_class}")
        continue

    for img_path in image_files:

        mask_path = MASK_DIR / img_path.name

        if not mask_path.exists():
            print(f"Mask not found for image: {img_path.name}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(str(img_path))

        if mask is None or original_img is None:
            print(f"Failed to read image or mask: {img_path.name}")
            continue

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        roi_index = 0

        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            cv2.rectangle(
                original_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Extract ROI
            roi = original_img[y:y + h, x:x + w]

            roi_name = f"{img_path.stem}_roi_{roi_index}.jpg"
            cv2.imwrite(str(class_roi_dir / roi_name), roi)

            roi_index += 1

        # Save visualization image
        cv2.imwrite(
            str(CONTOUR_OUTPUT_DIR / img_path.name),
            original_img
        )

        print(f"Contours processed: {img_path.name}")

print("\nAll classes processed successfully.")
