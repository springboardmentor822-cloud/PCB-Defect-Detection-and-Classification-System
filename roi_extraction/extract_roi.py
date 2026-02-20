import cv2
import os

MASK_PATH = "preprocessing/output/defect_mask.png"
DEFECT_IMG_DIR = "PCB_DATASET/PCB_DATASET/images/Missing_hole"
DEFECT_IMG_PATH = os.path.join(DEFECT_IMG_DIR, os.listdir(DEFECT_IMG_DIR)[0])

OUTPUT_DIR = "roi_extraction/output"
ROI_DIR = os.path.join(OUTPUT_DIR, "rois")

os.makedirs(ROI_DIR, exist_ok=True)

# Read images
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
defect_img = cv2.imread(DEFECT_IMG_PATH)

# Find contours
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

boxed_img = defect_img.copy()
roi_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    # Ignore tiny noise
    if area < 50:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    # Draw bounding box
    cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop ROI
    roi = defect_img[y:y + h, x:x + w]
    roi_path = os.path.join(ROI_DIR, f"roi_{roi_count}.png")
    cv2.imwrite(roi_path, roi)

    roi_count += 1

cv2.imwrite(os.path.join(OUTPUT_DIR, "bounding_boxes.png"), boxed_img)

print(f"Module 2 completed: {roi_count} defect ROIs extracted.")
