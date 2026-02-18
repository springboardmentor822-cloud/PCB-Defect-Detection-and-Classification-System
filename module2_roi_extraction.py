import cv2
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MASK_PATH = os.path.join(PROJECT_ROOT, "images", "defects_final.jpg")
ALIGNED_IMG_PATH = os.path.join(PROJECT_ROOT, "images", "test_aligned.jpg")
ROI_DIR = os.path.join(PROJECT_ROOT, "roi_output")

os.makedirs(ROI_DIR, exist_ok=True)

mask = cv2.imread(MASK_PATH, 0)
aligned = cv2.imread(ALIGNED_IMG_PATH)

if mask is None or aligned is None:
    print("❌ Input images not loaded")
    exit()

# ---- CONTOUR EXTRACTION ----
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

MIN_AREA = 800  # acceptable for PCB defects

roi_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    # Draw bounding box (visualization)
    cv2.rectangle(aligned, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Crop ROI
    roi = aligned[y:y+h, x:x+w]
    roi_path = os.path.join(ROI_DIR, f"roi_{roi_count}.png")
    cv2.imwrite(roi_path, roi)

    roi_count += 1

# Save visualization
cv2.imwrite(os.path.join(PROJECT_ROOT, "images", "module2_bounded_output.jpg"), aligned)

print(f"✅ Module 2 completed")
print(f"   → Total ROIs extracted: {roi_count}")
print(f"   → ROI folder: roi_output/")
print(f"   → Bounding box image: images/module2_bounded_output.jpg")

cv2.imshow("Module 2 – Bounding Boxes", aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()
