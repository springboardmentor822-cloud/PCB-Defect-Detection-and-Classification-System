import cv2
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

OTSU_PATH = os.path.join(PROJECT_ROOT, "images", "otsu_output.jpg")
OUT_PATH = os.path.join(PROJECT_ROOT, "images", "defects_final.jpg")

img = cv2.imread(OTSU_PATH, 0)

if img is None:
    print("❌ Image not loaded")
    exit()

# Morphological opening (remove thin tracks)
kernel_open = np.ones((9, 9), np.uint8)
clean = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open, iterations=2)

# Morphological closing (merge defect parts)
kernel_close = np.ones((13, 13), np.uint8)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close, iterations=2)

# Contour filtering
contours, _ = cv2.findContours(
    clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

final = np.zeros_like(clean)

MIN_AREA = 1000     # 🔥 increased
MIN_WIDTH = 15
MIN_HEIGHT = 15

for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    if area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT:
        cv2.drawContours(final, [cnt], -1, 255, -1)

cv2.imwrite(OUT_PATH, final)
print("✅ Final defect mask saved →", OUT_PATH)

cv2.imshow("Final Defects Only", final)
cv2.waitKey(0)
cv2.destroyAllWindows()

