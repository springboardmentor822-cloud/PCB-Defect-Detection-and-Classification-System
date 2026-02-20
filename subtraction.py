import cv2
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

REF_PATH = os.path.join(PROJECT_ROOT, "images", "template", "05.JPG")
ALIGNED_PATH = os.path.join(PROJECT_ROOT, "images", "test_aligned.jpg")
OUT_PATH = os.path.join(PROJECT_ROOT, "images", "difference.jpg")

ref = cv2.imread(REF_PATH, 0)
aligned = cv2.imread(ALIGNED_PATH, 0)

# Normalize
ref = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX)
aligned = cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX)

# Strong blur to suppress PCB tracks
ref_blur = cv2.GaussianBlur(ref, (11, 11), 0)
aligned_blur = cv2.GaussianBlur(aligned, (11, 11), 0)

# Subtraction
diff = cv2.absdiff(ref_blur, aligned_blur)

cv2.imwrite(OUT_PATH, diff)
print("✅ Improved difference saved →", OUT_PATH)

cv2.imshow("Improved Difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
