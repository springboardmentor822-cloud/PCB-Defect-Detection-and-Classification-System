import cv2
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

REF_PATH = os.path.join(PROJECT_ROOT, "images", "template", "05.JPG")
TEST_PATH = os.path.join(PROJECT_ROOT, "images", "test", "01_mouse_bite_02.jpg")
OUT_PATH = os.path.join(PROJECT_ROOT, "images", "test_aligned.jpg")

# Load images
ref = cv2.imread(REF_PATH, 0)
test = cv2.imread(TEST_PATH, 0)

if ref is None or test is None:
    print("❌ Image not loaded")
    exit()

# Resize test image to reference size (IMPORTANT)
test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

# Convert to float32 (required for ECC)
ref_f = ref.astype(np.float32)
test_f = test.astype(np.float32)

# Initialize warp matrix (AFFINE is enough for PCB)
warp_matrix = np.eye(2, 3, dtype=np.float32)

# ECC criteria
criteria = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    5000,
    1e-6
)

try:
    cc, warp_matrix = cv2.findTransformECC(
        ref_f,
        test_f,
        warp_matrix,
        cv2.MOTION_AFFINE,
        criteria
    )
except cv2.error as e:
    print("❌ ECC alignment failed:", e)
    exit()

# Apply alignment
aligned = cv2.warpAffine(
    test,
    warp_matrix,
    (ref.shape[1], ref.shape[0]),
    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
)

# Save output
cv2.imwrite(OUT_PATH, aligned)
print("✅ Proper alignment completed →", OUT_PATH)

# Show results
cv2.imshow("Reference", ref)
cv2.imshow("Test (Original)", test)
cv2.imshow("Aligned (Correct)", aligned)

cv2.waitKey(0)
cv2.destroyAllWindows()
