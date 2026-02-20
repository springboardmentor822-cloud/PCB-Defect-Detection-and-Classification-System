import cv2
import numpy as np
import time
import os

AREA_THRESHOLD = 80

def detect_and_annotate(template_path, test_path, output_dir, defect_label):
    os.makedirs(output_dir, exist_ok=True)

    timings = {}
    start_total = time.time()

    # -------------------------
    # READ IMAGES
    # -------------------------
    img = cv2.imread(test_path)
    template = cv2.imread(template_path)

    h, w, _ = img.shape
    template = cv2.resize(template, (w, h))

    # -------------------------
    # STEP 1: PREPROCESSING (DEFECT MASK)
    # -------------------------
    t0 = time.time()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_template, gray_img)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_path = os.path.join(output_dir, "defect_mask.png")
    cv2.imwrite(mask_path, mask)

    timings["Preprocessing"] = time.time() - t0

    # -------------------------
    # STEP 2: ROI LOCALIZATION
    # -------------------------
    t1 = time.time()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_only_img = img.copy()
    final_annotated_img = img.copy()
    roi_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < AREA_THRESHOLD:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        roi_count += 1

        # ROI localization image (NO LABEL)
        cv2.rectangle(
            roi_only_img,
            (x, y),
            (x + bw, y + bh),
            (255, 0, 0),
            2
        )

        # Final annotated image (WITH LABEL)
        cv2.rectangle(
            final_annotated_img,
            (x, y),
            (x + bw, y + bh),
            (0, 200, 0),
            3
        )

        cv2.rectangle(
            final_annotated_img,
            (x, y - 45),
            (x + bw, y),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            final_annotated_img,
            f"DEFECT: {defect_label}",
            (x + 8, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            3
        )

    roi_path = os.path.join(output_dir, "roi_localization.png")
    final_path = os.path.join(output_dir, "final_annotated.png")

    cv2.imwrite(roi_path, roi_only_img)
    cv2.imwrite(final_path, final_annotated_img)

    timings["Localization"] = time.time() - t1
    timings["Total"] = time.time() - start_total

    return {
        "mask_path": mask_path,
        "roi_path": roi_path,
        "final_path": final_path,
        "roi_count": roi_count,
        "timings": timings
    }
