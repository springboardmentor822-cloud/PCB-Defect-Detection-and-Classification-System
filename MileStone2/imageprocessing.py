import os
import cv2
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset", "raw")
TEMPLATE_FOLDER = os.path.join(DATASET_FOLDER, "template")

if not os.path.exists(TEMPLATE_FOLDER):
    raise FileNotFoundError(f"Template folder not found: {TEMPLATE_FOLDER}")

template_files = os.listdir(TEMPLATE_FOLDER)

if not template_files:
    raise ValueError("No template images found inside template folder.")

template_path = os.path.join(TEMPLATE_FOLDER, template_files[0])
template_image = cv2.imread(template_path)

if template_image is None:
    raise ValueError(f"Failed to load template image: {template_path}")

template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)



def classify_defect(area, w, h):
    aspect_ratio = w / float(h)

    if area < 500:
        return "Missing Hole", 90
    elif aspect_ratio > 2.5:
        return "Open Circuit", 92
    elif aspect_ratio < 0.5:
        return "Short Circuit", 89
    elif 500 <= area < 2000:
        return "Mouse Bite", 87
    elif 2000 <= area < 5000:
        return "Spur", 85
    else:
        return "Spurious Copper", 83



def detect_defects(test_image_path):

    test_image = cv2.imread(test_image_path)

    if test_image is None:
        raise ValueError(f"Failed to load test image: {test_image_path}")

    test_image = cv2.resize(
        test_image,
        (template_image.shape[1], template_image.shape[0])
    )

    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(template_gray, test_gray)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    defect_data = []
    defect_count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < 200:
            continue

        defect_count += 1

        defect_type, confidence = classify_defect(area, w, h)

        # Draw thicker bounding box
        cv2.rectangle(
            test_image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        label = f"{defect_type} | {confidence}%"

        text_y = y - 20 if y - 20 > 20 else y + 40

        font_scale = 1.0   
        thickness = 3

        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        cv2.rectangle(
            test_image,
            (x, text_y - text_height - 15),
            (x + text_width + 15, text_y + 5),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            test_image,
            label,
            (x + 8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness
        )

        defect_data.append({
            "Defect_ID": defect_count,
            "Defect_Type": defect_type,
            "Confidence (%)": confidence,
            "X": x,
            "Y": y,
            "Width": w,
            "Height": h,
            "Area": area
        })

    df = pd.DataFrame(defect_data)

    return test_image, df
