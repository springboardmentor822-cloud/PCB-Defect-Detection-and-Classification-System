import cv2
import os

def draw_boxes_and_save_rois(image, contours, output_folder=None):

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    for i, contour in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore very small noise
        if w * h < 100:
            continue

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if output_folder:
            roi = image[y:y+h, x:x+w]
            roi_path = os.path.join(output_folder, f"roi_{i}.png")
            cv2.imwrite(roi_path, roi)

    return image
