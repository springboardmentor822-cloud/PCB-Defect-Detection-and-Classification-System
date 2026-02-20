import cv2
import os
import numpy as np

TEMPLATE_DIR = "PCB_DATASET/PCB_DATASET/PCB_USED"
DEFECT_ROOT = "PCB_DATASET/PCB_DATASET/images"
OUTPUT_ROOT = "bounding_box_outputs"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

AREA_THRESHOLD = 50

for defect_type in os.listdir(DEFECT_ROOT):
    defect_dir = os.path.join(DEFECT_ROOT, defect_type)
    save_dir = os.path.join(OUTPUT_ROOT, defect_type)

    os.makedirs(save_dir, exist_ok=True)

    print(f"\nProcessing defect type: {defect_type}")

    for img_name in os.listdir(defect_dir):
        defect_path = os.path.join(defect_dir, img_name)

        template_id = img_name.split("_")[0]
        template_path = os.path.join(TEMPLATE_DIR, f"{template_id}.JPG")

        if not os.path.exists(template_path):
            print(f"Template missing for {img_name}")
            continue

        template = cv2.imread(template_path)
        defect = cv2.imread(defect_path)

        defect = cv2.resize(defect, (template.shape[1], template.shape[0]))

        temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        defect_gray = cv2.cvtColor(defect, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(temp_gray, defect_gray)

        _, mask = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxed_img = defect.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) < AREA_THRESHOLD:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Draw bounding box
            cv2.rectangle(
                boxed_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # 👇 PUT DEFECT LABEL TEXT
            cv2.putText(
                boxed_img,
                defect_type,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        save_path = os.path.join(
            save_dir,
            img_name.replace(".jpg", "").replace(".png", "") + "_labeled.png"
        )

        cv2.imwrite(save_path, boxed_img)
        print(f"Saved labeled bounding box for {img_name}")

print("\n✅ Bounding boxes with labels generated successfully.")
