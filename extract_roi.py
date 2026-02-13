import os
import cv2
import xml.etree.ElementTree as ET

DATASET_PATH = "dataset"
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
ANNOTATION_PATH = os.path.join(DATASET_PATH, "Annotations")
OUTPUT_PATH = "roi_dataset"


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


for defect_class in os.listdir(ANNOTATION_PATH):

    class_xml_folder = os.path.join(ANNOTATION_PATH, defect_class)
    class_image_folder = os.path.join(IMAGE_PATH, defect_class)

    output_class_folder = os.path.join(OUTPUT_PATH, defect_class)
    os.makedirs(output_class_folder, exist_ok=True)

    print(f"Processing class: {defect_class}")

    for xml_file in os.listdir(class_xml_folder):

        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(class_xml_folder, xml_file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        image_path = os.path.join(class_image_folder, filename)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            continue

        count = 0

      
        for obj in root.findall("object"):

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            roi = image[ymin:ymax, xmin:xmax]

            if roi.size == 0:
                continue

            roi = cv2.resize(roi, (128, 128))

            roi_filename = f"{filename.replace('.jpg','')}_{count}.jpg"
            save_path = os.path.join(output_class_folder, roi_filename)

            cv2.imwrite(save_path, roi)
            count += 1

print("ROI extraction completed successfully.")
