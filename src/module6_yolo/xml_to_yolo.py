import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random


BASE_DIR = Path(__file__).resolve().parents[2]

XML_ROOT = BASE_DIR / "data/raw/annotations_xml"
IMAGE_ROOT = BASE_DIR / "data/raw/test_images"
YOLO_ROOT = BASE_DIR / "data/yolo_dataset"

TRAIN_SPLIT = 0.8

CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

class_to_id = {name: i for i, name in enumerate(CLASSES)}


for split in ["train", "val"]:
    (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


xml_files = list(XML_ROOT.rglob("*.xml"))

if len(xml_files) == 0:
    print("No XML files found. Check folder path.")
    exit()

random.shuffle(xml_files)

split_index = int(len(xml_files) * TRAIN_SPLIT)
train_files = xml_files[:split_index]
val_files = xml_files[split_index:]

print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")


def convert(xml_file, split):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    class_name = xml_file.parent.name

    if class_name not in class_to_id:
        print(f"Skipping unknown class: {class_name}")
        return

    class_id = class_to_id[class_name]

    image_path = None
    for defect_folder in IMAGE_ROOT.iterdir():
        candidate = defect_folder / filename
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"⚠ Image not found for {filename}")
        return

    shutil.copy(
        image_path,
        YOLO_ROOT / "images" / split / filename
    )

    label_path = (
        YOLO_ROOT
        / "labels"
        / split
        / filename.replace(".jpg", ".txt")
    )

    with open(label_path, "w") as f:
        for obj in root.findall("object"):

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            f.write(
                f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"
            )


for xml_file in train_files:
    convert(xml_file, "train")

for xml_file in val_files:
    convert(xml_file, "val")

print(" Conversion Completed Successfully.")
