import cv2
from pathlib import Path
from xml_utils import parse_xml

BASE = Path(__file__).resolve().parents[2]

IMG_ROOT = BASE / "data/raw/test_images"
XML_ROOT = BASE / "data/raw/annotations_xml"
ROI_ROOT = BASE / "data/processed/rois"
VIS_ROOT = BASE / "data/processed/roi_visuals"

ROI_ROOT.mkdir(parents=True, exist_ok=True)
VIS_ROOT.mkdir(parents=True, exist_ok=True)

total_rois = 0

for defect_dir in XML_ROOT.iterdir():
    if not defect_dir.is_dir():
        continue

    defect = defect_dir.name
    print(f"\n Processing defect: {defect}")

    (ROI_ROOT / defect).mkdir(exist_ok=True)
    (VIS_ROOT / defect).mkdir(exist_ok=True)

    for xml_file in defect_dir.glob("*.xml"):
        img_name = xml_file.stem + ".jpg"
        img_path = IMG_ROOT / defect / img_name

        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        boxes = parse_xml(xml_file)

        vis = image.copy()

        for label, x1, y1, x2, y2 in boxes:
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_name = f"{xml_file.stem}_{label}.jpg"
            cv2.imwrite(str(ROI_ROOT / defect / roi_name), roi)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

            total_rois += 1

        cv2.imwrite(str(VIS_ROOT / defect / img_name), vis)

print(f"\n Module 2 completed successfully.")
print(f" Total ROIs extracted: {total_rois}")
