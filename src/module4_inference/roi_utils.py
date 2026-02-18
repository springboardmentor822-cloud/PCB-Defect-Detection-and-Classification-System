import cv2
from src.module2_roi.xml_utils import parse_xml

def extract_rois_from_xml(image, xml_path):
    """
    Returns list of:
    (roi_image, (x, y, w, h), label_from_xml)
    """
    h, w = image.shape[:2]
    boxes = parse_xml(xml_path)

    rois = []
    for label, xmin, ymin, xmax, ymax in boxes:
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        roi = image[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            continue

        rois.append((roi, (xmin, ymin, xmax - xmin, ymax - ymin), label))

    return rois
