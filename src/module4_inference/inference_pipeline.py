import cv2
import torch
from pathlib import Path
from typing import Dict, List

from src.module3_training.model import build_model
from src.module2_roi.xml_utils import parse_xml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#  PATHS 

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "efficientnet_pcb.pth"
OUT_DIR = BASE_DIR / "outputs" / "module4_annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]

model = build_model(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def auto_find_template(test_path: Path) -> Path:
    """
    Extract template number from test image name.
    Example:
        01_mouse_bite_03.jpg → template = 01.jpg
    """

    filename = test_path.name
    parts = filename.split("_")

    if len(parts) < 1:
        raise ValueError(f"Invalid filename format: {filename}")

    template_id = parts[0]   # "01"

    template_path = (
        BASE_DIR
        / "data"
        / "raw"
        / "templates"
        / f"{template_id}.jpg"
    )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found for ID {template_id} at {template_path}"
        )

    return template_path


def run_single_pair(template_path=None, test_path=None, save_outputs=True):

    test_path = Path(test_path)
    # If template not given → auto detect
    if template_path is None:
        template_path = auto_find_template(test_path)
    else:
        template_path = Path(template_path)

    # Infer defect class from filename
    fname = test_path.name.lower()

    if "missing_hole" in fname:
        defect_name = "Missing_hole"
    elif "mouse_bite" in fname:
        defect_name = "Mouse_bite"
    elif "open_circuit" in fname:
        defect_name = "Open_circuit"
    elif "short" in fname:
        defect_name = "Short"
    elif "spur" in fname:
        defect_name = "Spur"
    elif "spurious_copper" in fname:
        defect_name = "Spurious_copper"
    else:
        raise ValueError(
            f"Cannot infer defect class from filename: {test_path.name}"
        )

    xml_path = (
        BASE_DIR
        / "data"
        / "raw"
        / "annotations_xml"
        / defect_name
        / test_path.with_suffix(".xml").name
    )

    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    image = cv2.imread(str(test_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {test_path}")

    annotated = image.copy()
    roi_only = image.copy()

    detections: List[Dict] = []

    boxes = parse_xml(xml_path)

    for _, xmin, ymin, xmax, ymax in boxes:
        roi = image[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            continue

        roi_resized = cv2.resize(roi, (128, 128))
        roi_tensor = roi_resized / 255.0
        roi_tensor = roi_tensor.transpose(2, 0, 1)
        roi_tensor = torch.tensor(
            roi_tensor, dtype=torch.float
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(roi_tensor)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, dim=1)

        label = CLASSES[pred.item()]
        confidence = float(conf.item())

        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": [xmin, ymin, xmax, ymax],
        })

        # green box
        cv2.rectangle(roi_only, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Final annotated view 
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        
        font_scale = 0.9
        thickness = 2
        text = f"{label} {confidence:.2f}"

        (label_w, label_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        cv2.rectangle(
            annotated,
            (xmin, max(0, ymin - label_h - 12)),
            (xmin + label_w + 8, ymin),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            annotated,
            text,
            (xmin + 4, ymin - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    annotated_path = OUT_DIR / f"annotated_{test_path.name}"
    roi_path = OUT_DIR / f"roi_{test_path.name}"

    if save_outputs:
        cv2.imwrite(str(annotated_path), annotated)
        cv2.imwrite(str(roi_path), roi_only)

    return {
        "annotated_image": str(annotated_path),
        "roi_image": str(roi_path),
        "detections": detections,
        "model_info": {
            "model": "Custom CNN (PyTorch)",
            "input_size": "128x128",
            "optimizer": "Adam",
            "loss": "CrossEntropy",
            "accuracy": "84%",
        },
    }



def run_inference_pipeline(template_path, test_path, save_outputs=True):
    return run_single_pair(template_path, test_path, save_outputs)
