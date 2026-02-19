"""
PIPELINE 4 — ROBUST INDUSTRIAL AOI INFERENCE
CUSTOM CNN VARIANT
(SAME AOI LOGIC, FIXED MODEL)
"""

import sys
from pathlib import Path
# -------------------------------------------------
# PROJECT ROOT RESOLUTION (CRITICAL)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
import torch.nn as nn
import time

# ============================================================
# CONFIG
#  ============================================================

# BASE_DIR = Path(".")
# MODEL_PATH = BASE_DIR / "models" / "custom_cnn_pcb.pth"

# from config.config_loader import MODEL_PATHS
# -------------------------------------------------
# MODEL PATH (CUSTOM CNN)
# -------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "custom_cnn_pcb.pth"

# Safety check (recommended)
assert MODEL_PATH.exists(), f"Custom CNN model not found: {MODEL_PATH}"



TEMPLATE_DIR = BASE_DIR / "PCB_DATASET" / "PCB_DATASET" / "PCB_USED"
STORED_MASK_DIR = BASE_DIR / "data" / "processed" / "diff_masks"

CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_ROI_SIZE = 15
MIN_CONFIDENCE = 0.10
MASK_IOU_THRESHOLD = 0.3
BORDER_MARGIN = 20



# ============================================================
# PIPELINE METADATA (ROUTER / FRONTEND COMPATIBILITY)
# ============================================================

PIPELINE_NAME = "pipeline4_customcnn"
PIPELINE_MODEL_KEY = "custom"
PIPELINE_MODEL_NAME = "Custom CNN"


# ============================================================
# TEMPLATE REGISTRY
# ============================================================

def list_templates():
    templates = sorted(TEMPLATE_DIR.glob("*"))
    if not templates:
        raise RuntimeError("No templates found")
    return templates

# ============================================================
# TEMPLATE SELECTION (AUTO + MANUAL)
# ============================================================

def select_template(test_img, mode="auto", manual_index=None):
    templates = list_templates()

    if mode == "auto":
        best_score, best_tpl = -1, None
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        for tpl_path in templates:
            tpl = cv2.imread(str(tpl_path))
            if tpl is None:
                continue

            gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, test_gray.shape[::-1])

            score = cv2.matchTemplate(
                test_gray, gray, cv2.TM_CCOEFF_NORMED
            )[0][0]

            if score > best_score:
                best_score, best_tpl = score, tpl

        if best_tpl is None:
            raise RuntimeError("Automatic template selection failed")

        return best_tpl

    if manual_index is None or manual_index < 0 or manual_index >= len(templates):
        raise ValueError("Invalid manual template index")

    return cv2.imread(str(templates[manual_index]))

# ============================================================
# MODEL DEFINITION — CUSTOM CNN (MATCH TRAINING)
# ============================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================
# LOAD MODEL (FIXED)
# ============================================================

MODEL = CustomCNN(len(CLASSES))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

MODEL.load_state_dict(state_dict, strict=False)
MODEL.to(DEVICE).eval()

print(f"[INFO] Custom CNN loaded on {DEVICE}")

# ============================================================
# TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# AOI UTILS (UNCHANGED)
# ============================================================

def register_image(template, test_img):
    try:
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50, 1e-6
        )

        _, warp = cv2.findTransformECC(
            tpl_gray, img_gray, warp,
            cv2.MOTION_AFFINE, criteria
        )

        return cv2.warpAffine(
            test_img, warp,
            (template.shape[1], template.shape[0]),
            flags=cv2.INTER_LINEAR
        )
    except:
        return test_img

def generate_diff_mask(template, img):
    diff = cv2.absdiff(template, img)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
    mask = cv2.dilate(mask, kernel, 1)
    return mask

def load_stored_mask(image_path):
    mask_path = STORED_MASK_DIR / Path(image_path).name
    if mask_path.exists():
        return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return None

def validate_mask(auto_mask, stored_mask):
    if stored_mask is None:
        return auto_mask

    a = auto_mask > 0
    s = stored_mask > 0

    inter = np.logical_and(a, s).sum()
    union = np.logical_or(a, s).sum() + 1e-6
    iou = inter / union

    if iou < MASK_IOU_THRESHOLD:
        return stored_mask

    return cv2.bitwise_or(auto_mask, stored_mask)

def suppress_mask_borders(mask, margin):
    h, w = mask.shape
    mask[:margin, :] = 0
    mask[h-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, w-margin:] = 0
    return mask

def extract_rois(mask, image):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_ROI_SIZE or h < MIN_ROI_SIZE:
            continue
        roi = image[y:y+h, x:x+w]
        rois.append((cnt, roi, (x, y, w, h)))
    return rois

def classify_roi(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(roi).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(MODEL(tensor), dim=1)[0]

    conf, idx = torch.max(probs, dim=0)
    if conf.item() < MIN_CONFIDENCE:
        return None, conf.item()

    return CLASSES[idx.item()], conf.item()

# ============================================================
# PIPELINE 4 — CUSTOM CNN
# ============================================================

def run_inference(
    image_path,
    template_mode="auto",
    manual_template_index=None,
    model_key=None,          # 🔹 ignored (for router compatibility)
    min_confidence=None      # 🔹 optional override
):

    timing = {}
    t_total = time.perf_counter()

    test_img = cv2.imread(str(image_path))
    if test_img is None:
        raise ValueError("Image not found")

    template = select_template(
        test_img,
        mode=template_mode,
        manual_index=manual_template_index
    )

    template = cv2.resize(template, test_img.shape[1::-1])
    aligned = register_image(template, test_img)

    auto_mask = generate_diff_mask(template, aligned)
    stored_mask = load_stored_mask(image_path)
    final_mask = validate_mask(auto_mask, stored_mask)
    final_mask = suppress_mask_borders(final_mask, BORDER_MARGIN)

    rois = extract_rois(final_mask, aligned)

    annotated = test_img.copy()
    counts = {c: 0 for c in CLASSES}
    detections = []

    H, W = test_img.shape[:2]

    for _, roi, (x, y, w, h) in rois:
        label, conf = classify_roi(roi)

        if label is None:
            continue


        if label == "Open_circuit" and (
            x <= BORDER_MARGIN or
            y <= BORDER_MARGIN or
            x + w >= W - BORDER_MARGIN or
            y + h >= H - BORDER_MARGIN
        ):
            continue

        detections.append({
            "label": label,
            "confidence": float(conf),
            "bbox": (x, y, w, h)
        })

        counts[label] += 1
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            annotated, f"{label} ({conf:.2f})",
            (x, y-6), cv2.FONT_HERSHEY_SIMPLEX,
            1.1, (0,255,0), 2
        )

    timing["total"] = time.perf_counter() - t_total

    return {
        "annotated": annotated,
        "counts": counts,
        "detections": detections,
        "timing": timing,
        "template": template,
        "auto_mask": auto_mask,
        "stored_mask": stored_mask,
        "final_mask": final_mask,
        "rois": [
            {
                "bbox": (x, y, w, h),
                "roi": roi
            }
            for _, roi, (x, y, w, h) in rois
        ]
    }


# ============================================================
# DEFECT HEATMAP (SAME AS PIPELINE 4)
# ============================================================
def generate_defect_heatmap(image, detections):
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)

    for d in detections:
        x, y, w, h = d["bbox"]

        # Shape-aware heat
        heatmap[y:y+h, x:x+w] += 1

    # Wider halo
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=22)

    # Dark background, bright defects
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.clip(heatmap * 1.3 + 5, 0, 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    overlay = cv2.addWeighted(
        image, 0.6,
        heatmap_color, 0.4,
        0
    )

    return overlay

# ============================================================
# DEBUG / TERMINAL MODE (SAME AS PIPELINE 4)
# ============================================================
if __name__ == "__main__":

    print("\n[PIPELINE 4 – CUSTOM CNN]")

    # ---------------- TEMPLATE MODE ----------------
    print("\n[TEMPLATE SELECTION MODE]")
    print(" 1. Automatic (default)")
    print(" 2. Manual")

    choice = input("Select mode [1/2]: ").strip()

    if choice == "2":
        templates = list_templates()
        print("\n[AVAILABLE TEMPLATES]")
        for i, p in enumerate(templates, start=1):
            print(f" {i}. {p.name}")

        while True:
            try:
                idx = int(input("Select template number: ")) - 1
                if 0 <= idx < len(templates):
                    template_mode = "manual"
                    manual_index = idx
                    break
            except:
                pass
    else:
        template_mode = "auto"
        manual_index = None

    # ---------------- TEST IMAGE ----------------
    test_image = Path(
        r"PCB_DATASET\PCB_DATASET\images\Missing_hole\01_missing_hole_01.jpg"
    )

    print(f"\n[INFO] Running AOI on: {test_image.name}")
    print("[INFO] Model        : Custom CNN")
    print(f"[INFO] Device       : {DEVICE}")

    # ---------------- RUN INFERENCE ----------------
    start = time.perf_counter()

    result = run_inference(
        image_path=test_image,
        template_mode=template_mode,
        manual_template_index=manual_index
    )

    total_time = time.perf_counter() - start

    # ---------------- TIMING ----------------
    print("\n[AOI TIMING — PRODUCTION MODE]")
    for k, v in result["timing"].items():
        print(f" {k:<18}: {v:.3f}s")

    print(f" total             : {total_time:.3f}s")

    # ---------------- COUNTS ----------------
    print("\n[COUNTS]")
    for k, v in result["counts"].items():
        print(f" {k:<18}: {v}")

    # ---------------- DETECTIONS ----------------
    print("\n[DETECTIONS]")
    if not result["detections"]:
        print(" No defects detected")
    else:
        for i, d in enumerate(result["detections"], start=1):
            print(
                f" {i}. {d['label']} | "
                f"conf={d['confidence']:.2f} | "
                f"bbox={d['bbox']}"
            )

    # ---------------- SAVE OUTPUT ----------------
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    out_img = out_dir / "pipeline4_customcnn_result.jpg"
    cv2.imwrite(str(out_img), result["annotated"])

    print(f"\n[INFO] Result saved to {out_img}")

    # ---------------- HEATMAP ----------------
    if result["detections"]:
        heatmap = generate_defect_heatmap(
            result["annotated"],
            result["detections"]
        )

        heatmap_out = out_dir / "pipeline4_customcnn_heatmap.jpg"
        cv2.imwrite(str(heatmap_out), heatmap)

        print(f"[INFO] Heatmap saved to {heatmap_out}")
    else:
        print("[INFO] No detections — heatmap not generated")

    # ---------------- SYSTEM INFO ----------------
    h, w = result["annotated"].shape[:2]

    print("\n[SYSTEM INFO]")
    print(f" Device           : {DEVICE}")
    print(f" Model            : Custom CNN")
    print(f" Image Resolution : {w}x{h}")
    print(f" ROIs Processed   : {len(result['rois'])}")
    print(f" Detections       : {len(result['detections'])}")

# ============================================================
# ROUTER EXPORT (FOR MODEL ROUTER & FRONTEND)
# ============================================================

__all__ = [
    "run_inference",
    "generate_defect_heatmap",
    "list_templates",
    "PIPELINE_NAME",
    "PIPELINE_MODEL_KEY",
    "PIPELINE_MODEL_NAME"
]
