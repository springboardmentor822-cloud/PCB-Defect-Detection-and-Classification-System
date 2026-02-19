"""
PIPELINE 4 — ROBUST INDUSTRIAL AOI INFERENCE
(PRODUCTION TIMING – SAME LOGIC)
"""
import sys
from pathlib import Path

# ============================================================
# PATH FIX — ENSURE PROJECT ROOT IS IMPORTABLE
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BASE_DIR = ROOT_DIR  # ✅ Alias for backward compatibility

import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.models import efficientnet_b0
import torch.nn as nn
import time

# ============================================================
# CONFIG
# ============================================================
from config.config_loader import (
    TEMPLATE_DIR,
    STORED_MASK_DIR,
    MODEL_PATHS
)


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
MIN_CONFIDENCE = 0.3
MASK_IOU_THRESHOLD = 0.3
BORDER_MARGIN = 20

# ============================================================
# TEMPLATE REGISTRY
# ============================================================

def list_templates():
    templates = sorted(TEMPLATE_DIR.glob("*"))
    if not templates:
        raise RuntimeError("No templates found")
    return templates

# ============================================================
# TEMPLATE SELECTION (AUTO + MANUAL)  ✅ NEW
# ============================================================

def select_template(test_img, mode="auto", manual_index=None):
    """
    mode:
        - 'auto'   : automatic best-match (default)
        - 'manual' : operator-selected template
    """
    templates = list_templates()

    # ---------------- AUTO (DEFAULT) ----------------
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

    # ---------------- MANUAL ----------------
    if manual_index is None or manual_index < 0 or manual_index >= len(templates):
        raise ValueError("Invalid manual template index")

    return cv2.imread(str(templates[manual_index]))

# ============================================================
# TERMINAL TEMPLATE MODE SELECTOR  ✅ NEW
# ============================================================

def select_template_mode_interactive():
    print("\n[TEMPLATE SELECTION MODE]")
    print(" 1. Automatic (default)")
    print(" 2. Manual")

    choice = input("Select mode [1/2]: ").strip()
    if choice != "2":
        return {"mode": "auto", "index": None}

    templates = list_templates()
    print("\n[AVAILABLE TEMPLATES]")
    for i, p in enumerate(templates, start=1):
        print(f" {i}. {p.name}")

    while True:
        try:
            idx = int(input("Select template number: ")) - 1
            if 0 <= idx < len(templates):
                return {"mode": "manual", "index": idx}
        except:
            pass

# ============================================================
# CONFIDENCE THRESHOLD SELECTOR  ✅ NEW
# ============================================================

def select_confidence_threshold(default=MIN_CONFIDENCE):
    """
    Terminal + Frontend compatible confidence selector
    """
    try:
        print(f"\n[CONFIDENCE THRESHOLD]")
        print(f"Current default: {default}")
        val = input(
            "Enter confidence threshold (0.0 – 1.0) "
            f"[Press Enter to keep {default}]: "
        ).strip()

        if val == "":
            return default

        val = float(val)
        if 0.0 <= val <= 1.0:
            return val

    except:
        pass

    print(f"[INFO] Invalid input. Using default {default}")
    return default


# ============================================================
# MODEL REGISTRY (FIXED OPTIONS)
# ============================================================

MODEL_REGISTRY = {
    "efficientnet": {
        "name": "EfficientNet-B0",
        "type": "efficientnet",
        "path": BASE_DIR / "models" / "efficientnet_b0_pcb_v2.pth"
    },
    "custom": {
        "name": "Custom CNN",
        "type": "custom_cnn",
        "path": BASE_DIR / "models" / "custom_cnn_pcb.pth"
    }
}

DEFAULT_MODEL_KEY = "efficientnet"

# ============================================================
# MODEL SELECTOR
# ============================================================

def select_model(model_key=None):
    if model_key is None:
        print("\n[MODEL SELECTOR]")
        print(" 1. EfficientNet-B0 (default)")
        print(" 2. Custom CNN")

        choice = input("Select model [1/2] (default=1): ").strip()
        if choice == "2":
            return MODEL_REGISTRY["custom"]
        return MODEL_REGISTRY["efficientnet"]

    return MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])

# ============================================================
# MODEL DEFINITIONS
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

        # 🔴 MUST MATCH TRAINED MODEL
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
# MODEL LOADER
# ============================================================

def load_model(model_cfg):
    if model_cfg["type"] == "efficientnet":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, len(CLASSES)
        )
    else:
        model = CustomCNN(len(CLASSES))

    model.load_state_dict(
        torch.load(model_cfg["path"], map_location=DEVICE)
    )
    model.to(DEVICE).eval()

    print(f"[INFO] Model loaded on {DEVICE}")
    print(f"[INFO] Selected model: {model_cfg['name']}")

    return model

# ============================================================
# INITIALIZATION (MODEL + TEMPLATE MODE)  ✅ NEW
# ============================================================
if __name__ == "__main__":
    SELECTED_MODEL = select_model()
    TEMPLATE_MODE = select_template_mode_interactive()
    MIN_CONFIDENCE = select_confidence_threshold(MIN_CONFIDENCE)
else:
    SELECTED_MODEL = select_model("efficientnet")
    TEMPLATE_MODE = {"mode": "auto", "index": None}
    # Frontend can override this
    MIN_CONFIDENCE = MIN_CONFIDENCE

# ============================================================
# GLOBAL MODEL STATE (FOR FRONTEND SWITCHING)
# ============================================================
CURRENT_MODEL_KEY = DEFAULT_MODEL_KEY

MODEL = load_model(SELECTED_MODEL)


# ============================================================
# PIPELINE CONTEXT (ROUTER-COMPATIBLE) ✅ NEW
# ============================================================

def set_active_model(model):
    """
    Allows external router to hot-switch models
    without reloading the entire pipeline.
    """
    global MODEL
    MODEL = model

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
# UTILS (UNCHANGED CORE AOI LOGIC)
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


def classify_roi(roi, min_confidence):

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(roi).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(MODEL(tensor), dim=1)[0]

    conf, idx = torch.max(probs, dim=0)
    if conf.item() < min_confidence:

        return None, conf.item()

    return CLASSES[idx.item()], conf.item()

# ============================================================
# PIPELINE 4 — CORE LOGIC UNCHANGED
# ============================================================
def run_inference(
    image_path,
    template_mode="auto",
    manual_template_index=None,
    min_confidence=MIN_CONFIDENCE,
    model_key=DEFAULT_MODEL_KEY   # ✅ ADD THIS
):



    timing = {}
    t_total = time.perf_counter()

    # IMAGE LOAD
    t = time.perf_counter()
    test_img = cv2.imread(str(image_path))
    timing["image_load"] = time.perf_counter() - t
    if test_img is None:
        raise ValueError("Image not found")

    # TEMPLATE + REGISTRATION
    t = time.perf_counter()
    template = select_template(
        test_img,
        mode=template_mode,
        manual_index=manual_template_index
    )

    template = cv2.resize(template, test_img.shape[1::-1])
    aligned = register_image(template, test_img)
    timing["template_stage"] = time.perf_counter() - t

    # MASK PIPELINE
    t = time.perf_counter()
    auto_mask = generate_diff_mask(template, aligned)
    stored_mask = load_stored_mask(image_path)
    final_mask = validate_mask(auto_mask, stored_mask)
    final_mask = suppress_mask_borders(final_mask, BORDER_MARGIN)
    timing["mask_stage"] = time.perf_counter() - t

    # ROI EXTRACTION
    t = time.perf_counter()
    rois = extract_rois(final_mask, aligned)
    timing["roi_extraction"] = time.perf_counter() - t

    # CNN + DRAW
    t = time.perf_counter()
    annotated = test_img.copy()
    counts = {c: 0 for c in CLASSES}
    detections = []
    H, W = test_img.shape[:2]

    for cnt, roi, (x, y, w, h) in rois:
        label, conf = classify_roi(roi, min_confidence)

        # Reject low-confidence
        if label is None:
            continue

        # Reject border Open-circuit (AOI rule)
        if label == "Open_circuit" and (
            x <= BORDER_MARGIN or
            y <= BORDER_MARGIN or
            x + w >= W - BORDER_MARGIN or
            y + h >= H - BORDER_MARGIN
        ):
            continue

        # ✅ VALID detection — record it
        detections.append({
            "label": label,
            "confidence": float(conf),
            "bbox": (int(x), int(y), int(w), int(h))
        })

        # Draw & count
        counts[label] += 1
        cv2.rectangle(
            annotated, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )
        cv2.putText(
            annotated,
            f"{label} ({conf:.2f})",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

    timing["cnn_stage"] = time.perf_counter() - t
    timing["total"] = time.perf_counter() - t_total

    return {
        "annotated": annotated,
        "counts": counts,
        "timing": timing,
        "detections": detections,

        # 🔽 DEBUG / FRONTEND SUPPORT (NO LOGIC CHANGE)
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

#
# HEATMAP
#
def generate_defect_heatmap(image, detections):
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)

    for d in detections:
        x, y, w, h = d["bbox"]

        # 🔥 Shape-aware heat (NO point-based heat)
        heatmap[y:y+h, x:x+w] += 1

    # 🟡 Wider halo (yellow spread)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=22)

    # 🔵 Dark background, bright defects
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.clip(heatmap * 1.3 + 5, 0, 255).astype(np.uint8)

    # 🎨 Balanced colormap (blue → green → yellow → red)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    # 🟢 Overlay
    overlay = cv2.addWeighted(
        image, 0.6,
        heatmap_color, 0.4,
        0
    )

    return overlay


# ============================================================
# DEBUG
# ============================================================
if __name__ == "__main__":
    test_image = Path(
        r"original.jpg"
    )

    result = run_inference(
    image_path=test_image,
    template_mode=TEMPLATE_MODE["mode"],
    manual_template_index=TEMPLATE_MODE["index"]
    )


    # 🔹 Per-process timing print
    if "timing" in result:
        print("\n[AOI TIMING — PRODUCTION MODE]")
        for k, v in result["timing"].items():
            print(f" {k:<18}: {v:.3f}s")

    # Existing prints (UNCHANGED)
    print("\nCounts:", result["counts"])

    # 🔹 NEW: Print detections
    if "detections" in result:
        print("\n[DETECTIONS]")
        if len(result["detections"]) == 0:
            print(" No defects detected")
        else:
            for i, d in enumerate(result["detections"], start=1):
                print(
                    f" {i}. {d['label']} | "
                    f"conf={d['confidence']:.2f} | "
                    f"bbox={d['bbox']}"
                )
    
    # Inference time
    if "timing" in result and "total" in result["timing"]:
        print(f"\nInference time: {result['timing']['total']:.2f}s")
    else:
        print(f"\nInference time: {result['inference_time']:.2f}s")

    # Save output
    out = Path("outputs/pipeline4_result.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result["annotated"])
    # 🔹 SYSTEM INFO
    h, w = result["annotated"].shape[:2]

    print("\n[SYSTEM INFO]")
    print(f" Device           : {DEVICE}")
    print(f" Model            : {SELECTED_MODEL['name']}")
    print(f" Image Resolution : {w}x{h}")
    print(f" ROIs Processed   : {len(result.get('detections', []))}")
    print(f" Detections       : {len(result.get('detections', []))}")
        
    # Inference time
    if "timing" in result and "total" in result["timing"]:
        print(f"\nInference time: {result['timing']['total']:.2f}s")
    else:
        print(f"\nInference time: {result['inference_time']:.2f}s")

        
    print(f"[INFO] Result saved to {out}")
        # 🔥 DEFECT HEATMAP OVERLAY
    if "detections" in result and len(result["detections"]) > 0:
        heatmap_overlay = generate_defect_heatmap(
            result["annotated"],  
            result["detections"]
        )


        heatmap_out = Path("outputs/pipeline4_heatmap.jpg")
        cv2.imwrite(str(heatmap_out), heatmap_overlay)

        print(f"[INFO] Defect heatmap saved to {heatmap_out}")
    else:
        print("[INFO] No detections — heatmap not generated")

# ============================================================
# ROUTER EXPORT CONTRACT (DO NOT MODIFY)
# ============================================================

__all__ = [
    "run_inference",
    "generate_defect_heatmap",
    "list_templates",
]

