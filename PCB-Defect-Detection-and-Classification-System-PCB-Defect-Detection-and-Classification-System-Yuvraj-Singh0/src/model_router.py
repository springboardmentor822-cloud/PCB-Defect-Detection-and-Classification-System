"""
MODEL ROUTER
Routes inference to correct pipeline and enforces output schema
"""

from src.inference_pipeline import run_inference as run_efficientnet
from src.pipelinecnn import run_inference as run_customcnn


# ============================================================
# OUTPUT NORMALIZER
# ============================================================

def normalize_output(result):
    """
    Enforce identical output schema for frontend
    """

    # Ensure mandatory keys always exist
    result.setdefault("detections", [])
    result.setdefault("counts", {})
    result.setdefault("timing", {})
    result.setdefault("rois", [])

    # Normalize ROI format ONLY if needed (EfficientNet pipeline)
    if result["rois"] and isinstance(result["rois"][0], tuple):
        # (cnt, roi, (x,y,w,h)) → dict format
        result["rois"] = [
            {
                "bbox": bbox,
                "roi": roi
            }
            for _, roi, bbox in result["rois"]
        ]

    return result


# ============================================================
# MODEL ROUTER — SINGLE ENTRY POINT
# ============================================================

def run_model_router(
    image_path,
    model_key="efficientnet",
    template_mode="auto",
    manual_template_index=None,
    min_confidence=0.3
):
    """
    Unified inference entry point (HOT-SWITCH SAFE)
    """

    # ---------------- EfficientNet Pipeline ----------------
    if model_key == "efficientnet":
        result = run_efficientnet(
            image_path=image_path,
            template_mode=template_mode,
            manual_template_index=manual_template_index,
            min_confidence=min_confidence
        )

    # ---------------- Custom CNN Pipeline ----------------
    elif model_key == "custom":
        result = run_customcnn(
            image_path=image_path,
            template_mode=template_mode,
            manual_template_index=manual_template_index
        )

    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    # ---------------- Normalize Output ----------------
    result = normalize_output(result)

    # Attach model info (frontend-safe)
    result["model"] = model_key

    return result
