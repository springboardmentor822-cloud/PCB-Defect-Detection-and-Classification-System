from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.logging import get_logger
from app.services.pipeline import run_detection

router = APIRouter()
log = get_logger(__name__)


@router.post("/detect")
async def detect(
    mode: str = Form("auto"),
    image: UploadFile = File(...),
    template_image: UploadFile | None = File(None),
) -> dict:
    """
    Run PCB inspection.

    - mode="auto": YOLOv5 detection only
    - mode="template": golden template comparison + YOLO (combined)
    """
    mode = (mode or "auto").strip().lower()
    if mode not in {"auto", "template"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'auto' or 'template'.")

    if not image.filename:
        raise HTTPException(status_code=400, detail="No image filename provided.")

    if mode == "template" and (template_image is None or not template_image.filename):
        raise HTTPException(status_code=400, detail="template_image is required for template mode.")

    try:
        result = await run_detection(
            mode=mode,
            image_upload=image,
            template_upload=template_image,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Detection failed: %s", e)
        raise HTTPException(status_code=500, detail="Detection failed. Check server logs.")
