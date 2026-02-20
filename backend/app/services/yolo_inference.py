from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.services.model_manager import ModelManager

log = get_logger(__name__)


@dataclass
class Detection:
    det_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[int]  # [x1,y1,x2,y2] in pixels


def _load_image_bgr(image_path: Path) -> np.ndarray:
    # YOLOv5 internally uses cv2 from its utils.general, but we keep image IO here.
    import cv2  # lazy import

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to read image.")
    return img


def run_yolo_detection(image_path: Path, out_dir: Path) -> dict:
    """
    Single-image YOLOv5 inference using the local YOLOv5 repo utilities.
    Produces:
      - annotated.jpg
      - rois/*.jpg (cropped detections)
    Returns structured results used by API.
    """
    mm = ModelManager.get()
    st = mm.state()

    # Local imports from YOLOv5 repo
    # NOTE: We intentionally import from `utils.plots` inside the local YOLOv5 repo
    # instead of `ultralytics.utils.plotting` to avoid requiring the external
    # `ultralytics` pip package.
    from utils.plots import Annotator, colors, save_one_box  # type: ignore
    from utils.general import non_max_suppression, scale_boxes  # type: ignore
    from utils.augmentations import letterbox  # type: ignore

    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    rois_dir = out_dir / "rois"
    rois_dir.mkdir(parents=True, exist_ok=True)

    im0 = _load_image_bgr(image_path)
    h0, w0 = im0.shape[:2]

    # Preprocess
    t_pre0 = time.perf_counter()
    img = letterbox(im0, new_shape=settings.img_size, stride=st.stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(st.model.device)
    im = im.half() if st.model.fp16 else im.float()
    im /= 255.0
    if im.ndim == 3:
        im = im[None]
    t_pre = time.perf_counter() - t_pre0

    # Inference + NMS
    t_inf0 = time.perf_counter()
    pred = st.model(im, augment=False, visualize=False)
    pred = non_max_suppression(
        pred,
        conf_thres=settings.conf_thres,
        iou_thres=settings.iou_thres,
        classes=None,
        agnostic=False,
        max_det=settings.max_det,
    )
    t_inf = time.perf_counter() - t_inf0

    # Postprocess: scale boxes, annotate, crops
    t_post0 = time.perf_counter()
    detections: List[Detection] = []
    annotator = Annotator(im0.copy(), line_width=3, example=str(list(st.names.values())))

    det_id = 0
    det = pred[0]
    if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in det.tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            class_name = st.names.get(class_id, str(class_id))
            confidence = float(conf)
            detections.append(
                Detection(
                    det_id=det_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=[x1, y1, x2, y2],
                )
            )
            label = f"{class_name} {confidence:.2f}"
            annotator.box_label([x1, y1, x2, y2], label, color=colors(class_id, True))

            # Save ROI crop
            # NOTE: YOLOv5's save_one_box expects a Path-like object for `file`,
            # because it uses file.parent. Passing a plain string causes the
            # AttributeError you saw ("'str' object has no attribute 'parent'").
            roi_path = rois_dir / f"roi_{det_id:03d}_{class_name}.jpg"
            save_one_box([x1, y1, x2, y2], im0, file=roi_path, BGR=True)
            det_id += 1

    annotated = annotator.result()

    import cv2  # lazy import

    annotated_path = out_dir / "annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    t_post = time.perf_counter() - t_post0

    per_class: Dict[str, int] = {}
    for d in detections:
        per_class[d.class_name] = per_class.get(d.class_name, 0) + 1

    avg_conf = float(np.mean([d.confidence for d in detections])) if detections else 0.0

    return {
        "image": {
            "path": str(image_path),
            "width": int(w0),
            "height": int(h0),
        },
        "detections": [
            {
                "id": d.det_id,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox_xyxy": d.bbox_xyxy,
            }
            for d in detections
        ],
        "counts": {
            "total_defects": len(detections),
            "roi_count": len(detections),
            "per_class": per_class,
        },
        "metrics": {
            "prediction_accuracy": None,  # requires ground truth
            "average_confidence": avg_conf,
        },
        "timing": {
            "preprocess_s": t_pre,
            "inference_s": t_inf,
            "postprocess_s": t_post,
            "localization_s": t_inf,  # requested: localization time (proxy = model forward + NMS)
            "total_s": t_pre + t_inf + t_post,
        },
        "outputs": {
            "annotated_image": str(annotated_path),
            "rois_dir": str(rois_dir),
        },
        "model": mm.model_details(),
    }
