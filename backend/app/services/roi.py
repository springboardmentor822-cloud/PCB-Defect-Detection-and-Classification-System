from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def extract_rois(image_bgr: np.ndarray, detections: List[Dict[str, Any]], rois_dir: Path) -> List[str]:
    """
    Save ROI crops for each detection bbox.
    This is separated for extensibility (e.g., ROI feature extraction / secondary classifier).
    """
    import cv2

    rois_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    h, w = image_bgr.shape[:2]

    for d in detections:
        bbox = d.get("bbox_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        out = rois_dir / f"roi_{int(d.get('id', 0)):03d}.jpg"
        cv2.imwrite(str(out), crop)
        paths.append(str(out))
    return paths

