from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.core.config import settings


@dataclass
class DiffRegion:
    region_id: int
    bbox_xyxy: List[int]
    area: int


def _read_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def align_and_diff(template_path: Path, test_path: Path, out_dir: Path) -> dict:
    """
    Align test image to template (golden PCB), compute structural differences,
    and save visualizations:
      - diff.jpg (difference heat/overlay)
      - overlay.jpg (test overlaid with diff regions)
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    tpl = _read_bgr(template_path)
    tst = _read_bgr(test_path)

    # Resize test to template resolution if needed (keeps it simple for demos)
    if tpl.shape[:2] != tst.shape[:2]:
        tst = cv2.resize(tst, (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_AREA)

    # Alignment (ORB + homography)
    t_align0 = time.perf_counter()
    gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    gray_tst = cv2.cvtColor(tst, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray_tpl, None)
    kp2, des2 = orb.detectAndCompute(gray_tst, None)
    aligned = tst
    H = None

    if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:300]
        if len(matches) >= 8:
            src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                aligned = cv2.warpPerspective(tst, H, (tpl.shape[1], tpl.shape[0]))

    # If ORB failed, fall back to simple identity alignment
    t_align = time.perf_counter() - t_align0

    # Diff
    t_diff0 = time.perf_counter()
    g1 = cv2.GaussianBlur(gray_tpl, (5, 5), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(diff, settings.template_diff_threshold, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=2)

    # Find regions
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[DiffRegion] = []
    rid = 0
    overlay = aligned.copy()
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < settings.template_min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        regions.append(DiffRegion(region_id=rid, bbox_xyxy=[int(x), int(y), int(x + w), int(y + h)], area=area))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rid += 1

    # Visualize diff as heatmap overlay
    heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    diff_vis = cv2.addWeighted(aligned, 0.65, heat, 0.35, 0)

    diff_path = out_dir / "diff.jpg"
    overlay_path = out_dir / "overlay.jpg"
    cv2.imwrite(str(diff_path), diff_vis)
    cv2.imwrite(str(overlay_path), overlay)
    t_diff = time.perf_counter() - t_diff0

    return {
        "template": {"path": str(template_path)},
        "alignment": {
            "homography_found": H is not None,
            "time_s": t_align,
        },
        "diff": {
            "time_s": t_diff,
            "regions": [{"id": r.region_id, "bbox_xyxy": r.bbox_xyxy, "area": r.area} for r in regions],
            "region_count": len(regions),
        },
        "outputs": {"diff_image": str(diff_path), "overlay_image": str(overlay_path)},
        "timing": {"total_s": time.perf_counter() - t0},
    }
