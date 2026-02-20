from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from fastapi import UploadFile

from app.core.logging import get_logger
from app.services.report_writer import write_csv_report, write_json_report
from app.services.template_compare import align_and_diff
from app.services.yolo_inference import run_yolo_detection
from app.utils.files import new_run_id, run_output_dir, save_upload

log = get_logger(__name__)


def _public_outputs(run_id: str) -> Dict[str, str]:
    base = f"/outputs/{run_id}"
    return {
        "annotated_image_url": f"{base}/annotated.jpg",
        "diff_image_url": f"{base}/diff.jpg",
        "overlay_image_url": f"{base}/overlay.jpg",
        "report_json_url": f"{base}/report.json",
        "report_csv_url": f"{base}/report.csv",
        "rois_dir_url": f"{base}/rois",
    }


async def run_detection(
    mode: str,
    image_upload: UploadFile,
    template_upload: UploadFile | None,
) -> Dict[str, Any]:
    run_id = new_run_id()
    out_dir = run_output_dir(run_id)

    t_total0 = time.perf_counter()
    test_path = await save_upload(image_upload, run_id=run_id, role="test")

    template_path: Path | None = None
    if mode == "template" and template_upload is not None:
        template_path = await save_upload(template_upload, run_id=run_id, role="template")

    # YOLO detection
    t_yolo0 = time.perf_counter()
    yolo = run_yolo_detection(test_path, out_dir=out_dir)
    t_yolo = time.perf_counter() - t_yolo0

    # Template comparison (optional)
    template_result: Dict[str, Any] | None = None
    if mode == "template" and template_path is not None:
        t_tpl0 = time.perf_counter()
        template_result = align_and_diff(template_path, test_path, out_dir=out_dir)
        t_tpl = time.perf_counter() - t_tpl0
    else:
        t_tpl = 0.0

    # Compose evaluation panel metrics
    detections = yolo.get("detections", [])
    avg_conf = yolo.get("metrics", {}).get("average_confidence", 0.0)
    total_defects = yolo.get("counts", {}).get("total_defects", 0)

    # Structured payload
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "status": "success",
        "inputs": {
            "test_image": str(test_path),
            "template_image": str(template_path) if template_path else None,
        },
        "results": {
            "yolo": yolo,
            "template_comparison": template_result,
        },
        "summary": {
            "total_defects": int(total_defects),
            "average_confidence": float(avg_conf),
        },
        "evaluation": {
            "prediction_accuracy": None,  # requires ground-truth labels
            "average_confidence": float(avg_conf),
            "inference_performance_s": float(yolo.get("timing", {}).get("inference_s", 0.0)),
        },
        "pipeline_breakdown": [
            {"stage": "preprocessing", "time_s": float(yolo["timing"]["preprocess_s"])},
            {"stage": "detection", "time_s": float(yolo["timing"]["inference_s"])},
            {"stage": "localization", "time_s": float(yolo["timing"]["localization_s"])},
            {"stage": "classification", "time_s": 0.0},  # YOLO provides class labels
            {"stage": "postprocessing", "time_s": float(yolo["timing"]["postprocess_s"])},
            {"stage": "template_comparison", "time_s": float(t_tpl)},
        ],
        "timing": {
            "yolo_total_s": float(t_yolo),
            "template_total_s": float(t_tpl),
            "total_processing_s": float(time.perf_counter() - t_total0),
        },
        "outputs": {
            "run_dir": str(out_dir),
            "annotated_image": yolo["outputs"]["annotated_image"],
            "diff_image": (template_result or {}).get("outputs", {}).get("diff_image") if template_result else None,
            "overlay_image": (template_result or {}).get("outputs", {}).get("overlay_image") if template_result else None,
            "report_json": str(out_dir / "report.json"),
            "report_csv": str(out_dir / "report.csv"),
            "public_urls": _public_outputs(run_id),
        },
        "model": yolo.get("model"),
    }

    # Persist reports
    write_json_report(out_dir, payload)
    write_csv_report(out_dir, detections)

    log.info("Run %s complete: defects=%s time=%.2fs", run_id, total_defects, payload["timing"]["total_processing_s"])
    return payload
