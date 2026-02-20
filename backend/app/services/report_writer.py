from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def write_json_report(run_dir: Path, payload: Dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "report.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def write_csv_report(run_dir: Path, detections: List[Dict[str, Any]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "report.csv"
    fieldnames = ["id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in detections:
            x1, y1, x2, y2 = d.get("bbox_xyxy", [None, None, None, None])
            w.writerow(
                {
                    "id": d.get("id"),
                    "class_id": d.get("class_id"),
                    "class_name": d.get("class_name"),
                    "confidence": d.get("confidence"),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return path
