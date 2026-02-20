from __future__ import annotations

from typing import Any, Dict, List


def summarize_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_class: Dict[str, int] = {}
    for d in detections:
        name = str(d.get("class_name", "unknown"))
        per_class[name] = per_class.get(name, 0) + 1
    return {"total_defects": len(detections), "per_class": per_class}

