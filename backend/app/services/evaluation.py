from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def evaluate_predictions(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight evaluation for demo dashboards.
    True accuracy requires ground-truth labels; we expose proxy metrics only.
    """
    confs = [float(d.get("confidence", 0.0)) for d in detections]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return {
        "prediction_accuracy": None,
        "average_confidence": avg_conf,
        "num_predictions": len(detections),
    }

