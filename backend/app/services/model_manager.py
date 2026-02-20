from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class _ModelState:
    device: str
    names: Dict[int, str]
    stride: int
    model: Any
    weights_path: str
    fp16: bool


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self) -> None:
        self._state: _ModelState | None = None

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    @property
    def device(self) -> str:
        return self._state.device if self._state else "unloaded"

    def _ensure_weights_present(self) -> Path:
        requested = Path(settings.model_path)
        if requested.exists():
            return requested

        # If missing, try to auto-copy from a common local location found on this machine.
        # fallback = Path("/Users/haseeb/Downloads/content/yolov5/runs/train/pcb_1st/weights/best.pt")
        fallback = Path("/Users/haseeb/Downloads/PCB-Defect-Detection-and-Classification-System/yolov5/runs/train/pcb_1st/weights/best.pt")
        if fallback.exists():
            requested.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(fallback), str(requested))
            log.warning("Model weights were missing; copied from %s to %s", fallback, requested)
            return requested

        raise FileNotFoundError(
            f"Model weights not found at '{requested}'. "
            "Place best.pt there (or set MODEL_PATH env var)."
        )

    def warmup(self) -> None:
        if self._state is not None:
            return

        weights = self._ensure_weights_present()
        yolov5_dir = Path(settings.yolov5_dir)

        # Ensure YOLOv5 modules are importable
        if str(yolov5_dir) not in sys.path:
            sys.path.append(str(yolov5_dir))

        # Local imports from YOLOv5 repo
        from models.common import DetectMultiBackend  # type: ignore
        from utils.general import check_img_size  # type: ignore
        from utils.torch_utils import select_device  # type: ignore

        device = select_device(settings.device)
        model = DetectMultiBackend(str(weights), device=device, dnn=False, data=None, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        _ = check_img_size((settings.img_size, settings.img_size), s=stride)
        model.warmup(imgsz=(1 if pt or model.triton else 1, 3, settings.img_size, settings.img_size))

        self._state = _ModelState(
            device=str(device),
            names={int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {i: str(n) for i, n in enumerate(names)},
            stride=int(stride),
            model=model,
            weights_path=str(weights),
            fp16=bool(getattr(model, "fp16", False)),
        )

        log.info("Loaded YOLOv5 model from %s on device=%s", weights, device)

    def state(self) -> _ModelState:
        if self._state is None:
            self.warmup()
        assert self._state is not None
        return self._state

    def model_details(self) -> dict:
        s = self._state
        return {
            "loaded": bool(s),
            "weights_path": s.weights_path if s else settings.model_path,
            "device": s.device if s else "unloaded",
            "framework": "PyTorch",
            "yolo_version": "YOLOv5",
            "img_size": settings.img_size,
            "classes": s.names if s else None,
            "optimizer": "N/A (inference-only)",
            "dataset_size": "N/A (inference-only)",
            "achieved_accuracy": None,
        }
