# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import List

# from pydantic_settings import BaseSettings, SettingsConfigDict


# class Settings(BaseSettings):
#     model_config = SettingsConfigDict(env_file=".env", extra="ignore")

#     # Project paths
#     project_root: str = str(Path(__file__).resolve().parents[3])  # .../project 10 /
#     yolov5_dir: str = ""  # default resolved below
#     uploads_dir: str = ""
#     outputs_dir: str = ""

#     # Model
#     model_path: str = ""  # requested default resolved below
#     img_size: int = 640
#     conf_thres: float = 0.25
#     iou_thres: float = 0.45
#     max_det: int = 1000
#     device: str = ""  # "" => auto (cuda if available, else cpu)

#     # Template comparison
#     template_diff_threshold: int = 30
#     template_min_area: int = 120

#     # API – allow common dev ports so frontend works from any port (5500, 5501, 3000, etc.)
#     cors_allow_origins: List[str] = [
#         "http://localhost:5500", "http://127.0.0.1:5500",
#         "http://localhost:5501", "http://127.0.0.1:5501",
#         "http://localhost:3000", "http://127.0.0.1:3000",
#         "http://localhost:8080", "http://127.0.0.1:8080",
#         "null", "*",
#     ]


# def _default_paths() -> Settings:
#     s = Settings()
#     root = Path(s.project_root)

#     yolov5_dir = root / "YOLOv5_PCB_Defects_Detection-main" / "yolov5"
#     uploads_dir = root / "uploads"
#     outputs_dir = root / "outputs"
#     requested_model = yolov5_dir / "runs" / "train" / "pcb_1st" / "weights" / "best.pt"

#     s.yolov5_dir = os.getenv("YOLOV5_DIR", str(yolov5_dir))
#     s.uploads_dir = os.getenv("UPLOADS_DIR", str(uploads_dir))
#     s.outputs_dir = os.getenv("OUTPUTS_DIR", str(outputs_dir))
#     s.model_path = os.getenv("MODEL_PATH", str(requested_model))
#     return s


# settings = _default_paths()

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Project paths
    project_root: str = str(Path(__file__).resolve().parents[3])

    yolov5_dir: str = ""
    uploads_dir: str = ""
    outputs_dir: str = ""

    # Model
    model_path: str = ""
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 1000
    device: str = ""

    # Template comparison
    template_diff_threshold: int = 30
    template_min_area: int = 120

    # API – allow common dev ports so frontend works
    cors_allow_origins: List[str] = [
        "http://localhost:5500", "http://127.0.0.1:5500",
        "http://localhost:5501", "http://127.0.0.1:5501",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8080", "http://127.0.0.1:8080",
        "null", "*",
    ]


def _default_paths() -> Settings:
    s = Settings()

    # Project root folder
    root = Path(s.project_root)

    # Correct folders (as per your project structure)
    yolov5_dir = root / "yolov5"
    uploads_dir = root / "uploads"
    outputs_dir = root / "outputs"

    # Correct trained model path
    requested_model = (
        yolov5_dir
        / "runs"
        / "train"
        / "pcb_1st"
        / "weights"
        / "best.pt"
    )

    # Environment override (if set)
    s.yolov5_dir = os.getenv("YOLOV5_DIR", str(yolov5_dir))
    s.uploads_dir = os.getenv("UPLOADS_DIR", str(uploads_dir))
    s.outputs_dir = os.getenv("OUTPUTS_DIR", str(outputs_dir))
    s.model_path = os.getenv("MODEL_PATH", str(requested_model))

    return s


settings = _default_paths()