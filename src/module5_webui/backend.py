import sys
from pathlib import Path
import uuid
import shutil


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.module4_inference.inference_pipeline import run_inference_pipeline


def run_inference(template_path, test_path):
    """
    Runs Module 4 inference using file paths
    and returns structured results for Web UI.
    Supports optional template auto-detection.
    """

    test_path = Path(test_path)

    if template_path is not None:
        template_path = Path(template_path)
    else:
        template_path = None  

    results = run_inference_pipeline(
        template_path=template_path,
        test_path=test_path,
        save_outputs=True
    )

    return results
