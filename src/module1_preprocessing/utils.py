import cv2
import re
from pathlib import Path

def extract_pcb_id(filename: str) -> str:
    """
    Extract PCB ID from image name.
    Example: 04_mouse_bite_12.jpg → '04'
    """
    match = re.match(r"(\d+)_", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract PCB ID from {filename}")


def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    return img
