from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path("models/yolo_pcb.pt")
model = YOLO(MODEL_PATH)


def run_yolo_detection(image_path):
    """
    Runs YOLO detection on a single image.
    Returns path to annotated image.
    """

    results = model(image_path, conf=0.25, save=True)

    # YOLO saves inside runs/detect/predict/
    save_dir = Path(results[0].save_dir)
    output_image = save_dir / Path(image_path).name

    return str(output_image)
