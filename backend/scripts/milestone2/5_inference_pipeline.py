import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def get_classifier(model_path, num_classes, device):
    """Loads the trained EfficientNet-B0 classifier."""
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def process_pipeline(image_path, detector, classifier, class_names, device, output_dir):
    """
    Runs the full detection + classification pipeline on a single image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image {image_path}")
        return

    # 1. Detection (YOLOv8)
    results = detector(img, verbose=False)
    
    # 2. Extract ROIs and Classify
    classification_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop ROI
            roi = img[y1:y2, x1:x2]
            if roi.size == 0: continue
            
            # Prepare for classifier
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_tensor = classification_transforms(roi_pil).unsqueeze(0).to(device)
            
            # 3. Classification
            with torch.no_grad():
                outputs = classifier(roi_tensor)
                _, preds = torch.max(outputs, 1)
                label = class_names[preds[0]]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()

            # 4. Draw results
            color = (0, 255, 0) # Green for overall success
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} ({confidence:.2f})"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save final output
    output_path = Path(output_dir) / image_path.name
    cv2.imwrite(str(output_path), img)
    print(f"Pipeline complete for {image_path.name}. Saved to {output_path}")

if __name__ == "__main__":
    # Current script: backend/scripts/milestone2/5_inference_pipeline.py
    # BASE_DIR should be the 'backend' folder
    BASE_DIR = Path(__file__).parent.parent.parent
    DETECTOR_PATH = BASE_DIR / "models" / "best_pcb.pt" # YOLOv8 model from Module 1
    CLASSIFIER_PATH = BASE_DIR / "models" / "classifier_best.pt" # EfficientNet model from Milestone 2
    TEST_DATA_DIR = BASE_DIR / "data" / "external_test"
    OUTPUT_DIR = BASE_DIR / "results" / "milestone2_module4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

    print("--- Initializing End-to-End Pipeline ---")
    
    # Load models
    print("Loading YOLOv8 Detector...")
    detector = YOLO(DETECTOR_PATH)
    
    print("Loading Optimized Classifier...")
    classifier = get_classifier(CLASSIFIER_PATH, len(CLASS_NAMES), DEVICE)
    
    # Process images
    test_images = list(TEST_DATA_DIR.glob("*.jpg")) + list(TEST_DATA_DIR.glob("*.png"))
    if not test_images:
        print(f"No test images found in {TEST_DATA_DIR}!")
    else:
        for img_path in test_images:
            process_pipeline(img_path, detector, classifier, CLASS_NAMES, DEVICE, OUTPUT_DIR)

    print("--- Milestone 2 Module 4 Pipeline Complete ---")
