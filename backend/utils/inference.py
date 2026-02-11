"""
Inference utilities for PCB defect detection and classification.
Integrates YOLOv8 detection with EfficientNet-B0 classification.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
import streamlit as st
import os
import signal

# Suppress ultralytics signal handler issue with Streamlit
os.environ['YOLO_VERBOSE'] = 'False'
# Disable signal handlers before importing YOLO
if hasattr(signal, 'SIGTERM'):
    original_signal = signal.signal
    signal.signal = lambda sig, handler: None
    from ultralytics import YOLO
    signal.signal = original_signal
else:
    from ultralytics import YOLO


class ModelLoader:
    """Singleton class to load and cache models."""
    
    _detector = None
    _classifier = None
    _device = None
    _class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    
    @classmethod
    def get_device(cls):
        """Get the computation device (GPU if available, else CPU)."""
        if cls._device is None:
            cls._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return cls._device
    
    @classmethod
    @st.cache_resource
    def load_detector(_cls, model_path: str):
        """Load YOLOv8 detector model."""
        if _cls._detector is None:
            _cls._detector = YOLO(model_path)
        return _cls._detector
    
    @classmethod
    @st.cache_resource
    def load_classifier(_cls, model_path: str, num_classes: int):
        """Load EfficientNet-B0 classifier model."""
        if _cls._classifier is None:
            device = _cls.get_device()
            model = models.efficientnet_b0()
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            _cls._classifier = model
        return _cls._classifier
    
    @classmethod
    def get_class_names(_cls):
        """Get the list of defect class names."""
        return _cls._class_names


def preprocess_for_classification(roi_image: np.ndarray) -> torch.Tensor:
    """
    Preprocess ROI image for EfficientNet classification.
    
    Args:
        roi_image: ROI image in BGR format (OpenCV)
    
    Returns:
        Preprocessed tensor ready for classification
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
    roi_pil = Image.fromarray(roi_rgb)
    roi_tensor = transform(roi_pil).unsqueeze(0)
    
    return roi_tensor


def run_detection_pipeline(
    image: np.ndarray,
    detector_path: str,
    classifier_path: str
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run the complete detection and classification pipeline.
    
    Args:
        image: Input image in BGR format
        detector_path: Path to YOLOv8 model
        classifier_path: Path to EfficientNet classifier
    
    Returns:
        Tuple of (annotated_image, detections_list)
        detections_list contains dicts with keys: bbox, label, confidence
    """
    device = ModelLoader.get_device()
    class_names = ModelLoader.get_class_names()
    
    # Load models
    detector = ModelLoader.load_detector(detector_path)
    classifier = ModelLoader.load_classifier(classifier_path, len(class_names))
    
    # Run detection with lower confidence threshold to catch more defects
    # conf=0.15 means detect boxes with 15% confidence or higher (default is 0.25)
    # iou=0.45 for non-maximum suppression (default is 0.7)
    print(f"[DEBUG] Running YOLO detection with conf=0.15, iou=0.45")
    results = detector(image, conf=0.15, iou=0.45, verbose=False)
    
    detections = []
    annotated_image = image.copy()
    
    # Debug: Count total boxes detected by YOLO
    total_boxes = 0
    for result in results:
        if result.boxes is not None:
            total_boxes += len(result.boxes)
    
    print(f"[DEBUG] YOLO detected {total_boxes} bounding boxes")
    
    # Process each detection
    processed_count = 0
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop ROI
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                print(f"[DEBUG] Skipping empty ROI at ({x1}, {y1}, {x2}, {y2})")
                continue
            
            # Prepare for classification
            roi_tensor = preprocess_for_classification(roi).to(device)
            
            # Run classification
            with torch.no_grad():
                outputs = classifier(roi_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
                label = class_names[pred_idx.item()]
                conf_score = confidence.item()
            
            processed_count += 1
            print(f"[DEBUG] Detection {processed_count}: {label} ({conf_score:.2%}) at ({x1}, {y1}, {x2}, {y2})")
            
            # Store detection info
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'confidence': conf_score
            })
            
            # Draw on image
            color = get_color_for_label(label)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Add label with background
            label_text = f"{label}: {conf_score:.2%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_image,
                label_text,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
    
    print(f"[DEBUG] Total detections processed: {processed_count}, Total in list: {len(detections)}")
    return annotated_image, detections


def get_color_for_label(label: str) -> Tuple[int, int, int]:
    """
    Get a consistent color for each defect type.
    
    Args:
        label: Defect class name
    
    Returns:
        BGR color tuple
    """
    color_map = {
        'missing_hole': (0, 255, 255),      # Yellow
        'mouse_bite': (255, 0, 255),        # Magenta
        'open_circuit': (0, 0, 255),        # Red
        'short': (0, 165, 255),             # Orange
        'spur': (255, 0, 0),                # Blue
        'spurious_copper': (0, 255, 0)      # Green
    }
    return color_map.get(label, (128, 128, 128))  # Gray as default


def read_template_and_test(template_image: np.ndarray, test_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read and prepare template and test images for comparison.
    
    Args:
        template_image: Template (golden) PCB image in BGR
        test_image: Test PCB image in BGR
    
    Returns:
        Tuple of (aligned_template, aligned_test)
    """
    # Resize test image to match template if needed
    if template_image.shape != test_image.shape:
        test_image = cv2.resize(test_image, (template_image.shape[1], template_image.shape[0]))
    
    return template_image, test_image


def find_defects_with_subtraction(
    template_image: np.ndarray,
    test_image: np.ndarray,
    threshold: int = 30
) -> np.ndarray:
    """
    Find defects using image subtraction between template and test images.
    
    Args:
        template_image: Template PCB image in BGR
        test_image: Test PCB image in BGR
        threshold: Threshold for difference detection
    
    Returns:
        Difference mask highlighting potential defect areas
    """
    # Convert to grayscale
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(template_gray, test_gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh


def run_template_comparison_pipeline(
    template_image: np.ndarray,
    test_image: np.ndarray,
    detector_path: str,
    classifier_path: str,
    use_subtraction: bool = True
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run template comparison pipeline with optional image subtraction.
    
    Args:
        template_image: Template PCB image in BGR
        test_image: Test PCB image in BGR
        detector_path: Path to YOLOv8 model
        classifier_path: Path to EfficientNet classifier
        use_subtraction: Whether to use image subtraction for preprocessing
    
    Returns:
        Tuple of (annotated_test_image, detections_list)
    """
    device = ModelLoader.get_device()
    class_names = ModelLoader.get_class_names()
    
    # Align images
    template_aligned, test_aligned = read_template_and_test(template_image, test_image)
    
    # Load models
    detector = ModelLoader.load_detector(detector_path)
    classifier = ModelLoader.load_classifier(classifier_path, len(class_names))
    
    # Optional: Use image subtraction to highlight differences
    if use_subtraction:
        diff_mask = find_defects_with_subtraction(template_aligned, test_aligned)
        # This can be used to filter or prioritize detections, but we'll still run YOLO
    
    # Run detection on test image with lower confidence threshold
    print(f"[DEBUG] Running YOLO detection (template mode) with conf=0.15, iou=0.45")
    results = detector(test_aligned, conf=0.15, iou=0.45, verbose=False)
    
    detections = []
    annotated_image = test_aligned.copy()
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop ROI from test image
            roi = test_aligned[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Prepare for classification
            roi_tensor = preprocess_for_classification(roi).to(device)
            
            # Run classification
            with torch.no_grad():
                outputs = classifier(roi_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
                label = class_names[pred_idx.item()]
                conf_score = confidence.item()
            
            # Store detection info
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'confidence': conf_score
            })
            
            # Draw on image with color-coded boxes
            color = get_color_for_label(label)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Add label with background
            label_text = f"{label}: {conf_score:.2%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_image,
                label_text,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
    
    return annotated_image, detections

