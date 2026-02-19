# PCB-Defect-Detection-and-Classification-System
Automated PCB Defect Inspection System

## 1. Introduction

Printed Circuit Board (PCB) inspection is a crucial quality assurance step in electronics manufacturing. Even minor defects such as open circuits, spurious copper, or missing holes can lead to complete product failure.

The implemented system provides an automated inspection pipeline that combines classical computer vision techniques with deep learning classification. Unlike traditional AOI systems that rely purely on template matching, this system introduces intelligent defect classification using EfficientNet.

### The developed codebase performs:

1) Template–test image alignment

2) Difference-based defect localization

3) ROI extraction

4) Deep learning classification

5) Quantitative evaluation using ground truth annotations

6) Interactive Streamlit-based inspection interface

The primary goal is to simulate an end-to-end industrial AOI workflow.

## 2. Objectives

The implemented code aims to:

-Automate PCB defect detection and classification

-Reduce manual inspection effort

-Extract ROIs automatically from difference masks

-Evaluate predictions using IoU matching

-Provide an interactive web interface

-Support deployment-ready modular design

## 3. System Methodology

The implemented system operates in two major pipelines.

### Phase A — Defect Detection and Evaluation Pipeline

Used in Modules 1 to 4.

*Workflow:*

Template Image + Test Image
→ Image Subtraction
→ OTSU Thresholding
→ Morphological Cleaning
→ Contour Detection
→ ROI Extraction
→ EfficientNet Classification
→ Non-Maximum Suppression
→ IoU Matching with XML Ground Truth
→ Metrics Computation

### Phase B — Web-Based Inference Pipeline

Used in Module_5_&_6.

*Workflow:*

User Upload
→ Image Alignment (ORB Homography)
→ Difference Mask Generation
→ ROI Extraction
→ EfficientNet Classification
→ Bounding Box Annotation
→ Streamlit Visualization
→ Downloadable Output

## 4. Module-Wise Implementation
### Module 1 & 2 — Core Detection Utilities

These modules implement the fundamental image processing functions.

#### Key Functions

detect_defects(template, test)
Purpose: Localize potential defect regions.

#### Steps performed:

-Resize template to match test image

-Compute absolute difference

-Apply OTSU threshold

-Perform morphological opening

-Extract contours

-Filter small regions

-Return ROIs and bounding boxes

*Why this is done:*

-Difference imaging highlights manufacturing defects

-Morphology removes noise

-Contour filtering reduces false positives

**classify(model, rois, conf_thresh)**
Purpose: Classify each detected ROI.

*Steps:*

-Convert ROI to PIL image

-Apply EfficientNet preprocessing

-Run forward pass

-Apply softmax

-Filter by confidence threshold

*Why:*

-Converts detection pipeline into semantic defect classification

-Confidence filtering reduces weak predictions

**read_xml(xml_path)**
Purpose: Parse ground truth annotations.

Extracts:

-Defect class

-Bounding box coordinates

*Why:*

-Required for quantitative evaluation

-Enables IoU-based matching

**iou(boxA, boxB)**
Purpose: Compute Intersection over Union.

*Why:*

-Standard metric for detection evaluation

-Used to match predictions with ground truth

**nms(boxes, scores)**
Purpose: Remove duplicate detections.

*Why:*

-Difference imaging often produces overlapping boxes

-NMS keeps highest-confidence prediction

### Module 3 — ROI Classification Evaluation

This module performs full dataset evaluation.

*Process Flow*

For each PCB image:

-Load template and test image

-Detect ROIs

-Classify defects

-Apply NMS

-Match with ground truth using IoU

-Update TP, FP, FN

-Save visualization

**Metrics Computed**

-True Positives (TP)

-False Positives (FP)

-False Negatives (FN)

-Precision

-Recall

-F1 Score

-Prediction Match Rate

-Visualization

**The code generates:**

-Green boxes → predictions

-Blue boxes → ground truth

-Saved debug image

*Reason:*

-Helps visually inspect model behavior

-Useful for error analysis

### Module 4 — Performance Measurement

Final metrics are computed as:

-Precision = TP / (TP + FP)

-Recall = TP / (TP + FN)

-F1 Score = harmonic mean

-Match Rate = detection coverage

This module validates the effectiveness of the hybrid AOI pipeline.

### Module 5 — Streamlit Deployment

This module provides the user interface.

**Model Loading**

Uses:

-EfficientNet-B0 backbone

-Custom classifier head

-GPU/CPU auto selection

-Streamlit resource caching

*Why:*

-Ensures fast repeated inference

-Production-style deployment

**Image Alignment**

Function: align_images()

*Technique:*

-ORB feature detection

-Brute-force matching

-Homography estimation

-Perspective warping

*Why:*

-Real PCB captures may be slightly shifted

-Alignment improves subtraction quality

-Improved Defect Detection

-Enhancements over training pipeline:

-Gaussian blur for noise reduction

-Morphological open + close

-Area filtering

-Thin-region rejection

*Purpose:*

Reduce false positives in real-world images

### Module 6 — End-to-End Web Pipeline

**Main function: run_pipeline()**

*Steps:*

-Read images

-Align test to template

-Detect defects

-Display mask

-Classify ROIs

-Draw bounding boxes

-Count defects

-Return annotated image

**Streamlit Interface Features**

The web app provides:

-Template upload

-Test image upload

-One-click detection

-ROI count display

-Annotated image preview

-Download button

-Cloudflare public link support

## 5. Defect Categories

The classifier predicts six PCB defect classes:

1) Missing Hole

2) Mouse Bite

3) Open Circuit

4) Short

5) Spur

6) Spurious Copper

## 6. Model Used
EfficientNet-B0 

**Role:**

-ROI classification

-Production inference

**Advantages:**

-Lightweight

-High accuracy

-Transfer learning friendly

## 7. Output

The implemented system generates:

-Annotated PCB image

-Predicted defect labels

-Confidence scores

-ROI count

-Evaluation metrics

-Debug visualization

## 8. Special Implementation Features

Special design elements in the codes:

-Hybrid CV + Deep Learning pipeline

-OTSU thresholding

-ORB-based alignment

-Non-Maximum Suppression

-XML-based evaluation

-Streamlit interactive UI

-Cloudflare public deployment

-Confidence-based filtering

## 9. Future Scope

The current implementation can be extended with:

-Better ROI filtering

-Advanced augmentation

-Hard negative mining

-Real-time camera input

-Batch inspection

-REST API deployment

-GPU optimization

## 10. License

MIT License

## 11. Author

Prachi Sisodia

Github link: https://github.com/pstellar
