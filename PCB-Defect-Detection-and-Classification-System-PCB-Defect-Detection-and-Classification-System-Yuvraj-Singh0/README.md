# Automated PCB Defect Inspection System

### Intelligent AOI Application using Computer Vision and Deep Learning

---

## 1. Overview

Printed Circuit Boards (PCBs) form the backbone of modern electronic systems. Even minor defects such as open circuits, missing holes, or short connections can lead to complete device malfunction. Traditional manual inspection methods are slow, inconsistent, and dependent on operator experience. Conventional Automated Optical Inspection (AOI) systems improve speed but often lack adaptability to varying defect patterns.

This project presents a complete automated PCB inspection pipeline that integrates classical image processing techniques with deep learning–based classification. Instead of focusing solely on model training, the system simulates an end-to-end industrial inspection workflow — from raw image processing to final annotated output generation.

The application detects defect regions, classifies them, and generates a structured inspection result through an interactive interface.

---

## 2. Project Goals

* Automate PCB defect detection and classification
* Minimize reliance on manual inspection
* Automatically generate training datasets using annotation files
* Design a modular and scalable AOI pipeline
* Provide an interactive inspection dashboard
* Enable future industrial integration

---

## 3. System Workflow

The system operates in two major stages:

### Stage 1 — Dataset Construction

Training data is generated automatically using XML annotation files.

Raw PCB Image + Template + XML Annotation
→ Image Subtraction
→ Difference Mask Generation
→ Contour Detection
→ ROI Extraction
→ Structured Training Dataset

This approach eliminates manual cropping and prepares a clean dataset for training.

---

### Stage 2 — Defect Classification

Extracted defect regions (ROIs) are passed to a trained deep learning model.

ROI Image
→ Neural Network Classifier
→ Defect Label + Confidence Score

---

## 4. Inference Pipeline Architecture

1. User uploads PCB image
2. Image preprocessing (alignment + subtraction + filtering)
3. Difference region detection
4. Contour extraction
5. ROI generation
6. Deep learning classification
7. Annotated result generation
8. Display in Streamlit interface

---

## 5. Training Pipeline Architecture

Raw Images + Templates + XML
→ Subtraction
→ Mask Generation
→ Contour Extraction
→ ROI Cropping
→ Structured Dataset
→ Model Training
→ Saved Trained Model

---

## 6. Supported Defect Categories

The system classifies the following PCB defects:

* Missing Hole
* Mouse Bite
* Open Circuit
* Short Circuit
* Spur
* Spurious Copper

---

## 7. Project Structure

```
project/
│
├── frontend/              Streamlit user interface
├── src/                   Backend inference pipeline
├── config/                Configuration management
├── trainings/             Dataset generation & training scripts
├── models/                Saved trained models
│
├── data/raw/              Original images, templates, annotations
├── data/processed/        Generated masks and contour outputs
├── processed_dataset/     Final structured dataset
│
└── README.md
```

---

## 8. Models Implemented

### EfficientNet (Primary Model)

The main production model used during inference. It provides strong accuracy while maintaining computational efficiency.

### Custom CNN (Experimental Model)

A lightweight convolutional network implemented for comparative experimentation. Optional and not required for application execution.

---

## 9. Installation

Clone the repository:

```
git clone https://github.com/springboardmentor822-cloud/PCB-Defect-Detection-and-Classification-System.git
cd PCB-Defect-Detection-and-Classification-System
```

Create virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 10. Running the System

Execute the following steps sequentially:

### 1️⃣ Generate Difference Masks

```
python trainings/subtract.py
```

Output: `data/processed/diff_masks/`

---

### 2️⃣ Detect Contours

```
python trainings/contours.py
```

Output: `data/processed/contours/`

---

### 3️⃣ Generate Training Dataset

```
python trainings/preprocess_xml_to_rois.py
```

Output:
`processed_dataset/train/`
`processed_dataset/val/`

---

### 4️⃣ Train EfficientNet Model

```
python trainings/train_classifier.py
```

Output:
`models/efficientnet_model.pth`

---

### Optional: Train Custom CNN

```
python trainings/train_custom_cnn.py
```

---

### 5️⃣ Run Backend Inference

```
python src/inference_pipeline.py
```

---

### 6️⃣ Launch Web Interface

```
streamlit run frontend/app.py
```

Open in browser:
`http://localhost:8501`

---

## 11. Quick Start (Without Training)

If a trained model already exists in the `models` directory:

```
streamlit run frontend/app.py
```

---

## 12. Output

The system generates:

* Annotated PCB image
* Detected defect class
* Prediction confidence score
* Final inspection result

---

## 13. Key Features

* Automatic dataset generation from XML annotations
* Modular configuration-based architecture
* Hybrid classical vision + deep learning approach
* Multi-model training framework
* End-to-end inspection simulation

---

## 14. Future Enhancements

* Real-time camera-based inspection
* Batch PCB analysis support
* REST API deployment
* GPU optimization
* Manufacturing line hardware integration

---

## 15. License

MIT License

---

## 16. Author

Yuvraj Singh
GitHub: [https://github.com/Yuvi-Specs](https://github.com/Yuvi-Specs)
