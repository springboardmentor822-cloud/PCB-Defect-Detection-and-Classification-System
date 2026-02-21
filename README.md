# Automated PCB Defect Inspection System

Hybrid Computer Vision and Deep Learning Based AOI Application

---

## 1. Introduction

Printed Circuit Board (PCB) inspection is a critical step in electronics manufacturing. Even a minor defect such as a short circuit, missing hole, or open trace can cause complete device failure. Traditional manual inspection is slow, inconsistent, and prone to human error, while conventional Automated Optical Inspection (AOI) systems rely heavily on static templates and lack adaptability.

This project implements a complete intelligent inspection software that automates the inspection workflow using a combination of classical image processing and deep learning classification. The system processes PCB images, extracts defect regions, classifies the defects, and produces an annotated inspection report.

The goal of this project is to simulate an industrial inspection pipeline rather than only training a machine learning model.

---

## 2. Objectives

* Automate PCB defect inspection
* Reduce dependency on manual inspection
* Generate training datasets automatically using annotation files
* Implement a modular AOI pipeline
* Provide an interactive inspection interface
* Support extensibility for industrial deployment

---

## 3. System Methodology

The system operates in two major phases:

### Phase A: Dataset Generation Using Annotations

Annotation files are used to automatically construct a training dataset.

Raw PCB Image + Template + XML Annotation
→ Image Subtraction
→ Contour Detection
→ Region of Interest Extraction
→ Structured Training Dataset

### Phase B: Defect Classification

The trained model learns defect categories from cropped defect images.

ROI Images → Deep Learning Classifier → Defect Label


## System Methodology

Visual Architecture Diagram

flowchart TD


A[User Upload PCB Image] --> B[Streamlit Frontend]

B --> C[Inference Pipeline]

C --> D[Preprocessing]
D --> D1[Template Alignment]
D --> D2[Image Subtraction]
D --> D3[Noise Filtering]

D3 --> E[Contour Detection]
E --> F[ROI Extraction]

F --> G[Classification Model]

G --> G1[EfficientNet - Primary]
G --> G2[Custom CNN - Optional]

G1 --> H[Defect Label]
G2 --> H

H --> I[Annotated Image Generation]
I --> J[Display Result to User]

Training Pipeline Architecture

flowchart TD

A[Raw Images + Templates + XML] --> B[Subtraction]
B --> C[Difference Masks]
C --> D[Contour Extraction]
D --> E[ROI Cropping]
E --> F[Structured Dataset]

F --> G1[Train EfficientNet]
F --> G2[Train Custom CNN]

G1 --> H[Primary Model Saved]
G2 --> I[Optional Model Saved]


---

## 4. Defect Categories

The system classifies the following PCB defects:

* Missing Hole
* Mouse Bite
* Open Circuit
* Short
* Spur
* Spurious Copper

---

## 5. System Architecture

User Upload
→ Image Processing Engine
→ Difference Detection
→ Contour Extraction
→ ROI Generation
→ Deep Learning Classification
→ Annotated Output

---

## 6. Project Structure

```
project/
│
├── frontend/              User interface (Streamlit application)
├── src/                   Backend inference pipeline
├── config/                Configuration and path management
├── trainings/             Dataset generation and model training scripts
├── models/                Stored trained models
│
├── data/raw/              Original images, templates and annotations
├── data/processed/        Generated masks and contours
├── processed_dataset/     Final training dataset
│
└── README.md
```

---

## 7. Models Used

### EfficientNet (Primary Model)

This is the main production model used by the application during inference.

### Custom CNN (Optional Model)

An alternative architecture implemented for comparison and experimentation. It is not required for running the application.

---

## 8. Installation

Clone the repository

```
[git clone https://github.com/<username>/pcb-defect-inspection-system.git](https://github.com/springboardmentor822-cloud/PCB-Defect-Detection-and-Classification-System.git)
cd pcb-defect-inspection-system
```

Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

## 9. Running the Project

The steps must be executed in order.

### Step 1: Generate Difference Masks

```
python trainings/subtract.py
```

Output:
data/processed/diff_masks/

---

### Step 2: Detect Contours

```
python trainings/contours.py
```

Output:
data/processed/contours/

---

### Step 3: Generate Training Dataset from Annotations

```
python trainings/preprocess_xml_to_rois.py
```

Output:
processed_dataset/train/
processed_dataset/val/

---

### Step 4: Train the Main Model (EfficientNet)

```
python trainings/train_classifier.py
```

Output:
models/efficientnet_model.pth

---

### Optional: Train Custom CNN

```
python trainings/train_custom_cnn.py
```

---

### Step 5: Test Backend Inference

```
python src/inference_pipeline.py
```

---

### Step 6: Launch User Interface

```
streamlit run frontend/app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 10. Quick Execution (Without Training)

If the trained EfficientNet model already exists in the models folder:

```
streamlit run frontend/app.py
```

---

## 11. Output

The application produces:

* Annotated PCB image
* Defect class label
* Prediction confidence
* Inspection result

---

## 12. Special Implementation Features

* Automatic dataset creation from XML annotations
* Config-driven modular architecture
* Hybrid classical vision and deep learning pipeline
* Multi-model training framework (EfficientNet primary, Custom CNN optional)
* End-to-end inspection application

---

## 13. Future Work

* Real-time camera inspection
* Batch PCB analysis
* REST API deployment
* Hardware integration in manufacturing line
* GPU acceleration support

---

## 14. License

MIT License

---

## 15. Author

Sivaraj V
GitHub: [https://github.com/sivarajv04](https://github.com/sivarajv04)



