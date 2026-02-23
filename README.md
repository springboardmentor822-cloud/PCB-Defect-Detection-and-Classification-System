# Automated PCB Defect Inspection System

Hybrid Computer Vision and Deep Learning Based AOI Application

---

## 1. Introduction

Printed Circuit Boards (PCBs) are an essential part of modern electronic devices. Any small defect such as a short circuit, open track, spur, or missing hole can cause system failure. Therefore, accurate inspection is very important in PCB manufacturing.

Traditional manual inspection is slow and depends on human judgment. Conventional AOI systems mainly use fixed templates, which limits their ability to adapt to new designs and defects.

This project presents an intelligent PCB inspection system that combines classical computer vision techniques with deep learning models. The system automatically analyzes PCB images, detects defect regions, classifies defect types, and generates annotated inspection results.

The main purpose of this project is to build a complete industrial-style inspection pipeline instead of focusing only on model training.

---

## 2. Objectives

* Automate PCB defect inspection
* Reduce dependency on manual inspection
* Improve defect detection accuracy
* Generate training datasets using annotation files
* Build a modular and scalable AOI pipeline
* Provide a user-friendly inspection interface
* Support future industrial deployment

---

## 3. System Methodology

The proposed system works in two major phases.

### Phase A: Dataset Generation

Annotated PCB images are processed to create a structured training dataset.

Workflow:

Raw PCB Image + Template + XML File  
→ Image Alignment  
→ Difference Detection  
→ Noise Removal  
→ Contour Extraction  
→ ROI Cropping  
→ Training Dataset Creation  

This phase ensures that defect regions are accurately extracted for model training.

---

### Phase B: Defect Classification

In this phase, cropped defect images are classified using trained deep learning models.

Workflow:

ROI Images  
→ Deep Learning Classifier  
→ Defect Category Prediction  

The model outputs the defect type along with confidence score.

---

## 4. Defect Categories

The system is designed to detect the following PCB defects:

* Missing Hole
* Mouse Bite
* Open Circuit
* Short Circuit
* Spur
* Spurious Copper

---

## 5. System Architecture

The overall system architecture follows this workflow:

User Upload  
→ Preprocessing  
→ Template Matching  
→ Image Subtraction  
→ Contour Detection  
→ ROI Extraction  
→ Deep Learning Classification  
→ Annotated Output  

---

## 6. Project Structure


PCB-Defect-Detection-and-Classification-System/

├── backend/ # FastAPI backend
├── frontend/ # Web dashboard
├── yolov5/ # YOLOv5 framework
├── uploads/ # Uploaded images
├── XmlToTxt/ # Annotation converter
├── notebooks/ # Jupyter experiments
├── train.py # Training script
└── README.md # Project documentation


---

## 7. Models Used

### EfficientNet (Primary Model)

EfficientNet is used as the main classification model due to its high accuracy and optimized performance.

### Custom CNN (Optional Model)

A custom convolutional neural network is implemented for experimentation and comparison.

---

## 8. Installation

Clone the repository:


git clone https://github.com/springboardmentor822-cloud/PCB-Defect-Detection-and-Classification-System.git

cd PCB-Defect-Detection-and-Classification-System


Create virtual environment:


python -m venv venv

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate


Install dependencies:


pip install -r requirements.txt


---

## 9. Running the Project

Follow the steps in sequence.

### Step 1: Generate Difference Images


python trainings/subtract.py


---

### Step 2: Extract Contours


python trainings/contours.py


---

### Step 3: Prepare Training Dataset


python trainings/preprocess_xml_to_rois.py


---

### Step 4: Train Classification Model


python trainings/train_classifier.py


---

### Step 5: Run Backend Inference


python src/inference_pipeline.py


---

### Step 6: Launch User Interface


streamlit run frontend/app.py


Open in browser:


http://localhost:8501


---

## 10. Output

The system generates the following outputs:

* Annotated PCB image
* Detected defect label
* Prediction confidence
* Inspection summary

---

## 11. Key Features

* Automatic dataset generation
* Hybrid computer vision and deep learning system
* Modular and scalable architecture
* Multi-model support
* End-to-end inspection workflow

---

## 12. Future Scope

* Real-time camera-based inspection
* Batch PCB processing
* Cloud-based REST API
* Industrial hardware integration
* GPU optimization

---

## 13. License

This project is licensed under the MIT License.

---
