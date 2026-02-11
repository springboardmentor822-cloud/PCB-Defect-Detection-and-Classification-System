Automated PCB Defect Inspection System

Hybrid Computer Vision + Deep Learning AOI Application



|| Overview

Printed Circuit Board (PCB) inspection in manufacturing requires extremely high accuracy because even a microscopic defect can cause total circuit failure. Manual inspection is slow and error-prone, while traditional Automated Optical Inspection (AOI) systems depend heavily on static templates and fail when boards vary or references are unavailable.

This project implements a complete intelligent inspection software that automates the full inspection workflow — from raw PCB images to defect reporting — using classical image processing and deep learning classification.

The project is designed as an industrial-style inspection application, not just a machine learning experiment.



|| Objectives

Automate defect detection and classification

Eliminate manual inspection dependency

Build a real AOI pipeline rather than isolated ML model

Automatically generate training dataset from annotations

Provide a user-friendly inspection interface




|| Special Contributions (What I Implemented)


1️⃣ Automatic Dataset Generation Engine

Created a pipeline that converts annotations into training data automatically:

Template + PCB image + XML annotation
        ↓
Difference detection
        ↓
Contour detection
        ↓
ROI extraction
        ↓
Structured classification dataset

2️⃣ Hybrid Vision Pipeline

Implemented multi-stage inspection combining:

Classical image subtraction for localization

Contour extraction for defect regions

Deep learning classifier for understanding

3️⃣ Config-Driven Modular Architecture

Central configuration system:

config/paths.yaml


Allows project to run without editing code paths.

4️⃣ Multi-Model Training Framework

Implemented two classifiers trained on the same dataset:

Model	Role
EfficientNet	Primary production model
Custom CNN	Optional experimental model

The application uses EfficientNet by default.
Custom CNN is included only for comparison and experimentation.

5️⃣ Complete Inference Application

Built full inspection software:

Upload Image → Processing → Prediction → Annotated Output

6️⃣ Industrial Workflow Simulation

Replicates manufacturing inspection stages:

Pre-processing → Region detection → Classification → Reporting




|| Defect Classes

Missing Hole

Mouse Bite

Open Circuit

Short

Spur

Spurious Copper

     System Architecture


                User Upload
                    │
                    ▼
           Image Processing Engine
                    │
        ┌───────────┼───────────┐
        │                       │
   Subtraction            Contour Detection
        │                              │
        └──────────► ROI Extraction ◄──
                           │
                   EfficientNet Classifier
                       (Custom CNN Optional)
                           │
                   Defect Classification
                           │
                   Annotated Output

📁 Project Structure
project/
│
├── frontend/               # Streamlit UI
├── src/                    # Backend inference pipeline
├── config/                 # Path configuration system
├── trainings/              # Dataset generation + training
├── models/                 # Trained models
│
├── data/raw/               # Original dataset
├── data/processed/         # Generated masks & contours
├── processed_dataset/      # Final ML dataset
│
└── README.md

|| Installation

git clone 
cd pcb-defect-inspection-system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

|| How to Run (Complete Workflow)


Step 1 — Generate Difference Masks
python trainings/subtract.py

Step 2 — Detect Contours
python trainings/contours.py

Step 3 — Create Dataset from Annotations
python trainings/preprocess_xml_to_rois.py

Step 4 — Train Model

Train Primary Model (EfficientNet):

python trainings/train_classifier.py


Optional: Train Custom CNN (Experimental):

python trainings/train_custom_cnn.py

Step 5 — Test Backend
python src/inference_pipeline.py

Step 6 — Launch UI
streamlit run frontend/app.py



|| Quick Run (Skip Training)

If EfficientNet model already exists:

streamlit run frontend/app.py



||Models

Model	Usage
EfficientNet	Main inference model
Custom CNN	Optional experimental model



|| Output

Annotated PCB image

    .Defect type

    .Confidence score

Inspection result



|| Key Features

    .Automatic dataset generation from annotations

    .Modular architecture

    .Industrial inspection workflow

    .Graphical interface

    .Extendable pipeline

|| Future Improvements

.Real-time camera inspection

.Batch PCB analysis

.API deployment

.Hardware integration




|| License

MIT License

|| Author

Sivaraj V
AI / Machine Learning Engineer


