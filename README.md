📌 Project Overview

Printed Circuit Boards (PCBs) are essential components in electronic systems. Even minor defects can cause device failure. Manual inspection is time-consuming and error-prone.

This project presents an AI-based PCB Defect Detection System that uses a Convolutional Neural Network (CNN) integrated into a backend inference pipeline to automatically detect defects in PCB images. The system performs confidence-based detection, defect localization, and performance measurement, making it suitable for academic and industrial inspection scenarios.

🎯 Objectives

Automate PCB defect detection using AI

Reduce manual inspection effort

Provide confidence-based decision making

Localize defect regions on PCB images

Implement a real-time backend inference pipeline

🚀 Key Features

CNN-based PCB defect detection

Patch-based defect localization

Confidence threshold slider (UI control)

False positive suppression mechanism

Small red bounding boxes for defects

Confidence percentage display

Detection time measurement

Backend deployment using Streamlit

🏗️ System Architecture
Input PCB Image
        ↓
Image Preprocessing
        ↓
Patch-Based Scanning
        ↓
CNN Inference
        ↓
Confidence Filtering
        ↓
False Positive Suppression
        ↓
Defect Localization
        ↓
Final Output + Detection Time
🧪 Technologies Used

Programming Language: Python

Deep Learning: TensorFlow / Keras

Computer Vision: OpenCV

Backend Framework: Streamlit

Model Format: HDF5 (.h5)

Visualization: OpenCV + Streamlit

Version Control: Git & GitHub
🧠 Model Description

A CNN model trained to detect defective PCB patterns

Stored in HDF5 (.h5) format

Used only for inference, not retraining

Backend supports binary and multi-class model outputs

🔍 Defect Detection Strategy
🔹 Patch-Based Inference

PCB image is divided into overlapping patches

Each patch is analyzed independently by the CNN

Enables detection of small and localized defects

🔹 Confidence-Based Filtering

Predictions below a confidence threshold are ignored

Threshold can be adjusted using a UI slider

🔹 False Positive Suppression

A defect is confirmed only if multiple nearby patches agree

Prevents clean PCBs from being falsely marked as defective

🖼️ Output Visualization

Defects are highlighted using small red bounding boxes

Only confidence values (%) are displayed near defects

Clean visualization suitable for demos and reports

⏱️ Performance Measurement

Detection time is calculated for each uploaded image

Helps evaluate real-time feasibility of the system
🔮 Future Scope

Semantic segmentation models (U-Net) for pixel-level localization

Reference-based PCB comparison

REST API deployment for industrial pipelines

Heatmap visualization of defect confidence

Cloud or edge deployment
