Project Overview
This project implements a comprehensive AI-driven pipeline for detecting and classifying manufacturing defects in Printed Circuit Boards (PCBs). By combining Classical Computer Vision (OpenCV) for precise localization and Deep Learning (EfficientNet-B0) for intelligent classification, the system automates the quality inspection process.


Project Architecture
src/
├── module1_preprocessing/   # Template subtraction & Morphological cleaning
├── module2_roi/             # ROI extraction & Padding-based cropping
├── module3_training/        # EfficientNet-B0 training & Weighted sampling
├── module4_evaluation/      # Confusion matrix & Classification reports
└── module5_web_ui/          # Streamlit-based interactive dashboard (app.py)

Modules Description

Module 1 – Image Preprocessing
Template Alignment: Aligning "Golden Template" with the test PCB image.

Image Subtraction: Using cv2.absdiff and Otsu’s Thresholding to generate difference masks.

Morphological Cleaning: Applying Opening/Closing filters to remove noise from the masks.

Module 2 – ROI Extraction
Contour Detection: Identifying defect boundaries using OpenCV.

ROI Cropping: Automated extraction of defect regions with a 25px padding for better context.

Dataset Prep: Standardizing segments to 128x128 for the classification model.

Module 3 & 4 – Model Training & Evaluation
Architecture: Utilizes EfficientNet-B0 with transfer learning for high-accuracy classification.

Optimization: Implements AdamW optimizer and OneCycleLR scheduler for stable training.

Metrics: Detailed evaluation using Accuracy curves, Loss plots, and Confusion Matrices.

Module 5-7 – Web UI & Results Export
app.py: A real-time Streamlit interface for image uploads and instant inference.

Analytics: Visual metric cards for "Precision Match Rate" and "Inference Latency."

Export: One-click download for annotated images and prediction logs (CSV).

Technologies Used
Languages: Python 3.x

Computer Vision: OpenCV, googlcolab

Deep Learning: PyTorch, Torchvision (EfficientNet-B0)

Frontend/UI: Streamlit

Deployment: Localtunnel / Ngrok

Performance Metrics
The system achieves high-fidelity results for automated PCB inspection:

Classification Accuracy: ~80% (Best Val Acc: 79.57%)

Defect Classes: Missing_hole, Mouse_bite, Open_circuit, Short, Spur, Spurious_copper.

Processing Speed: Real-time inference with low latency per ROI.

Sample UI Preview
The Web Interface allows users to upload "Golden Template" and "Test Subject" images to visualize detection results interactively.
<img width="1026" height="530" alt="image" src="https://github.com/user-attachments/assets/d2ca8d56-4e8c-486e-af37-e03a87d7d640" />
