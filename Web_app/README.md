# PCB Defect Detection System – Web Application

## Overview
The **PCB Defect Detection System** is a web-based application designed to compare a test PCB image reference with a template (correct) PCB image and automatically detect defects. The interface is clean, intuitive, and focused on fast visual inspection and result interpretation.

---

## User Interface Description

### 1) Main Upload Interface

![Main Interface](Web_app/App_interface_1.png)

The main screen allows users to configure and run defect analysis.

#### 🔹 Model Confidence Threshold
- A slider located on the left panel.
- Allows users to adjust the detection confidence level (e.g., 0.25).
- Lower values detect more potential defects.
- Higher values reduce false positives.

#### 🔹 Upload Section
There are two upload panels:

- **Upload TEMPLATE**
  - Used to upload the reference PCB image.
- **Upload TEST**
  - Used to upload the PCB image to be inspected.

Each upload area supports:
- Drag-and-drop functionality
- Manual file browsing
- File size display after selection

#### 🔹 Run Analysis Button
- Once both images are uploaded, users click **Run Analysis**.
- The system processes the images and performs defect detection.

---

### 2️⃣ Analysis Result Interface

![Result Interface](Web_app/App_interface_2.png)

After processing, the system displays the results in a structured layout.

#### 🔹 Computation Time & Summary
At the top, the system displays:
- Total computation time (e.g., **Analysis complete in 2.264s**)
- Number of detected defects (e.g., **Defects: 3**)

This gives immediate performance feedback and detection summary.

#### 🔹 Difference Mask (Left Panel)
- Shows the binary difference mask.
- Highlights regions where discrepancies were detected.
- Helps visualize raw pixel-level differences.

#### 🔹 Final Result (Right Panel)
- Displays the test PCB with bounding boxes around detected defects.
- Each defect is labeled with:
  - Class name
  - Confidence score

This provides clear and interpretable defect localization.

#### 🔹 Download Result Button
- Allows users to download the processed result image with annotations.

---
## Performance

- The application performs defect detection in approximately **2–3 seconds** depending on image size and system performance.
- Computation time is displayed automatically after each analysis.

---
## Summary

The web application provides:

- Adjustable detection sensitivity
- Side-by-side template and test comparison
- Visual defect highlighting
- Real-time performance reporting
- Downloadable annotated results

The interface is designed for fast industrial inspection workflows with minimal user interaction.

