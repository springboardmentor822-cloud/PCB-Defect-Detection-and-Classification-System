# 🟢 PCB Defect Detection & Classification System

## Internship Project

An intelligent automated system to detect, localize, and classify PCB defects using Image Processing and Deep Learning techniques.

---

## Project Overview

This system performs:

1. Image Preprocessing
2. Defect Localization using Image Differencing
3. Noise Removal & Mask Generation
4. ROI Extraction
5. Defect Classification
6. Bounding Box Annotation
7. Processing Time Measurement

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Streamlit (UI)
- Image Processing Techniques

---

## Project Structure

```
PCB-Defect-Detection-and-Classification-System
│
├── roi_extraction/
│   ├── backend_detect.py
│   ├── extract_roi.py
│   ├── draw_bounding_boxes.py
│
├── ui/
│   ├── app.py
│
├── requirements.txt
├── README.md
```

---

## Features

- Upload Template (Defect-Free PCB)
- Upload Test PCB
- Automatic defect detection
- Mask visualization
- ROI extraction display
- Multiple defect handling
- Annotated output with labels
- Processing time per stage
- Model accuracy display

---

## Model Information

- Method: Image Differencing + Thresholding
- Morphological Noise Removal
- Contour-based ROI Detection
- Classification Label Annotation

---

## Performance

- Real-time defect detection
- Displays total processing time
- Shows number of detected ROIs

---

## Author

Nandini  
Internship Project Submission
GitHub link: https://github.com/Afkloppen

---

## Future Improvements

- Deep Learning Model Integration (CNN / EfficientNet)
- GPU Acceleration
- Industrial Deployment Ready Version