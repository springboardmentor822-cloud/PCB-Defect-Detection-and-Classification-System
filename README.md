 🔍 PCB Defect Detection and Classification System
---

An automated system that detects and classifies defects in Printed Circuit Board (PCB) images using computer vision and deep learning techniques. The system compares a reference PCB template with a test image, highlights defect regions, classifies defects, and provides downloadable results through a user-friendly interface.


---

About the Project
This project aims to automate PCB inspection, reducing manual effort and increasing accuracy in defect detection. By integrating classical image processing with a deep learning classifier, the system detects defect regions, identifies the defect type, and presents results in an intuitive web application.

---

Demo Screenshot
<img width="866" height="844" alt="image" src="https://github.com/user-attachments/assets/95a2d072-b18e-4bbf-80df-6ce41ec63593" />

<img width="795" height="760" alt="image" src="https://github.com/user-attachments/assets/e2d690ce-3437-49af-8398-f590e18d0690" />

---

Features
-  Upload PCB images through web interface  
-  Auto defect localization using image subtraction  
-  Classification of defect types  
-  Confidence scores and statistics  
-  Shows detection time in UI  
-  Download annotated image  
-  Export detailed CSV logs

---

Defect Types Detected
The system supports detection and classification of the following PCB defects:
- **Missing Hole**  
- **Mouse Bite**  
- **Open Circuit**  
- **Short Circuit**  
- **Spur**  
- **Spurious Copper**

---

System Architecture

1. Image Upload (Streamlit UI)  
2. Image Preprocessing (Grayscale + Resize)  
3. Image Subtraction  
4. Thresholding & Mask Generation  
5. Contour Detection  
6. ROI Extraction  
7. Defect Classification  
8. Annotated Output + CSV Export


---

Output

The system produces:
Annotated PCB image with bounding boxes
Defect type and confidence score
Detection time
CSV log file with defect details
Total defect count and statistics

---

Technologies Used

Python
OpenCV
TensorFlow / Keras
EfficientNetB0
Streamlit
NumPy
Pandas
scikit-learn

---

Author

Thanmai-Pamala  
https://github.com/Thanmai-Pamala


