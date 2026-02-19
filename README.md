 🔍 PCB Defect Detection and Classification System
---

An automated system that detects and classifies defects in Printed Circuit Board (PCB) images using computer vision and deep learning techniques. The system compares a reference PCB template with a test image, highlights defect regions, classifies defects, and provides downloadable results through a user-friendly interface.

---

  Table of Contents
1. [About the Project](#about-the-project)  
2. [Demo Screenshot](#demo-screenshot)  
3. [Features](#features)  
4. [Defect Types](#defect-types)  
5. [Folder Structure](#folder-structure)  
6. [Installation](#installation)  
7. [Running the App](#running-the-app)  
8. [Outputs](#outputs)  
9. [ Performance & Contributions](#performance--contributions)  
10. [Technologies Used](#technologies-used)  
11. [Future Improvements](#future-improvements)  
12. [Author](#author)

---

About the Project
This project aims to automate PCB inspection, reducing manual effort and increasing accuracy in defect detection. By integrating classical image processing with a deep learning classifier, the system detects defect regions, identifies the defect type, and presents results in an intuitive web application.

---

Demo Screenshot
*(Add a screenshot image of your Streamlit UI here if you have one)*

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

Folder Structure
pcb defect detection/
│
├── app/
│ └── app.py # Web UI
├── src/
│ ├── imageprocessing.py # Detection & ROI logic
│ └── train_model.py # Model training
├── dataset/ # Image dataset
├── output/
│ ├── images/ # Annotated output
│ └── logs/ # CSV logs
├── pcb_defect_model.h5 # Trained model
├── requirements.txt # Dependencies
└── README.md # Project overview



