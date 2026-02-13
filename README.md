# PCB-Defect-Detection-and-Classification-System

## 📌 Project Overview

This project presents an end-to-end system for detecting and classifying defects in Printed Circuit Boards (PCBs) using computer vision and deep learning techniques.

The system performs:

- Image preprocessing and defect region extraction  
- Classification of defect types using a trained CNN model  
- Performance evaluation using accuracy, confusion matrix, and classification report  
- A simple web interface for visual inspection  

The goal of this project is to automate PCB inspection and reduce manual quality control effort.

## 🎯 Problem Statement

Manual PCB inspection is time-consuming and prone to human error. Even small defects such as missing holes or open circuits can cause functional failure.

This system aims to:
- Detect defect regions automatically
- Classify the defect type accurately
- Provide evaluation metrics for performance validation

## 🧠 Defect Classes

The model classifies PCB defects into the following categories:

- Missing_hole  
- Mouse_bite  
- Open_circuit  
- Short  
- Spur  
- Spurious_copper  

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit (for UI)

## 📂 Project Structure

The repository is organized as follows:

```
PCB-Defect-Detection-and-Classification-System/
│
├── models/
│   └── pcb_defect_model.h5
│       Trained deep learning model used for defect classification.
│
├── outputs/
│   ├── output_result.jpg
│   ├── confusion_matrix.png
│   └── evaluation_report.txt
│       Contains generated results, visualizations, and evaluation metrics.
│
├── test_images/
│   └── sample PCB test images
│       Example PCB images used for inference and demonstration.
│
├── app.py
│       Streamlit web application for interactive defect detection.
│
├── detect_and_classify.py
│       Core pipeline for:
│       - Template alignment
│       - Image subtraction
│       - ROI extraction
│       - Defect classification
│
├── extract_roi.py
│       Logic for detecting Regions of Interest (defect areas).
│
├── train_model.py
│       Script used to train the CNN model on PCB defect dataset.
│
├── evaluate_model.py
│       Evaluates trained model and generates:
│       - Accuracy
│       - Precision
│       - Recall
│       - F1-score
│       - Confusion matrix
│
└── requirements.txt
        List of required Python dependencies.
```

---

### 🔎 Explanation

- **models/** → Stores trained model weights  
- **outputs/** → Stores generated results and evaluation reports  
- **test_images/** → Sample images for testing the system  
- **Core Scripts** → Handle training, detection, evaluation, and UI  

This modular structure ensures clean separation between:
- Model
- Processing logic
- Evaluation
- Interface
- Results


## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model 

```bash
python train_model.py
```

### 3️⃣ Evaluate the Model

```bash
python evaluate_model.py
```

### 4️⃣ Run the Web Application

```bash
streamlit run app.py
```

## 📊 Model Performance

The trained model achieved approximately:

- **Test Accuracy:** ~91%
- Strong precision and recall across most defect classes
- Confusion matrix visualization available in the `outputs/` folder


## 📈 Evaluation Metrics Used

The following metrics were used to validate performance:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

These metrics ensure the model is evaluated properly on unseen test data.

