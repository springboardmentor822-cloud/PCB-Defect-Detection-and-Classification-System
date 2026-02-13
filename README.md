# PCB Defect Detection and Classification System

## 📌 Project Overview

This project presents an end-to-end system for detecting and classifying defects in Printed Circuit Boards (PCBs) using computer vision and deep learning techniques.

The system performs:

- Image preprocessing and defect region extraction  
- Classification of defect types using a trained CNN model  
- Performance evaluation using accuracy, confusion matrix, and classification report  
- A simple web interface for visual inspection  

The main objective of this project is to automate the PCB inspection process and reduce dependency on manual quality checking.

## 🎯 Problem Statement

Manual inspection of PCBs is time-consuming and can lead to human errors, especially when defects are small or subtle. Even minor issues like open circuits or missing holes can cause device failure.

This project aims to:
-Detect defect regions automatically using image processing
-Classify detected defects into predefined categories
-Evaluate model performance using standard metrics
-Provide a simple interface for easy testing and visualization

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

##📦 Dataset
The model was trained using a labeled PCB defect dataset containing images categorized into six defect classes. The dataset was split into training, validation, and test sets to ensure proper evaluation on unseen data.


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

-Test Accuracy on unseen test dataset: ~91%
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

##🔮 Future Improvements

• Improve accuracy using advanced architectures (e.g., EfficientNet variants)  
• Add real-time PCB inspection support  
• Deploy as a cloud-based quality inspection service  
• Integrate defect localization heatmaps for better visualization  


