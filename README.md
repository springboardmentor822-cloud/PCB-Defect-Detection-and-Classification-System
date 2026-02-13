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

---

## 📂 Project Structure


models/ → Trained model files
outputs/ → Generated output images & confusion matrix
test_images/ → Sample PCB test images
app.py → Streamlit web application
detect_and_classify.py → Defect detection & classification logic
evaluate_model.py → Model evaluation script
extract_roi.py → ROI extraction logic
train_model.py → Model training script
requirements.txt → Required Python libraries


---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

bash
pip install -r requirements.txt
2️⃣ Train the Model (Optional)
python train_model.py

3️⃣ Evaluate the Model
python evaluate_model.py


This will generate:

Test Accuracy

Confusion Matrix

Classification Report

evaluation_report.txt

4️⃣ Run the Web Application
streamlit run app.py

📊 Model Performance

The trained model achieved approximately:

Test Accuracy: ~91%

Strong precision and recall across most defect classes

Confusion matrix analysis included in outputs folder

📈 Evaluation Metrics Used

Accuracy

Precision

Recall

F1-score

Confusion Matrix

These metrics ensure the model’s performance is validated on unseen test data.

📌 Key Highlights

ROI-based defect detection

CNN-based classification

Automated performance reporting

Clean and simple UI for demonstration

Modular code structure for scalability

🔮 Future Improvements

Improve performance for Spur and Mouse_bite classes

Add real-time PCB camera integration

Deploy as a cloud-based inspection service

