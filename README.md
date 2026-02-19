PCB Defect Detection and Classification System
Project Overview

This project is an automated PCB (Printed Circuit Board) Defect Detection and Classification System that uses computer vision and deep learning techniques to detect and classify defects in PCB images.

The system compares a test PCB image with a reference template, identifies defect regions, classifies them into predefined categories, and displays results through a Streamlit web interface.

Features

Template-based defect detection

Automatic contour detection

Multi-class defect classification

Confidence scoring

Real-time UI using Streamlit

Downloadable annotated images

CSV defect logs generation

Detection time display

Defect Types Detected

Missing Hole

Mouse Bite

Open Circuit

Short Circuit

Spur

Spurious Copper

System Architecture

Image Upload (Streamlit UI)

Image Preprocessing (Grayscale + Resize)

Image Subtraction

Thresholding & Mask Generation

Contour Detection

ROI Extraction

Defect Classification

Annotated Output + CSV Export

Project Structure
pcb defect detection/
│
├── app/
│   └── app.py
│
├── src/
│   ├── imageprocessing.py
│   └── train_model.py
│
├── dataset/
│
├── output/
│   ├── images/
│   └── logs/
│
├── pcb_defect_model.h5
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Clone the Repository
git clone <your-repo-link>
cd pcb defect detection

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
cd app
streamlit run app.py


Then open:

http://localhost:8501

📊 Output

The system generates:

Annotated PCB image with bounding boxes

Defect type and confidence score

Detection time

CSV log file with defect details

Total defect count and statistics

🛠 Technologies Used

Python

OpenCV

TensorFlow / Keras

EfficientNetB0

Streamlit

NumPy

Pandas

scikit-learn

Project Performance
Average Detection Time: ~0.842 seconds per image
Average Prediction Confidence: ~89 %
Overall Model Accuracy: ~87 %


🔮 Future Improvements

Real-time camera-based PCB inspection

YOLO-based object detection

Deployment on cloud server

Improved classification accuracy

Industrial-scale dataset integration

👨‍💻 Author

Thanmai
