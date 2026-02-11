# PCB Defect Detection and Classification System

This project is an end-to-end system for detecting and classifying defects in Printed Circuit Boards (PCBs) using deep learning (EfficientNet-B0) and computer vision techniques.

## Features
- **Image Processing**: Alignment, subtraction, and thresholding of PCB images.
- **Defect Detection**: Automated extraction of Regions of Interest (ROIs).
- **Classification**: Deep learning model to classify defects into predefined categories.
- **Web UI**: Streamlit-based interface for easy upload and visualization.

## Project Structure
- `backend/`: Contains processing scripts and model utilities.
    - `scripts/`: Milestone-specific implementation scripts.
    - `utils/`: Core inference and visualization modules.
- `frontend/`: Streamlit application code.
- `data/`: Dataset storage (ignored in repository).
- `models/`: Trained model checkpoints (ignored in repository).

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

## Usage
To start the web application:
```bash
./start_webapp.sh
```

## Milestones
- **Milestone 1**: Dataset Preparation and Image Processing.
- **Milestone 2**: Model Training and Evaluation.
- **Milestone 3**: Frontend and Backend Integration.
- **Milestone 4**: Finalization and Delivery.

## Acknowledgements
- DeepPCB Dataset.
- OpenCV for image processing.
- PyTorch / TensorFlow for deep learning.
