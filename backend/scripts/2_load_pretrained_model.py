import os
import sys
from ultralytics import YOLO

def load_model():
    print("--- 2_load_pretrained_model.py ---")
    
    model_path = os.path.join('models', 'best_pcb.pt')
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} missing.")
        return False
        
    try:
        model = YOLO(model_path)
        print("YOLOv8 Model loaded successfully.")
        print(f"Task: {model.task}")
        print(f"Model Names: {model.names}")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

if __name__ == "__main__":
    if not load_model():
        sys.exit(1)
