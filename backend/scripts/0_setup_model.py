import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def setup_model():
    print("--- 0_setup_model.py ---")
    
    # Get the backend directory (parent of scripts)
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    model_path = backend_dir / 'models' / 'best_pcb.pt'
    model_path = str(model_path)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    try:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device being used: {device}")
        
        # Verify classes
        names = model.names
        print(f"Model Classes detected ({len(names)}):")
        for idx, name in names.items():
            print(f"  {idx}: {name}")
            
        print("\nSetup complete. Ready for inference!")
        return True
        
    except Exception as e:
        print(f"An error occurred during model setup: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_model()
    if not success:
        sys.exit(1)
