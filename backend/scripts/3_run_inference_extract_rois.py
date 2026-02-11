import os
import cv2
from ultralytics import YOLO
import glob
from tqdm import tqdm

def extract_rois():
    print("--- 3_run_inference_extract_rois.py ---")
    
    model_path = os.path.join('models', 'best_pcb.pt')
    raw_dir = 'data/raw'
    roi_base_dir = 'data/rois'
    
    model = YOLO(model_path)
    
    test_images = glob.glob(os.path.join(raw_dir, "*.jpg"))
    print(f"Running inference on {len(test_images)} images...")
    
    total_rois = 0
    
    for img_path in tqdm(test_images):
        results = model(img_path, stream=True, verbose=False)
        img = cv2.imread(img_path)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class name
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] # e.g., missing_hole, spur, etc.
                
                # Normalize folder name
                folder_name = cls_name.replace(' ', '_').lower()
                class_dir = os.path.join(roi_base_dir, folder_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                    
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Expand Roi slightly for context (optional 5px)
                h, w, _ = img.shape
                x1 = max(0, x1 - 5)
                y1 = max(0, y1 - 5)
                x2 = min(w, x2 + 5)
                y2 = min(h, y2 + 5)
                
                # Crop
                roi = img[y1:y2, x1:x2]
                
                # Save
                roi_name = f"{os.path.basename(img_path).split('.')[0]}_roi_{total_rois}.jpg"
                cv2.imwrite(os.path.join(class_dir, roi_name), roi)
                total_rois += 1
                
    print(f"ROI Extraction complete. Total ROIs extracted: {total_rois}")
    print(f"ROIs saved to {roi_base_dir}")

if __name__ == "__main__":
    extract_rois()
