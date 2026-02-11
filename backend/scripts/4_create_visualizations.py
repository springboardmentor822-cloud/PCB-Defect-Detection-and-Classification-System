import os
import cv2
import matplotlib.pyplot as plt
import glob

def create_visualizations():
    print("--- 4_create_visualizations.py ---")
    
    mask_dir = 'data/masks'
    raw_dir = 'data/raw'
    roi_dir = 'data/rois'
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # 1. Mask Overlays
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))[:4]
    if mask_files:
        plt.figure(figsize=(20, 10))
        for i, mask_path in enumerate(mask_files):
            filename = os.path.basename(mask_path).replace('.png', '.jpg')
            raw_path = os.path.join(raw_dir, filename)
            
            raw_img = cv2.imread(raw_path)
            mask_img = cv2.imread(mask_path, 0)
            
            # Create overlay
            overlay = raw_img.copy()
            overlay[mask_img > 0] = [0, 255, 0] # Green for defects
            
            combined = np.hstack((cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), 
                                 cv2.cvtColor(cv2.merge([mask_img, mask_img, mask_img]), cv2.COLOR_BGR2RGB)))
            
            plt.subplot(2, 2, i+1)
            plt.imshow(combined)
            plt.title(f"Test vs Mask: {filename}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '02_mask_overlays.png'))
        print(f"Saved mask overlay visualization to {results_dir}/02_mask_overlays.png")

    # 2. ROI Samples
    roi_classes = [d for d in os.listdir(roi_dir) if os.path.isdir(os.path.join(roi_dir, d))]
    
    for cls in roi_classes:
        roi_files = glob.glob(os.path.join(roi_dir, cls, "*.jpg"))[:16]
        if not roi_files: continue
        
        plt.figure(figsize=(15, 15))
        cols = 4
        rows = (len(roi_files) + cols - 1) // cols
        
        for i, roi_path in enumerate(roi_files):
            roi_img = cv2.imread(roi_path)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(rows, cols, i+1)
            plt.imshow(roi_img)
            plt.title(f"{cls} sample")
            plt.axis('off')
            
        plt.suptitle(f"Extracted ROIs: {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'03_rois_{cls}.png'))
        print(f"Saved ROI visualization for {cls} to {results_dir}/03_rois_{cls}.png")

import numpy as np # Added for hstack
if __name__ == "__main__":
    create_visualizations()
