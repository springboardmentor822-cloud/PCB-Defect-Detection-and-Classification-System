import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

def process_subtraction():
    print("--- 2_image_subtraction.py ---")
    
    raw_dir = 'data/raw'
    template_dir = 'data/PCB_USED'
    mask_dir = 'data/masks'
    diff_dir = 'data/diffs'
    
    for d in [mask_dir, diff_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    test_images = glob.glob(os.path.join(raw_dir, "*.jpg"))
    print(f"Processing {len(test_images)} images...")
    
    for img_path in tqdm(test_images):
        filename = os.path.basename(img_path)
        prefix = filename.split('_')[0]
        template_path = os.path.join(template_dir, f"{prefix}.JPG")
        
        if not os.path.exists(template_path):
            # Try lowercase extension
            template_path = os.path.join(template_dir, f"{prefix}.jpg")
            if not os.path.exists(template_path):
                print(f"Warning: Template for {filename} not found.")
                continue
                
        # Load images
        test_img = cv2.imread(img_path)
        template_img = cv2.imread(template_path)
        
        if test_img is None or template_img is None:
            continue
            
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        # Template Matching Logic
        h, w = test_gray.shape
        H, W = template_gray.shape
        
        if (h, w) != (H, W):
            # Assume test is smaller and we need to find it in template
            res = cv2.matchTemplate(template_gray, test_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            top_left = max_loc
            template_crop = template_gray[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
        else:
            template_crop = template_gray
            
        # Image Subtraction
        diff = cv2.absdiff(test_gray, template_crop)
        
        # Otsu's Thresholding
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Cleanup small noise
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Save results
        mask_filename = os.path.join(mask_dir, filename.replace('.jpg', '.png'))
        diff_filename = os.path.join(diff_dir, filename.replace('.jpg', '.png'))
        
        cv2.imwrite(mask_filename, mask)
        cv2.imwrite(diff_filename, diff)
        
    print(f"Subtraction complete. Masks saved to {mask_dir}")

if __name__ == "__main__":
    process_subtraction()
