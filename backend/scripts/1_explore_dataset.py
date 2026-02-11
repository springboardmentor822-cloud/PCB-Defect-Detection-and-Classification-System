import os
import cv2
import matplotlib.pyplot as plt
import glob

def explore_dataset():
    print("--- 1_explore_dataset.py ---")
    
    raw_dir = 'data/raw'
    template_dir = 'data/PCB_USED'
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    test_images = glob.glob(os.path.join(raw_dir, "*.jpg"))
    templates = glob.glob(os.path.join(template_dir, "*.JPG"))
    
    print(f"Found {len(test_images)} test images in {raw_dir}")
    print(f"Found {len(templates)} master templates in {template_dir}")
    
    if not test_images:
        print("ERROR: No test images found!")
        return False
        
    # Pick sample test images to visualize
    samples = test_images[:4]
    
    plt.figure(figsize=(20, 10))
    for i, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '01_sample_images.png'))
    print(f"Saved exploration visualization to {results_dir}/01_sample_images.png")
    
    # Verify mapping logic for one sample
    sample_name = os.path.basename(samples[0])
    prefix = sample_name.split('_')[0]
    template_path = os.path.join(template_dir, f"{prefix}.JPG")
    
    if os.path.exists(template_path):
        print(f"Mapping Check: {sample_name} -> {prefix}.JPG (FOUND)")
    else:
        print(f"Mapping Check: {sample_name} -> {prefix}.JPG (NOT FOUND)")
        
    return True

if __name__ == "__main__":
    explore_dataset()
