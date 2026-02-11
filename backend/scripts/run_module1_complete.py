import subprocess
import os
import sys
import logging

def run_pipeline():
    # Setup logging
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'module1_run.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Module 1 Pipeline (Complete)")
    
    scripts = [
        "0_setup_model.py",
        "1_explore_dataset.py",
        "2_image_subtraction.py",
        "3_run_inference_extract_rois.py",
        "4_create_visualizations.py"
    ]
    
    python_exe = sys.executable # Use current environment
    
    for script in scripts:
        script_path = os.path.join('scripts', script)
        logger.info(f"Running {script}...")
        
        try:
            result = subprocess.run([python_exe, script_path], check=True, capture_output=True, text=True)
            logger.info(f"Successfully finished {script}")
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {script}: {e.stderr}")
            logger.info("Pipeline aborted.")
            return False
            
    logger.info("Module 1 Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    run_pipeline()
