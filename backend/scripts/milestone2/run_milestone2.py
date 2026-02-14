import subprocess
import os
from pathlib import Path

def run_script(script_path):
    print(f"\n>>> Running: {script_path} ...")
    result = subprocess.run(["python", str(script_path)], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}")
        return False
    return True

if __name__ == "__main__":
    MILESTONE2_DIR = Path(__file__).parent

    scripts = [
        MILESTONE2_DIR / "1_prepare_data.py",
        MILESTONE2_DIR / "2_train_classifier.py",
        MILESTONE2_DIR / "3_evaluate_model.py"
    ]

    print("========================================")
    print("   Starting Milestone 2 Master Pipeline")
    print("========================================")

    for script in scripts:
        if not run_script(script):
            print("Pipeline failed!")
            break
    else:
        print("\n========================================")
        print("   Milestone 2 Pipeline Complete! 🎉")
        print("========================================")
