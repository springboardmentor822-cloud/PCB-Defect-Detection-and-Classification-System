from pathlib import Path

# ✅ Final Base Project Directory
BASE_DIR = Path("F:/PCB_Defect_Detection_System")

# Folder structure definition
folders = [
    # Raw data
    "data/raw/images",
    "data/raw/annotations",
    "data/raw/templates",

    # Processed outputs
    "data/processed/diff_masks",
    "data/processed/contours",
    "data/processed/rois",

    # Source code
    "src",

    # Model & UI (later milestones)
    "models",
    "streamlit_app",

    # Logs & docs
    "logs",
]

# Create folders
for folder in folders:
    path = BASE_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created folder: {path}")

# Create placeholder files
files = [
    "src/preprocess.py",
    "src/subtract.py",
    "src/contours.py",
    "src/utils.py",
    "streamlit_app/app.py",
    "requirements.txt",
    "README.md"
]

for file in files:
    file_path = BASE_DIR / file
    file_path.touch(exist_ok=True)
    print(f"Created file: {file_path}")

print("\n✅ Project folder structure created successfully!")
