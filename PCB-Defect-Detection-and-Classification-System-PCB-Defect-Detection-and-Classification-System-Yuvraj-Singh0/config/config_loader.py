from pathlib import Path
import yaml

# -------------------------------------------------
# LOAD CONFIG
# -------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "paths.yaml"

if not CONFIG_PATH.exists():
    raise RuntimeError(
        "Missing config/paths.yaml. "
        "Please configure paths before running."
    )

with open(CONFIG_PATH, "r") as f:
    _CFG = yaml.safe_load(f)

# -------------------------------------------------
# RESOLVE PATHS
# -------------------------------------------------
BASE_DIR = Path(_CFG["BASE_DIR"]).resolve()

DATASET_DIR = BASE_DIR / _CFG["DATASET_DIR"]
TEMPLATE_DIR = BASE_DIR / _CFG["TEMPLATE_DIR"]
IMAGE_DIR = BASE_DIR / _CFG["IMAGE_DIR"]
STORED_MASK_DIR = BASE_DIR / _CFG["STORED_MASK_DIR"]

MODEL_PATHS = {
    k: BASE_DIR / v
    for k, v in _CFG["MODEL_PATHS"].items()
}

OUTPUT_DIR = BASE_DIR / _CFG["OUTPUT_DIR"]
TEMP_UPLOAD_DIR = BASE_DIR / _CFG["TEMP_UPLOAD_DIR"]

# Auto-create safe dirs
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
