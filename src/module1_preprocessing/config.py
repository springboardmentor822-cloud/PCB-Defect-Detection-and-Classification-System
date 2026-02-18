from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
TEMPLATE_DIR = RAW_DIR / "templates"
TEST_IMG_DIR = RAW_DIR / "test_images"
ANNOTATION_DIR = RAW_DIR / "annotations_xml"

OUTPUT_DIR = BASE_DIR / "outputs"
DIFF_MASK_DIR = OUTPUT_DIR / "diff_masks"

TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
DIFF_MASK_DIR.mkdir(parents=True, exist_ok=True)
