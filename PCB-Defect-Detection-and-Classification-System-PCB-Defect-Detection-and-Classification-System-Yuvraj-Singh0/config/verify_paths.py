from pathlib import Path
from config_loader import (
    BASE_DIR,
    DATASET_DIR,
    TEMPLATE_DIR,
    IMAGE_DIR,
    STORED_MASK_DIR,
    MODEL_PATHS,
    OUTPUT_DIR,
    TEMP_UPLOAD_DIR
)

def check_dir(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"[❌] Missing directory: {name} → {path}")
    if not path.is_dir():
        raise RuntimeError(f"[❌] Not a directory: {name} → {path}")
    print(f"[✅] {name} directory OK")

def check_file(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"[❌] Missing file: {name} → {path}")
    if not path.is_file():
        raise RuntimeError(f"[❌] Not a file: {name} → {path}")
    print(f"[✅] {name} file OK")

def main():
    print("\n🔍 VERIFYING PROJECT PATH CONFIGURATION\n")

    print(f"[INFO] BASE_DIR → {BASE_DIR}\n")

    # ---------- DIRECTORIES ----------
    check_dir(DATASET_DIR, "DATASET_DIR")
    check_dir(TEMPLATE_DIR, "TEMPLATE_DIR")
    check_dir(IMAGE_DIR, "IMAGE_DIR")

    # Stored mask is optional
    if STORED_MASK_DIR.exists():
        check_dir(STORED_MASK_DIR, "STORED_MASK_DIR")
    else:
        print("[⚠️] STORED_MASK_DIR not found (optional)")

    check_dir(OUTPUT_DIR, "OUTPUT_DIR")
    check_dir(TEMP_UPLOAD_DIR, "TEMP_UPLOAD_DIR")

    # ---------- MODEL FILES ----------
    for key, path in MODEL_PATHS.items():
        check_file(path, f"MODEL_PATH ({key})")

    # ---------- PERMISSIONS ----------
    try:
        test_file = OUTPUT_DIR / "_write_test.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print("[✅] OUTPUT_DIR write permission OK")
    except Exception as e:
        raise RuntimeError("[❌] OUTPUT_DIR is not writable") from e

    print("\n🎉 ALL PATHS VERIFIED SUCCESSFULLY\n")

if __name__ == "__main__":
    main()
