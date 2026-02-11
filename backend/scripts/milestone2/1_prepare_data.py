import os
import shutil
import random
from pathlib import Path

def prepare_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    """
    Splits the ROI dataset into train and test sets.
    """
    source_path = Path(source_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    # Clean previous splits if they exist
    if train_path.exists():
        shutil.rmtree(train_path)
    if test_path.exists():
        shutil.rmtree(test_path)

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_source = source_path / cls
        cls_train = train_path / cls
        cls_test = test_path / cls

        cls_train.mkdir(parents=True, exist_ok=True)
        cls_test.mkdir(parents=True, exist_ok=True)

        images = list(cls_source.glob("*.jpg")) + list(cls_source.glob("*.png"))
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        print(f"Class {cls}: {len(train_images)} train, {len(test_images)} test")

        for img in train_images:
            shutil.copy(img, cls_train / img.name)
        
        for img in test_images:
            shutil.copy(img, cls_test / img.name)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    SOURCE_DIR = BASE_DIR / "data" / "rois"
    TRAIN_DIR = BASE_DIR / "data" / "train"
    TEST_DIR = BASE_DIR / "data" / "test"

    print("--- Starting Data Preparation ---")
    prepare_data(SOURCE_DIR, TRAIN_DIR, TEST_DIR)
    print("--- Data Preparation Complete ---")
