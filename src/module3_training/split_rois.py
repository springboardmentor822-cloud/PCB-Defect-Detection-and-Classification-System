from pathlib import Path
import shutil
import random

SRC_DIR = Path("data/processed/rois")
OUT_DIR = Path("data/processed/rois_split")
SPLIT = 0.8

# Clean old split
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

total_train = 0
total_test = 0

for cls_dir in SRC_DIR.iterdir():
    if not cls_dir.is_dir():
        continue

    images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))

    if len(images) == 0:
        print(f"{cls_dir.name}: NO IMAGES FOUND")
        continue

    random.shuffle(images)
    split_idx = int(len(images) * SPLIT)

    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    train_dir = OUT_DIR / "train" / cls_dir.name
    test_dir = OUT_DIR / "test" / cls_dir.name

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, train_dir / img.name)

    for img in test_imgs:
        shutil.copy(img, test_dir / img.name)

    total_train += len(train_imgs)
    total_test += len(test_imgs)

    print(f"{cls_dir.name}: Train={len(train_imgs)} | Test={len(test_imgs)}")

print("\nROI dataset split completed")
print(f"Total Train: {total_train}")
print(f"Total Test : {total_test}")
