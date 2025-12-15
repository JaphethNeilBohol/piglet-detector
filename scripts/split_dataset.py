# split_dataset.py
# Usage: run inside your project venv:
# python scripts/split_dataset.py
#
# It will read dataset/train/ and create dataset/test/, moving ~20% files there.

import os
import random
import shutil
from pathlib import Path

# CONFIG: adjust these if your folders differ
PROJECT_ROOT = Path.cwd()
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOT_EXTS = {".xml"}  # Pascal VOC annotations

SPLIT = 0.2  # fraction to move to test (20%)

def find_pairs(train_dir: Path):
    # map basename -> (image_path, ann_path)
    items = {}
    for p in train_dir.iterdir():
        if p.is_file():
            stem = p.stem
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                items.setdefault(stem, {})["image"] = p
            elif ext in ANNOT_EXTS:
                items.setdefault(stem, {})["ann"] = p
    # keep only pairs that have both
    pairs = []
    for stem, d in items.items():
        if "image" in d and "ann" in d:
            pairs.append((d["image"], d["ann"]))
        else:
            # report unmatched files
            if "image" in d and "ann" not in d:
                print(f"[WARN] image without annotation: {d['image'].name}")
            if "ann" in d and "image" not in d:
                print(f"[WARN] annotation without image: {d['ann'].name}")
    return pairs

def main():
    if not TRAIN_DIR.exists() or not TRAIN_DIR.is_dir():
        print(f"ERROR: train directory not found: {TRAIN_DIR}")
        return

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(TRAIN_DIR)
    total = len(pairs)
    if total == 0:
        print("No image+annotation pairs found in train/ — nothing to split.")
        return

    random.shuffle(pairs)
    n_test = max(1, int(total * SPLIT))
    test_pairs = pairs[:n_test]
    keep_pairs = pairs[n_test:]

    print(f"Total pairs found: {total}")
    print(f"Will move {n_test} pairs to test/  (split={SPLIT})")

    moved = 0
    for img_path, ann_path in test_pairs:
        # move both files to test folder
        try:
            shutil.move(str(img_path), str(TEST_DIR / img_path.name))
            shutil.move(str(ann_path), str(TEST_DIR / ann_path.name))
            moved += 1
        except Exception as e:
            print(f"[ERROR] moving {img_path.name} / {ann_path.name}: {e}")

    print(f"Moved {moved}/{n_test} pairs to {TEST_DIR}")
    print("Done. You can now generate test.record from dataset/test/")

if __name__ == "__main__":
    main()
