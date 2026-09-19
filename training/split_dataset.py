import random
import shutil
from pathlib import Path

random.seed(42)

IMAGE_DIR = Path("dataset/all/images")
MASK_DIR = Path("dataset/all/masks")
DATASET_DIR = Path("dataset")

extensions = {".jpg", ".jpeg", ".png"}
images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in extensions]
pairs = [p for p in images if (MASK_DIR / f"{p.stem}.png").exists()]

random.shuffle(pairs)

total = len(pairs)
train_count = int(total * 0.8)
val_count = int(total * 0.1)

train_files = pairs[:train_count]
val_files = pairs[train_count:train_count + val_count]
test_files = pairs[train_count + val_count:]


def copy_split(files, split):
    split_dir = DATASET_DIR / split
    if split_dir.exists():
        shutil.rmtree(split_dir)

    image_out = split_dir / "images"
    mask_out = split_dir / "masks"
    image_out.mkdir(parents=True)
    mask_out.mkdir(parents=True)

    for image in files:
        mask = MASK_DIR / f"{image.stem}.png"
        shutil.copy2(image, image_out / image.name)
        shutil.copy2(mask, mask_out / mask.name)


copy_split(train_files, "train")
copy_split(val_files, "val")
copy_split(test_files, "test")

print("Total:", total)
print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))
