from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("dataset")
SPLITS = ["train", "val", "test"]
CLASS_NAMES = [
    "Bare Soil",
    "Other Non-Veg",
    "Herbaceous",
    "Shrub",
    "Trees",
    "Cultivated",
]
NUM_CLASSES = 6


def analyze_split(split):
    mask_dir = ROOT / split / "masks"
    if not mask_dir.exists():
        print(f"[SKIP] {mask_dir} 不存在")
        return None

    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        print(f"[SKIP] {mask_dir} 沒有 mask")
        return None

    total_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    invalid_files = []

    for mask_path in masks:
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)
        unique = np.unique(mask)

        if np.any((unique < 0) | (unique >= NUM_CLASSES)):
            invalid_files.append(
                (mask_path.name, unique.tolist())
            )
            continue

        counts = np.bincount(
            mask.reshape(-1),
            minlength=NUM_CLASSES,
        )
        total_counts += counts
        image_class_counts += (counts > 0).astype(np.int64)

    total_pixels = int(total_counts.sum())

    print("=" * 78)
    print(f"{split.upper()} SET")
    print("=" * 78)
    print(f"Mask 數量: {len(masks)}")
    print(f"有效 pixel 數量: {total_pixels:,}\n")

    for i, name in enumerate(CLASS_NAMES):
        pct = (
            total_counts[i] / total_pixels * 100.0
            if total_pixels
            else 0.0
        )
        print(
            f"{i} {name:<16} "
            f"pixels={total_counts[i]:>12,} "
            f"ratio={pct:>7.2f}% "
            f"images={image_class_counts[i]}/{len(masks)}"
        )

    if invalid_files:
        print("\n[WARNING] 發現非法 class id:")
        for name, values in invalid_files:
            print(f"  {name}: {values}")

    return total_counts


def main():
    grand_total = np.zeros(NUM_CLASSES, dtype=np.int64)

    for split in SPLITS:
        counts = analyze_split(split)
        if counts is not None:
            grand_total += counts

    total = int(grand_total.sum())
    print("\n" + "=" * 78)
    print("ALL DATASET")
    print("=" * 78)

    for i, name in enumerate(CLASS_NAMES):
        pct = (
            grand_total[i] / total * 100.0
            if total
            else 0.0
        )
        print(
            f"{i} {name:<16} "
            f"pixels={grand_total[i]:>12,} "
            f"ratio={pct:>7.2f}%"
        )


if __name__ == "__main__":
    main()
