from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("dataset")
SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["Background", "Grassland", "Barren"]
NUM_CLASSES = 3


def analyze_split(split: str):
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
    shape_counts = {}

    for mask_path in masks:
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)

        shape_counts[mask.shape] = shape_counts.get(mask.shape, 0) + 1

        unique = np.unique(mask)

        if np.any((unique < 0) | (unique >= NUM_CLASSES)):
            invalid_files.append((mask_path.name, unique.tolist()))
            continue

        counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
        total_counts += counts
        image_class_counts += (counts > 0).astype(np.int64)

    total_pixels = int(total_counts.sum())

    print("=" * 72)
    print(f"{split.upper()} SET")
    print("=" * 72)
    print(f"Mask 數量: {len(masks)}")
    print(f"有效 pixel 數量: {total_pixels:,}")
    print()

    for i, name in enumerate(CLASS_NAMES):
        pct = (
            total_counts[i] / total_pixels * 100.0
            if total_pixels
            else 0.0
        )
        print(
            f"{i} {name:<12} "
            f"pixels={total_counts[i]:>12,}  "
            f"ratio={pct:>7.2f}%  "
            f"images_with_class={image_class_counts[i]}/{len(masks)}"
        )

    print()
    print("Mask 尺寸分布:")
    for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        print(f"  {shape}: {count} 張")

    if invalid_files:
        print()
        print("[WARNING] 發現非法 class id:")
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

    print()
    print("=" * 72)
    print("ALL DATASET")
    print("=" * 72)

    for i, name in enumerate(CLASS_NAMES):
        pct = grand_total[i] / total * 100.0 if total else 0.0
        print(
            f"{i} {name:<12} "
            f"pixels={grand_total[i]:>12,}  ratio={pct:>7.2f}%"
        )

    if total:
        nonzero = grand_total[grand_total > 0]

        if len(nonzero) >= 2:
            imbalance = nonzero.max() / nonzero.min()
            print()
            print(f"最大 / 最小類別 pixel 比例: {imbalance:.2f}x")

            if imbalance >= 5:
                print("判斷: 類別不平衡明顯，建議考慮 class weights / sampling。")
            elif imbalance >= 2:
                print("判斷: 有一定類別不平衡，建議持續觀察。")
            else:
                print("判斷: 類別比例相對均衡。")


if __name__ == "__main__":
    main()
