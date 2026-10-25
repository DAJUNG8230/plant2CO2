from pathlib import Path
import csv
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_ROOT = Path(r"D:\FLAIR\flair_aerial_train\flair_aerial_train")
LABEL_ROOT = Path(r"D:\FLAIR\flair_labels_train\flair_labels_train")
OUTPUT_MASK_ROOT = Path(r"D:\FLAIR\flair_6class_masks")
MANIFEST_DIR = Path("flair_manifests")

SEED = 42

CLASS_NAMES = [
    "Bare Soil",
    "Other Non-Veg",
    "Herbaceous",
    "Shrub",
    "Trees",
    "Cultivated",
]

# FLAIR #1 original label values:
# 1 building
# 2 pervious surface
# 3 impervious surface
# 4 bare soil
# 5 water
# 6 coniferous
# 7 deciduous
# 8 brushwood
# 9 vineyard
# 10 herbaceous vegetation
# 11 agricultural land
# 12 plowed land
# 13 swimming pool
# 14 snow
# 15 clear cut
# 16 mixed
# 17 ligneous
# 18 greenhouse
# 19 other
#
# New 6-class mapping:
# 0 Bare Soil
# 1 Other Non-Veg
# 2 Herbaceous
# 3 Shrub
# 4 Trees
# 5 Cultivated
# 255 Ignore
FLAIR_TO_6 = {
    1: 1,   # building -> Other Non-Veg
    2: 1,   # pervious surface -> Other Non-Veg
    3: 1,   # impervious surface -> Other Non-Veg
    4: 0,   # bare soil -> Bare Soil
    5: 1,   # water -> Other Non-Veg
    6: 4,   # coniferous -> Trees
    7: 4,   # deciduous -> Trees
    8: 3,   # brushwood -> Shrub
    9: 5,   # vineyard -> Cultivated
    10: 2,  # herbaceous vegetation -> Herbaceous
    11: 5,  # agricultural land -> Cultivated
    12: 5,  # plowed land -> Cultivated
    13: 1,  # swimming pool -> Other Non-Veg
    14: 1,  # snow -> Other Non-Veg
    15: 255, # clear cut -> Ignore (ambiguous)
    16: 4,  # mixed -> Trees
    17: 3,  # ligneous -> Shrub
    18: 1,  # greenhouse -> Other Non-Veg
    19: 255, # other -> Ignore
}


def build_image_index():
    images = list(IMAGE_ROOT.rglob("IMG_*.tif"))
    index = {p.stem.replace("IMG_", ""): p for p in images}
    return images, index


def get_domain(path: Path):
    rel = path.relative_to(LABEL_ROOT)
    return rel.parts[0]


def split_domains(domains):
    domains = sorted(domains)
    rng = random.Random(SEED)
    rng.shuffle(domains)

    n = len(domains)
    if n < 3:
        raise RuntimeError("至少需要 3 個 domain 才能分 train/val/test")

    n_val = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))
    n_train = n - n_val - n_test

    return {
        "train": set(domains[:n_train]),
        "val": set(domains[n_train:n_train + n_val]),
        "test": set(domains[n_train + n_val:]),
    }


def convert_mask(label_path: Path, output_path: Path):
    raw = np.asarray(Image.open(label_path), dtype=np.uint8)

    lut = np.full(256, 255, dtype=np.uint8)
    for old_id, new_id in FLAIR_TO_6.items():
        lut[old_id] = new_id

    converted = lut[raw]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(converted).save(output_path)

    return np.unique(raw), np.unique(converted)


def main():
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"找不到影像資料夾: {IMAGE_ROOT}")
    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"找不到標記資料夾: {LABEL_ROOT}")

    labels = sorted(LABEL_ROOT.rglob("MSK_*.tif"))
    images, image_index = build_image_index()

    print(f"Images found: {len(images):,}")
    print(f"Labels found: {len(labels):,}")

    if len(images) != len(labels):
        print("[WARNING] 影像與標記數量不同，會依檔名實際配對。")

    domains = sorted({get_domain(p) for p in labels})
    splits = split_domains(domains)

    print("\nDomain split:")
    for split, values in splits.items():
        print(f"  {split}: {sorted(values)}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    writers = {}
    files = {}

    for split in ["train", "val", "test"]:
        f = open(
            MANIFEST_DIR / f"{split}.csv",
            "w",
            newline="",
            encoding="utf-8-sig",
        )
        writer = csv.writer(f)
        writer.writerow(["image", "mask", "domain"])
        files[split] = f
        writers[split] = writer

    matched = 0
    missing = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    try:
        for label_path in tqdm(labels, desc="Converting FLAIR masks"):
            key = label_path.stem.replace("MSK_", "")
            image_path = image_index.get(key)

            if image_path is None:
                missing += 1
                continue

            domain = get_domain(label_path)
            split = next(
                name
                for name, ds in splits.items()
                if domain in ds
            )

            rel = label_path.relative_to(LABEL_ROOT)
            out_path = (
                OUTPUT_MASK_ROOT
                / rel.parent
                / f"{label_path.stem}.png"
            )

            convert_mask(label_path, out_path)

            writers[split].writerow([
                str(image_path),
                str(out_path),
                domain,
            ])

            matched += 1
            split_counts[split] += 1

    finally:
        for f in files.values():
            f.close()

    print("\nFinished")
    print(f"Matched: {matched:,}")
    print(f"Missing image: {missing:,}")
    print(f"Output masks: {OUTPUT_MASK_ROOT}")
    print(f"Manifests: {MANIFEST_DIR.resolve()}")

    for split, count in split_counts.items():
        print(f"{split}: {count:,}")


if __name__ == "__main__":
    main()
