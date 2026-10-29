from pathlib import Path
import csv
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import rasterio
except ImportError:
    rasterio = None

from tqdm import tqdm


IMAGE_ROOT = Path(
    r"D:\FLAIR\flair_aerial_train\flair_aerial_train"
)
LABEL_ROOT = Path(
    r"D:\FLAIR\flair_labels_train\flair_labels_train"
)
OUTPUT_ROOT = Path("flair_class_analysis")

SAMPLES_PER_CLASS = 12
MIN_PIXELS_PER_SAMPLE = 256
RANDOM_SEED = 42

CLASS_NAMES = {
    1: "building",
    2: "pervious_surface",
    3: "impervious_surface",
    4: "bare_soil",
    5: "water",
    6: "coniferous",
    7: "deciduous",
    8: "brushwood",
    9: "vineyard",
    10: "herbaceous_vegetation",
    11: "agricultural_land",
    12: "plowed_land",
    13: "swimming_pool",
    14: "snow",
    15: "clear_cut",
    16: "mixed",
    17: "ligneous",
    18: "greenhouse",
    19: "other",
}

TARGET_6CLASS = {
    1: "Other Non-Veg",
    2: "Other Non-Veg",
    3: "Other Non-Veg",
    4: "Bare Soil",
    5: "Other Non-Veg",
    6: "Trees",
    7: "Trees",
    8: "Shrub",
    9: "Cultivated",
    10: "Herbaceous",
    11: "Cultivated",
    12: "Cultivated",
    13: "Other Non-Veg",
    14: "Other Non-Veg",
    15: "Ignore",
    16: "Trees",
    17: "Shrub",
    18: "Other Non-Veg",
    19: "Ignore",
}

PALETTE = {
    1: (239, 68, 68),
    2: (249, 115, 22),
    3: (234, 179, 8),
    4: (154, 103, 64),
    5: (14, 165, 233),
    6: (21, 128, 61),
    7: (34, 197, 94),
    8: (22, 163, 74),
    9: (168, 85, 247),
    10: (132, 204, 22),
    11: (234, 179, 8),
    12: (202, 138, 4),
    13: (6, 182, 212),
    14: (226, 232, 240),
    15: (148, 163, 184),
    16: (16, 185, 129),
    17: (5, 150, 105),
    18: (100, 116, 139),
    19: (71, 85, 105),
}


def load_rgb(path):
    path = Path(path)

    if path.suffix.lower() in {".tif", ".tiff"}:
        if rasterio is None:
            raise ImportError(
                "Need rasterio. Run: "
                "python -m pip install rasterio"
            )

        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError(
                    f"{path} has only {src.count} band(s)"
                )

            arr = src.read([1, 2, 3])

        arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    return Image.open(path).convert("RGB")


def find_pairs():
    pairs = []

    for mask_path in LABEL_ROOT.rglob("MSK_*.tif"):
        rel = mask_path.relative_to(LABEL_ROOT)
        parts = list(rel.parts)

        try:
            msk_index = parts.index("msk")
        except ValueError:
            continue

        image_parts = parts[:]
        image_parts[msk_index] = "img"

        image_name = mask_path.name.replace("MSK_", "IMG_", 1)
        image_parts[-1] = image_name

        image_path = IMAGE_ROOT.joinpath(*image_parts)

        if image_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def read_mask(path):
    with Image.open(path) as im:
        return np.asarray(im, dtype=np.int16)


def class_presence(mask):
    counts = np.bincount(
        mask.reshape(-1),
        minlength=max(CLASS_NAMES) + 1,
    )

    result = {}

    for class_id in CLASS_NAMES:
        value = int(counts[class_id])
        if value >= MIN_PIXELS_PER_SAMPLE:
            result[class_id] = value

    return result


def make_visual(image, mask, class_id, pixel_count):
    image_np = np.asarray(image, dtype=np.uint8)
    hit = mask == class_id

    overlay = image_np.copy().astype(np.float32)
    color = np.array(PALETTE[class_id], dtype=np.float32)
    overlay[hit] = (
        0.35 * overlay[hit]
        + 0.65 * color
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    binary = np.zeros_like(image_np)
    binary[hit] = PALETTE[class_id]

    original = Image.fromarray(image_np)
    mask_vis = Image.fromarray(binary)
    overlay = Image.fromarray(overlay)

    w, h = original.size
    title_h = 72

    canvas = Image.new(
        "RGB",
        (w * 3, h + title_h),
        "white",
    )
    canvas.paste(original, (0, title_h))
    canvas.paste(mask_vis, (w, title_h))
    canvas.paste(overlay, (w * 2, title_h))

    draw = ImageDraw.Draw(canvas)

    title = (
        f"FLAIR class {class_id}: "
        f"{CLASS_NAMES[class_id]} | "
        f"current 6-class: "
        f"{TARGET_6CLASS[class_id]} | "
        f"pixels={pixel_count:,}"
    )
    draw.text((10, 8), title, fill="black")

    draw.text(
        (10, 38),
        "Original",
        fill="black",
    )
    draw.text(
        (w + 10, 38),
        "Class mask",
        fill="black",
    )
    draw.text(
        (w * 2 + 10, 38),
        "Overlay",
        fill="black",
    )

    return canvas


def main():
    random.seed(RANDOM_SEED)

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image root not found: {IMAGE_ROOT}"
        )

    if not LABEL_ROOT.exists():
        raise FileNotFoundError(
            f"Label root not found: {LABEL_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = find_pairs()

    if not pairs:
        raise RuntimeError(
            "No matching FLAIR image/mask pairs found."
        )

    print(f"Pairs found: {len(pairs):,}")

    per_class_candidates = {
        class_id: []
        for class_id in CLASS_NAMES
    }

    total_pixels = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    files_with_class = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    for image_path, mask_path in tqdm(
        pairs,
        desc="Scanning original FLAIR classes",
    ):
        mask = read_mask(mask_path)
        present = class_presence(mask)

        counts = np.bincount(
            mask.reshape(-1),
            minlength=max(CLASS_NAMES) + 1,
        )

        for class_id in CLASS_NAMES:
            count = int(counts[class_id])
            total_pixels[class_id] += count
            if count > 0:
                files_with_class[class_id] += 1

        for class_id, pixel_count in present.items():
            per_class_candidates[class_id].append(
                (
                    pixel_count,
                    image_path,
                    mask_path,
                )
            )

    summary_path = OUTPUT_ROOT / "class_summary.csv"

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id",
            "class_name",
            "current_6class_mapping",
            "total_pixels",
            "files_with_class",
            "candidate_files",
        ])

        for class_id, class_name in CLASS_NAMES.items():
            writer.writerow([
                class_id,
                class_name,
                TARGET_6CLASS[class_id],
                total_pixels[class_id],
                files_with_class[class_id],
                len(per_class_candidates[class_id]),
            ])

    for class_id, class_name in CLASS_NAMES.items():
        class_dir = OUTPUT_ROOT / (
            f"{class_id:02d}_{class_name}"
        )
        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidates = per_class_candidates[class_id]

        if not candidates:
            print(
                f"Class {class_id:02d} "
                f"{class_name}: no candidates"
            )
            continue

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        pool_size = min(
            len(candidates),
            max(
                SAMPLES_PER_CLASS * 8,
                SAMPLES_PER_CLASS,
            ),
        )

        pool = candidates[:pool_size]

        selected = random.sample(
            pool,
            k=min(SAMPLES_PER_CLASS, len(pool)),
        )

        for index, (
            pixel_count,
            image_path,
            mask_path,
        ) in enumerate(selected, start=1):
            image = load_rgb(image_path)
            mask = read_mask(mask_path)

            visual = make_visual(
                image,
                mask,
                class_id,
                pixel_count,
            )

            out_name = (
                f"{index:02d}_"
                f"{image_path.stem}_"
                f"{pixel_count}px.jpg"
            )

            visual.save(
                class_dir / out_name,
                quality=95,
            )

        print(
            f"Class {class_id:02d} "
            f"{class_name}: "
            f"saved {len(selected)} sample(s)"
        )

    print("\nDone.")
    print("Output:", OUTPUT_ROOT.resolve())
    print("Summary:", summary_path.resolve())


if __name__ == "__main__":
    main()
