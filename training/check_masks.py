from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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

COLORS = np.array(
    [
        [154, 103, 64],
        [100, 116, 139],
        [132, 204, 22],
        [22, 163, 74],
        [21, 128, 61],
        [234, 179, 8],
    ],
    dtype=np.uint8,
)

OUT_DIR = Path("mask_checks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image(image_dir, stem):
    for ext in [
        ".png", ".jpg", ".jpeg",
        ".PNG", ".JPG", ".JPEG",
    ]:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def resize_keep_aspect(img, width=420):
    ratio = width / img.width
    return img.resize(
        (width, int(img.height * ratio)),
        Image.Resampling.BILINEAR,
    )


def add_title(img, title):
    canvas = Image.new(
        "RGB",
        (img.width, img.height + 42),
        "white",
    )
    canvas.paste(img, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill="black")
    return canvas


def create_check(split, mask_path):
    image_dir = ROOT / split / "images"
    image_path = find_image(image_dir, mask_path.stem)
    if image_path is None:
        print(f"[SKIP] 找不到原圖: {mask_path.stem}")
        return

    rgb_img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(rgb_img)
    mask = np.asarray(
        Image.open(mask_path),
        dtype=np.uint8,
    )

    if mask.shape != rgb.shape[:2]:
        print(f"[ERROR] 尺寸不一致: {mask_path.name}")
        return

    unique = np.unique(mask)
    invalid = unique[(unique < 0) | (unique > 5)]
    if len(invalid):
        print(
            f"[ERROR] {mask_path.name} 非法 class: "
            f"{invalid.tolist()}"
        )
        return

    color_mask = COLORS[mask]
    mask_img = Image.fromarray(color_mask)

    overlay = (
        rgb.astype(np.float32) * 0.58
        + color_mask.astype(np.float32) * 0.42
    ).clip(0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    panels = [
        add_title(resize_keep_aspect(rgb_img), "Original"),
        add_title(resize_keep_aspect(mask_img), "Ground Truth Mask"),
        add_title(resize_keep_aspect(overlay_img), "Overlay"),
    ]

    canvas = Image.new(
        "RGB",
        (sum(p.width for p in panels), max(p.height for p in panels)),
        "white",
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width

    out_path = OUT_DIR / f"{split}_{mask_path.stem}_check.jpg"
    canvas.save(out_path, quality=92)

    counts = np.bincount(
        mask.reshape(-1),
        minlength=6,
    )
    total = counts.sum()
    percentages = counts / total * 100.0

    parts = [
        f"{i}:{CLASS_NAMES[i]}={percentages[i]:.1f}%"
        for i in range(6)
    ]
    print(
        f"[OK] {split}/{mask_path.name} | "
        + " | ".join(parts)
    )


def main():
    created = 0
    for split in SPLITS:
        mask_dir = ROOT / split / "masks"
        if not mask_dir.exists():
            continue

        for mask_path in sorted(mask_dir.glob("*.png"))[:10]:
            create_check(split, mask_path)
            created += 1

    print(f"\n完成，共處理 {created} 張")
    print(f"輸出位置: {OUT_DIR.resolve()}")
    print("\n六類顏色:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i}: {name} -> {COLORS[i].tolist()}")


if __name__ == "__main__":
    main()
