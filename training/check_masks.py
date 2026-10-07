from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("dataset")
SPLITS = ["train", "val", "test"]

# 0 Background, 1 Grassland, 2 Barren
COLORS = np.array(
    [
        [71, 85, 105],
        [23, 143, 89],
        [154, 103, 64],
    ],
    dtype=np.uint8,
)

OUT_DIR = Path("mask_checks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image(image_dir: Path, stem: str):
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def resize_keep_aspect(img: Image.Image, width=420):
    ratio = width / img.width
    return img.resize((width, int(img.height * ratio)), Image.Resampling.BILINEAR)


def add_title(img: Image.Image, title: str):
    canvas = Image.new("RGB", (img.width, img.height + 42), "white")
    canvas.paste(img, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill="black")
    return canvas


def make_overlay(rgb: np.ndarray, color_mask: np.ndarray):
    overlay = (
        rgb.astype(np.float32) * 0.58
        + color_mask.astype(np.float32) * 0.42
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def create_check(split: str, mask_path: Path):
    image_dir = ROOT / split / "images"
    image_path = find_image(image_dir, mask_path.stem)

    if image_path is None:
        print(f"[SKIP] 找不到原圖: {split}/{mask_path.stem}")
        return

    rgb_img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(rgb_img)

    mask = np.asarray(Image.open(mask_path), dtype=np.uint8)

    if mask.shape != rgb.shape[:2]:
        print(
            f"[ERROR] 尺寸不一致: {mask_path.name} "
            f"image={rgb.shape[:2]} mask={mask.shape}"
        )
        return

    unique = np.unique(mask)
    invalid = unique[(unique < 0) | (unique > 2)]
    if len(invalid):
        print(f"[ERROR] {mask_path.name} 有未知 class id: {invalid.tolist()}")
        return

    color_mask = COLORS[mask]
    mask_img = Image.fromarray(color_mask)
    overlay_img = make_overlay(rgb, color_mask)

    panels = [
        add_title(resize_keep_aspect(rgb_img), "Original"),
        add_title(resize_keep_aspect(mask_img), "Ground Truth Mask"),
        add_title(resize_keep_aspect(overlay_img), "Overlay"),
    ]

    h = max(p.height for p in panels)
    total_w = sum(p.width for p in panels)
    canvas = Image.new("RGB", (total_w, h), "white")

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width

    out_path = OUT_DIR / f"{split}_{mask_path.stem}_check.jpg"
    canvas.save(out_path, quality=92)

    counts = np.bincount(mask.reshape(-1), minlength=3)
    total = counts.sum()
    pct = counts / total * 100.0

    print(
        f"[OK] {split}/{mask_path.name} | "
        f"BG={pct[0]:.2f}% Grass={pct[1]:.2f}% Barren={pct[2]:.2f}%"
    )


def main():
    created = 0

    for split in SPLITS:
        mask_dir = ROOT / split / "masks"

        if not mask_dir.exists():
            print(f"[SKIP] 不存在: {mask_dir}")
            continue

        masks = sorted(mask_dir.glob("*.png"))

        # 每個 split 先輸出前 10 張，避免一次產生太多圖
        for mask_path in masks[:10]:
            create_check(split, mask_path)
            created += 1

    print()
    print("完成。")
    print(f"檢查圖輸出位置: {OUT_DIR.resolve()}")
    print(f"共處理: {created} 張")
    print()
    print("顏色:")
    print("Background = 灰色")
    print("Grassland  = 綠色")
    print("Barren     = 棕色")


if __name__ == "__main__":
    main()
