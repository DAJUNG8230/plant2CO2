"""
plant2CO2 inference layer.

Current behavior:
- If models/best_model.pth is not available, the app uses a lightweight RGB
  demonstration segmentation so the full Flask website works immediately.
- Replace predict_with_model() with the trained DeepLabV3+ pipeline when the
  final checkpoint is ready.

Class ids:
0 = Background / Other
1 = Grassland
2 = Barren
"""

from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

CLASS_COLORS = np.array(
    [
        [71, 85, 105],
        [23, 143, 89],
        [154, 103, 64],
    ],
    dtype=np.uint8,
)


def _demo_segmentation(rgb: np.ndarray) -> np.ndarray:
    """Simple RGB rules used only until a trained model is connected."""
    arr = rgb.astype(np.float32)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    brightness = (r + g + b) / 3.0

    mask = np.zeros(r.shape, dtype=np.uint8)

    grass = (
        (brightness >= 28)
        & (brightness <= 242)
        & (g > r * 1.08)
        & (g > b * 1.06)
        & ((g - r) > 10)
    )

    barren = (
        (brightness >= 28)
        & (brightness <= 242)
        & (~grass)
        & (r > b * 1.05)
        & (r >= g * 0.92)
        & ((r - g) < 55)
        & (r > 70)
        & (g > 45)
    )

    mask[grass] = 1
    mask[barren] = 2
    return mask


def predict_with_model(rgb: np.ndarray) -> np.ndarray:
    """
    TODO: Connect the trained DeepLabV3+ checkpoint here.

    Return H x W uint8 class ids:
      0 = Background
      1 = Grassland
      2 = Barren
    """
    raise NotImplementedError("DeepLabV3+ checkpoint is not connected yet.")


def _save_visuals(
    rgb: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    job_id: str,
) -> tuple[str, str]:
    color_mask = CLASS_COLORS[mask]

    mask_filename = f"{job_id}_mask.png"
    overlay_filename = f"{job_id}_overlay.png"

    Image.fromarray(color_mask).save(output_dir / mask_filename)

    overlay = (
        rgb.astype(np.float32) * 0.58
        + color_mask.astype(np.float32) * 0.42
    ).clip(0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(output_dir / overlay_filename)

    return mask_filename, overlay_filename


def analyze_image(
    image_path: Path,
    output_dir: Path,
    job_id: str,
    gsd_cm_per_pixel: float,
    carbon_coefficient: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)

    if MODEL_PATH.exists():
        try:
            mask = predict_with_model(rgb)
            mode = "model"
        except NotImplementedError:
            mask = _demo_segmentation(rgb)
            mode = "demo"
    else:
        mask = _demo_segmentation(rgb)
        mode = "demo"

    if mask.shape != rgb.shape[:2]:
        raise ValueError("模型輸出的 mask 尺寸與原始影像不一致")

    counts = np.bincount(mask.reshape(-1), minlength=3)
    background_pixels, grassland_pixels, barren_pixels = map(int, counts[:3])
    total_pixels = int(mask.size)

    grassland_pct = grassland_pixels / total_pixels * 100.0
    barren_pct = barren_pixels / total_pixels * 100.0
    background_pct = background_pixels / total_pixels * 100.0

    meters_per_pixel = gsd_cm_per_pixel / 100.0
    square_meters_per_pixel = meters_per_pixel ** 2
    vegetation_area_m2 = grassland_pixels * square_meters_per_pixel
    carbon_sink = vegetation_area_m2 * carbon_coefficient

    mask_filename, overlay_filename = _save_visuals(
        rgb=rgb,
        mask=mask,
        output_dir=output_dir,
        job_id=job_id,
    )

    return {
        "mode": mode,
        "model": "DeepLabV3+ · ResNet50" if mode == "model" else "Demo RGB segmentation",
        "validation_miou": None,
        "width": image.width,
        "height": image.height,
        "gsd_cm_per_pixel": gsd_cm_per_pixel,
        "carbon_coefficient": carbon_coefficient,
        "background_pixels": background_pixels,
        "grassland_pixels": grassland_pixels,
        "barren_pixels": barren_pixels,
        "background_pct": round(background_pct, 4),
        "grassland_pct": round(grassland_pct, 4),
        "barren_pct": round(barren_pct, 4),
        "vegetation_area_m2": round(vegetation_area_m2, 4),
        "carbon_sink_kg_co2e": round(carbon_sink, 4),
        "mask_filename": mask_filename,
        "overlay_filename": overlay_filename,
    }
