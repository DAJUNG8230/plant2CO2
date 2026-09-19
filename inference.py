"""
plant2CO2 inference layer.

Class ids:
0 = Background / Other
1 = Grassland
2 = Barren

When models/best_model.pth exists, this module automatically loads the
DeepLabV3+ checkpoint produced by training/train.py. If no checkpoint exists,
the web demo falls back to a lightweight RGB rule-based segmentation.
"""

from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_COLORS = np.array(
    [
        [71, 85, 105],   # Background
        [23, 143, 89],   # Grassland
        [154, 103, 64],  # Barren
    ],
    dtype=np.uint8,
)

_MODEL = None
_MODEL_META = {}
_MODEL_MTIME = None


def _demo_segmentation(rgb: np.ndarray) -> np.ndarray:
    """Simple RGB rules used only until a trained model is available."""
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


def _load_checkpoint():
    try:
        return torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(MODEL_PATH, map_location=DEVICE)


def load_model():
    """
    Load and cache the trained DeepLabV3+ model.

    The cache is refreshed automatically when best_model.pth is replaced.
    """
    global _MODEL, _MODEL_META, _MODEL_MTIME

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型權重：{MODEL_PATH}. "
            "請先執行 training/train.py。"
        )

    current_mtime = MODEL_PATH.stat().st_mtime

    if _MODEL is not None and _MODEL_MTIME == current_mtime:
        return _MODEL

    checkpoint = _load_checkpoint()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        encoder = checkpoint.get("encoder", "resnet50")
        num_classes = int(checkpoint.get("num_classes", 3))
        image_size = int(checkpoint.get("image_size", 512))
        class_names = checkpoint.get(
            "class_names",
            ["Background", "Grassland", "Barren"],
        )
        best_miou = checkpoint.get("best_miou")
    else:
        # Also support a raw state_dict checkpoint.
        state_dict = checkpoint
        encoder = "resnet50"
        num_classes = 3
        image_size = 512
        class_names = ["Background", "Grassland", "Barren"]
        best_miou = None

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    _MODEL = model
    _MODEL_MTIME = current_mtime
    _MODEL_META = {
        "encoder": encoder,
        "num_classes": num_classes,
        "image_size": image_size,
        "class_names": class_names,
        "best_miou": float(best_miou) if best_miou is not None else None,
        "device": str(DEVICE),
    }

    return _MODEL


def _prepare_tensor(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    image = Image.fromarray(rgb).resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=tensor.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=tensor.dtype,
    ).view(1, 3, 1, 1)

    tensor = (tensor - mean) / std
    return tensor.to(DEVICE)


def predict_with_model(rgb: np.ndarray) -> np.ndarray:
    """
    Run DeepLabV3+ inference and return an H x W uint8 class-index mask.
    """
    model = load_model()
    image_size = int(_MODEL_META.get("image_size", 512))

    input_tensor = _prepare_tensor(rgb, image_size)

    with torch.inference_mode():
        logits = model(input_tensor)

        # Resize logits back to the original image resolution before argmax.
        logits = F.interpolate(
            logits,
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        mask = torch.argmax(logits, dim=1)[0]

    return mask.detach().cpu().numpy().astype(np.uint8)


def _save_visuals(
    rgb: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    job_id: str,
) -> tuple[str, str]:
    if int(mask.max()) >= len(CLASS_COLORS):
        raise ValueError("模型輸出了未定義的 class id")

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


def get_runtime_status() -> dict:
    """Return lightweight runtime information for the Flask status endpoint."""
    return {
        "checkpoint_exists": MODEL_PATH.exists(),
        "checkpoint_path": str(MODEL_PATH),
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "loaded": _MODEL is not None,
        "model_meta": _MODEL_META or None,
    }


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

    model_error = None

    if MODEL_PATH.exists():
        try:
            mask = predict_with_model(rgb)
            mode = "model"
        except Exception as exc:
            # Keep the website usable during development, but expose the error
            # instead of silently pretending the fallback is the real model.
            mask = _demo_segmentation(rgb)
            mode = "demo_fallback"
            model_error = str(exc)
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

    validation_miou = (
        _MODEL_META.get("best_miou")
        if mode == "model"
        else None
    )

    return {
        "mode": mode,
        "model": (
            f"DeepLabV3+ · {_MODEL_META.get('encoder', 'resnet50')}"
            if mode == "model"
            else "Demo RGB segmentation"
        ),
        "device": str(DEVICE),
        "validation_miou": validation_miou,
        "model_error": model_error,
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
