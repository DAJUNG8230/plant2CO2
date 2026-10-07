from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "deeplabv3plus": {
        "display_name": "DeepLabV3+",
        "default_encoder": "resnet50",
        "paths": [
            MODELS_DIR / "deeplabv3plus_best.pth",
            MODELS_DIR / "best_model.pth",
        ],
    },
    "unet": {
        "display_name": "U-Net",
        "default_encoder": "resnet34",
        "paths": [MODELS_DIR / "unet_best.pth"],
    },
    "segformer": {
        "display_name": "SegFormer",
        "default_encoder": "MiT",
        "paths": [
            MODELS_DIR / "best_model_segformer",
            MODELS_DIR / "segformer_best",
            MODELS_DIR / "segformer_best.pth",
        ],
    },
}

CLASS_COLORS = np.array(
    [
        [71, 85, 105],   # Background
        [23, 143, 89],   # Grassland
        [154, 103, 64],  # Barren
    ],
    dtype=np.uint8,
)

_MODELS = {}
_PROCESSORS = {}
_MODEL_META = {}
_MODEL_MTIMES = {}


def _demo_segmentation(rgb):
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
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


def _get_model_path(model_name):
    for path in MODEL_CONFIG[model_name]["paths"]:
        if path.exists():
            return path
    return None


def _path_mtime(path):
    if path.is_file():
        return path.stat().st_mtime

    mtimes = [path.stat().st_mtime]
    for child in path.rglob("*"):
        if child.is_file():
            mtimes.append(child.stat().st_mtime)
    return max(mtimes)


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _build_smp_model(model_name, encoder, num_classes):
    if model_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )

    if model_name == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )

    raise ValueError(f"未知 SMP 模型：{model_name}")


def _load_segformer(model_path):
    try:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "SegFormer 需要 transformers 套件，請執行 "
            "python -m pip install -r requirements.txt"
        ) from exc

    if not model_path.is_dir():
        raise ValueError(
            "SegFormer 建議使用 Hugging Face save_pretrained() 資料夾格式。"
            "請將 config.json、model.safetensors、preprocessor_config.json "
            "放在 models/best_model_segformer/。"
        )

    required = [
        model_path / "config.json",
        model_path / "preprocessor_config.json",
    ]
    if not all(p.exists() for p in required):
        raise FileNotFoundError(
            "SegFormer 資料夾缺少 config.json 或 preprocessor_config.json"
        )

    has_weights = (
        (model_path / "model.safetensors").exists()
        or (model_path / "pytorch_model.bin").exists()
    )
    if not has_weights:
        raise FileNotFoundError(
            "SegFormer 資料夾找不到 model.safetensors 或 pytorch_model.bin"
        )

    processor = SegformerImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()

    config = model.config
    id2label = {
        int(k): v for k, v in dict(config.id2label or {}).items()
    }

    _PROCESSORS["segformer"] = processor
    _MODEL_META["segformer"] = {
        "encoder": getattr(config, "model_type", "segformer"),
        "num_classes": int(config.num_labels),
        "image_size": None,
        "best_miou": None,
        "checkpoint_path": str(model_path),
        "id2label": id2label,
    }

    return model


def load_model(model_name):
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"未知模型：{model_name}")

    model_path = _get_model_path(model_name)
    if model_path is None:
        raise FileNotFoundError(f"找不到 {model_name} 模型權重")

    mtime = _path_mtime(model_path)
    if model_name in _MODELS and _MODEL_MTIMES.get(model_name) == mtime:
        return _MODELS[model_name]

    if model_name == "segformer":
        model = _load_segformer(model_path)
        _MODELS[model_name] = model
        _MODEL_MTIMES[model_name] = mtime
        return model

    checkpoint = _load_checkpoint(model_path)
    config = MODEL_CONFIG[model_name]

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        encoder = checkpoint.get("encoder", config["default_encoder"])
        num_classes = int(checkpoint.get("num_classes", 3))
        image_size = int(checkpoint.get("image_size", 512))
        best_miou = checkpoint.get("best_miou")
        class_names = checkpoint.get(
            "class_names",
            ["Background", "Grassland", "Barren"],
        )
    else:
        state_dict = checkpoint
        encoder = config["default_encoder"]
        num_classes = 3
        image_size = 512
        best_miou = None
        class_names = ["Background", "Grassland", "Barren"]

    model = _build_smp_model(
        model_name=model_name,
        encoder=encoder,
        num_classes=num_classes,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    _MODELS[model_name] = model
    _MODEL_MTIMES[model_name] = mtime
    _MODEL_META[model_name] = {
        "encoder": encoder,
        "num_classes": num_classes,
        "image_size": image_size,
        "best_miou": float(best_miou) if best_miou is not None else None,
        "checkpoint_path": str(model_path),
        "id2label": {i: name for i, name in enumerate(class_names)},
    }

    return model


def _prepare_tensor(rgb, image_size):
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

    return ((tensor - mean) / std).to(DEVICE)


def _predict_segformer(rgb):
    model = load_model("segformer")
    processor = _PROCESSORS["segformer"]

    image = Image.fromarray(rgb)
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.argmax(logits, dim=1)[0]

    return mask.cpu().numpy().astype(np.uint8)


def predict_with_model(rgb, model_name):
    if model_name == "segformer":
        return _predict_segformer(rgb)

    model = load_model(model_name)
    image_size = int(_MODEL_META[model_name].get("image_size", 512))
    x = _prepare_tensor(rgb, image_size)

    with torch.inference_mode():
        logits = model(x)
        logits = F.interpolate(
            logits,
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.argmax(logits, dim=1)[0]

    return mask.cpu().numpy().astype(np.uint8)


def _save_visuals(rgb, mask, output_dir, job_id, model_name):
    if int(mask.max()) >= len(CLASS_COLORS):
        raise ValueError(
            "模型輸出的 class id 超出網站目前的 3 類設定。"
            "請確認模型類別順序為 "
            "0=Background, 1=Grassland, 2=Barren。"
        )

    color_mask = CLASS_COLORS[mask]

    mask_filename = f"{job_id}_{model_name}_mask.png"
    overlay_filename = f"{job_id}_{model_name}_overlay.png"

    Image.fromarray(color_mask).save(output_dir / mask_filename)

    overlay = (
        rgb.astype(np.float32) * 0.58
        + color_mask.astype(np.float32) * 0.42
    ).clip(0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(output_dir / overlay_filename)
    return mask_filename, overlay_filename


def get_runtime_status():
    models = {}

    for name, config in MODEL_CONFIG.items():
        path = _get_model_path(name)
        models[name] = {
            "display_name": config["display_name"],
            "checkpoint_exists": path is not None,
            "checkpoint_path": str(path) if path else None,
            "loaded": name in _MODELS,
            "model_meta": _MODEL_META.get(name),
        }

    return {
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "models": models,
    }


def analyze_image(
    image_path,
    output_dir,
    job_id,
    gsd_cm_per_pixel,
    carbon_coefficient,
    model_name="deeplabv3plus",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in MODEL_CONFIG:
        raise ValueError(f"未知模型：{model_name}")

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)

    model_error = None

    if _get_model_path(model_name) is not None:
        try:
            mask = predict_with_model(rgb, model_name)
            mode = "model"
        except Exception as exc:
            mask = _demo_segmentation(rgb)
            mode = "demo_fallback"
            model_error = str(exc)
    else:
        mask = _demo_segmentation(rgb)
        mode = "demo"

    if mask.shape != rgb.shape[:2]:
        raise ValueError("模型輸出的 mask 尺寸與原始影像不一致")

    counts = np.bincount(mask.reshape(-1), minlength=3)
    background_pixels, grassland_pixels, barren_pixels = map(
        int,
        counts[:3],
    )
    total_pixels = int(mask.size)

    grassland_pct = grassland_pixels / total_pixels * 100.0
    barren_pct = barren_pixels / total_pixels * 100.0
    background_pct = background_pixels / total_pixels * 100.0

    m_per_pixel = gsd_cm_per_pixel / 100.0
    vegetation_area_m2 = grassland_pixels * (m_per_pixel ** 2)
    carbon_sink = vegetation_area_m2 * carbon_coefficient

    mask_filename, overlay_filename = _save_visuals(
        rgb,
        mask,
        output_dir,
        job_id,
        model_name,
    )

    meta = _MODEL_META.get(model_name, {})
    config = MODEL_CONFIG[model_name]

    return {
        "mode": mode,
        "selected_model": model_name,
        "model": config["display_name"],
        "encoder": meta.get("encoder", config["default_encoder"]),
        "device": str(DEVICE),
        "validation_miou": (
            meta.get("best_miou")
            if mode == "model"
            else None
        ),
        "id2label": meta.get("id2label"),
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
