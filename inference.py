from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Bare Soil",
    "Other Non-Veg",
    "Herbaceous",
    "Shrub",
    "Trees",
    "Cultivated",
]
NUM_CLASSES = 6
VEGETATION_CLASS_IDS = [2, 3, 4, 5]

CLASS_COLORS = np.array(
    [
        [154, 103, 64],   # Bare Soil
        [100, 116, 139],  # Other Non-Veg
        [132, 204, 22],   # Herbaceous
        [22, 163, 74],    # Shrub
        [21, 128, 61],    # Trees
        [234, 179, 8],    # Cultivated
    ],
    dtype=np.uint8,
)

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

_MODELS = {}
_PROCESSORS = {}
_MODEL_META = {}
_MODEL_MTIMES = {}


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
            "SegFormer 需要 transformers，請執行 "
            "python -m pip install -r requirements.txt"
        ) from exc

    if not model_path.is_dir():
        raise ValueError(
            "SegFormer 請使用 Hugging Face save_pretrained() 資料夾格式"
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

    if int(model.config.num_labels) != NUM_CLASSES:
        raise ValueError(
            f"SegFormer 必須是 {NUM_CLASSES} 類，"
            f"目前 config.json num_labels={model.config.num_labels}"
        )

    model.to(DEVICE)
    model.eval()

    raw_id2label = dict(model.config.id2label or {})
    id2label = {
        int(k): str(v)
        for k, v in raw_id2label.items()
    }
    if len(id2label) != NUM_CLASSES:
        id2label = {i: name for i, name in enumerate(CLASS_NAMES)}

    _PROCESSORS["segformer"] = processor
    _MODEL_META["segformer"] = {
        "encoder": getattr(model.config, "model_type", "segformer"),
        "num_classes": NUM_CLASSES,
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
        num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))
        image_size = int(checkpoint.get("image_size", 512))
        best_miou = checkpoint.get("best_miou")
        class_names = checkpoint.get("class_names", CLASS_NAMES)
    else:
        state_dict = checkpoint
        encoder = config["default_encoder"]
        num_classes = NUM_CLASSES
        image_size = 512
        best_miou = None
        class_names = CLASS_NAMES

    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"{config['display_name']} checkpoint 是 {num_classes} 類，"
            f"目前網站需要 {NUM_CLASSES} 類，請換成重新訓練的模型。"
        )

    model = _build_smp_model(model_name, encoder, num_classes)
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
    inputs = processor(
        images=Image.fromarray(rgb),
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = F.interpolate(
            outputs.logits,
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
    if int(mask.max()) >= NUM_CLASSES:
        raise ValueError("模型輸出出現 0~5 以外的 class id")

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


def _build_class_stats(mask, id2label):
    counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
    total = int(mask.size)
    stats = []

    for class_id in range(NUM_CLASSES):
        color = CLASS_COLORS[class_id]
        pixels = int(counts[class_id])
        pct = pixels / total * 100.0 if total else 0.0
        stats.append({
            "id": class_id,
            "label": id2label.get(class_id, CLASS_NAMES[class_id]),
            "pixels": pixels,
            "pct": round(pct, 4),
            "color": "#{:02X}{:02X}{:02X}".format(*map(int, color)),
            "is_vegetation": class_id in VEGETATION_CLASS_IDS,
        })

    return stats, counts


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
        "class_names": CLASS_NAMES,
        "vegetation_class_ids": VEGETATION_CLASS_IDS,
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

    mask = predict_with_model(rgb, model_name)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("模型輸出的 mask 尺寸與原始影像不一致")

    meta = _MODEL_META.get(model_name, {})
    id2label = meta.get(
        "id2label",
        {i: name for i, name in enumerate(CLASS_NAMES)},
    )

    class_stats, counts = _build_class_stats(mask, id2label)

    total_pixels = int(mask.size)
    vegetation_pixels = int(
        sum(counts[i] for i in VEGETATION_CLASS_IDS)
    )
    vegetation_pct = (
        vegetation_pixels / total_pixels * 100.0
        if total_pixels
        else 0.0
    )

    meters_per_pixel = gsd_cm_per_pixel / 100.0
    vegetation_area_m2 = (
        vegetation_pixels * (meters_per_pixel ** 2)
    )
    carbon_sink = vegetation_area_m2 * carbon_coefficient

    mask_filename, overlay_filename = _save_visuals(
        rgb,
        mask,
        output_dir,
        job_id,
        model_name,
    )

    config = MODEL_CONFIG[model_name]

    return {
        "mode": "model",
        "selected_model": model_name,
        "model": config["display_name"],
        "encoder": meta.get("encoder", config["default_encoder"]),
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES,
        "validation_miou": meta.get("best_miou"),
        "id2label": id2label,
        "class_stats": class_stats,
        "width": image.width,
        "height": image.height,
        "gsd_cm_per_pixel": gsd_cm_per_pixel,
        "carbon_coefficient": carbon_coefficient,
        "vegetation_pixels": vegetation_pixels,
        "vegetation_pct": round(vegetation_pct, 4),
        "vegetation_area_m2": round(vegetation_area_m2, 4),
        "carbon_sink_kg_co2e": round(carbon_sink, 4),
        "carbon_sink_kg_co2e_per_year": round(carbon_sink, 4),
        "carbon_method": "area_times_annual_coefficient",
        "carbon_formula": "vegetation_area_m2 × carbon_coefficient_kg_co2e_m2_year",
        "mask_filename": mask_filename,
        "overlay_filename": overlay_filename,
    }
