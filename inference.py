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

INFERENCE_TILE_SIZE = 512
TILE_OVERLAP_RATIO = 0.50

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
        "default_encoder": "resnet50",
        "paths": [MODELS_DIR / "unet_best.pth"],
    },
    "pspnet": {
        "display_name": "PSPNet",
        "default_encoder": "resnet50",
        "paths": [MODELS_DIR / "pspnet_resnet50_6class.pth"],
    },
    "segformer": {
        "display_name": "SegFormer",
        "default_encoder": "MiT",
        "paths": [
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
    if model_name == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
    raise ValueError(f"未知 SMP 模型：{model_name}")


def _load_segformer(model_path):
    """Load the selected .pth weights using the matching local HF configuration."""
    try:
        from transformers import (
            SegformerConfig,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "SegFormer 需要 transformers，請執行 "
            "python -m pip install -r requirements.txt"
        ) from exc

    if not model_path.is_file() or model_path.suffix.lower() != ".pth":
        raise ValueError(
            "SegFormer 現在指定 models/segformer_best.pth；"
            "請確認檔案存在且副檔名為 .pth"
        )

    checkpoint = _load_checkpoint(model_path)
    if not isinstance(checkpoint, dict):
        raise ValueError("SegFormer .pth 必須包含 state_dict 或 checkpoint 字典")

    state_dict = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            break

    if not state_dict or not all(isinstance(key, str) for key in state_dict):
        raise ValueError("SegFormer checkpoint 沒有有效的權重字典")

    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    # A matching HF config is required: a plain state_dict has no architecture
    # information. Do not silently infer MiT depth or initialize random heads.
    config_dir = MODELS_DIR / "best_model_segformer"
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        config = SegformerConfig.from_dict(checkpoint_config)
    elif config_dir.is_dir() and (config_dir / "config.json").exists():
        config = SegformerConfig.from_pretrained(
            str(config_dir), local_files_only=True
        )
    else:
        raise FileNotFoundError(
            "segformer_best.pth 缺少網路架構資訊。請保留 "
            "models/best_model_segformer/config.json，或在 checkpoint "
            "內提供 config 字典。不可只依檔名推測模型架構。"
        )

    if int(config.num_labels) != NUM_CLASSES:
        raise ValueError(
            f"SegFormer checkpoint config 為 {config.num_labels} 類，"
            f"網站要求 {NUM_CLASSES} 類"
        )

    processor_dir = MODELS_DIR / "best_model_segformer"
    if not (processor_dir / "preprocessor_config.json").exists():
        raise FileNotFoundError(
            "SegFormer 缺少 preprocessor_config.json；"
            "請保留 models/best_model_segformer 資料夾供影像前處理使用"
        )
    processor = SegformerImageProcessor.from_pretrained(
        str(processor_dir), local_files_only=True
    )

    model = SegformerForSemanticSegmentation(config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "segformer_best.pth 與目前的 SegFormer config 不相容，"
            "請確認使用相同 MiT 架構與六類輸出，且權重為 "
            "Hugging Face SegformerForSemanticSegmentation 格式"
        ) from exc

    model.to(DEVICE)
    model.eval()

    raw_id2label = dict(config.id2label or {})
    id2label = {int(k): str(v) for k, v in raw_id2label.items()}
    if len(id2label) != NUM_CLASSES:
        id2label = {i: name for i, name in enumerate(CLASS_NAMES)}

    _PROCESSORS["segformer"] = processor
    _MODEL_META["segformer"] = {
        "encoder": getattr(config, "model_type", "segformer"),
        "num_classes": NUM_CLASSES,
        "image_size": None,
        "best_miou": checkpoint.get("best_miou"),
        "checkpoint_path": str(model_path),
        "id2label": id2label,
    }
    return model


def _infer_resnet_encoder_from_state_dict(state_dict, fallback):
    keys = set(state_dict.keys())

    # ResNet50/101/152 use Bottleneck blocks and therefore contain conv3.
    has_bottleneck = any(
        key.startswith("encoder.layer1.")
        and ".conv3.weight" in key
        for key in keys
    )

    if not has_bottleneck:
        return "resnet34" if fallback.startswith("resnet") else fallback

    def max_block_index(layer_name):
        prefix = f"encoder.{layer_name}."
        indexes = []
        for key in keys:
            if key.startswith(prefix):
                rest = key[len(prefix):]
                block = rest.split(".", 1)[0]
                if block.isdigit():
                    indexes.append(int(block))
        return max(indexes) if indexes else -1

    layer3_max = max_block_index("layer3")

    # layer3 blocks: ResNet50=6 (0..5), ResNet101=23 (0..22),
    # ResNet152=36 (0..35).
    if layer3_max >= 35:
        return "resnet152"
    if layer3_max >= 22:
        return "resnet101"
    return "resnet50"


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

        if model_name == "unet" and "encoder" not in checkpoint:
            encoder = _infer_resnet_encoder_from_state_dict(
                state_dict,
                encoder,
            )

        num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))
        image_size = int(checkpoint.get("image_size", 512))
        best_miou = checkpoint.get("best_miou")
        class_names = checkpoint.get("class_names", CLASS_NAMES)
    else:
        state_dict = checkpoint
        encoder = config["default_encoder"]

        if model_name == "unet":
            encoder = _infer_resnet_encoder_from_state_dict(
                state_dict,
                encoder,
            )

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


def _prepare_tensor(rgb, image_size, normalize=True):
    image = Image.fromarray(rgb).resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    if normalize:
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


def _tile_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size

    if positions[-1] != last:
        positions.append(last)

    return positions


def _make_tile_weight(tile_size):
    window = np.hanning(tile_size).astype(np.float32)
    window = np.maximum(window, 0.05)
    weight = np.outer(window, window)
    return weight.astype(np.float32)


def _predict_tiled_smp(rgb, model_name, normalize=True):
    model = load_model(model_name)
    # Use a smaller fixed tile at native resolution to preserve tiny
    # vegetation and soil gaps, even when the checkpoint was trained with
    # a larger nominal input size.
    tile_size = INFERENCE_TILE_SIZE
    overlap = max(
        32,
        int(tile_size * TILE_OVERLAP_RATIO),
    )
    stride = max(1, tile_size - overlap)

    height, width = rgb.shape[:2]
    padded_h = max(height, tile_size)
    padded_w = max(width, tile_size)

    padded = np.zeros(
        (padded_h, padded_w, 3),
        dtype=np.uint8,
    )
    padded[:height, :width] = rgb

    if height < padded_h:
        padded[height:, :width] = rgb[height - 1:height, :width]

    if width < padded_w:
        padded[:, width:] = padded[:, width - 1:width]

    y_positions = _tile_positions(
        padded_h,
        tile_size,
        stride,
    )
    x_positions = _tile_positions(
        padded_w,
        tile_size,
        stride,
    )

    logits_sum = np.zeros(
        (NUM_CLASSES, padded_h, padded_w),
        dtype=np.float32,
    )
    weight_sum = np.zeros(
        (padded_h, padded_w),
        dtype=np.float32,
    )
    tile_weight = _make_tile_weight(tile_size)

    amp_enabled = DEVICE.type == "cuda"

    with torch.inference_mode():
        for y in y_positions:
            for x in x_positions:
                tile = padded[
                    y:y + tile_size,
                    x:x + tile_size,
                ]

                tensor = _prepare_tensor(
                    tile,
                    tile_size,
                    normalize=normalize,
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=amp_enabled,
                ):
                    logits = model(tensor)

                if logits.shape[-2:] != (
                    tile_size,
                    tile_size,
                ):
                    logits = F.interpolate(
                        logits,
                        size=(tile_size, tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                tile_logits = (
                    logits[0]
                    .float()
                    .cpu()
                    .numpy()
                )

                logits_sum[
                    :,
                    y:y + tile_size,
                    x:x + tile_size,
                ] += tile_logits * tile_weight[None, :, :]

                weight_sum[
                    y:y + tile_size,
                    x:x + tile_size,
                ] += tile_weight

    logits_sum /= np.maximum(
        weight_sum[None, :, :],
        1e-6,
    )

    mask = np.argmax(
        logits_sum[:, :height, :width],
        axis=0,
    )

    return mask.astype(np.uint8)


def _dominant_class_fraction(mask):
    counts = np.bincount(
        mask.reshape(-1),
        minlength=NUM_CLASSES,
    )
    return float(counts.max()) / max(int(mask.size), 1)


def _predict_pspnet_with_preprocessing_fallback(rgb):
    # This PSPNet checkpoint is a raw state_dict and does not store
    # preprocessing metadata. Try ImageNet normalization first, then
    # automatically retry raw [0,1] input only if the result collapses
    # almost entirely into a single class.
    mask_normalized = _predict_tiled_smp(
        rgb,
        "pspnet",
        normalize=True,
    )
    normalized_dominance = _dominant_class_fraction(
        mask_normalized
    )

    if normalized_dominance < 0.985:
        _MODEL_META["pspnet"]["preprocessing"] = "imagenet"
        _MODEL_META["pspnet"]["dominant_class_fraction"] = (
            normalized_dominance
        )
        return mask_normalized

    mask_raw = _predict_tiled_smp(
        rgb,
        "pspnet",
        normalize=False,
    )
    raw_dominance = _dominant_class_fraction(mask_raw)

    if raw_dominance + 0.02 < normalized_dominance:
        _MODEL_META["pspnet"]["preprocessing"] = "raw_0_1"
        _MODEL_META["pspnet"]["dominant_class_fraction"] = (
            raw_dominance
        )
        return mask_raw

    _MODEL_META["pspnet"]["preprocessing"] = "imagenet"
    _MODEL_META["pspnet"]["dominant_class_fraction"] = (
        normalized_dominance
    )
    return mask_normalized


def predict_with_model(rgb, model_name):
    if model_name == "segformer":
        return _predict_segformer(rgb)

    if model_name == "pspnet":
        return _predict_pspnet_with_preprocessing_fallback(
            rgb
        )

    return _predict_tiled_smp(
        rgb,
        model_name,
        normalize=True,
    )


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
        "inference_tile_size": INFERENCE_TILE_SIZE,
        "tile_overlap_ratio": TILE_OVERLAP_RATIO,
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
        "preprocessing": meta.get("preprocessing"),
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
