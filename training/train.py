from pathlib import Path
import csv
import random
import argparse
import hashlib
import json
from datetime import datetime, timezone

from level2_checkpoint import save_full, restore_full, read_approved, apply_approved, save_best_protected
from experiment_templates import get_experiment, BASE_CLASS_WEIGHTS, EXPERIMENTS

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rasterio
except ImportError:
    rasterio = None

import torch

try:
    import pynvml
except ImportError:
    pynvml = None

from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    WeightedRandomSampler,
)
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import segmentation_models_pytorch as smp
from tqdm import tqdm

NUM_CLASSES = 6
IGNORE_INDEX = 255

CLASS_NAMES = [
    "Bare Soil",
    "Other Non-Veg",
    "Herbaceous",
    "Shrub",
    "Trees",
    "Cultivated",
]
VEGETATION_CLASS_IDS = [2, 3, 4, 5]

IMAGE_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 3e-5
EARLY_STOPPING_PATIENCE = 12

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True
VALIDATE_EVERY = 2
GRAD_NORM_EVERY = 100
USE_AMP = True
RESUME_FROM_BEST = False

GPU_TEMP_LIMIT_C = 80
GPU_TEMP_CHECK_EVERY_BATCHES = 50
TARGET_VEGETATION_MIOU = 0.80

USE_FLAIR = True
USE_LABEL_STUDIO = True

# 每個 training epoch 預期抽樣比例。
# 即使 Label Studio 圖片數遠少於 FLAIR，也讓自有場景有足夠影響力。
LABEL_STUDIO_TRAIN_SHARE = 0.30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Native-detail training settings for Windows + rasterio + RTX 3070 Ti.
torch.set_num_threads(6)

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

OUTPUT_DIR = Path("outputs_training_v5_level2")
PLOT_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


BATCH_SIZE = 4


def load_image(path):
    path = Path(path)

    if path.suffix.lower() in {".tif", ".tiff"}:
        if rasterio is None:
            raise ImportError(
                "讀取 FLAIR GeoTIFF 需要 rasterio。"
                "請執行 python -m pip install -r requirements.txt"
            )

        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError(
                    f"{path} 只有 {src.count} 個 band，至少需要 RGB 3 bands"
                )

            rgb = src.read([1, 2, 3])
            rgb = np.transpose(rgb, (1, 2, 0))

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return Image.fromarray(rgb)

    return Image.open(path).convert("RGB")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.images = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".tif", ".tiff"
            }
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        return load_pair(image_path, mask_path, self.augment)


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, augment=False):
        self.augment = augment
        self.rows = []

        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = Path(row["image"])
                mask_path = Path(row["mask"])
                if image_path.exists() and mask_path.exists():
                    self.rows.append((image_path, mask_path))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path, mask_path = self.rows[index]
        return load_pair(image_path, mask_path, self.augment)


def _pad_to_crop_size(image, mask):
    width, height = image.size
    pad_right = max(0, IMAGE_SIZE - width)
    pad_bottom = max(0, IMAGE_SIZE - height)

    if pad_right > 0 or pad_bottom > 0:
        image = TF.pad(
            image,
            [0, 0, pad_right, pad_bottom],
            fill=0,
        )
        mask = TF.pad(
            mask,
            [0, 0, pad_right, pad_bottom],
            fill=IGNORE_INDEX,
        )

    return image, mask


def _native_crop_pair(image, mask, augment):
    image, mask = _pad_to_crop_size(image, mask)
    width, height = image.size

    if augment:
        top = random.randint(0, height - IMAGE_SIZE)
        left = random.randint(0, width - IMAGE_SIZE)
    else:
        top = max(0, (height - IMAGE_SIZE) // 2)
        left = max(0, (width - IMAGE_SIZE) // 2)

    image = TF.crop(
        image,
        top,
        left,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    mask = TF.crop(
        mask,
        top,
        left,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    return image, mask


def load_pair(image_path, mask_path, augment):
    try:
        image = load_image(image_path)
        mask = Image.open(mask_path)
    except Exception as exc:
        raise RuntimeError(
            f"讀取資料失敗\nImage: {image_path}\nMask: {mask_path}\n"
            f"原因: {type(exc).__name__}: {exc}"
        ) from exc

    # Preserve native pixel detail instead of resizing the whole image.
    # FLAIR 512x512 stays untouched; larger custom images are cropped at
    # native resolution so tiny vegetation is not smoothed away.
    image, mask = _native_crop_pair(
        image,
        mask,
        augment,
    )

    image = TF.to_tensor(image)
    mask = torch.from_numpy(
        np.asarray(mask, dtype=np.int64).copy()
    )

    unique = torch.unique(mask)
    valid_values = (
        ((unique >= 0) & (unique < NUM_CLASSES))
        | (unique == IGNORE_INDEX)
    )
    if not torch.all(valid_values):
        raise ValueError(
            f"{mask_path} 出現非法 class id: {unique.tolist()}，"
            "只允許 0~5 或 255(ignore)。"
        )

    if augment:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])

        k = random.randint(0, 3)
        image = torch.rot90(image, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[0, 1])

        if random.random() < 0.4:
            image = TF.adjust_brightness(
                image,
                random.uniform(0.90, 1.10),
            )
        if random.random() < 0.4:
            image = TF.adjust_contrast(
                image,
                random.uniform(0.90, 1.10),
            )
        if random.random() < 0.3:
            image = TF.adjust_saturation(
                image,
                random.uniform(0.90, 1.10),
            )

    image = TF.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return image, mask


def masked_dice_loss(logits, targets):
    probs = torch.softmax(logits, dim=1)
    valid = targets != IGNORE_INDEX

    safe_targets = targets.clone()
    safe_targets[~valid] = 0

    one_hot = torch.nn.functional.one_hot(
        safe_targets,
        NUM_CLASSES,
    ).permute(0, 3, 1, 2).float()

    valid_mask = valid.unsqueeze(1).float()
    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)

    dice = (2.0 * intersection + 1e-6) / (
        denominator + 1e-6
    )

    # Dice weighting is fixed across all Level 3 pilot/control runs.
    # The one allowed experiment variable is the CrossEntropy class weight
    # configured in main(). Never reference main()-local experiment here.
    class_weights = torch.tensor(
        BASE_CLASS_WEIGHTS,
        device=logits.device,
        dtype=dice.dtype,
    )
    class_weights = class_weights / class_weights.sum()

    return 1.0 - (dice * class_weights).sum()

def boundary_loss(logits, targets):
    probs = torch.softmax(logits, dim=1)
    valid = targets != IGNORE_INDEX

    safe_targets = targets.clone()
    safe_targets[~valid] = 0

    one_hot = torch.nn.functional.one_hot(
        safe_targets,
        NUM_CLASSES,
    ).permute(0, 3, 1, 2).float()

    valid_mask = valid.unsqueeze(1).float()

    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    def morphological_gradient(x):
        dilated = torch.nn.functional.max_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        eroded = -torch.nn.functional.max_pool2d(
            -x,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return (dilated - eroded).clamp(0.0, 1.0)

    pred_edges = morphological_gradient(probs)
    target_edges = morphological_gradient(one_hot)

    diff = torch.abs(pred_edges - target_edges) * valid_mask
    denominator = valid_mask.sum() * NUM_CLASSES

    return diff.sum() / denominator.clamp_min(1.0)


def has_valid_pixels(targets):
    return bool((targets != IGNORE_INDEX).any().item())


def safe_segmentation_loss(
    outputs,
    masks,
    cross_entropy,
    lovasz_loss,
):
    if not has_valid_pixels(masks):
        return None

    ce = cross_entropy(outputs, masks)
    lovasz = lovasz_loss(outputs, masks)
    dice = masked_dice_loss(outputs, masks)
    boundary = boundary_loss(outputs, masks)

    # Lovasz can return an empty tensor for batches dominated by ignore
    # pixels. Keep boundary supervision active in that case.
    if lovasz.numel() == 0:
        return (
            0.50 * ce
            + 0.25 * dice
            + 0.25 * boundary
        )

    return (
        0.30 * ce
        + 0.30 * lovasz
        + 0.20 * dice
        + 0.20 * boundary
    )


def calculate_iou(confusion_matrix):
    cm = confusion_matrix.float().cpu()
    intersection = torch.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    iou = torch.full((NUM_CLASSES,), float("nan"))
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou


def calculate_binary_vegetation_metrics(confusion_matrix):
    cm = confusion_matrix.float().cpu()
    veg = VEGETATION_CLASS_IDS
    non_veg = [0, 1]

    tp = cm[veg][:, veg].sum().item()
    fn = cm[veg][:, non_veg].sum().item()
    fp = cm[non_veg][:, veg].sum().item()
    tn = cm[non_veg][:, non_veg].sum().item()

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def safe_mean(values):
    valid = ~torch.isnan(values)
    return values[valid].mean().item() if valid.any() else 0.0


def iter_mask_paths(dataset):
    if isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            yield from iter_mask_paths(sub_dataset)
        return

    if isinstance(dataset, ManifestDataset):
        for _, mask_path in dataset.rows:
            yield mask_path
        return

    if isinstance(dataset, SegmentationDataset):
        for image_path in dataset.images:
            yield dataset.mask_dir / f"{image_path.stem}.png"
        return

    raise TypeError(
        f"不支援的 dataset 類型: {type(dataset).__name__}"
    )


def audit_validation_masks(dataset):
    """Fail before training if ground truth cannot evaluate all six classes."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_image = np.zeros(NUM_CLASSES, dtype=np.int64)
    paths = list(iter_mask_paths(dataset))
    for mask_path in tqdm(paths, desc="V5 validation label audit"):
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)
        valid = (mask >= 0) & (mask < NUM_CLASSES)
        if valid.any():
            image_counts = np.bincount(mask[valid], minlength=NUM_CLASSES)
            counts += image_counts
            per_image += (image_counts > 0).astype(np.int64)
    print("\nV5 validation class support (full masks):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name}: {counts[i]:,} pixels, "
              f"{per_image[i]:,}/{len(paths):,} masks")
    missing = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if counts[i] == 0]
    if missing:
        raise RuntimeError(
            "Validation GT missing classes: " + ", ".join(missing)
            + ". Add labeled examples or change the split before training."
        )


def aggregate_vegetation_coverage_error(cm):
    """Dataset-wide coverage bias; individual images can differ substantially."""
    veg = VEGETATION_CLASS_IDS
    gt_pixels = cm[veg, :].sum().item()
    predicted_pixels = cm[:, veg].sum().item()
    total = cm.sum().item()
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        100.0 * gt_pixels / total,
        100.0 * predicted_pixels / total,
        100.0 * (predicted_pixels - gt_pixels) / total,
    )


def compute_class_weights(dataset):
    print("\nComputing automatic class weights from masks...")
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    mask_paths = list(iter_mask_paths(dataset))

    for mask_path in tqdm(
        mask_paths,
        desc="Class statistics",
    ):
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)
        valid = (
            (mask >= 0)
            & (mask < NUM_CLASSES)
            & (mask != IGNORE_INDEX)
        )
        if np.any(valid):
            counts += np.bincount(
                mask[valid].reshape(-1),
                minlength=NUM_CLASSES,
            )

    counts = np.maximum(counts, 1)
    frequencies = counts / counts.sum()

    weights = np.sqrt(
        frequencies.mean() / frequencies
    )
    weights = np.clip(weights, 0.5, 3.0)
    weights = weights / weights.mean()

    print("Class pixel distribution / weights:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {i} {name:<16} "
            f"{frequencies[i] * 100:6.2f}%  "
            f"weight={weights[i]:.3f}"
        )

    return torch.tensor(weights, dtype=torch.float32)


def save_confusion_matrix(cm, epoch):
    cm = cm.detach().cpu().numpy().astype(np.int64)
    path = OUTPUT_DIR / f"confusion_matrix_epoch_{epoch:03d}.csv"

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["GT / Pred", *CLASS_NAMES])
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *cm[i].tolist()])

    latest = OUTPUT_DIR / "confusion_matrix_latest.csv"
    with open(latest, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["GT / Pred", *CLASS_NAMES])
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *cm[i].tolist()])



def calculate_weight_norm(model):
    total = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                total += parameter.detach().float().norm(2).item() ** 2
    return total ** 0.5


def calculate_grad_norm(model):
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5


def save_class_weight_plot(class_weights):
    values = class_weights.detach().cpu().numpy()

    plt.figure(figsize=(9, 5))
    plt.bar(CLASS_NAMES, values)
    plt.title("Training Class Weights")
    plt.xlabel("Class")
    plt.ylabel("Weight")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_weights.png", dpi=160)
    plt.close()


def save_training_plots(history):
    if not history["epoch"]:
        return

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for class_index, class_name in enumerate(CLASS_NAMES):
        plt.plot(
            epochs,
            [row[class_index] * 100 for row in history["ious"]],
            marker="o",
            label=class_name,
        )
    plt.xlabel("Epoch")
    plt.ylabel("IoU (%)")
    plt.title("Per-Class IoU")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_iou_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [v * 100 for v in history["miou_all"]],
        marker="o",
        label="mIoU All",
    )
    plt.plot(
        epochs,
        [v * 100 for v in history["miou_vegetation"]],
        marker="o",
        label="mIoU Vegetation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("mIoU (%)")
    plt.title("mIoU Trend")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "miou_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [v * 100 for v in history["binary_vegetation_iou"]],
        marker="o",
        label="IoU",
    )
    plt.plot(
        epochs,
        [v * 100 for v in history["binary_vegetation_f1"]],
        marker="o",
        label="F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title("Binary Vegetation Metrics")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "binary_vegetation_metrics.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["learning_rate"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "learning_rate_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["weight_norm"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")
    plt.title("Model Weight Norm")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "weight_norm_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["grad_norm"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average L2 Norm")
    plt.title("Gradient Norm")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "gradient_norm_curve.png", dpi=160)
    plt.close()


def get_gpu_temperature_c():
    if DEVICE.type != "cuda":
        return None

    if pynvml is None:
        return None

    try:
        pynvml.nvmlInit()
        index = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        temp = pynvml.nvmlDeviceGetTemperature(
            handle,
            pynvml.NVML_TEMPERATURE_GPU,
        )
        return int(temp)
    except Exception:
        return None


def check_gpu_temperature_or_stop():
    temp = get_gpu_temperature_c()

    if temp is None:
        return False, None

    if temp >= GPU_TEMP_LIMIT_C:
        print(
            f"\n[SAFETY STOP] GPU temperature reached "
            f"{temp}°C (limit {GPU_TEMP_LIMIT_C}°C)."
        )
        print(
            "Training will stop now. "
            "Previously saved best_model.pth is kept."
        )
        return True, temp

    return False, temp


def build_datasets():
    train_sets = []
    val_sets = []

    flair_train_dataset = None
    label_studio_train_dataset = None

    if USE_FLAIR:
        flair_train = Path("flair_manifests/train.csv")
        flair_val = Path("flair_manifests/val.csv")

        if not flair_train.exists() or not flair_val.exists():
            raise FileNotFoundError(
                "找不到 flair_manifests。請先執行 "
                "python training\\prepare_flair_6class.py"
            )

        flair_train_dataset = ManifestDataset(
            flair_train,
            augment=True,
        )
        train_sets.append(flair_train_dataset)
        val_sets.append(
            ManifestDataset(flair_val, augment=False)
        )

    if USE_LABEL_STUDIO:
        custom_train_images = Path("dataset/train/images")
        custom_train_masks = Path("dataset/train/masks")
        custom_val_images = Path("dataset/val/images")
        custom_val_masks = Path("dataset/val/masks")

        if (
            custom_train_images.exists()
            and custom_train_masks.exists()
        ):
            label_studio_train_dataset = SegmentationDataset(
                custom_train_images,
                custom_train_masks,
                augment=True,
            )
            if len(label_studio_train_dataset) > 0:
                train_sets.append(label_studio_train_dataset)

        if (
            custom_val_images.exists()
            and custom_val_masks.exists()
        ):
            custom_val_dataset = SegmentationDataset(
                custom_val_images,
                custom_val_masks,
                augment=False,
            )
            if len(custom_val_dataset) > 0:
                val_sets.append(custom_val_dataset)

    if not train_sets or not val_sets:
        raise RuntimeError("沒有可用的 train/val dataset")

    train_dataset = (
        train_sets[0]
        if len(train_sets) == 1
        else ConcatDataset(train_sets)
    )
    val_dataset = (
        val_sets[0]
        if len(val_sets) == 1
        else ConcatDataset(val_sets)
    )

    train_sampler = None

    if (
        flair_train_dataset is not None
        and label_studio_train_dataset is not None
        and len(flair_train_dataset) > 0
        and len(label_studio_train_dataset) > 0
    ):
        flair_share = 1.0 - LABEL_STUDIO_TRAIN_SHARE

        flair_weight = (
            flair_share / len(flair_train_dataset)
        )
        label_studio_weight = (
            LABEL_STUDIO_TRAIN_SHARE
            / len(label_studio_train_dataset)
        )

        sample_weights = (
            [flair_weight] * len(flair_train_dataset)
            + [label_studio_weight]
            * len(label_studio_train_dataset)
        )

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

        print(
            "Training sampling target: "
            f"FLAIR {flair_share * 100:.0f}% / "
            f"Label Studio "
            f"{LABEL_STUDIO_TRAIN_SHARE * 100:.0f}%"
        )
        print(
            "Source samples: "
            f"FLAIR={len(flair_train_dataset):,}, "
            f"Label Studio="
            f"{len(label_studio_train_dataset):,}"
        )

    return train_dataset, val_dataset, train_sampler


def main():
    parser = argparse.ArgumentParser(description="DeepLab V5 controlled Level 2 experiment")
    parser.add_argument("--resume-full", type=Path,
                        help="Resume from full latest_state.pth (trusted local file only)")
    parser.add_argument("--approved-config", type=Path,
                        help="Approved JSON from Level 2 manual or Codex-controller workflow")
    parser.add_argument("--allow-auto-approved", action="store_true",
                        help="Explicit opt-in to bounded autonomous Codex proposal")
    parser.add_argument("--max-epochs", type=int, default=EPOCHS)
    parser.add_argument("--experiment-id", choices=tuple(EXPERIMENTS),
                        help="Whitelisted single-variable experiment, isolated output only")
    parser.add_argument("--experiment-seed", type=int, default=42)
    parser.add_argument("--experiment-output", type=Path,
                        help="Unique output path under outputs_training_v5_experiments")
    parser.add_argument("--init-from-best", type=Path,
                        help="Trusted local best checkpoint, weights only and new optimizer")
    args = parser.parse_args()
    experiment = get_experiment(args.experiment_id) if args.experiment_id else None
    if experiment:
        if not 0 <= args.experiment_seed <= 2147483647:
            parser.error("Experiment seed out of range")
        # Each paired run reinitializes Python / NumPy / torch RNG identically.
        # CUDA kernels and worker prefetch may remain nondeterministic.
        random.seed(args.experiment_seed)
        np.random.seed(args.experiment_seed)
        torch.manual_seed(args.experiment_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.experiment_seed)
        if args.resume_full or args.approved_config:
            parser.error("Experiment mode cannot resume or apply unrelated approval")
        if not args.experiment_output or not args.init_from_best:
            parser.error("Experiment requires --experiment-output and --init-from-best")
        root = (Path.cwd() / "outputs_training_v5_experiments").resolve()
        target = args.experiment_output.resolve()
        if root not in target.parents or target == root or len(target.relative_to(root).parts) != 1:
            parser.error("Experiment output must be a single unique subdirectory within outputs_training_v5_experiments")
        global OUTPUT_DIR, PLOT_DIR
        OUTPUT_DIR = target
        PLOT_DIR = OUTPUT_DIR / "plots"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        if not 1 <= args.max_epochs <= 20:
            parser.error("Pilot experiments are limited to 1..20 epochs")
    if not 1 <= args.max_epochs <= EPOCHS:
        parser.error(f"--max-epochs must be between 1 and {EPOCHS}; Level 2 is capped at 50")
    approved = read_approved(args.approved_config, allow_autonomous=args.allow_auto_approved)
    print("Level 2 changes approved for launch:", approved or "none")
    print("Device:", DEVICE)
    print("Classes:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")

    train_dataset, val_dataset, train_sampler = build_datasets()

    print("Dataset source: FLAIR + Label Studio")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"DataLoader workers: {NUM_WORKERS}")
    print(f"Pin memory: {PIN_MEMORY and DEVICE.type == 'cuda'}")
    print(f"AMP: {USE_AMP and DEVICE.type == 'cuda'}")
    print(f"GPU temp limit: {GPU_TEMP_LIMIT_C} C")
    print(
        "GPU temp check every: "
        f"{GPU_TEMP_CHECK_EVERY_BATCHES} batches"
    )
    print(f"Validate every: {VALIDATE_EVERY} epoch(s)")
    print(f"Gradient norm every: {GRAD_NORM_EVERY} batch(es)")

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and DEVICE.type == "cuda",
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    audit_validation_masks(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    resume_path = OUTPUT_DIR / "best_model.pth"
    resumed_miou = -1.0
    if experiment:
        # New experiment runs are initialized from the preserved baseline
        # best weights, but optimizer/scheduler and early stopping are new.
        original = args.init_from_best.resolve()
        if original == (OUTPUT_DIR / "best_model.pth").resolve() or not original.is_file():
            raise ValueError("Experiment source must be an existing external best checkpoint")
        source = torch.load(original, map_location="cpu", weights_only=True)
        if not isinstance(source, dict) or "model_state_dict" not in source:
            raise ValueError("Invalid baseline checkpoint format")
        if source.get("num_classes") != NUM_CLASSES or source.get("encoder") != "resnet50":
            raise ValueError("Baseline encoder/classes mismatch")
        model.load_state_dict(source["model_state_dict"], strict=True)
        print("New independent experiment warm-started from:", original)

    if RESUME_FROM_BEST and resume_path.exists():
        checkpoint = torch.load(
            resume_path,
            map_location=DEVICE,
            weights_only=False,
        )
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )
        model.load_state_dict(state_dict)
        resumed_miou = float(
            checkpoint.get("best_miou", -1.0)
        )
        print(
            "\nResumed from:",
            resume_path,
            f"(best vegetation mIoU="
            f"{resumed_miou * 100:.2f}%)",
        )

    class_weights = torch.tensor(
        [1.20, 0.80, 1.25, 1.45, 1.00, 0.90],
        dtype=torch.float32,
        device=DEVICE,
    )
    print("\nManual class weights:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {i} {name:<16} "
            f"weight={class_weights[i].item():.2f}"
        )
    save_class_weight_plot(class_weights)

    cross_entropy = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
        label_smoothing=0.02,
    )
    lovasz_loss = smp.losses.LovaszLoss(
        mode="multiclass",
        per_image=False,
        ignore_index=IGNORE_INDEX,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.get("learning_rate", LEARNING_RATE) if experiment else LEARNING_RATE,
        weight_decay=1e-4,
    )

    amp_enabled = USE_AMP and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_vegetation_miou = resumed_miou
    epochs_without_improvement = 0
    history_path = OUTPUT_DIR / "history.csv"

    def digest(path):
        if not path.exists():
            return "missing"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    data_fingerprint = {
        "flair_train_manifest_sha256": digest(Path("flair_manifests/train.csv")),
        "flair_val_manifest_sha256": digest(Path("flair_manifests/val.csv")),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "encoder": "resnet50",
        "mapping": "v5_unet_stage3_flair_lut_aligned",
        "experiment_id": args.experiment_id or "baseline",
        "experiment_learning_rate": experiment.get("learning_rate", LEARNING_RATE) if experiment else LEARNING_RATE,
        "experiment_seed": args.experiment_seed if experiment else None,
        "class_weights": experiment["class_weights"] if experiment else BASE_CLASS_WEIGHTS,
    }

    start_epoch = 1
    if args.resume_full:
        if not history_path.exists():
            raise FileNotFoundError(
                "Resume history.csv 缺失，為避免覆寫訓練紀錄已停止；"
                "請還原相同實驗的 history.csv。"
            )
        previous_epoch, best_vegetation_miou, epochs_without_improvement = (
            restore_full(
                args.resume_full, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler,
                expected_metadata=data_fingerprint,
            )
        )
        with history_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or int(rows[-1]["epoch"]) != previous_epoch:
            raise RuntimeError(
                "history.csv 最後 Epoch 與完整 checkpoint 不一致；"
                "請先恢復相同版本的歷史紀錄。"
            )
        start_epoch = previous_epoch + 1
        print(f"Fully resumed through epoch {previous_epoch}")
    elif history_path.exists() or (OUTPUT_DIR / "latest_state.pth").exists():
        raise FileExistsError(
            "Level 2 目錄已有訓練紀錄或 checkpoint。"
            "請用 --resume-full outputs_training_v5_level2/latest_state.pth "
            "續訓，或手動備份後建立新的實驗目錄。"
        )

    # Approval is applied AFTER full-state restoration, so optimizer changes
    # cannot be silently undone by optimizer.load_state_dict.
    apply_approved(approved, optimizer, scheduler)
    if args.resume_full and approved:
        # The reviewer explicitly approved a new optimization stage. Retain
        # model and optimizer state, but reset early-stop patience to provide
        # a fair window to evaluate the changed hyperparameter.
        epochs_without_improvement = 0
        print("Approved next-stage settings applied; early-stop count reset to 0.")

    history_header = [
        "epoch",
        "train_loss",
        "val_loss",
        *[
            name.lower().replace(" ", "_") + "_iou"
            for name in CLASS_NAMES
        ],
        "miou_all",
        "miou_vegetation",
        "binary_vegetation_iou",
        "binary_vegetation_precision",
        "binary_vegetation_recall",
        "binary_vegetation_f1",
        *(["coverage_mae_pp"] if experiment else []),
        "learning_rate",
        "weight_norm",
        "grad_norm",
    ]

    if start_epoch == 1:
        with open(
            history_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerow(history_header)

    plot_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "ious": [],
        "miou_all": [],
        "miou_vegetation": [],
        "binary_vegetation_iou": [],
        "binary_vegetation_precision": [],
        "binary_vegetation_recall": [],
        "binary_vegetation_f1": [],
        "learning_rate": [],
        "weight_norm": [],
        "grad_norm": [],
    }

    stop_for_temperature = False
    finished_marker = OUTPUT_DIR / "training_finished.json"
    # Remove old completion marker just before the training loop actually resumes.
    finished_marker.unlink(missing_ok=True)
    completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        train_loss = 0.0
        valid_train_batches = 0
        skipped_train_batches = 0
        grad_norm_sum = 0.0
        grad_norm_steps = 0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{args.max_epochs} Train",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
        )

        for batch_index, (images, masks) in enumerate(
            train_progress,
            start=1,
        ):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
            ):
                outputs = model(images)
                loss = safe_segmentation_loss(
                    outputs,
                    masks,
                    cross_entropy,
                    lovasz_loss,
                )

            if loss is None:
                skipped_train_batches += 1
                continue

            scaler.scale(loss).backward()

            should_measure_grad = (
                batch_index % GRAD_NORM_EVERY == 0
            )

            if should_measure_grad:
                scaler.unscale_(optimizer)
                grad_norm_sum += calculate_grad_norm(model)
                grad_norm_steps += 1

            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            valid_train_batches += 1

            if (
                batch_index
                % GPU_TEMP_CHECK_EVERY_BATCHES
                == 0
            ):
                should_stop, gpu_temp = (
                    check_gpu_temperature_or_stop()
                )

                if gpu_temp is not None:
                    train_progress.set_postfix(
                        temp=f"{gpu_temp}C",
                        refresh=False,
                    )

                if should_stop:
                    train_progress.set_postfix(
                        temp=f"{gpu_temp}C",
                        status="THERMAL STOP",
                        refresh=True,
                    )
                    stop_for_temperature = True
                    break

        if stop_for_temperature:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            break

        train_loss /= max(valid_train_batches, 1)

        if skipped_train_batches > 0:
            print(
                f"Skipped all-ignore train batches: "
                f"{skipped_train_batches}"
            )

        should_validate = (
            epoch % VALIDATE_EVERY == 0
            or epoch == 1
            or epoch == args.max_epochs
        )

        val_loss = float("nan")
        ious = torch.full((NUM_CLASSES,), float("nan"))
        miou_all = float("nan")
        miou_vegetation = float("nan")
        binary_metrics = {
            "iou": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
        cm = None
        coverage_mae_pp = float("nan")

        if should_validate:
            model.eval()
            val_loss = 0.0
            cm = torch.zeros(
                (NUM_CLASSES, NUM_CLASSES),
                dtype=torch.int64,
                device=DEVICE,
            )

            valid_val_batches = 0
            coverage_error_sum = 0.0
            coverage_valid_images = 0

            with torch.no_grad():
                for images, masks in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch:02d}/{EPOCHS} Val",
                    dynamic_ncols=True,
                    leave=True,
                    mininterval=0.5,
                ):
                    images = images.to(DEVICE, non_blocking=True)
                    masks = masks.to(DEVICE, non_blocking=True)

                    with torch.amp.autocast(
                        "cuda",
                        enabled=amp_enabled,
                    ):
                        outputs = model(images)
                        loss = safe_segmentation_loss(
                            outputs,
                            masks,
                            cross_entropy,
                            lovasz_loss,
                        )

                    if loss is None:
                        continue

                    val_loss += loss.item()
                    valid_val_batches += 1

                    predictions = torch.argmax(outputs, dim=1)
                    valid = (
                        (masks >= 0)
                        & (masks < NUM_CLASSES)
                        & (masks != IGNORE_INDEX)
                    )

                    indexes = (
                        NUM_CLASSES * masks[valid]
                        + predictions[valid]
                    )
                    cm += torch.bincount(
                        indexes,
                        minlength=NUM_CLASSES * NUM_CLASSES,
                    ).reshape(NUM_CLASSES, NUM_CLASSES)

                    # A real per-image, GT-valid-pixel area error (percentage
                    # points); do not confuse with signed dataset-wide bias.
                    valid_per_image = valid.flatten(1).sum(1)
                    meaningful = valid_per_image > 0
                    gt_veg = ((masks >= 2) & (masks <= 5) & valid)
                    pred_veg = ((predictions >= 2) & (predictions <= 5) & valid)
                    gt_counts = gt_veg.flatten(1).sum(1).float()
                    pred_counts = pred_veg.flatten(1).sum(1).float()
                    diff_pp = 100.0 * (pred_counts - gt_counts).abs() / valid_per_image.clamp_min(1)
                    coverage_error_sum += diff_pp[meaningful].sum().item()
                    coverage_valid_images += int(meaningful.sum().item())

            coverage_mae_pp = coverage_error_sum / coverage_valid_images if coverage_valid_images else float("nan")
            val_loss /= max(valid_val_batches, 1)
            scheduler.step(val_loss)

            ious = calculate_iou(cm)
            miou_all = safe_mean(ious)
            miou_vegetation = safe_mean(
                ious[VEGETATION_CLASS_IDS]
            )
            binary_metrics = calculate_binary_vegetation_metrics(cm)
            gt_cov, pred_cov, cov_bias = aggregate_vegetation_coverage_error(cm)
            gt_support = cm.sum(dim=1).cpu().tolist()
            if any(count == 0 for count in gt_support):
                print("[WARNING] Missing GT after validation crop: "
                      + ", ".join(CLASS_NAMES[i] for i, count in enumerate(gt_support)
                                  if count == 0))


        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")

        if should_validate:
            print(f"Val Loss: {val_loss:.4f}")

            for i, name in enumerate(CLASS_NAMES):
                print(
                    f"{name:<16}: "
                    f"{ious[i].item() * 100:.2f}%"
                )

            print(
                f"mIoU (all 6 classes): "
                f"{miou_all * 100:.2f}%"
            )
            print(
                f"mIoU (vegetation): "
                f"{miou_vegetation * 100:.2f}%"
            )
            print(
                f"GT vegetation coverage: {gt_cov:.2f}% | "
                f"Predicted: {pred_cov:.2f}% | "
                f"Dataset-wide coverage bias: {cov_bias:+.2f} percentage points"
            )
            print(
                "Validation GT pixel support: "
                + ", ".join(f"{name}={count:,}" for name, count
                            in zip(CLASS_NAMES, gt_support))
            )
            print(
                "Binary vegetation IoU: "
                f"{binary_metrics['iou'] * 100:.2f}%"
            )
            print(
                "Binary vegetation Precision: "
                f"{binary_metrics['precision'] * 100:.2f}%"
            )
            print(
                "Binary vegetation Recall: "
                f"{binary_metrics['recall'] * 100:.2f}%"
            )
            print(
                "Binary vegetation F1: "
                f"{binary_metrics['f1'] * 100:.2f}%"
            )
            print(f"Per-image vegetation coverage MAE: {coverage_mae_pp:.3f} percentage points")
        else:
            print("Validation: skipped this epoch")
        current_lr = optimizer.param_groups[0]["lr"]
        current_weight_norm = calculate_weight_norm(model)
        average_grad_norm = (
            grad_norm_sum / max(grad_norm_steps, 1)
        )

        print("Learning rate:", current_lr)
        print(f"Weight norm: {current_weight_norm:.4f}")
        print(f"Gradient norm: {average_grad_norm:.4f}")

        if should_validate:
            save_confusion_matrix(cm, epoch)

        plot_history["epoch"].append(epoch)
        plot_history["train_loss"].append(train_loss)
        plot_history["val_loss"].append(val_loss)
        plot_history["ious"].append(
            [float(x.item()) for x in ious]
        )
        plot_history["miou_all"].append(miou_all)
        plot_history["miou_vegetation"].append(miou_vegetation)
        plot_history["binary_vegetation_iou"].append(binary_metrics["iou"])
        plot_history["binary_vegetation_precision"].append(binary_metrics["precision"])
        plot_history["binary_vegetation_recall"].append(binary_metrics["recall"])
        plot_history["binary_vegetation_f1"].append(binary_metrics["f1"])
        plot_history["learning_rate"].append(current_lr)
        plot_history["weight_norm"].append(current_weight_norm)
        plot_history["grad_norm"].append(average_grad_norm)

        save_training_plots(plot_history)

        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerow([
                epoch,
                train_loss,
                val_loss,
                *[x.item() for x in ious],
                miou_all,
                miou_vegetation,
                binary_metrics["iou"],
                binary_metrics["precision"],
                binary_metrics["recall"],
                binary_metrics["f1"],
                *([coverage_mae_pp] if experiment else []),
                current_lr,
                current_weight_norm,
                average_grad_norm,
            ])

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        stop_after_epoch = False
        if should_validate:
            if miou_vegetation > best_vegetation_miou + 1e-4:
                best_vegetation_miou = miou_vegetation
                epochs_without_improvement = 0

                save_best_protected(
                    OUTPUT_DIR / "best_model.pth",
                    {
                        "model_state_dict": model.state_dict(),
                        "encoder": "resnet50",
                        "num_classes": NUM_CLASSES,
                        "class_names": CLASS_NAMES,
                        "vegetation_class_ids": VEGETATION_CLASS_IDS,
                        "image_size": IMAGE_SIZE,
                        "crop_mode": "native_resolution_512",
                        "batch_size": BATCH_SIZE,
                        "num_workers": NUM_WORKERS,
                        "gpu_temp_limit_c": GPU_TEMP_LIMIT_C,
                        "amp": amp_enabled,
                        "loss": "0.30 CE + 0.30 Lovasz + 0.20 Dice + 0.20 Boundary",
                        "class_weights": experiment["class_weights"] if experiment else BASE_CLASS_WEIGHTS,
                        "label_studio_train_share": LABEL_STUDIO_TRAIN_SHARE,
                        "flair_mapping_revision": "v5_unet_stage3_flair_lut_aligned",
                        "best_miou": best_vegetation_miou,
                        "binary_vegetation_iou": binary_metrics["iou"],
                        "binary_vegetation_f1": binary_metrics["f1"],
                        "coverage_mae_pp": coverage_mae_pp,
                    },
                )

                print("Saved best model.")

                if best_vegetation_miou >= TARGET_VEGETATION_MIOU:
                    print(
                        "\nTarget reached: vegetation mIoU "
                        f">= {TARGET_VEGETATION_MIOU * 100:.0f}%"
                    )
                    stop_after_epoch = True
            else:
                epochs_without_improvement += 1
                print(
                    "No vegetation mIoU improvement: "
                    f"{epochs_without_improvement}/"
                    f"{EARLY_STOPPING_PATIENCE}"
                )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    "\nEarly stopping: vegetation mIoU "
                    "has not improved."
                )
                stop_after_epoch = True

        save_full(
            OUTPUT_DIR / "latest_state.pth",
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch,
            best_miou=best_vegetation_miou,
            no_improve=epochs_without_improvement,
            metadata=data_fingerprint,
        )
        print(f"Full resumable checkpoint saved: Epoch {epoch}")
        completed_epoch = epoch
        if stop_after_epoch:
            break

    if pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    # Written only on a normal completed stop, never on an interrupted epoch.
    # Atomic marker lets the autonomous watcher distinguish a finished run
    # from a process still training, crashing, or being killed.
    marker_data = {
        "last_epoch": completed_epoch,
        "max_epochs": args.max_epochs,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "gpu_temperature_limit" if stop_for_temperature else "normal_training_loop_exit",
    }
    marker_temp = finished_marker.with_suffix(".tmp")
    marker_temp.write_text(json.dumps(marker_data, ensure_ascii=False), encoding="utf-8")
    marker_temp.replace(finished_marker)
    print("\nTraining finished.")
    print(
        "Best vegetation mIoU:",
        f"{best_vegetation_miou * 100:.2f}%",
    )


if __name__ == "__main__":
    main()
