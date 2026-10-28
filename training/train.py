from pathlib import Path
import csv
import random

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
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
VALIDATE_EVERY = 2
GRAD_NORM_EVERY = 100
USE_AMP = True

USE_FLAIR = True
USE_LABEL_STUDIO = True

# 每個 training epoch 預期抽樣比例。
# 即使 Label Studio 圖片數遠少於 FLAIR，也讓自有場景有足夠影響力。
LABEL_STUDIO_TRAIN_SHARE = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs_training")
PLOT_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def choose_batch_size():
    if DEVICE.type != "cuda":
        return 2

    total_gb = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    if total_gb >= 5.5:
        return 4
    return 2


BATCH_SIZE = choose_batch_size()


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


def load_pair(image_path, mask_path, augment):
    try:
        image = load_image(image_path)
        mask = Image.open(mask_path)
    except Exception as exc:
        raise RuntimeError(
            f"讀取資料失敗\nImage: {image_path}\nMask: {mask_path}\n"
            f"原因: {type(exc).__name__}: {exc}"
        ) from exc

    image = TF.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=InterpolationMode.BILINEAR,
    )
    mask = TF.resize(
        mask,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=InterpolationMode.NEAREST,
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


def masked_tversky_loss(
    logits,
    targets,
    alpha=0.7,
    beta=0.3,
):
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
    tp = (probs * one_hot).sum(dims)
    fp = (probs * (1.0 - one_hot)).sum(dims)
    fn = ((1.0 - probs) * one_hot).sum(dims)

    score = (tp + 1e-6) / (
        tp + alpha * fp + beta * fn + 1e-6
    )

    class_weights = torch.tensor(
        [1.4, 0.7, 1.4, 1.7, 1.0, 0.8],
        device=logits.device,
        dtype=score.dtype,
    )
    class_weights = class_weights / class_weights.sum()

    return 1.0 - (score * class_weights).sum()


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

    pred_dx = torch.abs(
        probs[:, :, :, 1:] - probs[:, :, :, :-1]
    )
    pred_dy = torch.abs(
        probs[:, :, 1:, :] - probs[:, :, :-1, :]
    )
    true_dx = torch.abs(
        one_hot[:, :, :, 1:] - one_hot[:, :, :, :-1]
    )
    true_dy = torch.abs(
        one_hot[:, :, 1:, :] - one_hot[:, :, :-1, :]
    )

    valid_dx = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
    valid_dy = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]

    loss_x = (
        torch.abs(pred_dx - true_dx) * valid_dx
    ).sum() / valid_dx.sum().clamp_min(1.0)

    loss_y = (
        torch.abs(pred_dy - true_dy) * valid_dy
    ).sum() / valid_dy.sum().clamp_min(1.0)

    return 0.5 * (loss_x + loss_y)


def calculate_iou(confusion_matrix):
    cm = confusion_matrix.float().cpu()
    intersection = torch.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    iou = torch.full((NUM_CLASSES,), float("nan"))
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou


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
    plt.title("Automatic Class Weights")
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
    print(f"AMP: {USE_AMP and DEVICE.type == 'cuda'}")
    print(f"Validate every: {VALIDATE_EVERY} epoch(s)")
    print(f"Gradient norm every: {GRAD_NORM_EVERY} batch(es)")

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

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

    class_weights = torch.tensor(
        [1.4, 0.7, 1.4, 1.7, 1.0, 0.8],
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
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
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
        patience=3,
    )

    best_vegetation_miou = -1.0
    epochs_without_improvement = 0
    history_path = OUTPUT_DIR / "history.csv"

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
        "learning_rate",
        "weight_norm",
        "grad_norm",
    ]

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
        "learning_rate": [],
        "weight_norm": [],
        "grad_norm": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        grad_norm_sum = 0.0
        grad_norm_steps = 0

        for batch_index, (images, masks) in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{EPOCHS} Train",
            ),
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
                loss = (
                    0.4 * cross_entropy(outputs, masks)
                    + 0.4 * masked_tversky_loss(
                        outputs,
                        masks,
                    )
                    + 0.2 * boundary_loss(
                        outputs,
                        masks,
                    )
                )

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

        train_loss /= max(len(train_loader), 1)

        should_validate = (
            epoch % VALIDATE_EVERY == 0
            or epoch == 1
            or epoch == EPOCHS
        )

        val_loss = float("nan")
        ious = torch.full((NUM_CLASSES,), float("nan"))
        miou_all = float("nan")
        miou_vegetation = float("nan")
        cm = None

        if should_validate:
            model.eval()
            val_loss = 0.0
            cm = torch.zeros(
                (NUM_CLASSES, NUM_CLASSES),
                dtype=torch.int64,
                device=DEVICE,
            )

            with torch.no_grad():
                for images, masks in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch}/{EPOCHS} Val",
                ):
                    images = images.to(DEVICE, non_blocking=True)
                    masks = masks.to(DEVICE, non_blocking=True)

                    with torch.amp.autocast(
                        "cuda",
                        enabled=amp_enabled,
                    ):
                        outputs = model(images)
                        loss = (
                            0.5 * cross_entropy(outputs, masks)
                            + 0.5 * masked_dice_loss(outputs, masks)
                        )

                    val_loss += loss.item()

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

            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)

            ious = calculate_iou(cm)
            miou_all = safe_mean(ious)
            miou_vegetation = safe_mean(
                ious[VEGETATION_CLASS_IDS]
            )

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
                current_lr,
                current_weight_norm,
                average_grad_norm,
            ])

        if should_validate:
            if miou_vegetation > best_vegetation_miou + 1e-4:
                best_vegetation_miou = miou_vegetation
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "encoder": "resnet50",
                        "num_classes": NUM_CLASSES,
                        "class_names": CLASS_NAMES,
                        "vegetation_class_ids": VEGETATION_CLASS_IDS,
                        "image_size": IMAGE_SIZE,
                        "batch_size": BATCH_SIZE,
                        "amp": amp_enabled,
                        "loss": "0.4 CE + 0.4 Tversky + 0.2 Boundary",
                        "class_weights": [1.4, 0.7, 1.4, 1.7, 1.0, 0.8],
                        "label_studio_train_share": LABEL_STUDIO_TRAIN_SHARE,
                        "best_miou": best_vegetation_miou,
                    },
                    OUTPUT_DIR / "best_model.pth",
                )

                print("Saved best model.")
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
                break

    print("\nTraining finished.")
    print(
        "Best vegetation mIoU:",
        f"{best_vegetation_miou * 100:.2f}%",
    )


if __name__ == "__main__":
    main()
