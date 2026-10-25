from pathlib import Path
import csv
import random

import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 8

USE_FLAIR = True
USE_LABEL_STUDIO = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

        if random.random() < 0.8:
            image = TF.adjust_brightness(
                image,
                random.uniform(0.85, 1.15),
            )
        if random.random() < 0.8:
            image = TF.adjust_contrast(
                image,
                random.uniform(0.85, 1.15),
            )
        if random.random() < 0.6:
            image = TF.adjust_saturation(
                image,
                random.uniform(0.85, 1.15),
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
    return 1.0 - dice.mean()


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


def compute_class_weights(dataset):
    print("\nComputing automatic class weights...")
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    for index in tqdm(
        range(len(dataset)),
        desc="Class statistics",
    ):
        _, mask = dataset[index]
        valid = (
            (mask >= 0)
            & (mask < NUM_CLASSES)
            & (mask != IGNORE_INDEX)
        )
        if valid.any():
            values = torch.bincount(
                mask[valid].reshape(-1),
                minlength=NUM_CLASSES,
            ).double()
            counts += values

    counts = torch.clamp(counts, min=1.0)
    frequencies = counts / counts.sum()

    weights = torch.sqrt(
        frequencies.mean() / frequencies
    )
    weights = torch.clamp(weights, min=0.5, max=3.0)
    weights = weights / weights.mean()

    print("Class pixel distribution / weights:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {i} {name:<16} "
            f"{frequencies[i].item() * 100:6.2f}%  "
            f"weight={weights[i].item():.3f}"
        )

    return weights.float()


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


def build_datasets():
    train_sets = []
    val_sets = []

    if USE_FLAIR:
        flair_train = Path("flair_manifests/train.csv")
        flair_val = Path("flair_manifests/val.csv")

        if not flair_train.exists() or not flair_val.exists():
            raise FileNotFoundError(
                "找不到 flair_manifests。請先執行 "
                "python training\\prepare_flair_6class.py"
            )

        train_sets.append(
            ManifestDataset(flair_train, augment=True)
        )
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
            train_sets.append(
                SegmentationDataset(
                    custom_train_images,
                    custom_train_masks,
                    augment=True,
                )
            )

        if (
            custom_val_images.exists()
            and custom_val_masks.exists()
        ):
            val_sets.append(
                SegmentationDataset(
                    custom_val_images,
                    custom_val_masks,
                    augment=False,
                )
            )

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

    return train_dataset, val_dataset


def main():
    print("Device:", DEVICE)
    print("Classes:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")

    train_dataset, val_dataset = build_datasets()

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    class_weights = compute_class_weights(train_dataset).to(DEVICE)

    cross_entropy = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
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
    ]

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerow(history_header)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS} Train",
        ):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)

            loss = (
                0.5 * cross_entropy(outputs, masks)
                + 0.5 * masked_dice_loss(outputs, masks)
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        cm = torch.zeros(
            (NUM_CLASSES, NUM_CLASSES),
            dtype=torch.int64,
            device=DEVICE,
        )

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
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
            "Learning rate:",
            optimizer.param_groups[0]["lr"],
        )

        save_confusion_matrix(cm, epoch)

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
            ])

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
