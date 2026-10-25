from pathlib import Path
import csv
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import segmentation_models_pytorch as smp
from tqdm import tqdm

NUM_CLASSES = 6
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.images = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

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
        if torch.any((unique < 0) | (unique >= NUM_CLASSES)):
            raise ValueError(
                f"{mask_path} 出現非法 class id: {unique.tolist()}，"
                "六類模型只允許 0~5。"
            )

        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[1])
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[0])
            k = random.randint(0, 3)
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return image, mask


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


def main():
    print("Device:", DEVICE)
    print("Classes:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")

    train_dataset = SegmentationDataset(
        "dataset/train/images",
        "dataset/train/masks",
        augment=True,
    )
    val_dataset = SegmentationDataset(
        "dataset/val/images",
        "dataset/val/masks",
        augment=False,
    )

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

    cross_entropy = torch.nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(
        mode="multiclass",
        from_logits=True,
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
        patience=5,
    )

    best_vegetation_miou = -1.0
    history_path = OUTPUT_DIR / "history.csv"

    history_header = [
        "epoch",
        "train_loss",
        "val_loss",
        *[name.lower().replace(" ", "_") + "_iou" for name in CLASS_NAMES],
        "miou_all",
        "miou_vegetation",
    ]

    with open(history_path, "w", newline="", encoding="utf-8") as f:
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
                + 0.5 * dice_loss(outputs, masks)
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
                    + 0.5 * dice_loss(outputs, masks)
                )
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                valid = (masks >= 0) & (masks < NUM_CLASSES)
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
        vegetation_ious = ious[VEGETATION_CLASS_IDS]
        miou_vegetation = safe_mean(vegetation_ious)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name} IoU: {ious[i].item():.4f}")
        print(f"mIoU (all 6 classes): {miou_all:.4f}")
        print(
            "mIoU (vegetation 2/3/4/5): "
            f"{miou_vegetation:.4f}"
        )

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

        if miou_vegetation > best_vegetation_miou:
            best_vegetation_miou = miou_vegetation
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

    print("Training finished.")
    print(
        "Best vegetation mIoU:",
        best_vegetation_miou,
    )


if __name__ == "__main__":
    main()
