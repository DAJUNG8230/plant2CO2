"""Read-only evidence collection for DeepLab V5 Level 2 experiments.

Uses existing validation CSVs, local checkpoint metadata, and optional fixed-image
prediction masks. Never runs model inference or changes datasets/checkpoints.
"""
import csv
import json
import math
from pathlib import Path

CLASSES = ("Bare Soil", "Other Non-Veg", "Herbaceous", "Shrub", "Trees", "Cultivated")
VEGETATION = (2, 3, 4, 5)


def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def checkpoint_info(path, inspect=False):
    if not path.is_file():
        return {"available": False, "reason": "missing"}
    result = {"available": True, "path": str(path), "bytes": path.stat().st_size,
              "modified_unix": path.stat().st_mtime, "metadata_read": False}
    if not inspect:
        result["reason"] = "未載入權重；可使用 --inspect-checkpoint 讀取可信本地 checkpoint metadata"
        return result
    try:
        import torch
        # Load only trusted local checkpoint files. mmap avoids eager tensor copy.
        # weights_only disallows arbitrary user-supplied pickle classes.
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(ckpt, dict):
            result["reason"] = "checkpoint 格式非字典"
            return result
        allow = ("encoder", "num_classes", "class_names",
                 "flair_mapping_revision", "best_miou",
                 "binary_vegetation_iou", "binary_vegetation_f1",
                 "image_size", "loss", "epoch")
        result["metadata"] = {k: ckpt.get(k) for k in allow if k in ckpt}
        result["metadata_read"] = True
    except Exception as exc:
        result["reason"] = "Metadata unavailable: " + type(exc).__name__ + ": " + str(exc)[:180]
    return result


def read_confusions(folder, limit=70):
    matrices = []
    errors = []
    for path in sorted(folder.glob("confusion_matrix_epoch_*.csv"))[-limit:]:
        try:
            epoch = int(path.stem.split("_")[-1])
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                data = list(csv.reader(f))
            if len(data) != 7 or len(data[0]) != 7:
                raise ValueError("expected 6x6 confusion matrix")
            if data[0][1:] != list(CLASSES):
                raise ValueError("class header mismatch")
            if any(len(row) != 7 or row[0] != CLASSES[i]
                   for i, row in enumerate(data[1:])):
                raise ValueError("row labels mismatch")
            matrix = []
            for row in data[1:]:
                entries = []
                for cell in row[1:]:
                    val = int(cell)
                    if val < 0:
                        raise ValueError("negative pixels")
                    entries.append(val)
                matrix.append(entries)
            per_class = []
            for i in range(6):
                gt = sum(matrix[i])
                predicted = sum(matrix[r][i] for r in range(6))
                tp = matrix[i][i]
                union = gt + predicted - tp
                confusion = sorted(
                    [{"predicted_as": CLASSES[j], "pixels": count,
                      "fraction_of_gt": round(count / gt, 4) if gt else None}
                     for j, count in enumerate(matrix[i]) if j != i and count > 0],
                    key=lambda x: x["pixels"], reverse=True,
                )[:2]
                per_class.append({
                    "class": CLASSES[i], "gt_pixels": gt, "predicted_pixels": predicted,
                    "iou": round(tp / union, 6) if union else None,
                    "top_misclassifications": confusion,
                })
            matrices.append({"epoch": epoch, "total_pixels": sum(map(sum, matrix)),
                             "classes": per_class})
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    return {"available": bool(matrices), "epochs": matrices, "errors": errors}


def evaluate_fixed_pairs(manifest):
    """Manifest: image_id,gt_mask,pred_best_mask,pred_latest_mask.

    A pixel mask uses class IDs 0..5, ignore_index=255. Predictions must be
    precomputed with identical spatial coordinates; NO geometric registration.
    """
    if not manifest.is_file():
        return {"available": False, "reason": "固定影像預測清單不存在", "manifest": str(manifest)}
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {"available": False, "reason": "缺少 Pillow 或 NumPy"}
    records = []
    errors = []
    with manifest.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "gt_mask", "pred_best_mask", "pred_latest_mask"}
        if not required.issubset(reader.fieldnames or []):
            return {"available": False, "reason": "CSV 欄位缺少：" + ", ".join(sorted(required))}
        for item in reader:
            image_id = item["image_id"]
            def load(col):
                name = item.get(col, "").strip()
                if not name:
                    return None
                p = Path(name)
                if not p.is_absolute():
                    p = (manifest.parent / p).resolve()
                with Image.open(p) as im:
                    if im.mode not in ("L", "P", "I", "I;16"):
                        raise ValueError("mask must be single-channel class indices, not RGB")
                    return np.asarray(im).astype("int32")
            try:
                gt = load("gt_mask")
                best = load("pred_best_mask")
                latest = load("pred_latest_mask")
                if gt is None or best is None or latest is None:
                    raise ValueError("missing GT/best/latest prediction")
                if gt.shape != best.shape or gt.shape != latest.shape:
                    raise ValueError("mask shape mismatch; no implicit resizing allowed")
                if not np.all((gt == 255) | ((gt >= 0) & (gt < 6))):
                    raise ValueError("GT contains unknown class IDs")
                valid = gt != 255
                if not valid.any():
                    raise ValueError("GT contains no valid pixels")
                if not all(np.all((p >= 0) & (p < 6)) for p in (best[valid], latest[valid])):
                    raise ValueError("prediction contains invalid class IDs")
                gt_fraction = float(np.isin(gt[valid], VEGETATION).mean())
                values = {}
                for tag, pred in (("best", best), ("latest", latest)):
                    fraction = float(np.isin(pred[valid], VEGETATION).mean())
                    values[tag] = {
                        "predicted_coverage_pct": round(100 * fraction, 4),
                        "signed_error_pp": round(100 * (fraction - gt_fraction), 4),
                        "absolute_error_pp": round(100 * abs(fraction - gt_fraction), 4),
                    }
                records.append({"image_id": image_id,
                                "gt_coverage_pct": round(100 * gt_fraction, 4),
                                "best": values["best"], "latest": values["latest"]})
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"{image_id}: {exc}")
    aggregates = {}
    for tag in ("best", "latest"):
        absolute = sorted(x[tag]["absolute_error_pp"] for x in records)
        signed = [x[tag]["signed_error_pp"] for x in records]
        if absolute:
            p90_index = max(0, math.ceil(len(absolute) * .9) - 1)
            aggregates[tag] = {
                "images": len(absolute),
                "coverage_mae_pp": round(sum(absolute) / len(absolute), 4),
                "coverage_signed_bias_pp": round(sum(signed) / len(signed), 4),
                "coverage_p90_absolute_pp": round(absolute[p90_index], 4),
            }
    return {"available": bool(records), "manifest": str(manifest),
            "per_image": records[:200], "aggregates": aggregates, "errors": errors,
            "definition": "per-image valid-GT-pixel vegetation coverage error, percentage points"}


def collect_evidence(rows, output_dir, inspect_checkpoint=False, fixed_manifest=None):
    valid = [r for r in rows if r.get("miou_vegetation") is not None and r.get("val_loss") is not None]
    best = max(valid, key=lambda x: x["miou_vegetation"]) if valid else None
    last = valid[-1] if valid else None
    cm = read_confusions(output_dir)
    best_epoch = best["epoch"] if best else None
    checkpoint = checkpoint_info(output_dir / "best_model.pth", inspect=inspect_checkpoint)
    matches = [x for x in cm["epochs"] if x["epoch"] == best_epoch]
    latest_cm = [x for x in cm["epochs"] if x["epoch"] == last["epoch"]] if last else []
    fixed = evaluate_fixed_pairs(fixed_manifest or output_dir / "evaluation" / "fixed_predictions.csv")
    return {
        "source": "local read-only inspection",
        "history": {"rows": len(rows), "validated_epochs": len(valid),
                    "full": rows, "best_epoch": best_epoch,
                    "best_vegetation_miou": best["miou_vegetation"] if best else None,
                    "latest_valid_epoch": last["epoch"] if last else None,
                    "latest_vegetation_miou": last["miou_vegetation"] if last else None},
        "best_checkpoint": checkpoint,
        "confusion": cm,
        "best_confusion": matches[-1] if matches else None,
        "latest_confusion": latest_cm[-1] if latest_cm else None,
        "fixed_images": fixed,
        "limitations": [
            "A confusion matrix is validation performance only, not independent test data.",
            "No prediction images or labels are invented. Fixed mask pairs are optional.",
            "Class IDs must follow V5 six-class mapping and masks must be spatially aligned.",
            "best_model.pth records best-mIoU weights; latest_state.pth is not best weights.",
        ],
    }


def review_tasks(evidence):
    """Structured audit tasks, not executable training hyperparameter changes."""
    history = evidence["history"]
    if not history["validated_epochs"]:
        return []
    tasks = []
    if not evidence["best_confusion"] or not evidence["latest_confusion"]:
        tasks.append({
            "task_code": "compare_best_last_confusion",
            "reason": f"最佳驗證 Epoch {history['best_epoch']} 與最近驗證 Epoch {history['latest_valid_epoch']} 必須以相同六類混淆矩陣比較；目前缺少至少一個 CSV。",
            "evidence": ["history.best_epoch", "history.latest_valid_epoch", "confusion.epochs"],
        })
    else:
        last = evidence["latest_confusion"]
        weak = [x for x in last["classes"] if x["class"] in ("Shrub", "Cultivated") and
                x["iou"] is not None and x["iou"] < .5]
        for item in weak:
            confused = item["top_misclassifications"]
            summary = ", ".join(f"{x['predicted_as']} {x['fraction_of_gt']:.1%}" for x in confused
                                if x.get("fraction_of_gt") is not None)
            tasks.append({
                "task_code": "audit_" + item["class"].lower(),
                "reason": f"{item['class']} IoU {item['iou']:.2%}；GT 被混淆為 {summary or '無法判定'}。先檢查對應原圖／GT／預測，再討論類別權重。",
                "evidence": ["latest_confusion.classes", "history.full"],
            })
    if not evidence["fixed_images"]["available"]:
        tasks.append({
            "task_code": "collect_fixed_prediction_pairs",
            "reason": "缺少同一批固定圖片的 GT、最佳權重預測及最近輪預測 mask；不能比較逐張覆蓋率 MAE。先準備評估資料，禁止直接以最新 epoch 替換最佳模型。",
            "evidence": ["fixed_images.available", "best_checkpoint.available"],
        })
    else:
        aggregates = evidence["fixed_images"]["aggregates"]
        tasks.append({
            "task_code": "review_coverage_error",
            "reason": "固定測試圖片逐張覆蓋率 MAE：最佳權重 {:.2f} pp；最近輪 {:.2f} pp，需人工確認可比性後決定下一輪單變因實驗。".format(
                aggregates["best"]["coverage_mae_pp"], aggregates["latest"]["coverage_mae_pp"]),
            "evidence": ["fixed_images.aggregates", "fixed_images.per_image"],
        })
    return tasks
