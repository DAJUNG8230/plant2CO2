"""DeepLab V5 Level 2: atomic full-state checkpoint and human approval contract."""
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

ALLOWED_KEYS = {"learning_rate", "weight_decay", "scheduler_patience"}
BOUNDS = {
    "learning_rate": (1e-7, 1e-3),
    "weight_decay": (0.0, 0.02),
    "scheduler_patience": (1, 12),
}


def validate_changes(changes):
    if not isinstance(changes, dict) or not changes:
        raise ValueError("需要非空參數修改字典")
    if set(changes) - ALLOWED_KEYS:
        raise ValueError("不允許的參數：" + ", ".join(sorted(set(changes) - ALLOWED_KEYS)))
    accepted = {}
    for key, value in changes.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} 必須是數字")
        low, high = BOUNDS[key]
        if not low <= value <= high:
            raise ValueError(f"{key} 必須在 [{low}, {high}]")
        if key == "scheduler_patience" and int(value) != value:
            raise ValueError("scheduler_patience 必須是整數")
        accepted[key] = int(value) if key == "scheduler_patience" else float(value)
    return accepted


def read_approved(path):
    """Only explicitly approved local JSON is eligible for applying at launch."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "approved" or payload.get("approval") != "human-confirmed":
        raise ValueError("必須先透過人工確認流程核准參數修改")
    return validate_changes(payload.get("changes"))


def apply_approved(changes, optimizer, scheduler):
    """Apply only when initializing or resuming, never to a running loop."""
    if "learning_rate" in changes:
        for group in optimizer.param_groups:
            group["lr"] = changes["learning_rate"]
    if "weight_decay" in changes:
        for group in optimizer.param_groups:
            group["weight_decay"] = changes["weight_decay"]
    if "scheduler_patience" in changes:
        scheduler.patience = changes["scheduler_patience"]


def save_full(path, *, model, optimizer, scheduler, scaler, epoch,
              best_miou, no_improve, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "plant2co2-v5-full-state-v1",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": float(best_miou),
        "epochs_without_improvement": int(no_improve),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "metadata": metadata,
    }
    # Atomic replace; old complete checkpoint stays intact if writing fails.
    fd, temp_name = tempfile.mkstemp(prefix="checkpoint_", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def restore_full(path, *, model, optimizer, scheduler, scaler, expected_metadata):
    # torch.load is a pickle boundary for trusted local checkpoint only.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "scaler_state_dict", "rng_python", "rng_numpy", "rng_torch",
                "epoch", "best_miou", "epochs_without_improvement", "metadata"}
    if payload.get("format") != "plant2co2-v5-full-state-v1" or not required.issubset(payload):
        raise ValueError("不是完整的 V5 checkpoint；舊 best_model.pth 不能無縫續訓")
    for key, value in expected_metadata.items():
        if payload["metadata"].get(key) != value:
            raise ValueError(f"續訓設定不一致：{key}，避免混用不同資料或模型")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    # Move optimizer state tensors to the current model device.
    device = next(model.parameters()).device
    for values in optimizer.state.values():
        for key, item in values.items():
            if torch.is_tensor(item):
                values[key] = item.to(device)
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch"].cpu())
    if torch.cuda.is_available() and payload.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return (int(payload["epoch"]), float(payload["best_miou"]),
            int(payload["epochs_without_improvement"]))


def save_best_protected(path, payload):
    """Atomically replace the best weights while retaining one prior best backup."""
    import shutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="best_", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        if path.exists():
            old = path.with_name("best_model_previous.pth")
            backup_temp = old.with_suffix(".tmp")
            shutil.copy2(path, backup_temp)
            os.replace(backup_temp, old)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
