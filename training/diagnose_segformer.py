"""Check SegFormer config/weight shapes without loading incompatible weights.

PowerShell: .\.venv\Scripts\python.exe training\diagnose_segformer.py
Optional:   .\.venv\Scripts\python.exe training\diagnose_segformer.py --model models\best_model_segformer
Never rewrite the user's checkpoint or modify config.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Bare Soil", "Other Non-Veg", "Herbaceous",
               "Shrub", "Trees", "Cultivated"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models" / "best_model_segformer")
    args = p.parse_args()
    folder = args.model.resolve()
    config_path = folder / "config.json"
    weights_path = folder / "model.safetensors"
    bin_path = folder / "pytorch_model.bin"
    if not config_path.is_file():
        p.error("找不到 config.json: " + str(config_path))
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    print("模型資料夾：", folder)
    print("config model_type：", raw.get("model_type"))
    print("config num_labels：", raw.get("num_labels"))
    print("config id2label：", raw.get("id2label"))
    print("config hidden_sizes：", raw.get("hidden_sizes"))
    print("config decoder_hidden_size：", raw.get("decoder_hidden_size"))

    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    config = SegformerConfig.from_pretrained(str(folder), local_files_only=True)
    model = SegformerForSemanticSegmentation(config)
    expected_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    del model
    if weights_path.is_file():
        from safetensors import safe_open
        with safe_open(str(weights_path), framework="pt", device="cpu") as sf:
            actual_shapes = {key: tuple(sf.get_slice(key).get_shape()) for key in sf.keys()}
        selected = weights_path
    elif bin_path.is_file():
        import torch
        try:
            state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(bin_path), map_location="cpu")
        actual_shapes = {k: tuple(v.shape) for k, v in state.items()
                         if hasattr(v, "shape")}
        selected = bin_path
    else:
        p.error("找不到 model.safetensors 或 pytorch_model.bin")
    print("使用權重：", selected)
    mismatches = [(k, expected_shapes[k], shape)
                  for k, shape in actual_shapes.items()
                  if k in expected_shapes and expected_shapes[k] != shape]
    missing = sorted(set(expected_shapes) - set(actual_shapes))
    unexpected = sorted(set(actual_shapes) - set(expected_shapes))
    print("不相符的張量數：", len(mismatches))
    for key, expected, actual in mismatches[:30]:
        print(f"  {key}: config 預期 {expected}；實際權重 {actual}")
    print("缺少張量：", len(missing), "；額外張量：", len(unexpected))
    for key in missing[:8]:
        print("  缺少：", key)
    for key in unexpected[:8]:
        print("  額外：", key)
    head = [(k, sh) for k, sh in actual_shapes.items()
            if k.endswith("decode_head.classifier.weight")]
    if head:
        print("實際分類頭輸出通道數：", head[0][1][0], "；所需六類：6")
    labels = {int(k): str(v) for k, v in (config.id2label or {}).items()}
    expected_labels = {i: name for i, name in enumerate(CLASS_NAMES)}
    if labels != expected_labels:
        print("警告：id2label 與網站六類順序不同或未明確指定；需要核對訓練標籤順序。")
    if not mismatches and not missing and not unexpected and config.num_labels == 6:
        print("結構檢查：通過。仍需確認模型訓練時的六類順序。")
    else:
        print("結構檢查：未通過。請保留此輸出供排查，不要改成 ignore_mismatched_sizes=True。")
        sys.exit(1)


if __name__ == "__main__":
    main()
