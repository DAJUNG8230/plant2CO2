import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from label_studio_converter.brush import decode_rle

JSON_PATH = Path("export.json")
IMAGE_DIR = Path("original_images")
OUT_IMAGE_DIR = Path("dataset/all/images")
OUT_MASK_DIR = Path("dataset/all/masks")

CLASS_MAP = {
    "草地 (Grassland)": 1,
    "裸地 (Barren)": 2,
}

OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


def get_filename(image_ref: str) -> str:
    return image_ref.replace("\\", "/").split("/")[-1]


with open(JSON_PATH, "r", encoding="utf-8") as f:
    tasks = json.load(f)

print("Total tasks:", len(tasks))
success = 0
failed = 0

for task in tasks:
    image_ref = task.get("data", {}).get("image")
    if not image_ref:
        print("[ERROR] No image path")
        failed += 1
        continue

    filename = get_filename(image_ref)
    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        print("[ERROR] Image not found:", filename)
        failed += 1
        continue

    annotations = [
        a for a in task.get("annotations", [])
        if not a.get("was_cancelled", False)
    ]
    if not annotations:
        print("[SKIP] No annotation:", filename)
        failed += 1
        continue

    results = annotations[-1].get("result", [])

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    found_mask = False

    for result in results:
        if result.get("type") != "brushlabels":
            continue

        value = result.get("value", {})
        labels = value.get("brushlabels", [])
        rle = value.get("rle")
        if not labels or rle is None:
            continue

        label = labels[0]
        if label not in CLASS_MAP:
            print("[WARNING] Unknown label:", label)
            continue

        class_id = CLASS_MAP[label]
        width = int(result.get("original_width", image_width))
        height = int(result.get("original_height", image_height))

        decoded = np.asarray(decode_rle(rle), dtype=np.uint8)
        expected_size = width * height * 4

        if decoded.size != expected_size:
            print("[ERROR] RLE size mismatch:", filename, decoded.size, expected_size)
            continue

        rgba = decoded.reshape(height, width, 4)
        region = rgba[:, :, 3] > 0

        if width != image_width or height != image_height:
            temp = np.zeros((height, width), dtype=np.uint8)
            temp[region] = 255
            temp = Image.fromarray(temp).resize(
                (image_width, image_height),
                Image.Resampling.NEAREST,
            )
            region = np.asarray(temp) > 0

        mask[region] = class_id
        found_mask = True

    if not found_mask:
        print("[SKIP] No Brush mask:", filename)
        failed += 1
        continue

    shutil.copy2(image_path, OUT_IMAGE_DIR / filename)
    Image.fromarray(mask).save(OUT_MASK_DIR / f"{image_path.stem}.png")

    print("[OK]", filename, "classes =", np.unique(mask))
    success += 1

print("\nFinished")
print("Success:", success)
print("Failed:", failed)
