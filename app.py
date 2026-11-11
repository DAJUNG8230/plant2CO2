from io import BytesIO
from pathlib import Path
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from inference import analyze_image, get_runtime_status

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIFF_EXTENSIONS = {".tif", ".tiff"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"} | TIFF_EXTENSIONS
ALLOWED_MODELS = {"deeplabv3plus", "unet", "segformer", "pspnet"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(success=True, **get_runtime_status())


@app.get("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


def tiff_png(image):
    import numpy as np
    from PIL import Image
    from rasterio.enums import ColorInterp
    from rasterio.io import MemoryFile

    with MemoryFile(image.read()) as source, source.open() as dataset:
        if dataset.driver != "GTiff" or dataset.width * dataset.height > 25_000_000:
            raise ValueError("請上傳有效且不超過 2500 萬像素的 TIFF。")
        colors = dataset.colorinterp
        rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        if all(color in colors for color in rgb):
            bands = [colors.index(color) + 1 for color in rgb]
        elif dataset.count >= 3:
            bands = [1, 2, 3]  # FLAIR stores RGB first, followed by NIR and height.
        elif dataset.count == 1 and colors[0] != ColorInterp.palette:
            bands = [1]
        else:
            raise ValueError("無法辨識此 TIFF 的 RGB 波段。")
        if any(dataset.dtypes[i - 1] != "uint8" for i in bands):
            raise ValueError("目前支援 8 位元 TIFF。")
        pixels = dataset.read(bands)
    array = pixels[0] if len(bands) == 1 else np.moveaxis(pixels, 0, -1)
    output = BytesIO()
    Image.fromarray(array).convert("RGB").save(output, format="PNG")
    output.seek(0)
    return output


@app.post("/api/tiff-preview")
def tiff_preview():
    image = request.files.get("image")
    if not image or Path(image.filename or "").suffix.lower() not in TIFF_EXTENSIONS:
        return jsonify(message="請選擇 TIF 或 TIFF 影像。"), 400
    try:
        preview = tiff_png(image)
    except (ValueError, OSError) as exc:
        return jsonify(message=str(exc)), 400
    return send_file(preview, mimetype="image/png")


@app.post("/api/analyze")
def analyze():
    started = time.perf_counter()

    if "image" not in request.files:
        return jsonify(success=False, message="沒有收到圖片"), 400

    image = request.files["image"]
    if not image.filename:
        return jsonify(success=False, message="沒有選擇圖片"), 400

    suffix = Path(image.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(success=False, message="支援 JPG / JPEG / PNG / TIF / TIFF"), 400

    model_name = request.form.get("model", "deeplabv3plus").strip().lower()
    if model_name not in ALLOWED_MODELS:
        return jsonify(success=False, message=f"不支援的模型：{model_name}"), 400

    try:
        gsd_cm = float(request.form.get("gsd", "5.0"))
        carbon_coefficient = float(request.form.get("carbon_coefficient", "0.40"))
    except ValueError:
        return jsonify(success=False, message="GSD 或碳匯係數格式錯誤"), 400

    if gsd_cm <= 0:
        return jsonify(success=False, message="GSD 必須大於 0"), 400
    if carbon_coefficient < 0:
        return jsonify(success=False, message="碳匯係數不可小於 0"), 400

    safe_name = secure_filename(image.filename) or f"image{suffix}"
    job_id = uuid.uuid4().hex[:12]
    input_name = f"{job_id}_{safe_name}"
    input_path = UPLOAD_DIR / input_name
    if suffix in TIFF_EXTENSIONS:
        try:
            converted = tiff_png(image)
        except (ValueError, OSError) as exc:
            return jsonify(success=False, message=str(exc)), 400
        input_path = input_path.with_suffix(".png")
        input_path.write_bytes(converted.getvalue())
    else:
        image.save(input_path)

    try:
        result = analyze_image(
            image_path=input_path,
            output_dir=OUTPUT_DIR,
            job_id=job_id,
            gsd_cm_per_pixel=gsd_cm,
            carbon_coefficient=carbon_coefficient,
            model_name=model_name,
        )
    except Exception as exc:
        app.logger.exception("Analysis failed")
        return jsonify(success=False, message=f"分析失敗：{exc}"), 500

    result["success"] = True
    result["filename"] = safe_name
    result["inference_time"] = round(time.perf_counter() - started, 4)
    result["mask_url"] = f"/outputs/{result['mask_filename']}"
    result["overlay_url"] = f"/outputs/{result['overlay_filename']}"

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
