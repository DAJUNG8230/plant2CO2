from pathlib import Path
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from inference import analyze_image

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


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
        return jsonify(success=False, message="只支援 JPG / JPEG / PNG"), 400

    try:
        gsd_cm = float(request.form.get("gsd", "5.0"))
        carbon_coefficient = float(request.form.get("carbon_coefficient", "0.35"))
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
    image.save(input_path)

    try:
        result = analyze_image(
            image_path=input_path,
            output_dir=OUTPUT_DIR,
            job_id=job_id,
            gsd_cm_per_pixel=gsd_cm,
            carbon_coefficient=carbon_coefficient,
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
