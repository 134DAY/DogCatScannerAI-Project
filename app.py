from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__, static_folder="static")
CORS(app)

# ใช้ yolo11n.pt (COCO pretrained) — มี dog/cat อยู่แล้ว แม่นกว่า
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolo11n.pt")
model = YOLO(MODEL_PATH)

# ชื่อ class ของ model (ปรับตาม class ที่เทรนมา)
DOG_LABELS = {"dog", "dogs"}
CAT_LABELS = {"cat", "cats"}


def analyze_image(img_array):
    """รัน YOLO และสรุปผลว่าเจอหมา / แมว / ไม่ใช่ทั้งคู่"""
    results = model(img_array, verbose=False, conf=0.5)[0]

    detected_classes = set()
    boxes_info = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        conf = float(box.conf[0])
        detected_classes.add(label)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes_info.append({
            "label": model.names[cls_id],
            "confidence": round(conf * 100, 1),
            "bbox": [x1, y1, x2, y2]
        })

    # สรุปผลลัพธ์
    has_dog = any(l in DOG_LABELS for l in detected_classes)
    has_cat = any(l in CAT_LABELS for l in detected_classes)

    if has_dog and has_cat:
        verdict = "both"
        verdict_th = "พบทั้งหมาและแมว! 🐕🐈"
        verdict_en = "Both Dog & Cat Detected!"
    elif has_dog:
        verdict = "dog"
        verdict_th = "นี่คือหมา! 🐕"
        verdict_en = "It's a Dog!"
    elif has_cat:
        verdict = "cat"
        verdict_th = "นี่คือแมว! 🐈"
        verdict_en = "It's a Cat!"
    else:
        verdict = "none"
        verdict_th = "ไม่ใช่หมาหรือแมว ❌"
        verdict_en = "Not a Dog or Cat"

    # วาด bounding box บนรูป
    annotated = results.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".jpg", annotated_rgb)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "verdict": verdict,
        "verdict_th": verdict_th,
        "verdict_en": verdict_en,
        "boxes": boxes_info,
        "annotated_image": img_b64
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """รับรูปภาพ (base64) แล้ว return ผลวิเคราะห์"""
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # decode base64 → numpy array
        img_data = data["image"]
        if "," in img_data:
            img_data = img_data.split(",")[1]

        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        result = analyze_image(img)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🐕🐈  DoggyCatScaner  🐈🐕")
    print(f"Model: {MODEL_PATH}")
    print("Server: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
