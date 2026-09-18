import base64
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
MODEL_PATH = os.environ.get("VEHICLE_MODEL", "yolo12n.pt")
model = None

TARGET_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
REAL_VEHICLE_HEIGHTS = {2: 1.55, 3: 1.2, 5: 3.0, 7: 2.5}
WARNING_DISTANCES = {"LEFT": 1.0, "MAIN": 2.0, "RIGHT": 1.0}
ROI_ZONES = {
    "LEFT": [[240, 600], [925, 550], [312, 1100], [100, 1100]],
    "MAIN": [[925, 550], [1025, 550], [1712, 1100], [312, 1100]],
    "RIGHT": [[1025, 550], [1802, 600], [1942, 1100], [1712, 1100]],
}
OPTICAL_CENTERS = {"LEFT": (156, 1050), "MAIN": (1000, 1020), "RIGHT": (1868, 1050)}
BASE_WIDTH = 2042
BASE_HEIGHT = 1148
BASE_FOCAL_LENGTH = 500


def get_model():
    global model
    if model is None:
        model_path = Path(MODEL_PATH)
        if model_path.is_absolute() or model_path.exists():
            model = YOLO(str(model_path))
        else:
            # Ultralytics downloads a named model from its model registry when needed.
            model = YOLO(MODEL_PATH)
    return model


def scaled_configuration(width, height):
    scale_x = width / BASE_WIDTH
    scale_y = height / BASE_HEIGHT
    zones = {
        name: np.array([[int(x * scale_x), int(y * scale_y)] for x, y in points], dtype=np.int32)
        for name, points in ROI_ZONES.items()
    }
    centers = {
        name: (x * scale_x, y * scale_y)
        for name, (x, y) in OPTICAL_CENTERS.items()
    }
    return zones, centers, BASE_FOCAL_LENGTH * scale_y


def zone_for_point(point, zones):
    for name, polygon in zones.items():
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return name
    return None


def estimate_distance(box, class_id, zone, centers, focal_length):
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    if box_height <= 0:
        return None
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    optical_x, optical_y = centers[zone]
    displacement = np.hypot(center_x - optical_x, center_y - optical_y)
    distance = REAL_VEHICLE_HEIGHTS[class_id] * focal_length / box_height
    return float(distance * (1.0 + displacement * 0.0001))


def analyze_frame(frame):
    height, width = frame.shape[:2]
    zones, centers, focal_length = scaled_configuration(width, height)
    detections = []
    results = get_model()(frame, classes=list(TARGET_CLASSES), conf=0.7, verbose=False)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            zone = zone_for_point(center, zones)
            if zone is None:
                continue
            distance = estimate_distance((x1, y1, x2, y2), class_id, zone, centers, focal_length)
            if distance is None:
                continue
            warning = distance < WARNING_DISTANCES[zone]
            color = (0, 0, 255) if warning or (zone == "MAIN" and distance < 5) else (0, 200, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            label = f"{TARGET_CLASSES[class_id]}  {distance:.1f}m  {zone}"
            cv2.rectangle(frame, (int(x1), max(0, int(y1) - 34)), (int(x1) + 300, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1) + 8, max(22, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            detections.append({
                "object": TARGET_CLASSES[class_id],
                "confidence": round(confidence, 3),
                "zone": zone,
                "distance_m": round(distance, 2),
                "warning": warning,
            })

    if any(item["warning"] for item in detections):
        cv2.rectangle(frame, (20, 20), (620, 75), (0, 0, 220), -1)
        cv2.putText(frame, "WARNING: VEHICLE VERY CLOSE", (35, 57), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame, detections


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Please choose an image."}), 400
    image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "The uploaded file is not a supported image."}), 400
    try:
        annotated, detections = analyze_frame(image)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return jsonify({"error": "Could not encode the processed image."}), 500
    image_data = base64.b64encode(encoded).decode("ascii")
    return jsonify({"image": f"data:image/jpeg;base64,{image_data}", "detections": detections})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), debug=False)
