import io
import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Base directory for resolving relative paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Model file paths
YOLO_PATH = BASE_DIR / "best.pt"
HEALTH_MODEL_PATH = BASE_DIR / "mobilenetv2_healthy_best.h5"
DISEASE_MODEL_PATH = BASE_DIR / "mobilenetv2_soybean_best_old.h5"
YOLOV5_LOCAL_PATH = BASE_DIR / "yolov5"

# Class labels
HEALTH_LABELS = ['healthy', 'unhealthy']
DISEASE_LABELS = [
    "Healthy", "Mossaic Virus", "Southern blight", "Sudden Death Syndrome",
    "Yellow Mosaic", "bacterial_blight", "brown_spot", "ferrugen",
    "powdery_mildew", "septoria"
]

# Load YOLOv5 model
def load_yolo_model():
    model = torch.hub.load(str(YOLOV5_LOCAL_PATH), 'custom', path=str(YOLO_PATH), source='local')
    model.conf = 0.6  # confidence threshold
    return model

# Load models at startup
yolo_model = load_yolo_model()
health_model = load_model(HEALTH_MODEL_PATH)
disease_model = load_model(DISEASE_MODEL_PATH)

# Prediction pipeline
def predict_pipeline(image_bytes: bytes) -> dict:
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(image)

    # Run YOLOv5 detection
    results = yolo_model(img_array)
    detections = results.pandas().xyxy[0]
    leaf_detections = detections[detections['name'] == 'leaf']

    if leaf_detections.empty:
        return {
            "detected": False,
            "message": "No leaf detected"
        }

    # Get best detection (first one)
    best_leaf = leaf_detections.iloc[0]
    x1, y1, x2, y2 = map(int, [best_leaf['xmin'], best_leaf['ymin'], best_leaf['xmax'], best_leaf['ymax']])
    leaf_crop = img_array[y1:y2, x1:x2]

    # Preprocessing
    def preprocess(img):
        img = cv2.resize(img, (224, 224))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return preprocess_input(img)

    # Health model prediction
    health_input = preprocess(leaf_crop)
    health_pred = health_model.predict(np.expand_dims(health_input, axis=0), verbose=0)[0]
    health_idx = np.argmax(health_pred)
    health_status = HEALTH_LABELS[health_idx]
    health_confidence = float(health_pred[health_idx])

    if health_status == 'healthy':
        return {
            "detected": True,
            "health_status": "healthy",
            "confidence": health_confidence
        }

    # Disease model prediction
    disease_input = preprocess(leaf_crop)
    disease_pred = disease_model.predict(np.expand_dims(disease_input, axis=0), verbose=0)[0]
    disease_idx = np.argmax(disease_pred)
    disease_label = DISEASE_LABELS[disease_idx]
    disease_confidence = float(disease_pred[disease_idx])

    return {
        "detected": True,
        "health_status": "unhealthy",
        "disease": disease_label,
        "confidence": disease_confidence
    }
