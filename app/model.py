import io
import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Global variables to store loaded models
_yolo_model = None
_health_model = None
_disease_model = None

def load_yolo_model():
    """Load YOLOv5 model lazily"""
    global _yolo_model
    if _yolo_model is None:
        try:
            logger.info("Loading YOLO model...")
            _yolo_model = torch.hub.load(str(YOLOV5_LOCAL_PATH), 'custom', path=str(YOLO_PATH), source='local')
            _yolo_model.conf = 0.6  # confidence threshold
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    return _yolo_model

def load_health_model():
    """Load health classification model lazily"""
    global _health_model
    if _health_model is None:
        try:
            logger.info("Loading health model...")
            _health_model = load_model(HEALTH_MODEL_PATH)
            logger.info("Health model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load health model: {e}")
            raise
    return _health_model

def load_disease_model():
    """Load disease classification model lazily"""
    global _disease_model
    if _disease_model is None:
        try:
            logger.info("Loading disease model...")
            _disease_model = load_model(DISEASE_MODEL_PATH)
            logger.info("Disease model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load disease model: {e}")
            raise
    return _disease_model

def preprocess_image(img):
    """Preprocess image for model input"""
    img = cv2.resize(img, (224, 224))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return preprocess_input(img)

def predict_pipeline(image_bytes: bytes) -> dict:
    """Main prediction pipeline with lazy model loading"""
    try:
        logger.info("Starting prediction pipeline...")
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(image)
        logger.info(f"Image loaded with shape: {img_array.shape}")

        # Load YOLO model and run detection
        yolo_model = load_yolo_model()
        logger.info("Running YOLO detection...")
        results = yolo_model(img_array)
        detections = results.pandas().xyxy[0]
        leaf_detections = detections[detections['name'] == 'leaf']

        if leaf_detections.empty:
            logger.info("No leaf detected")
            return {
                "detected": False,
                "message": "No leaf detected"
            }

        # Get best detection (first one)
        best_leaf = leaf_detections.iloc[0]
        x1, y1, x2, y2 = map(int, [best_leaf['xmin'], best_leaf['ymin'], best_leaf['xmax'], best_leaf['ymax']])
        leaf_crop = img_array[y1:y2, x1:x2]
        logger.info(f"Leaf detected and cropped: {leaf_crop.shape}")

        # Load health model and make prediction
        health_model = load_health_model()
        health_input = preprocess_image(leaf_crop)
        logger.info("Running health prediction...")
        health_pred = health_model.predict(np.expand_dims(health_input, axis=0), verbose=0)[0]
        health_idx = np.argmax(health_pred)
        health_status = HEALTH_LABELS[health_idx]
        health_confidence = float(health_pred[health_idx])

        if health_status == 'healthy':
            logger.info(f"Prediction completed: {health_status}")
            return {
                "detected": True,
                "health_status": "healthy",
                "confidence": health_confidence
            }

        # Load disease model and make prediction
        disease_model = load_disease_model()
        disease_input = preprocess_image(leaf_crop)
        logger.info("Running disease prediction...")
        disease_pred = disease_model.predict(np.expand_dims(disease_input, axis=0), verbose=0)[0]
        disease_idx = np.argmax(disease_pred)
        disease_label = DISEASE_LABELS[disease_idx]
        disease_confidence = float(disease_pred[disease_idx])

        result = {
            "detected": True,
            "health_status": "unhealthy",
            "disease": disease_label,
            "confidence": disease_confidence
        }
        logger.info(f"Prediction completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}")
        return {
            "detected": False,
            "message": f"Prediction failed: {str(e)}",
            "error": str(e)
        }