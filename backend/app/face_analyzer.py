"""
Анализ формы лица через MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10.30).
Скачиваем модель автоматически при первом запуске.
"""

import cv2
import numpy as np
import urllib.request
import os
from typing import Tuple, Dict

MODEL_PATH = r"C:\models\face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Скачиваем модель MediaPipe FaceLandmarker (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Модель скачана.")

IDX_FOREHEAD_TOP = 10
IDX_CHIN         = 152
IDX_CHEEK_L      = 234
IDX_CHEEK_R      = 454
IDX_JAW_L        = 58
IDX_JAW_R        = 288
IDX_FOREHEAD_L   = 103
IDX_FOREHEAD_R   = 332

FACE_SHAPE_MAP = {
    "round":    {"recommended_shapes": ["Кошачий глаз", "Авиаторы", "Вайфареры"],  "shape_codes": ["cat", "aviator", "wayfarer"]},
    "oval":     {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],    "shape_codes": ["aviator", "round", "cat"]},
    "square":   {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],    "shape_codes": ["aviator", "round", "cat"]},
    "rect":     {"recommended_shapes": ["Авиаторы", "Круглые"],                     "shape_codes": ["aviator", "round"]},
    "triangle": {"recommended_shapes": ["Авиаторы", "Стрекозы", "Круглые"],        "shape_codes": ["aviator", "round"]},
    "heart":    {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],       "shape_codes": ["aviator", "wayfarer", "round"]},
    "diamond":  {"recommended_shapes": ["Авиаторы", "Овальные", "Кошачий глаз"],   "shape_codes": ["aviator", "oval", "cat"]},
}

def _dist(a, b) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

def _get_xy(landmark, w: int, h: int) -> Tuple[float, float]:
    return landmark.x * w, landmark.y * h

def classify_face_shape(face_ratio: float, jaw_ratio: float, forehead_ratio: float) -> Tuple[str, float]:
    def in_range(val, lo, hi):
        if lo <= val <= hi:
            return 1.0
        margin = (hi - lo) * 0.5 or 0.1
        return max(0.0, 1.0 - (abs(val - (lo if val < lo else hi)) / margin))

    scores = {
        "oval":     sum([in_range(face_ratio, 1.30, 1.60), in_range(jaw_ratio, 0.70, 0.85), in_range(forehead_ratio, 0.80, 0.95)]) / 3,
        "round":    sum([in_range(face_ratio, 1.00, 1.25), in_range(jaw_ratio, 0.75, 0.90), in_range(forehead_ratio, 0.80, 0.95)]) / 3,
        "square":   sum([in_range(face_ratio, 1.00, 1.30), in_range(jaw_ratio, 0.85, 1.00), in_range(forehead_ratio, 0.80, 1.00)]) / 3,
        "rect":     sum([in_range(face_ratio, 1.50, 2.00), in_range(jaw_ratio, 0.75, 0.95), in_range(forehead_ratio, 0.75, 0.95)]) / 3,
        "heart":    sum([in_range(face_ratio, 1.20, 1.60), in_range(jaw_ratio, 0.50, 0.72), in_range(forehead_ratio, 0.88, 1.10)]) / 3,
        "triangle": sum([in_range(face_ratio, 1.10, 1.50), in_range(jaw_ratio, 0.90, 1.10), in_range(forehead_ratio, 0.60, 0.80)]) / 3,
        "diamond":  sum([in_range(face_ratio, 1.30, 1.70), in_range(jaw_ratio, 0.55, 0.75), in_range(forehead_ratio, 0.65, 0.82)]) / 3,
    }
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    confidence = round(scores[best] / total, 3)
    return best, confidence

def analyze_face_image(image_bytes: bytes) -> dict:
    _ensure_model()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Не удалось декодировать изображение", "face_shape": None, "confidence": 0, "measurements": None}

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {"error": "Лицо не обнаружено на фото", "face_shape": None, "confidence": 0, "measurements": None}

    lm = result.face_landmarks[0]
    forehead_top = _get_xy(lm[IDX_FOREHEAD_TOP], w, h)
    chin         = _get_xy(lm[IDX_CHIN], w, h)
    cheek_l      = _get_xy(lm[IDX_CHEEK_L], w, h)
    cheek_r      = _get_xy(lm[IDX_CHEEK_R], w, h)
    jaw_l        = _get_xy(lm[IDX_JAW_L], w, h)
    jaw_r        = _get_xy(lm[IDX_JAW_R], w, h)
    forehead_l   = _get_xy(lm[IDX_FOREHEAD_L], w, h)
    forehead_r   = _get_xy(lm[IDX_FOREHEAD_R], w, h)

    face_height     = _dist(forehead_top, chin)
    cheekbone_width = _dist(cheek_l, cheek_r)
    jaw_width       = _dist(jaw_l, jaw_r)
    forehead_width  = _dist(forehead_l, forehead_r)

    if cheekbone_width == 0:
        return {"error": "Не удалось измерить пропорции лица", "face_shape": None, "confidence": 0, "measurements": None}

    face_ratio     = round(face_height / cheekbone_width, 3)
    jaw_ratio      = round(jaw_width / cheekbone_width, 3)
    forehead_ratio = round(forehead_width / cheekbone_width, 3)

    shape, confidence = classify_face_shape(face_ratio, jaw_ratio, forehead_ratio)

    return {
        "face_shape": shape,
        "confidence": confidence,
        "measurements": {
            "face_ratio": face_ratio,
            "jaw_ratio": jaw_ratio,
            "forehead_ratio": forehead_ratio,
        },
        "error": None,
    }
