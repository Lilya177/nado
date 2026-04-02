"""
face_analyzer.py — с промежуточными формами лица.
Если два класса близки по скору — возвращаем гибрид типа oval-square.
"""

import cv2
import numpy as np
import urllib.request
import os
import joblib
from typing import Tuple, Optional

MODEL_PATH = r"C:\models\face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

_ML_MODEL_CACHE = None

def _load_ml_model():
    global _ML_MODEL_CACHE
    if _ML_MODEL_CACHE is None:
        path = os.path.join(os.path.dirname(__file__), "face_shape_model.pkl")
        if os.path.exists(path):
            _ML_MODEL_CACHE = joblib.load(path)
    return _ML_MODEL_CACHE

# Маппинг всех форм включая гибриды
FACE_SHAPE_MAP = {
    # Чистые формы
    "oval":         {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],   "shape_codes": ["aviator", "round", "cat"]},
    "round":        {"recommended_shapes": ["Кошачий глаз", "Авиаторы", "Вайфареры"], "shape_codes": ["cat", "aviator", "wayfarer"]},
    "square":       {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],   "shape_codes": ["aviator", "round", "cat"]},
    "rect":         {"recommended_shapes": ["Авиаторы", "Круглые"],                    "shape_codes": ["aviator", "round"]},
    "heart":        {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],      "shape_codes": ["aviator", "wayfarer", "round"]},
    "triangle":     {"recommended_shapes": ["Авиаторы", "Стрекозы", "Круглые"],       "shape_codes": ["aviator", "round"]},
    "diamond":      {"recommended_shapes": ["Авиаторы", "Овальные", "Кошачий глаз"],  "shape_codes": ["aviator", "oval", "cat"]},
    # Гибридные формы
    "oval-square":  {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Круглые"],   "shape_codes": ["aviator", "cat", "round"]},
    "oval-round":   {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Вайфареры"], "shape_codes": ["aviator", "cat", "wayfarer"]},
    "oval-heart":   {"recommended_shapes": ["Авиаторы", "Вайфареры", "Кошачий глаз"], "shape_codes": ["aviator", "wayfarer", "cat"]},
    "oval-rect":    {"recommended_shapes": ["Авиаторы", "Круглые"],                    "shape_codes": ["aviator", "round"]},
    "square-round": {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Круглые"],   "shape_codes": ["aviator", "cat", "round"]},
    "heart-oval":   {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],      "shape_codes": ["aviator", "wayfarer", "round"]},
    "round-square": {"recommended_shapes": ["Авиаторы", "Кошачий глаз"],              "shape_codes": ["aviator", "cat"]},
}

def _dist(a, b) -> float:
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))

def _make_rotator(lm, w, h):
    angle = np.arctan2(lm[263].y*h - lm[33].y*h, lm[263].x*w - lm[33].x*w)
    cos_a = np.cos(-angle); sin_a = np.sin(-angle)
    cx, cy = w/2.0, h/2.0
    def rot(idx):
        x = lm[idx].x*w - cx; y = lm[idx].y*h - cy
        return x*cos_a - y*sin_a + cx, x*sin_a + y*cos_a + cy
    return rot

def extract_measurements(lm, w, h) -> Optional[dict]:
    rot = _make_rotator(lm, w, h)
    cheek_l   = rot(234); cheek_r   = rot(454)
    jaw_l     = rot(172); jaw_r     = rot(397)
    jaw_ang_l = rot(58);  jaw_ang_r = rot(288)
    brow_l    = rot(70);  brow_r    = rot(300)
    eye_l     = rot(33);  eye_r     = rot(263)
    temple_l  = rot(162); temple_r  = rot(389)
    forehead  = rot(9)
    chin      = rot(152)
    nose_tip  = rot(1)

    cheek_w   = _dist(cheek_l, cheek_r)
    jaw_w     = _dist(jaw_l, jaw_r)
    jaw_ang_w = _dist(jaw_ang_l, jaw_ang_r)
    brow_w    = _dist(brow_l, brow_r)
    eye_w     = _dist(eye_l, eye_r)
    temple_w  = _dist(temple_l, temple_r)
    face_h    = _dist(forehead, chin) * 1.60
    lower_h   = _dist(nose_tip, chin)

    if cheek_w == 0 or face_h == 0:
        return None

    return {
        "face_ratio":    round(face_h / cheek_w, 4),
        "jaw_ratio":     round(jaw_w / cheek_w, 4),
        "jaw_ang_ratio": round(jaw_ang_w / cheek_w, 4),
        "brow_ratio":    round(brow_w / cheek_w, 4),
        "eye_ratio":     round(eye_w / cheek_w, 4),
        "temple_ratio":  round(temple_w / cheek_w, 4),
        "lower_ratio":   round(lower_h / face_h, 4),
        "taper":         round(jaw_w / brow_w, 4) if brow_w else 0,
    }

def classify_face_shape(m: dict) -> Tuple[str, float]:
    fr  = m["face_ratio"]
    jr  = m["jaw_ratio"]
    jar = m["jaw_ang_ratio"]
    br  = m["brow_ratio"]
    er  = m["eye_ratio"]
    tr  = m["temple_ratio"]
    lr  = m["lower_ratio"]
    tp  = m["taper"]

    def g(val, center, sigma):
        return float(np.exp(-((val - center)**2) / (2 * sigma**2)))

    scores = {
        "oval": (
            g(fr, 1.38, 0.16) * 0.35 +
            g(jr, 0.76, 0.08) * 0.20 +
            g(br, 0.85, 0.08) * 0.15 +
            g(tp, 0.90, 0.09) * 0.20 +
            g(lr, 0.44, 0.06) * 0.10
        ),
        "round": (
            g(fr, 1.05, 0.10) * 0.35 +
            g(jr, 0.85, 0.07) * 0.25 +
            g(tp, 0.98, 0.07) * 0.25 +
            g(lr, 0.48, 0.06) * 0.15
        ),
        "square": (
            g(fr, 1.20, 0.13) * 0.25 +
            g(jar, 0.90, 0.07) * 0.30 +
            g(tp,  1.00, 0.07) * 0.25 +
            g(jr,  0.85, 0.07) * 0.20
        ),
        "rect": (
            g(fr, 1.75, 0.12) * 0.55 +
            g(jr, 0.82, 0.08) * 0.20 +
            g(tr, 0.95, 0.08) * 0.15 +
            g(br, 0.88, 0.08) * 0.10
        ),
        "heart": (
            g(br, 0.95, 0.07) * 0.30 +
            g(tp, 0.72, 0.08) * 0.35 +
            g(jr, 0.65, 0.08) * 0.25 +
            g(fr, 1.35, 0.12) * 0.10
        ),
        "triangle": (
            g(tp, 1.20, 0.10) * 0.40 +
            g(jar, 0.95, 0.07) * 0.30 +
            g(br, 0.72, 0.08) * 0.20 +
            g(fr, 1.20, 0.12) * 0.10
        ),
        "diamond": (
            g(er, 0.88, 0.06) * 0.30 +
            g(tp, 0.82, 0.07) * 0.25 +
            g(jr, 0.68, 0.07) * 0.25 +
            g(fr, 1.45, 0.12) * 0.20
        ),
    }

    # Сортируем по убыванию
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    best_name,  best_score  = sorted_scores[0]
    second_name, second_score = sorted_scores[1]

    total = sum(scores.values()) or 1.0
    best_share   = best_score / total
    second_share = second_score / total

    # Порог гибрида: если второй класс набрал > 75% от первого — гибрид
    HYBRID_THRESHOLD = 0.75

    if second_score / best_score > HYBRID_THRESHOLD:
        # Формируем гибридное название в алфавитном порядке чтобы не дублировать
        pair = tuple(sorted([best_name, second_name]))
        hybrid = f"{pair[0]}-{pair[1]}"
        # Проверяем есть ли такой гибрид в маппинге, иначе берём лучший
        if hybrid not in FACE_SHAPE_MAP:
            hybrid = f"{best_name}-{second_name}"
        if hybrid not in FACE_SHAPE_MAP:
            hybrid = best_name  # fallback
        confidence = round(min(0.95, (best_share + second_share) * 1.5), 3)
        return hybrid, confidence
    else:
        confidence = round(min(0.95, best_share * 2.2), 3)
        return best_name, confidence


def analyze_face_image(image_bytes: bytes) -> dict:
    _ensure_model()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Не удалось декодировать изображение",
                "face_shape": None, "confidence": 0, "measurements": None}

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
        return {"error": "Лицо не обнаружено на фото",
                "face_shape": None, "confidence": 0, "measurements": None}

    lm = result.face_landmarks[0]
    m = extract_measurements(lm, w, h)
    if m is None:
        return {"error": "Не удалось измерить пропорции",
                "face_shape": None, "confidence": 0, "measurements": None}

    ml = _load_ml_model()
    if ml:
        # ML not used
        features = _build_feature_vector(m)
        raw_shape = ml["encoder"].inverse_transform(ml["model"].predict(features))[0]
        shape = raw_shape.lower().replace("oblong", "rect")
        proba = ml["model"].predict_proba(features)[0].max()
        confidence = round(float(proba), 3)
    else:
        shape, confidence = classify_face_shape(m)

    return {
        "face_shape": shape,
        "confidence": confidence,
        "measurements": {
            "face_ratio":     m["face_ratio"],
            "jaw_ratio":      m["jaw_ratio"],
            "forehead_ratio": m["brow_ratio"],
        },
        "error": None,
    }