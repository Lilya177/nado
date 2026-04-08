"""
face_analyzer.py — финальная версия.
MediaPipe точки + предобработка + ансамбль CNN (70%) + геометрика (30%).
"""

import cv2
import numpy as np
import urllib.request
import os
from typing import Tuple, Optional

MODEL_PATH = r"C:\models\face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Скачиваем модель MediaPipe...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Готово.")

FACE_SHAPE_MAP = {
    "oval":         {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],   "shape_codes": ["aviator", "round", "cat"]},
    "round":        {"recommended_shapes": ["Кошачий глаз", "Авиаторы", "Вайфареры"], "shape_codes": ["cat", "aviator", "wayfarer"]},
    "square":       {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],   "shape_codes": ["aviator", "round", "cat"]},
    "rect":         {"recommended_shapes": ["Авиаторы", "Круглые"],                    "shape_codes": ["aviator", "round"]},
    "heart":        {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],      "shape_codes": ["aviator", "wayfarer", "round"]},
    "triangle":     {"recommended_shapes": ["Авиаторы", "Стрекозы", "Круглые"],       "shape_codes": ["aviator", "round"]},
    "diamond":      {"recommended_shapes": ["Авиаторы", "Овальные", "Кошачий глаз"],  "shape_codes": ["aviator", "oval", "cat"]},
    "oval-square":  {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Круглые"],   "shape_codes": ["aviator", "cat", "round"]},
    "oval-round":   {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Вайфареры"], "shape_codes": ["aviator", "cat", "wayfarer"]},
    "oval-heart":   {"recommended_shapes": ["Авиаторы", "Вайфареры", "Кошачий глаз"], "shape_codes": ["aviator", "wayfarer", "cat"]},
    "oval-rect":    {"recommended_shapes": ["Авиаторы", "Круглые"],                    "shape_codes": ["aviator", "round"]},
    "square-round": {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Круглые"],   "shape_codes": ["aviator", "cat", "round"]},
    "round-square": {"recommended_shapes": ["Авиаторы", "Кошачий глаз"],              "shape_codes": ["aviator", "cat"]},
    "heart-oval":   {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],      "shape_codes": ["aviator", "wayfarer", "round"]},
    "diamond-oval": {"recommended_shapes": ["Авиаторы", "Овальные", "Кошачий глаз"],  "shape_codes": ["aviator", "oval", "cat"]},
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
    cheek_w   = _dist(rot(234), rot(454))
    jaw_w     = _dist(rot(172), rot(397))
    jaw_ang_w = _dist(rot(58),  rot(288))
    brow_w    = _dist(rot(70),  rot(300))
    eye_w     = _dist(rot(33),  rot(263))
    temple_w  = _dist(rot(162), rot(389))
    face_h    = _dist(rot(9),   rot(152)) * 1.60
    lower_h   = _dist(rot(1),   rot(152))
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

def classify_face_shape(m: dict) -> dict:
    """Геометрический классификатор. Возвращает dict {shape: score} для ансамбля."""
    fr=m["face_ratio"]; jr=m["jaw_ratio"]; jar=m["jaw_ang_ratio"]
    br=m["brow_ratio"]; er=m["eye_ratio"]; tr=m["temple_ratio"]
    lr=m["lower_ratio"]; tp=m["taper"]

    def g(val, center, sigma):
        return float(np.exp(-((val-center)**2) / (2*sigma**2)))

    scores = {
        "oval":     g(fr,1.38,0.16)*0.35 + g(jr,0.76,0.08)*0.20 + g(br,0.85,0.08)*0.15 + g(tp,0.90,0.09)*0.20 + g(lr,0.44,0.06)*0.10,
        "round":    g(fr,1.05,0.10)*0.35 + g(jr,0.85,0.07)*0.25 + g(tp,0.98,0.07)*0.25 + g(lr,0.48,0.06)*0.15,
        "square":   g(fr,1.20,0.13)*0.25 + g(jar,0.90,0.07)*0.30 + g(tp,1.00,0.07)*0.25 + g(jr,0.85,0.07)*0.20,
        "rect":     g(fr,1.75,0.12)*0.55 + g(jr,0.82,0.08)*0.20 + g(tr,0.95,0.08)*0.15 + g(br,0.88,0.08)*0.10,
        "heart":    g(br,0.95,0.07)*0.30 + g(tp,0.72,0.08)*0.35 + g(jr,0.65,0.08)*0.25 + g(fr,1.35,0.12)*0.10,
        "triangle": g(tp,1.20,0.10)*0.40 + g(jar,0.95,0.07)*0.30 + g(br,0.72,0.08)*0.20 + g(fr,1.20,0.12)*0.10,
        "diamond":  g(er,0.88,0.06)*0.30 + g(tp,0.82,0.07)*0.25 + g(jr,0.68,0.07)*0.25 + g(fr,1.45,0.12)*0.20,
    }
    total = sum(scores.values()) or 1.0
    return {k: v/total for k, v in scores.items()}


def _preprocess_face(img_bgr, lm, w, h) -> bytes:
    """
    Предобработка для CNN:
    1. Поворот по линии глаз
    2. Кроп строго от лба до подбородка
    3. Ресайз до 380x380 (размер обучения EfficientNetB4)
    4. Нормализация яркости
    """
    # Угол поворота по глазам
    eye_l = (lm[33].x*w,  lm[33].y*h)
    eye_r = (lm[263].x*w, lm[263].y*h)
    angle = np.degrees(np.arctan2(eye_r[1]-eye_l[1], eye_r[0]-eye_l[0]))

    # Центр между глазами
    cx = int((eye_l[0] + eye_r[0]) / 2)
    cy = int((eye_l[1] + eye_r[1]) / 2)

    # Поворачиваем изображение
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h))

    # Трансформируем ключевые точки для кропа
    def transform_pt(idx):
        x, y = lm[idx].x*w, lm[idx].y*h
        pt = np.array([[[x, y]]], dtype=np.float32)
        return cv2.transform(pt, M)[0][0]

    forehead_pt = transform_pt(10)   # верх лба
    chin_pt     = transform_pt(152)  # подбородок
    cheek_l_pt  = transform_pt(234)  # левая скула
    cheek_r_pt  = transform_pt(454)  # правая скула

    face_h_px = _dist(forehead_pt, chin_pt)
    face_w_px = _dist(cheek_l_pt, cheek_r_pt)

    # Кроп с небольшим отступом
    pad_v = int(face_h_px * 0.12)
    pad_h = int(face_w_px * 0.15)

    x1 = max(0, int(cheek_l_pt[0]) - pad_h)
    x2 = min(w, int(cheek_r_pt[0]) + pad_h)
    y1 = max(0, int(forehead_pt[1]) - pad_v)
    y2 = min(h, int(chin_pt[1]) + pad_v)

    if x2 - x1 < 10 or y2 - y1 < 10:
        # Fallback — простой кроп по всем точкам
        xs = [lm[i].x*w for i in range(len(lm))]
        ys = [lm[i].y*h for i in range(len(lm))]
        x1, x2 = max(0, int(min(xs))-10), min(w, int(max(xs))+10)
        y1, y2 = max(0, int(min(ys))-10), min(h, int(max(ys))+10)

    face_crop = rotated[y1:y2, x1:x2]

    # Ресайз до 380x380
    face_crop = cv2.resize(face_crop, (380, 380))

    # Нормализация яркости через CLAHE
    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    face_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


def _ensemble(cnn_scores: dict, geo_scores: dict,
              cnn_weight: float = 0.70, geo_weight: float = 0.30) -> Tuple[str, float]:
    """
    Объединяем CNN и геометрику с весами.
    CNN знает 5 классов, геометрика знает 7 (включая rect и triangle).
    """
    all_shapes = set(list(cnn_scores.keys()) + list(geo_scores.keys()))
    combined = {}
    for shape in all_shapes:
        cnn_s = cnn_scores.get(shape, 0.0)
        geo_s = geo_scores.get(shape, 0.0)
        # Если CNN не знает класс (rect, triangle) — геометрика с полным весом
        if shape not in cnn_scores:
            combined[shape] = geo_s * geo_weight
        else:
            combined[shape] = cnn_s * cnn_weight + geo_s * geo_weight

    sorted_scores = sorted(combined.items(), key=lambda x: -x[1])
    best_name,   best_score   = sorted_scores[0]
    second_name, second_score = sorted_scores[1]

    HYBRID_THRESHOLD = 0.80
    if second_score / (best_score + 1e-9) > HYBRID_THRESHOLD:
        pair = tuple(sorted([best_name, second_name]))
        hybrid = f"{pair[0]}-{pair[1]}"
        if hybrid not in FACE_SHAPE_MAP:
            hybrid = f"{pair[1]}-{pair[0]}"
        if hybrid not in FACE_SHAPE_MAP:
            hybrid = best_name
        confidence = round(min(0.95, (best_score + second_score) * 1.1), 3)
        return hybrid, confidence

    total = sum(combined.values()) or 1.0
    confidence = round(min(0.95, best_score / total * 1.8), 3)
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
        base_options=base_options, running_mode=RunningMode.IMAGE, num_faces=1)

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

    # Геометрические скоры
    geo_scores = classify_face_shape(m)

    # Предобработка и CNN
    try:
        from cnn_classifier import classify_with_cnn
        face_bytes = _preprocess_face(img_bgr, lm, w, h)
        cnn_result = classify_with_cnn(face_bytes)
        cnn_scores = cnn_result.get("all_scores", {})
        # Ансамбль
        shape, confidence = _ensemble(cnn_scores, geo_scores)
    except Exception as e:
        print(f"CNN недоступна, используем геометрику: {e}")
        # Fallback — только геометрика
        sorted_geo = sorted(geo_scores.items(), key=lambda x: -x[1])
        best, best_s = sorted_geo[0]
        second, second_s = sorted_geo[1]
        if second_s / (best_s + 1e-9) > 0.80:
            pair = tuple(sorted([best, second]))
            hybrid = f"{pair[0]}-{pair[1]}"
            shape = hybrid if hybrid in FACE_SHAPE_MAP else best
            confidence = round(min(0.95, (best_s + second_s) * 1.1), 3)
        else:
            shape = best
            confidence = round(min(0.95, best_s * 2.2), 3)

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