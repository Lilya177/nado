import cv2
import numpy as np
import urllib.request
import os
from typing import Optional

MODEL_PATH = r"C:\models\face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

FACE_SHAPE_MAP = {
    "oval":         {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],    "shape_codes": ["aviator", "round", "cat"]},
    "round":        {"recommended_shapes": ["Кошачий глаз", "Авиаторы", "Вайфареры"],  "shape_codes": ["cat", "aviator", "wayfarer"]},
    "square":       {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],    "shape_codes": ["aviator", "round", "cat"]},
    "rect":         {"recommended_shapes": ["Авиаторы", "Круглые"],                     "shape_codes": ["aviator", "round"]},
    "heart":        {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],       "shape_codes": ["aviator", "wayfarer", "round"]},
    "triangle":     {"recommended_shapes": ["Авиаторы", "Стрекозы", "Круглые"],        "shape_codes": ["aviator", "round"]},
    "diamond":      {"recommended_shapes": ["Авиаторы", "Овальные", "Кошачий глаз"],   "shape_codes": ["aviator", "oval", "cat"]},
    "oval-square":  {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Круглые"],    "shape_codes": ["aviator", "cat", "round"]},
    "oval-round":   {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Вайфареры"],  "shape_codes": ["aviator", "cat", "wayfarer"]},
    "oval-heart":   {"recommended_shapes": ["Авиаторы", "Вайфареры", "Кошачий глаз"],  "shape_codes": ["aviator", "wayfarer", "cat"]},
    "oval-rect":    {"recommended_shapes": ["Авиаторы", "Круглые"],                     "shape_codes": ["aviator", "round"]},
    "square-round": {"recommended_shapes": ["Авиаторы", "Кошачий глаз", "Круглые"],    "shape_codes": ["aviator", "cat", "round"]},
    "round-square": {"recommended_shapes": ["Авиаторы", "Кошачий глаз"],                "shape_codes": ["aviator", "cat"]},
    "heart-oval":   {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],       "shape_codes": ["aviator", "wayfarer", "round"]},
    "diamond-oval": {"recommended_shapes": ["Авиаторы", "Овальные", "Кошачий глаз"],   "shape_codes": ["aviator", "oval", "cat"]},
}

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Скачиваем модель MediaPipe...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Готово.")

def _dist(a, b) -> float:
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))

def _get_landmark_px(lm, idx, w, h):
    return (lm[idx].x * w, lm[idx].y * h)

def _make_rotator(lm, w, h):
    """Нормализует угол наклона головы для точных замеров."""
    eye_l = _get_landmark_px(lm, 33, w, h)
    eye_r = _get_landmark_px(lm, 263, w, h)
    angle = np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    cx, cy = w / 2.0, h / 2.0

    def rot(idx):
        x = lm[idx].x * w - cx
        y = lm[idx].y * h - cy
        return x * cos_a - y * sin_a + cx, x * sin_a + y * cos_a + cy

    return rot

def extract_measurements(lm, w, h) -> Optional[dict]:
    """Извлекает геометрические замеры лица с нормализацией угла наклона."""
    rot = _make_rotator(lm, w, h)

    cheek_w   = _dist(rot(234),  rot(454))
    jaw_w     = _dist(rot(172),  rot(397))
    jaw_ang_w = _dist(rot(58),   rot(288))
    brow_w    = _dist(rot(70),   rot(300))
    eye_w     = _dist(rot(33),   rot(263))
    temple_w  = _dist(rot(162),  rot(389))
    face_h    = _dist(rot(9),    rot(152)) * 1.60
    lower_h   = _dist(rot(1),    rot(152))

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
    """Классифицирует форму лица по геометрическим замерам через гауссовы функции."""
    fr  = m["face_ratio"]
    jr  = m["jaw_ratio"]
    jar = m["jaw_ang_ratio"]
    tp  = m["taper"]
    lr  = m["lower_ratio"]
    br  = m["brow_ratio"]
    er  = m["eye_ratio"]

    def g(val, center, sigma):
        return float(np.exp(-((val - center)**2) / (2 * sigma**2)))

    scores = {
        "oval":     g(fr,1.38,0.16)*0.35 + g(jr,0.76,0.08)*0.20 + g(tp,0.90,0.09)*0.20 + g(lr,0.44,0.06)*0.10,
        "round":    g(fr,1.05,0.10)*0.35 + g(jr,0.85,0.07)*0.25 + g(tp,0.98,0.07)*0.25 + g(lr,0.48,0.06)*0.15,
        "square":   g(fr,1.20,0.13)*0.25 + g(jar,0.90,0.07)*0.30 + g(tp,1.00,0.07)*0.25 + g(jr,0.85,0.07)*0.20,
        "rect":     g(fr,1.75,0.12)*0.55 + g(jr,0.82,0.08)*0.20 + g(br,0.88,0.08)*0.10,
        "heart":    g(br,0.95,0.07)*0.30  + g(tp,0.72,0.08)*0.35 + g(jr,0.65,0.08)*0.25,
        "triangle": g(tp,1.20,0.10)*0.40  + g(jar,0.95,0.07)*0.30 + g(fr,1.20,0.12)*0.10,
        "diamond":  g(er,0.88,0.06)*0.30  + g(tp,0.82,0.07)*0.25 + g(jr,0.68,0.07)*0.25,
    }

    total = sum(scores.values()) or 1.0
    return {k: round(v / total, 4) for k, v in scores.items()}

def _ensemble(cnn_scores: dict, geo_scores: dict):
    """Смешивает результаты CNN (70%) и геометрии (30%)."""
    all_keys = set(list(cnn_scores.keys()) + list(geo_scores.keys()))
    combined = {
        k: cnn_scores.get(k, 0.0) * 0.7 + geo_scores.get(k, 0.0) * 0.3
        for k in all_keys
    }

    sorted_res = sorted(combined.items(), key=lambda x: -x[1])
    best_name, best_score = sorted_res[0]

    if len(sorted_res) > 1:
        second_name, second_score = sorted_res[1]
        if second_score / (best_score + 1e-9) > 0.85:
            pair    = tuple(sorted([best_name, second_name]))
            hybrid  = f"{pair[0]}-{pair[1]}"
            if hybrid in FACE_SHAPE_MAP:
                return hybrid, round(best_score * 1.1, 3), combined

    return best_name, round(best_score, 3), combined

def _run_mediapipe(img_bgr: np.ndarray):
    """
    Запускает MediaPipe FaceLandmarker и возвращает landmarks.
    Совместим с любой версией mediapipe 0.10+
    """
    _ensure_model()

    import mediapipe as mp

    # Универсальный способ получить BaseOptions — работает на всех версиях
    try:
        BaseOptions = mp.tasks.BaseOptions
    except AttributeError:
        from mediapipe.tasks.python.core.base_options import BaseOptions

    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )

    h, w = img_bgr.shape[:2]
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with vision.FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

    if not result.face_landmarks:
        return None, w, h

    return result.face_landmarks[0], w, h

def analyze_face_image(image_bytes: bytes) -> dict:
    """
    Анализирует ОДНО изображение.
    Возвращает форму лица, уверенность и все промежуточные данные.
    """
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"error": "Не удалось декодировать изображение", "face_shape": None}

    lm, w, h = _run_mediapipe(img_bgr)

    if lm is None:
        return {"error": "Лицо не найдено на изображении", "face_shape": None}

    measurements = extract_measurements(lm, w, h)
    if not measurements:
        return {"error": "Не удалось рассчитать замеры лица", "face_shape": None}

    geo_scores = classify_face_shape(measurements)

    # Пробуем подключить CNN — если нет, работаем только на геометрии
    try:
        from cnn_classifier import classify_with_cnn

        def _preprocess(img, landmarks, iw, ih) -> bytes:
            eye_l = (landmarks[33].x * iw,  landmarks[33].y * ih)
            eye_r = (landmarks[263].x * iw, landmarks[263].y * ih)
            angle = np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))
            cx    = int((eye_l[0] + eye_r[0]) / 2)
            cy    = int((eye_l[1] + eye_r[1]) / 2)
            M     = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rot   = cv2.warpAffine(img, M, (iw, ih))

            def tr(idx):
                p = np.array([[[landmarks[idx].x * iw, landmarks[idx].y * ih]]], dtype=np.float32)
                return cv2.transform(p, M)[0][0]

            f, c, l, r = tr(10), tr(152), tr(234), tr(454)
            x1 = max(0, int(l[0] - 20))
            x2 = min(iw, int(r[0] + 20))
            y1 = max(0, int(f[1] - 20))
            y2 = min(ih, int(c[1] + 20))
            crop = cv2.resize(rot[y1:y2, x1:x2], (380, 380))
            lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
            return buf.tobytes()

        cnn_res                    = classify_with_cnn(_preprocess(img_bgr, lm, w, h))
        shape, conf, all_scores    = _ensemble(cnn_res.get("all_scores", {}), geo_scores)

    except Exception as e:
        print(f"[CNN] недоступна, используем геометрию: {e}")
        all_scores = geo_scores
        shape      = max(geo_scores, key=geo_scores.get)
        conf       = geo_scores[shape]

    return {
        "face_shape":   shape,
        "confidence":   round(float(conf), 3),
        "all_scores":   all_scores,
        "measurements": measurements,
        "error":        None,
    }

def analyze_multi_frames(frames: list[bytes]) -> dict:
    """
    Принимает список байт-изображений (прямо, влево, вправо).
    Усредняет результаты и возвращает финальную форму лица.

    frames[0] = смотрит прямо
    frames[1] = повернул влево
    frames[2] = повернул вправо
    """
    if not frames:
        return {"error": "Нет кадров для анализа", "face_shape": None}

    # Веса для каждого угла: прямо важнее всего
    weights = [1.0, 0.75, 0.75]

    results     = []
    angle_votes = {}
    angle_names = ["front", "left", "right"]

    for i, frame_bytes in enumerate(frames):
        res = analyze_face_image(frame_bytes)

        if res.get("error") or not res.get("face_shape"):
            print(f"[Frame {i}] пропущен: {res.get('error')}")
            continue

        weight = weights[i] if i < len(weights) else 0.75
        results.append({
            "all_scores":  res["all_scores"],
            "face_shape":  res["face_shape"],
            "confidence":  res["confidence"],
            "weight":      weight,
        })
        angle_name              = angle_names[i] if i < len(angle_names) else f"frame_{i}"
        angle_votes[angle_name] = res["face_shape"]

    if not results:
        return {"error": "Лицо не обнаружено ни на одном кадре", "face_shape": None}

    # Взвешенное суммирование по всем классам
    final_scores: dict[str, float] = {}
    total_weight = sum(r["weight"] for r in results)

    for res in results:
        for shape, score in res["all_scores"].items():
            final_scores[shape] = final_scores.get(shape, 0.0) + score * res["weight"]

    # Нормализуем
    for k in final_scores:
        final_scores[k] = round(final_scores[k] / total_weight, 4)

    sorted_shapes  = sorted(final_scores.items(), key=lambda x: -x[1])
    best_shape     = sorted_shapes[0][0]
    best_conf      = sorted_shapes[0][1]

    # Проверяем нужен ли гибрид
    if len(sorted_shapes) > 1:
        second_shape = sorted_shapes[1][0]
        second_conf  = sorted_shapes[1][1]
        if second_conf / (best_conf + 1e-9) > 0.82:
            pair   = tuple(sorted([best_shape, second_shape]))
            hybrid = f"{pair[0]}-{pair[1]}"
            if hybrid in FACE_SHAPE_MAP:
                best_shape = hybrid

    # Рекомендации по оправам
    info = FACE_SHAPE_MAP.get(best_shape, FACE_SHAPE_MAP.get(best_shape.split("-")[0], {}))

    return {
        "face_shape":          best_shape,
        "confidence":          best_conf,
        "all_scores":          final_scores,
        "angle_votes":         angle_votes,
        "frames_analyzed":     len(results),
        "recommended_shapes":  info.get("recommended_shapes", []),
        "shape_codes":         info.get("shape_codes", []),
        "error":               None,
    }