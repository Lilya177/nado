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

# Порог гибрида — создаётся только если два класса реально неразличимы
HYBRID_THRESHOLD = 0.98

# ─────────────────────────────────────────────
# Допустимые диапазоны замеров для чистого лица
# Если значение вылетает за границы — скорее всего помеха (наушники, волосы, рука)
# ─────────────────────────────────────────────
MEASUREMENT_BOUNDS = {
    # (min, max)
    "face_ratio":    (0.70, 2.50),
    "jaw_ratio":     (0.40, 1.30),
    "jaw_ang_ratio": (0.40, 1.30),
    "brow_ratio":    (0.40, 1.30),
    "eye_ratio":     (0.20, 1.00),
    "temple_ratio":  (0.40, 1.30),
    "lower_ratio":   (0.15, 0.75),
    "taper":         (0.40, 1.60),
}

# Ключевые точки, видимость которых критична.
# Индексы MediaPipe: 234/454 = края щёк (перекрываются наушниками!),
# 70/300 = брови, 9 = лоб, 152 = подбородок
CRITICAL_LANDMARKS = [9, 70, 152, 162, 172, 234, 300, 389, 397, 454]


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


def validate_landmark_visibility(lm, w, h) -> tuple[bool, str]:
    """
    Проверяет видимость критических точек лица.
    Возвращает (ok, причина_отказа).

    Точки считаются подозрительными если:
    - находятся за пределами кадра
    - у MediaPipe есть поле visibility < 0.5 (если доступно)
    """
    for idx in CRITICAL_LANDMARKS:
        point = lm[idx]
        x_px  = point.x * w
        y_px  = point.y * h

        # Точка вылетела за пределы кадра
        if x_px < 0 or x_px > w or y_px < 0 or y_px > h:
            return False, f"Точка {idx} вне кадра (x={x_px:.0f}, y={y_px:.0f})"

        # MediaPipe даёт visibility если запрошено — проверяем если есть
        if hasattr(point, "visibility") and point.visibility is not None:
            if point.visibility < 0.5:
                return False, f"Точка {idx} перекрыта (visibility={point.visibility:.2f})"

    return True, ""


def validate_measurements(m: dict) -> tuple[bool, str]:
    """
    Проверяет что все замеры попадают в допустимые диапазоны.
    Если нет — скорее всего помеха: наушники, волосы, рука, плохой угол.
    Возвращает (ok, причина_отказа).
    """
    for key, (lo, hi) in MEASUREMENT_BOUNDS.items():
        val = m.get(key)
        if val is None:
            continue
        if not (lo <= val <= hi):
            return False, f"{key}={val:.3f} вне допустимого диапазона [{lo}, {hi}]"
    return True, ""


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
    """Определяет форму лица по face_ratio с мягким распределением."""
    fr = m["face_ratio"]

    centers = {
        "round": 1.00, "square": 1.15, "oval": 1.45,
        "rect": 1.90, "heart": 1.10, "triangle": 1.20, "diamond": 1.25,
    }

    raw = {}
    for shape, center in centers.items():
        dist = abs(fr - center)
        raw[shape] = max(0.0, 1.0 - dist * 2.5)

    sorted_shapes = sorted(raw.items(), key=lambda x: -x[1])
    winner, win_score = sorted_shapes[0]
    is_hybrid = False

    if len(sorted_shapes) > 1:
        second_score = sorted_shapes[1][1]
        if win_score > 0 and second_score / win_score > 0.40:
            pair = tuple(sorted([sorted_shapes[0][0], sorted_shapes[1][0]]))
            hybrid = f"{pair[0]}-{pair[1]}"
            if hybrid in FACE_SHAPE_MAP:
                winner = hybrid
                is_hybrid = True

    scores = {k: 0.0 for k in raw}
    total_raw = sum(max(0.0, v) for v in raw.values())
    if total_raw > 0:
        for k in raw:
            scores[k] = round(max(0.0, raw[k]) / total_raw, 4)
    else:
        scores[winner] = 1.0

    return scores


def classify_face_shape_side_only(m: dict) -> dict:
    """Боковой кадр — всегда возвращает равномерное распределение."""
    # Боковые кадры неточны, не влияем ими на итог
    n = 7
    return {s: round(1.0/n, 4) for s in ["oval","round","square","rect","heart","triangle","diamond"]}


def _ensemble(cnn_scores: dict, geo_scores: dict):
    """Смешивает результаты геометрии (70%) и CNN (30%). CNN часто ошибается."""
    all_keys = set(list(cnn_scores.keys()) + list(geo_scores.keys()))
    combined = {
        k: cnn_scores.get(k, 0.0) * 0.3 + geo_scores.get(k, 0.0) * 0.7
        for k in all_keys
    }

    sorted_res = sorted(combined.items(), key=lambda x: -x[1])
    best_name, best_score = sorted_res[0]

    if len(sorted_res) > 1:
        second_name, second_score = sorted_res[1]
        if second_score / (best_score + 1e-9) > HYBRID_THRESHOLD:
            pair   = tuple(sorted([best_name, second_name]))
            hybrid = f"{pair[0]}-{pair[1]}"
            if hybrid in FACE_SHAPE_MAP:
                return hybrid, round(best_score * 1.1, 3), combined

    return best_name, round(best_score, 3), combined


def _run_mediapipe(img_bgr: np.ndarray):
    """Запускает MediaPipe FaceLandmarker и возвращает landmarks."""
    _ensure_model()

    import mediapipe as mp

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
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with vision.FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

    if not result.face_landmarks:
        return None, w, h

    return result.face_landmarks[0], w, h


def analyze_face_image(image_bytes: bytes, is_side: bool = False) -> dict:
    """
    Анализирует одно изображение.

    Возвращает dict. Если что-то не так — поле "retry_reason" содержит
    человекочитаемую причину почему нужно переснять кадр.
    """
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {
            "error":        "Не удалось декодировать изображение",
            "retry_reason": "Изображение повреждено, попробуйте ещё раз",
            "face_shape":   None,
        }

    lm, w, h = _run_mediapipe(img_bgr)

    if lm is None:
        return {
            "error":        "Лицо не найдено",
            "retry_reason": "Лицо не обнаружено — встаньте ближе к камере и убедитесь что лицо хорошо освещено",
            "face_shape":   None,
        }

    # Проверка видимости ключевых точек (наушники, волосы, рука)
    landmarks_ok, landmark_reason = validate_landmark_visibility(lm, w, h)
    if not landmarks_ok:
        print(f"[Валидация landmarks] {landmark_reason}")
        return {
            "error":        f"Ключевые точки лица перекрыты: {landmark_reason}",
            "retry_reason": "Уберите наушники, волосы и руки от лица и повторите попытку",
            "face_shape":   None,
        }

    measurements = extract_measurements(lm, w, h)

    if not measurements:
        return {
            "error":        "Не удалось рассчитать замеры лица",
            "retry_reason": "Не удалось считать лицо — смотрите прямо в камеру",
            "face_shape":   None,
        }

    # Проверка диапазонов замеров — ловим искажения от помех
    meas_ok, meas_reason = validate_measurements(measurements)
    if not meas_ok:
        print(f"[Валидация замеров] {meas_reason}")
        # Определяем конкретную причину для пользователя
        if "jaw_ratio" in meas_reason or "temple_ratio" in meas_reason:
            user_hint = "Уберите наушники — они искажают контур лица. Повторите попытку"
        elif "face_ratio" in meas_reason:
            user_hint = "Лицо обрезано — отодвиньтесь от камеры чтобы лицо целиком попало в кадр"
        elif "brow_ratio" in meas_reason or "eye_ratio" in meas_reason:
            user_hint = "Уберите волосы с лица и повторите попытку"
        else:
            user_hint = "Что-то мешает анализу — уберите помехи с лица и повторите попытку"

        return {
            "error":        f"Замеры вне допустимого диапазона: {meas_reason}",
            "retry_reason": user_hint,
            "face_shape":   None,
        }

    # Всё чисто — классифицируем
    print(f"[MEASURE] {measurements}")
    if is_side:
        geo_scores = classify_face_shape_side_only(measurements)
    else:
        geo_scores = classify_face_shape(measurements)

    # Только геометрия — CNN ненадёжен (обучен только на анфас)
    all_scores = geo_scores
    shape      = max(geo_scores, key=geo_scores.get)
    conf       = geo_scores[shape]

    return {
        "face_shape":   shape,
        "confidence":   round(float(conf), 3),
        "all_scores":   all_scores,
        "measurements": measurements,
        "retry_reason": None,
        "error":        None,
    }


def _majority_vote(votes: list[str]) -> str | None:
    """Возвращает форму за которую проголосовало большинство кадров."""
    if not votes:
        return None

    flat_votes: list[str] = []
    for v in votes:
        flat_votes.extend(v.split("-"))

    counts: dict[str, int] = {}
    for v in flat_votes:
        counts[v] = counts.get(v, 0) + 1

    best_shape = max(counts, key=counts.__getitem__)
    best_count = counts[best_shape]

    if best_count > len(votes) / 2:
        return best_shape

    return None


def analyze_multi_frames(frames: list[bytes]) -> dict:
    """
    Принимает список байт-изображений (прямо, влево, вправо).

    Если кадр не прошёл валидацию — возвращает retry_reason вместо результата.
    Фронтальный кадр (frames[0]) обязателен — без него анализ не проводится.
    Боковые кадры опциональны — если не прошли валидацию, анализируем без них.

    Возвращает либо результат анализа, либо dict с retry_reason для конкретного кадра.
    """
    if not frames:
        return {"error": "Нет кадров для анализа", "face_shape": None}

    weights     = [1.0, 0.15, 0.15]
    side_flags  = [False, True, True]
    angle_names = ["front", "left", "right"]
    angle_labels = {
        "front": "прямо",
        "left":  "влево",
        "right": "вправо",
    }

    results     = []
    angle_votes = {}
    retry_hints = {}   # накапливаем подсказки по каждому кадру

    for i, frame_bytes in enumerate(frames):
        is_side    = side_flags[i] if i < len(side_flags) else True
        angle_name = angle_names[i] if i < len(angle_names) else f"frame_{i}"
        angle_label = angle_labels.get(angle_name, angle_name)

        res = analyze_face_image(frame_bytes, is_side=is_side)

        if res.get("retry_reason") or res.get("error") or not res.get("face_shape"):
            reason = res.get("retry_reason") or res.get("error") or "Неизвестная ошибка"
            print(f"[Frame {i} / {angle_name}] отклонён: {reason}")
            retry_hints[angle_name] = reason

            # Фронтальный кадр обязателен — без него сразу просим переснять
            if angle_name == "front":
                return {
                    "face_shape":   None,
                    "error":        f"Фронтальный кадр не принят: {reason}",
                    "retry_reason": reason,
                    "retry_frame":  "front",
                    "retry_label":  f"Пожалуйста, повторите съёмку ({angle_label}): {reason}",
                    "frames_analyzed": 0,
                }
            else:
                # Боковой кадр — пропускаем, но продолжаем
                print(f"[Frame {i}] боковой кадр пропущен, анализируем без него")
                continue

        weight = weights[i] if i < len(weights) else 0.35
        results.append({
            "all_scores":   res["all_scores"],
            "face_shape":   res["face_shape"],
            "confidence":   res["confidence"],
            "weight":       weight,
            "measurements": res.get("measurements"),
        })
        angle_votes[angle_name] = res["face_shape"]

    if not results:
        return {
            "face_shape":      None,
            "error":           "Лицо не обнаружено ни на одном кадре",
            "retry_reason":    "Уберите помехи с лица и повторите все три кадра",
            "retry_frame":     "all",
            "retry_label":     "Повторите все три кадра: уберите наушники, волосы и руки от лица",
            "frames_analyzed": 0,
        }

    # Берём результат только с фронтального кадра (он самый точный)
    front_result = results[0] if results else {"all_scores": {"oval": 0.8, "round": 0.2}}
    final_scores = front_result["all_scores"]
    sorted_shapes = sorted(final_scores.items(), key=lambda x: -x[1])
    best_shape    = sorted_shapes[0][0]
    best_conf     = sorted_shapes[0][1]

    print(f"[AI] Итог: {best_shape}={best_conf:.3f}, оценки: {dict(sorted_shapes)}")

    info = FACE_SHAPE_MAP.get(best_shape, FACE_SHAPE_MAP.get(best_shape.split("-")[0], {}))

    return {
        "face_shape":          best_shape,
        "confidence":          best_conf,
        "all_scores":          final_scores,
        "angle_votes":         angle_votes,
        "frames_analyzed":     len(results),
        "skipped_frames":      retry_hints,   # какие кадры были пропущены и почему
        "recommended_shapes":  info.get("recommended_shapes", []),
        "shape_codes":         info.get("shape_codes", []),
        "retry_reason":        None,
        "error":               None,
    }