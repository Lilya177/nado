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
HYBRID_THRESHOLD = 0.92

# ─────────────────────────────────────────────
# Допустимые диапазоны замеров для чистого лица
# Если значение вылетает за границы — скорее всего помеха (наушники, волосы, рука)
# ─────────────────────────────────────────────
MEASUREMENT_BOUNDS = {
    # (min, max)
    "face_ratio":    (0.85, 2.20),   # слишком низкий = лицо обрезано сверху/снизу
    "jaw_ratio":     (0.50, 1.10),   # слишком высокий = что-то раздувает щёки/уши
    "jaw_ang_ratio": (0.55, 1.15),
    "brow_ratio":    (0.55, 1.10),
    "eye_ratio":     (0.35, 0.85),
    "temple_ratio":  (0.55, 1.15),
    "lower_ratio":   (0.25, 0.65),
    "taper":         (0.50, 1.40),
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
    """Классифицирует форму лица — фронтальный кадр, все метрики надёжны."""
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


def classify_face_shape_side_only(m: dict) -> dict:
    """Классифицирует форму лица — боковой кадр, только надёжные метрики."""
    jr  = m["jaw_ratio"]
    jar = m["jaw_ang_ratio"]
    tp  = m["taper"]

    def g(val, center, sigma):
        return float(np.exp(-((val - center)**2) / (2 * sigma**2)))

    scores = {
        "oval":     g(tp, 0.90, 0.10) * 0.50 + g(jr, 0.76, 0.09) * 0.50,
        "round":    g(tp, 0.98, 0.08) * 0.50 + g(jr, 0.85, 0.08) * 0.50,
        "square":   g(jar, 0.90, 0.07) * 0.60 + g(tp, 1.00, 0.07) * 0.40,
        "rect":     g(jar, 0.88, 0.07) * 0.60 + g(tp, 0.95, 0.08) * 0.40,
        "heart":    g(tp, 0.72, 0.08) * 0.70 + g(jr, 0.65, 0.09) * 0.30,
        "triangle": g(tp, 1.20, 0.10) * 0.70 + g(jar, 0.95, 0.08) * 0.30,
        "diamond":  g(tp, 0.82, 0.08) * 0.60 + g(jr, 0.68, 0.08) * 0.40,
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
    if is_side:
        geo_scores = classify_face_shape_side_only(measurements)
    else:
        geo_scores = classify_face_shape(measurements)

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
            x1   = max(0, int(l[0] - 20))
            x2   = min(iw, int(r[0] + 20))
            y1   = max(0, int(f[1] - 20))
            y2   = min(ih, int(c[1] + 20))
            crop = cv2.resize(rot[y1:y2, x1:x2], (380, 380))
            lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
            return buf.tobytes()

        cnn_res                 = classify_with_cnn(_preprocess(img_bgr, lm, w, h))
        shape, conf, all_scores = _ensemble(cnn_res.get("all_scores", {}), geo_scores)

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

    weights     = [1.0, 0.35, 0.35]
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

    # Взвешенное суммирование
    final_scores: dict[str, float] = {}
    total_weight = sum(r["weight"] for r in results)

    for res in results:
        for shape, score in res["all_scores"].items():
            final_scores[shape] = final_scores.get(shape, 0.0) + score * res["weight"]

    for k in final_scores:
        final_scores[k] = final_scores[k] / total_weight

    # Temperature scaling
    temperature = 0.5
    sharpened   = {k: v ** (1.0 / temperature) for k, v in final_scores.items()}
    total_sharp = sum(sharpened.values()) or 1.0
    final_scores = {k: round(v / total_sharp, 4) for k, v in sharpened.items()}

    sorted_shapes = sorted(final_scores.items(), key=lambda x: -x[1])
    best_shape    = sorted_shapes[0][0]
    best_conf     = sorted_shapes[0][1]

    # Голосование как tiebreaker при близких scores
    if len(sorted_shapes) > 1:
        second_conf = sorted_shapes[1][1]
        if second_conf / (best_conf + 1e-9) > 0.75:
            majority = _majority_vote([r["face_shape"] for r in results])
            if majority and majority in final_scores:
                print(f"[Голосование] tiebreaker: {majority}")
                best_shape = majority
                best_conf  = final_scores[majority]

    # Проверка гибрида
    sorted_shapes = sorted(final_scores.items(), key=lambda x: -x[1])
    if len(sorted_shapes) > 1:
        top_shape    = sorted_shapes[0][0]
        top_conf     = sorted_shapes[0][1]
        second_shape = sorted_shapes[1][0]
        second_conf  = sorted_shapes[1][1]
        if second_conf / (top_conf + 1e-9) > HYBRID_THRESHOLD:
            pair   = tuple(sorted([top_shape, second_shape]))
            hybrid = f"{pair[0]}-{pair[1]}"
            if hybrid in FACE_SHAPE_MAP:
                best_shape = hybrid

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