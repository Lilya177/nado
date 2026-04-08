"""
CNN классификатор формы лица на базе EfficientNetB4 с Hugging Face.
Модель: fahd9999/face_shape_classification (85% accuracy)
При первом запуске скачивается автоматически (~80MB).
"""

from PIL import Image
import io

_pipe = None

LABEL_MAP = {
    "Oval":    "oval",
    "Round":   "round",
    "Square":  "square",
    "Heart":   "heart",
    "Diamond": "diamond",
}

# Гибридные пары которые есть в FACE_SHAPE_MAP
VALID_HYBRIDS = {
    "oval-square", "oval-round", "oval-heart", "oval-rect",
    "square-round", "heart-oval", "round-square", "diamond-oval",
}

def _load_pipeline():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        print("Загружаем CNN модель EfficientNetB4 (~80MB)...")
        _pipe = pipeline(
            "image-classification",
            model="fahd9999/face_shape_classification",
            device=-1,  # CPU. Поменяй на 0 если есть GPU
        )
        print("CNN модель загружена.")
    return _pipe


def classify_with_cnn(image_bytes: bytes) -> dict:
    """
    Классифицирует форму лица через CNN.

    Возвращает:
    {
        "face_shape": "oval" или "oval-square" (гибрид),
        "confidence": 0.82,
        "all_scores": {"oval": 0.45, "square": 0.38, ...},
        "is_hybrid": True/False
    }
    """
    pipe = _load_pipeline()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Кропаем центр если изображение сильно вытянутое — улучшает точность
    w, h = img.size
    if h > w * 1.5:
        # Берём верхние 2/3 изображения где обычно лицо
        img = img.crop((0, 0, w, int(h * 0.7)))

    results = pipe(img, top_k=5)

    # Маппим лейблы
    all_scores = {
        LABEL_MAP.get(r["label"], r["label"].lower()): round(r["score"], 3)
        for r in results
    }

    first  = results[0]
    second = results[1]

    first_shape  = LABEL_MAP.get(first["label"],  first["label"].lower())
    second_shape = LABEL_MAP.get(second["label"], second["label"].lower())

    first_score  = first["score"]
    second_score = second["score"]

    # Порог гибрида: второй класс набрал > 72% от первого
    HYBRID_THRESHOLD = 0.72

    if second_score / first_score > HYBRID_THRESHOLD:
        # Формируем гибрид в алфавитном порядке
        pair = tuple(sorted([first_shape, second_shape]))
        hybrid = f"{pair[0]}-{pair[1]}"

        # Проверяем что такой гибрид есть в нашем маппинге
        if hybrid not in VALID_HYBRIDS:
            # Пробуем обратный порядок
            hybrid_rev = f"{pair[1]}-{pair[0]}"
            if hybrid_rev in VALID_HYBRIDS:
                hybrid = hybrid_rev
            else:
                # Гибрид не предусмотрен — берём чистый первый класс
                return {
                    "face_shape": first_shape,
                    "confidence": round(first_score, 3),
                    "all_scores": all_scores,
                    "is_hybrid": False,
                }

        # Уверенность гибрида — среднее двух лучших
        confidence = round((first_score + second_score) / 2, 3)

        return {
            "face_shape": hybrid,
            "confidence": confidence,
            "all_scores": all_scores,
            "is_hybrid": True,
        }

    return {
        "face_shape": first_shape,
        "confidence": round(first_score, 3),
        "all_scores": all_scores,
        "is_hybrid": False,
    }
