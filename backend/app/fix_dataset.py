import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH   = r"C:\models\face_landmarker.task"
DATASET_PATH = r"C:\dataset"

BaseOptions       = mp.tasks.BaseOptions
FaceLandmarker    = vision.FaceLandmarker
FaceLandmarkerOpt = vision.FaceLandmarkerOptions
RunningMode       = vision.RunningMode

options = FaceLandmarkerOpt(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
)

def fix_rotation(img, detector):
    h, w = img.shape[:2]
    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if not result.face_landmarks:
        return img

    lm    = result.face_landmarks[0]
    eye_l = (lm[33].x * w,  lm[33].y * h)
    eye_r = (lm[263].x * w, lm[263].y * h)
    angle = np.degrees(np.arctan2(
        eye_r[1] - eye_l[1],
        eye_r[0] - eye_l[0]
    ))

    if abs(angle) < 5:
        return img

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


total, fixed, skipped = 0, 0, 0

with FaceLandmarker.create_from_options(options) as detector:
    for split in ["train", "val"]:
        split_path = os.path.join(DATASET_PATH, split)
        if not os.path.exists(split_path):
            print(f"Папка не найдена: {split_path}")
            continue

        for cls in os.listdir(split_path):
            folder = os.path.join(split_path, cls)
            if not os.path.isdir(folder):
                continue

            files = [f for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            print(f"  {split}/{cls} — {len(files)} фото")

            for fname in files:
                path = os.path.join(folder, fname)
                img  = cv2.imread(path)
                if img is None:
                    skipped += 1
                    continue

                out = fix_rotation(img, detector)
                cv2.imwrite(path, out)
                total += 1

                if total % 100 == 0:
                    print(f"    обработано {total}...")

print(f"\nГотово. Всего: {total}, пропущено: {skipped}")