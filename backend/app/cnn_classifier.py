import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

DATASET_PATH  = r"C:\dataset"
MODEL_SAVE_PATH = r"C:\models\face_cnn_custom.pth"
LEARNING_RATE = 0.00005
BATCH_SIZE    = 32
EPOCHS        = 40
IMG_SIZE      = 224
DROPOUT       = 0.3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def build_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Замораживаем только первые 5 блоков из 8
    for i, block in enumerate(model.features):
        if i < 5:
            for param in block.parameters():
                param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model.to(DEVICE)

def train():
    print("=== СТАРТ ОБУЧЕНИЯ ===")
    print(f"Устройство: {DEVICE}")

    train_path = os.path.join(DATASET_PATH, "train")
    val_path   = os.path.join(DATASET_PATH, "val")

    if not os.path.exists(train_path):
        print(f"ОШИБКА: папка не найдена: {train_path}")
        return

    train_ds = ImageFolder(train_path, transform=train_transform)
    val_ds   = ImageFolder(val_path,   transform=val_transform)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Классы: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model     = build_model(len(train_ds.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.3
    )

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()

        train_acc  = train_correct / len(train_ds)
        train_loss = train_loss    / len(train_ds)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out       = model(imgs)
                loss      = criterion(out, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()

        val_acc  = val_correct / len(val_ds)
        val_loss = val_loss    / len(val_ds)
        scheduler.step(val_loss)

        print(f"Эпоха {epoch:02d}/{EPOCHS} | "
              f"train={train_loss:.4f} acc={train_acc:.1%} | "
              f"val={val_loss:.4f} acc={val_acc:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes":     train_ds.classes,
            }, MODEL_SAVE_PATH)
            print(f"  Сохранено — val_acc={val_acc:.1%}")

    print(f"\nГотово. Лучшая точность: {best_val_acc:.1%}")
    print(f"Модель сохранена: {MODEL_SAVE_PATH}")

# ВОТ СЮДА — без отступов, на уровне корня файла
if __name__ == "__main__":
    train()

_cnn_model   = None
_cnn_classes = None

def _load_model():
    global _cnn_model, _cnn_classes
    if _cnn_model is None:
        checkpoint   = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        _cnn_classes = checkpoint["classes"]
        _cnn_model   = build_model(len(_cnn_classes))
        _cnn_model.load_state_dict(checkpoint["model_state"])
        _cnn_model.eval()
    return _cnn_model, _cnn_classes

def classify_with_cnn(image_bytes: bytes) -> dict:
    import io
    from PIL import Image

    model, classes = _load_model()

    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    return {
        "all_scores": {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}
    }