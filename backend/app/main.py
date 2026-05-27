import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import bcrypt
import models
import schemas
from database import SessionLocal, init_db
from auth import create_access_token, decode_token
from deps import get_db, get_current_user, require_user, require_doctor, require_admin
from journal import log_action
from face_analyzer import analyze_face_image, FACE_SHAPE_MAP, analyze_multi_frames

app = FastAPI(title="NOIR VISION API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "noir_vision_ready.html"))


@app.get("/scan")
def scan_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "scan.html"))


# ── Auto Seed ──────────────────────────────────────────

def parse_specs(specs: str):
    """Парсит строку 'Ключ: значение · Ключ: значение' в список атрибутов."""
    parts = [s.strip() for s in specs.split("·")]
    result = []
    for part in parts:
        if ":" in part:
            key, val = part.split(":", 1)
            result.append((key.strip(), val.strip()))
    return result

SEED_DATA = {
    "frames": [
        {"title": "Noir Round 01",    "brand": "NOIR VISION ATELIER", "price": 29000, "shape": "round",    "icon": "🕶️", "desc": "Классическая круглая оправа из ацетата. Диаметр 47 мм, дужки 145 мм.", "specs": "Материал: ацетат · Цвет: чёрный · УФ-защита: UV400"},
        {"title": "Noir Round 02",    "brand": "NOIR VISION ATELIER", "price": 30000, "shape": "round",    "icon": "🕶️", "desc": "Круглая оправа с тонким металлическим ободом. Диаметр 48 мм.", "specs": "Материал: металл + ацетат · Цвет: золото/чёрный · Вес: 18 г"},
        {"title": "Noir Round 03",    "brand": "NOIR VISION ATELIER", "price": 31000, "shape": "round",    "icon": "🕶️", "desc": "Увеличенная круглая оправа для акцентного образа. Диаметр 50 мм.", "specs": "Материал: ацетат · Цвет: тёмно-коричневый · Гибкие дужки"},
        {"title": "Noir Oval 01",     "brand": "NOIR VISION ATELIER", "price": 30000, "shape": "oval",     "icon": "🕶️", "desc": "Овальная оправа с мягкими линиями. Ширина 52 мм, высота 38 мм.", "specs": "Материал: ацетат · Цвет: матовый чёрный · Вес: 20 г"},
        {"title": "Noir Oval 02",     "brand": "NOIR VISION ATELIER", "price": 31000, "shape": "oval",     "icon": "🕶️", "desc": "Вытянутая овальная форма с металлическими дужками. Ширина 54 мм.", "specs": "Материал: комбинированный · Цвет: чёрный/серебро"},
        {"title": "Noir Oval 03",     "brand": "NOIR VISION ATELIER", "price": 32000, "shape": "oval",     "icon": "🕶️", "desc": "Классический овал в итальянском стиле. Ширина 52 мм.", "specs": "Материал: ацетат · Цвет: черепаховый · Дужки 140 мм"},
        {"title": "Noir Square 01",   "brand": "OBSIDIAN FRAMES",     "price": 32000, "shape": "square",   "icon": "🕶️", "desc": "Квадратная оправа с чётко выраженными углами. Ширина 52 мм.", "specs": "Материал: ацетат · Цвет: чёрный матовый · Вес: 22 г"},
        {"title": "Noir Square 02",   "brand": "OBSIDIAN FRAMES",     "price": 33000, "shape": "square",   "icon": "🕶️", "desc": "Массивная квадратная оправа из толстого ацетата. Ширина 54 мм.", "specs": "Материал: ацетат · Цвет: тёмно-серый · Гибкие петли"},
        {"title": "Noir Square 03",   "brand": "OBSIDIAN FRAMES",     "price": 34000, "shape": "square",   "icon": "🕶️", "desc": "Квадратная оправа с металлическими вставками на дужках.", "specs": "Материал: ацетат + металл · Цвет: чёрный/золото"},
        {"title": "Noir Rect 01",     "brand": "STUDIO ECLIPSE",      "price": 33000, "shape": "rect",     "icon": "🕶️", "desc": "Прямоугольная оправа в минималистичном стиле. Ширина 54 мм.", "specs": "Материал: ацетат · Цвет: чёрный · Тонкий обод"},
        {"title": "Noir Rect 02",     "brand": "STUDIO ECLIPSE",      "price": 34000, "shape": "rect",     "icon": "🕶️", "desc": "Широкая прямоугольная оправа. Ширина 56 мм.", "specs": "Материал: ацетат · Цвет: матовый чёрный · Вес: 24 г"},
        {"title": "Noir Rect 03",     "brand": "STUDIO ECLIPSE",      "price": 35000, "shape": "rect",     "icon": "🕶️", "desc": "Прямоугольная металлическая оправа. Элегантный деловой стиль.", "specs": "Материал: металл (титан) · Цвет: серебро · Вес: 14 г"},
        {"title": "Noir Aviator 01",  "brand": "NOIR VISION ATELIER", "price": 36000, "shape": "aviator",  "icon": "✈️", "desc": "Классический силуэт авиатора с тонким металлическим ободом.", "specs": "Материал: металл · Цвет: золото · Линзы: с градиентом"},
        {"title": "Noir Aviator 02",  "brand": "NOIR VISION ATELIER", "price": 37000, "shape": "aviator",  "icon": "✈️", "desc": "Авиатор с двойным ободом. Матовый чёрный.", "specs": "Материал: нержавеющая сталь · Цвет: матовый чёрный"},
        {"title": "Noir Aviator 03",  "brand": "NOIR VISION ATELIER", "price": 38000, "shape": "aviator",  "icon": "✈️", "desc": "Увеличенный силуэт авиатора для яркого образа.", "specs": "Материал: металл · Цвет: серебро/чёрный"},
        {"title": "Noir Cat 01",      "brand": "STUDIO ECLIPSE",      "price": 35000, "shape": "cat",      "icon": "🐱", "desc": "Кошачий глаз с выраженным углом. Женственная форма.", "specs": "Материал: ацетат · Цвет: матовый чёрный"},
        {"title": "Noir Cat 02",      "brand": "STUDIO ECLIPSE",      "price": 36000, "shape": "cat",      "icon": "🐱", "desc": "Изящный кошачий глаз с металлическим декором на углах.", "specs": "Материал: ацетат + металл · Цвет: чёрный/золото"},
        {"title": "Noir Wayfarer 01", "brand": "OBSIDIAN FRAMES",     "price": 34000, "shape": "wayfarer", "icon": "🕶️", "desc": "Классический Вайфарер с массивным ацетатным ободом.", "specs": "Материал: ацетат · Цвет: чёрный · Трапециевидная форма"},
        {"title": "Noir Wayfarer 02", "brand": "OBSIDIAN FRAMES",     "price": 35000, "shape": "wayfarer", "icon": "🕶️", "desc": "Вайфарер с тонкой рамкой. Современная интерпретация классики.", "specs": "Материал: ацетат · Цвет: тёмно-коричневый"},
    ],
    "lenses": [
        {"title": "Distance Standard",       "brand": "NOIR OPTICS",     "price": 2799,  "lens": "dal",   "icon": "🔬", "desc": "Базовые монофокальные линзы для дали.", "specs": "Индекс: 1.50 · УФ-фильтр · Антибликовое покрытие"},
        {"title": "Distance Slim 1.60",      "brand": "NOIR OPTICS",     "price": 3999,  "lens": "dal",   "icon": "🔬", "desc": "Тонкие линзы индекса 1.60 для умеренных диоптрий.", "specs": "Индекс: 1.60 · Лёгкие · Антибликовое покрытие"},
        {"title": "Distance Ultra Thin 1.67","brand": "NOIR OPTICS",     "price": 5499,  "lens": "dal",   "icon": "🔬", "desc": "Ультратонкие линзы 1.67 — минимальная толщина при высоких диоптриях.", "specs": "Индекс: 1.67 · Олеофобное покрытие · Лёгкие"},
        {"title": "Distance Hi-Index 1.74",  "brand": "OBSIDIAN OPTICS", "price": 7999,  "lens": "dal",   "icon": "🔬", "desc": "Премиальные линзы 1.74 для самых высоких рецептов.", "specs": "Индекс: 1.74 · Почти незаметны в оправе"},
        {"title": "Reading Classic",         "brand": "NOIR OPTICS",     "price": 2499,  "lens": "bliz",  "icon": "🔬", "desc": "Классические линзы для близи — чтение, работа с документами.", "specs": "Монофокальные · Для близи · Антибликовое покрытие"},
        {"title": "Reading Slim 1.60",       "brand": "NOIR OPTICS",     "price": 3499,  "lens": "bliz",  "icon": "🔬", "desc": "Тонкие линзы для близи индекса 1.60.", "specs": "Индекс: 1.60 · Комфорт при длительном чтении"},
        {"title": "Office Base",             "brand": "NOIR OPTICS",     "price": 3799,  "lens": "comp",  "icon": "💻", "desc": "Базовые офисные линзы для работы за ПК. 40–70 см.", "specs": "Оптимизированы для экрана · Антибликовые"},
        {"title": "Office Blue Block",       "brand": "NOIR OPTICS",     "price": 4799,  "lens": "comp",  "icon": "💻", "desc": "Офисные линзы с фильтром синего света.", "specs": "Blue Block · Снижают усталость глаз"},
        {"title": "Progressive Entry",       "brand": "NOIR OPTICS",     "price": 6999,  "lens": "prog",  "icon": "🔭", "desc": "Прогрессивные линзы начального уровня.", "specs": "3 зоны зрения · Удобная адаптация"},
        {"title": "Progressive Comfort",     "brand": "NOIR OPTICS",     "price": 8999,  "lens": "prog",  "icon": "🔭", "desc": "Улучшенный дизайн с расширенными боковыми зонами.", "specs": "Меньше периферийных искажений · FreeView"},
        {"title": "Progressive Premium",     "brand": "OBSIDIAN OPTICS", "price": 12999, "lens": "prog",  "icon": "🔭", "desc": "Премиальный прогрессивный дизайн.", "specs": "Минимум аберраций · Для всех дистанций"},
        {"title": "Drive Day",               "brand": "NOIR OPTICS",     "price": 4299,  "lens": "drive", "icon": "🚗", "desc": "Дневные линзы для вождения.", "specs": "Высокая контрастность · Поляризация"},
        {"title": "Drive Night",             "brand": "NOIR OPTICS",     "price": 5499,  "lens": "drive", "icon": "🌙", "desc": "Специальное покрытие для ночного вождения.", "specs": "Подавляет ореолы от фар встречных машин"},
        {"title": "Drive Polarized",         "brand": "OBSIDIAN OPTICS", "price": 6999,  "lens": "drive", "icon": "🚗", "desc": "Поляризационные линзы для вождения.", "specs": "Устраняют блики от асфальта"},
        {"title": "Photochromic Classic",    "brand": "NOIR CARE",       "price": 3999,  "lens": "photo", "icon": "🌤️", "desc": "Базовые фотохромные линзы.", "specs": "Прозрачные в помещении · Адаптируются за 30 сек"},
        {"title": "Photochromic Fast Adapt", "brand": "NOIR CARE",       "price": 5299,  "lens": "photo", "icon": "🌤️", "desc": "Ускоренная адаптация: затемнение за 20 сек.", "specs": "Просветление за 3 мин · Blue Block"},
        {"title": "Sun Classic",             "brand": "NOIR CARE",       "price": 2999,  "lens": "sun",   "icon": "☀️", "desc": "Базовые солнцезащитные линзы с UV400-защитой.", "specs": "Нейтральный серый оттенок · UV400"},
        {"title": "Sun Gradient Noir",       "brand": "STUDIO ECLIPSE",  "price": 4999,  "lens": "sun",   "icon": "☀️", "desc": "Градиентные линзы: тёмные сверху, светлые снизу.", "specs": "Стиль и функциональность · UV400"},
        {"title": "Polarized Gray",          "brand": "NOIR CARE",       "price": 4499,  "lens": "polar", "icon": "🌊", "desc": "Серые поляризационные линзы.", "specs": "Устраняют блики от воды и дороги"},
        {"title": "Blue Light Basic",        "brand": "NOIR CARE",       "price": 1799,  "lens": "blue",  "icon": "📱", "desc": "Базовый фильтр синего света.", "specs": "Снижает нагрузку при работе за ПК"},
        {"title": "Blue Light Premium Slim", "brand": "STUDIO ECLIPSE",  "price": 5499,  "lens": "blue",  "icon": "📱", "desc": "Тонкие линзы 1.67 с фильтром синего света.", "specs": "Комфорт весь день · Индекс 1.67"},
    ],
    "liquids": [
        {"title": "Clean Drops 10 мл",     "brand": "NOIR CARE", "price": 890,  "icon": "💧", "desc": "Увлажняющие капли для контактных линз. Снимают усталость.", "specs": "Объём: 10 мл · Состав: HA 0.1% · Гипоаллергенно"},
        {"title": "Clean Drops 15 мл",     "brand": "NOIR CARE", "price": 1190, "icon": "💧", "desc": "Увлажняющие капли увеличенного объёма.", "specs": "Объём: 15 мл · Состав: HA 0.1% · Гипоаллергенно"},
        {"title": "Pro Solution 100 мл",   "brand": "NOIR CARE", "price": 1490, "icon": "🧴", "desc": "Многофункциональный раствор для мягких линз.", "specs": "Объём: 100 мл · Дезинфекция · Хранение"},
        {"title": "Pro Solution 360 мл",   "brand": "NOIR CARE", "price": 2490, "icon": "🧴", "desc": "Многофункциональный раствор для мягких линз — увеличенный.", "specs": "Объём: 360 мл · Дезинфекция · Хранение"},
        {"title": "Frame Cleaner Spray",   "brand": "NOIR CARE", "price": 990,  "icon": "🌀", "desc": "Спрей для очистки оправ и жёстких оптических линз.", "specs": "Объём: 50 мл · Не оставляет разводов"},
        {"title": "Ultra Clean Mist",      "brand": "NOIR CARE", "price": 3500, "icon": "✨", "desc": "Спрей для быстрой очистки и увлажнения.", "specs": "Объём: 30 мл · Микрофибра в комплекте"},
        {"title": "Sensitive Solution",    "brand": "NOIR CARE", "price": 1890, "icon": "🧴", "desc": "Раствор для чувствительных глаз. Без консервантов.", "specs": "Объём: 200 мл · Без консервантов · Гипоаллергенно"},
    ],
    "cases": [
        {"title": "Stone Case 01",   "brand": "OBSIDIAN CASES", "price": 4500, "icon": "🗃️", "desc": "Жёсткий чехол тёмно-серого цвета. Защита от ударов.", "specs": "Материал: поликарбонат · Застёжка: кнопка · 165×70×40 мм"},
        {"title": "Stone Case 02",   "brand": "OBSIDIAN CASES", "price": 4900, "icon": "🗃️", "desc": "Матовый чёрный футляр с магнитной застёжкой.", "specs": "Материал: поликарбонат · Застёжка: магнит · Мягкая подкладка"},
        {"title": "Leather Fold 01", "brand": "NOIR LEATHER",   "price": 5200, "icon": "👜", "desc": "Плоский кожаный чехол‑конверт. Минималистичный дизайн.", "specs": "Материал: натуральная кожа · Застёжка: кнопка · 170×75 мм"},
        {"title": "Leather Box 02",  "brand": "NOIR LEATHER",   "price": 5500, "icon": "👜", "desc": "Объёмный кожаный футляр с тиснением логотипа.", "specs": "Материал: натуральная кожа · Застёжка: металл · Вес: 95 г"},
        {"title": "Concrete Box 01", "brand": "STUDIO ECLIPSE", "price": 5800, "icon": "🧱", "desc": "Дизайнерский чехол с бетонной текстурой. Ограниченная серия.", "specs": "Материал: полимер бетон-стиль · Магнит · 170×70×42 мм"},
    ],
}

def seed_products(db: Session):
    if db.query(models.Product).count() > 0:
        return

    product_id_map = {}
    for tab_name, product_list in SEED_DATA.items():
        ptype = tab_name if tab_name in ("frame", "lens", "liquid", "case") else "accessory"
        if tab_name == "frames": ptype = "frame"
        elif tab_name == "lenses": ptype = "lens"
        elif tab_name == "liquids": ptype = "liquid"
        elif tab_name == "cases": ptype = "case"

        for item in product_list:
            p = models.Product(
                title=item["title"],
                brand=item["brand"],
                type=ptype,
                shape=item.get("shape") or item.get("lens"),
                description=item["desc"],
                price=item["price"],
                stock=999,
                in_stock=True,
                image_url=item.get("icon"),
            )
            db.add(p)
            db.flush()

            for key, val in parse_specs(item.get("specs", "")):
                db.add(models.Attribute(product_id=p.id, key=key, value=val))

    db.commit()


@app.on_event("startup")
def on_startup():
    init_db()
    db = SessionLocal()
    try:
        seed_products(db)
    finally:
        db.close()


# ───── AUTH /api/auth ──────────────────────────────────

@app.post("/api/auth/register", response_model=schemas.TokenResponse, tags=["Auth"])
def register(body: schemas.UserCreate, request: Request, db: Session = Depends(get_db)):
    errors = {}
    if db.query(models.User).filter(models.User.nickname == body.nickname).first():
        errors["nickname"] = "Этот никнейм уже используется"
    if db.query(models.User).filter(models.User.email == body.email).first():
        errors["email"] = "Пользователь с такой почтой уже существует"
    if db.query(models.User).filter(models.User.phone == body.phone).first():
        errors["phone"] = "Этот номер телефона уже зарегистрирован"
    if errors:
        raise HTTPException(409, detail={"errors": errors})
    hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    user = models.User(
        nickname=body.nickname,
        phone=body.phone,
        email=body.email,
        hashed_password=hashed,
        is_active=True,
        role=models.UserRole.client
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return schemas.TokenResponse(access_token=token, user=user)


@app.post("/api/auth/token", response_model=schemas.TokenResponse, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    username = form.username.strip()
    digits = re.sub(r'\D', '', username)
    is_phone = 11 == len(digits) and (digits.startswith('7') or digits.startswith('8'))
    if is_phone:
        username = '+' + digits if digits.startswith('7') else '+7' + digits[1:]
    user = (db.query(models.User).filter(
        (models.User.email == username) | (models.User.phone == username)
    ).first())
    if not user or not bcrypt.checkpw(form.password.encode(), user.hashed_password.encode()):
        raise HTTPException(401, "Неверный email/телефон или пароль")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return schemas.TokenResponse(access_token=token, user=user)


# ───── USERS ──────────────────────────────────────────

@app.get("/api/auth/me", response_model=schemas.UserOut, tags=["Auth"])
def get_me(current_user: models.User = Depends(get_current_user)):
    return current_user


# ───── PRODUCTS ────────────────────────────────────────

TYPE_MAP = {
    "frames": "frame", "lenses": "lens", "liquids": "liquid", "cases": "case",
    "frame": "frame", "lens": "lens", "liquid": "liquid", "case": "case", "accessory": "accessory",
}

@app.get("/api/products", response_model=schemas.ProductPaginationResponse, tags=["Products"])
def list_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    type: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    price_min: Optional[float] = Query(None, ge=0),
    price_max: Optional[float] = Query(None, ge=0),
    shape: Optional[str] = Query(None),
    lens_type: Optional[str] = Query(None),
    in_stock: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(models.Product)

    db_type = TYPE_MAP.get(type) if type else None
    if db_type:
        q = q.filter(models.Product.type == db_type)
    if brand:
        q = q.filter(models.Product.brand.ilike(f"%{brand}%"))
    if search:
        like = f"%{search}%"
        q = q.filter(
            models.Product.title.ilike(like) |
            models.Product.description.ilike(like) |
            models.Product.brand.ilike(like)
        )
    if price_min is not None:
        q = q.filter(models.Product.price >= price_min)
    if price_max is not None:
        q = q.filter(models.Product.price <= price_max)
    if shape:
        q = q.filter(models.Product.shape == shape)
    if lens_type:
        q = q.filter(models.Product.shape == lens_type)
    if in_stock is not None:
        q = q.filter(models.Product.in_stock == in_stock)

    total = q.count()
    items = q.offset((page - 1) * limit).limit(limit).all()

    return schemas.ProductPaginationResponse(
        items=items, total=total, page=page, limit=limit
    )


@app.get("/api/products/{product_id}", response_model=schemas.ProductOut, tags=["Products"])
def get_product(product_id: int, db: Session = Depends(get_db)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    return p


# ───── ADMIN PRODUCTS ──────────────────────────────────

@app.post("/admin/products", tags=["Admin"])
def create_product(body: schemas.ProductCreate, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = models.Product(**body.dict())
    db.add(p); db.commit(); db.refresh(p)
    return p


@app.put("/admin/products/{product_id}", tags=["Admin"])
def update_product(product_id: int, body: schemas.ProductCreate, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    for k, v in body.dict().items():
        setattr(p, k, v)
    db.commit(); db.refresh(p)
    return p


@app.delete("/admin/products/{product_id}", tags=["Admin"])
def delete_product(product_id: int, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    db.delete(p); db.commit()
    return {"ok": True}


# ───── ORDERS ──────────────────────────────────────────

@app.post("/orders", response_model=schemas.OrderRead, tags=["Orders"])
def create_order(body: schemas.OrderCreate, request: Request,
                 db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    order = models.Order(user_id=current_user.id)
    db.add(order); db.flush()
    total = 0
    for item in body.items:
        product = db.query(models.Product).get(item.product_id)
        if not product:
            raise HTTPException(404, f"Товар {item.product_id} не найден")
        oi = models.OrderItem(order_id=order.id, product_id=item.product_id,
                              quantity=item.quantity, price=product.price)
        db.add(oi)
        total += product.price * item.quantity
    order.total_price = total
    db.commit(); db.refresh(order)
    return order


@app.get("/orders", response_model=List[schemas.OrderRead], tags=["Orders"])
def get_my_orders(db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    return db.query(models.Order).filter(models.Order.user_id == current_user.id).all()


@app.get("/admin/orders", response_model=List[schemas.OrderRead], tags=["Admin"])
def list_all_orders(db: Session = Depends(get_db), _: models.User = Depends(require_admin)):
    return db.query(models.Order).all()


@app.patch("/admin/orders/{order_id}", tags=["Admin"])
def update_order_status(order_id: int, status: str, request: Request,
                        db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    order = db.query(models.Order).get(order_id)
    if not order:
        raise HTTPException(404, "Заказ не найден")
    order.status = status
    db.commit()
    return {"ok": True}


# ───── APPOINTMENTS ────────────────────────────────────

@app.post("/appointments", response_model=schemas.AppointmentRead, tags=["Appointments"])
def create_appointment(body: schemas.AppointmentCreate, request: Request,
                       db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    apt = models.Appointment(user_id=current_user.id, date=body.date, time=body.time)
    db.add(apt); db.commit(); db.refresh(apt)
    return apt


@app.get("/appointments", response_model=List[schemas.AppointmentRead], tags=["Appointments"])
def get_my_appointments(db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    return db.query(models.Appointment).filter(models.Appointment.user_id == current_user.id).all()


@app.get("/doctor/appointments", response_model=List[schemas.AppointmentRead], tags=["Doctor"])
def list_all_appointments(db: Session = Depends(get_db), _: models.User = Depends(require_doctor)):
    return db.query(models.Appointment).all()


@app.patch("/doctor/appointments/{appointment_id}", tags=["Doctor"])
def update_appointment(appointment_id: int, status: str, request: Request,
                       db: Session = Depends(get_db), current_user: models.User = Depends(require_doctor)):
    appo = db.query(models.Appointment).get(appointment_id)
    if not appo:
        raise HTTPException(404, "Запись не найдена")
    appo.status = status
    db.commit()
    return {"ok": True}


# ───── JOURNAL ─────────────────────────────────────────

@app.get("/admin/journal", response_model=List[schemas.JournalEntryRead], tags=["Admin"])
def get_journal(limit: int = Query(100, le=500), action: Optional[str] = Query(None),
                db: Session = Depends(get_db), _: models.User = Depends(require_admin)):
    q = db.query(models.JournalEntry)
    if action:
        q = q.filter(models.JournalEntry.action == action)
    return q.order_by(models.JournalEntry.id.desc()).limit(limit).all()


# ───── AI ──────────────────────────────────────────────

@app.post("/ai/face-analyze", response_model=schemas.FaceAnalysisResult, tags=["AI"])
async def analyze_face(request: Request, file: UploadFile = File(...),
                       db: Session = Depends(get_db)):
    image_bytes = await file.read()
    result = analyze_face_image(image_bytes)
    if result.get("error"):
        raise HTTPException(422, detail=result["error"])
    return _format_ai_response(result, db, request)


@app.post("/ai/analyze-video")
async def analyze_video(
    frame_front: UploadFile = File(...),
    frame_left: UploadFile = File(None),
    frame_right: UploadFile = File(None),
):
    frames = []
    for upload in [frame_front, frame_left, frame_right]:
        if upload is not None:
            frames.append(await upload.read())
    result = analyze_multi_frames(frames)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


def _format_ai_response(result, db, request):
    face_shape = result["face_shape"]
    confidence = result["confidence"]
    measurements = result.get("measurements", {"face_ratio": 0, "jaw_ratio": 0, "forehead_ratio": 0})

    meta = FACE_SHAPE_MAP.get(face_shape, FACE_SHAPE_MAP["oval"])
    products = (db.query(models.Product)
                .filter(models.Product.type == "frame",
                        models.Product.shape.in_(meta["shape_codes"]))
                .limit(6).all())

    return schemas.FaceAnalysisResult(
        face_shape=face_shape,
        confidence=confidence,
        measurements=schemas.FaceMeasurements(**measurements),
        recommended_shapes=meta["recommended_shapes"],
        recommended_products=products,
    )


# ───── PAGES ───────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse, tags=["Admin"])
def admin_panel():
    html = open("admin_panel.html", encoding="utf-8").read()
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
