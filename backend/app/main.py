import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import text
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
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


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
        {"title": "Noir Round 04",    "brand": "NOIR VISION ATELIER", "price": 25000, "shape": "round",    "icon": "🕶️", "desc": "Лёгкая полуободковая круглая оправа. Диаметр 46 мм.", "specs": "Материал: металл · Цвет: розовое золото · Вес: 16 г"},
        {"title": "Noir Round 05",    "brand": "STUDIO ECLIPSE",      "price": 35000, "shape": "round",    "icon": "🕶️", "desc": "Премиальная круглая оправа с титановыми дужками.", "specs": "Материал: титан · Цвет: матовый графит · Вес: 12 г"},
        {"title": "Noir Oval 01",     "brand": "NOIR VISION ATELIER", "price": 30000, "shape": "oval",     "icon": "🕶️", "desc": "Овальная оправа с мягкими линиями. Ширина 52 мм, высота 38 мм.", "specs": "Материал: ацетат · Цвет: матовый чёрный · Вес: 20 г"},
        {"title": "Noir Oval 02",     "brand": "NOIR VISION ATELIER", "price": 31000, "shape": "oval",     "icon": "🕶️", "desc": "Вытянутая овальная форма с металлическими дужками. Ширина 54 мм.", "specs": "Материал: комбинированный · Цвет: чёрный/серебро"},
        {"title": "Noir Oval 03",     "brand": "NOIR VISION ATELIER", "price": 32000, "shape": "oval",     "icon": "🕶️", "desc": "Классический овал в итальянском стиле. Ширина 52 мм.", "specs": "Материал: ацетат · Цвет: черепаховый · Дужки 140 мм"},
        {"title": "Noir Oval 04",     "brand": "OBSIDIAN FRAMES",     "price": 28000, "shape": "oval",     "icon": "🕶️", "desc": "Овальная оправа в ретро-стиле. Ширина 50 мм.", "specs": "Материал: ацетат · Цвет: тёмно-зелёный · Ручная работа"},
        {"title": "Noir Oval 05",     "brand": "OBSIDIAN FRAMES",     "price": 42000, "shape": "oval",     "icon": "🕶️", "desc": "Дизайнерская овальная оправа с градиентом цвета.", "specs": "Материал: ацетат · Цвет: градиент серый/чёрный · Ограниченная серия"},
        {"title": "Noir Square 01",   "brand": "OBSIDIAN FRAMES",     "price": 32000, "shape": "square",   "icon": "🕶️", "desc": "Квадратная оправа с чётко выраженными углами. Ширина 52 мм.", "specs": "Материал: ацетат · Цвет: чёрный матовый · Вес: 22 г"},
        {"title": "Noir Square 02",   "brand": "OBSIDIAN FRAMES",     "price": 33000, "shape": "square",   "icon": "🕶️", "desc": "Массивная квадратная оправа из толстого ацетата. Ширина 54 мм.", "specs": "Материал: ацетат · Цвет: тёмно-серый · Гибкие петли"},
        {"title": "Noir Square 03",   "brand": "OBSIDIAN FRAMES",     "price": 34000, "shape": "square",   "icon": "🕶️", "desc": "Квадратная оправа с металлическими вставками на дужках.", "specs": "Материал: ацетат + металл · Цвет: чёрный/золото"},
        {"title": "Noir Square 04",   "brand": "STUDIO ECLIPSE",      "price": 36000, "shape": "square",   "icon": "🕶️", "desc": "Узкая квадратная оправа для овального лица.", "specs": "Материал: ацетат · Цвет: хаки · Ширина 48 мм"},
        {"title": "Noir Square 05",   "brand": "NOIR VISION ATELIER", "price": 27000, "shape": "square",   "icon": "🕶️", "desc": "Доступная квадратная оправа в повседневном стиле.", "specs": "Материал: ацетат · Цвет: чёрный глянцевый"},
        {"title": "Noir Rect 01",     "brand": "STUDIO ECLIPSE",      "price": 33000, "shape": "rect",     "icon": "🕶️", "desc": "Прямоугольная оправа в минималистичном стиле. Ширина 54 мм.", "specs": "Материал: ацетат · Цвет: чёрный · Тонкий обод"},
        {"title": "Noir Rect 02",     "brand": "STUDIO ECLIPSE",      "price": 34000, "shape": "rect",     "icon": "🕶️", "desc": "Широкая прямоугольная оправа. Ширина 56 мм.", "specs": "Материал: ацетат · Цвет: матовый чёрный · Вес: 24 г"},
        {"title": "Noir Rect 03",     "brand": "STUDIO ECLIPSE",      "price": 35000, "shape": "rect",     "icon": "🕶️", "desc": "Прямоугольная металлическая оправа. Элегантный деловой стиль.", "specs": "Материал: металл (титан) · Цвет: серебро · Вес: 14 г"},
        {"title": "Noir Rect 04",     "brand": "OBSIDIAN FRAMES",     "price": 29000, "shape": "rect",     "icon": "🕶️", "desc": "Прямоугольная оправа в стиле ретро. Ширина 50 мм.", "specs": "Материал: ацетат · Цвет: бордовый · Дужки 145 мм"},
        {"title": "Noir Rect 05",     "brand": "NOIR VISION ATELIER", "price": 45000, "shape": "rect",     "icon": "🕶️", "desc": "Премиальная прямоугольная оправа с покрытием из палладия.", "specs": "Материал: палладий · Цвет: платина · Вес: 11 г"},
        {"title": "Noir Aviator 01",  "brand": "NOIR VISION ATELIER", "price": 36000, "shape": "aviator",  "icon": "✈️", "desc": "Классический силуэт авиатора с тонким металлическим ободом.", "specs": "Материал: металл · Цвет: золото · Линзы: с градиентом"},
        {"title": "Noir Aviator 02",  "brand": "NOIR VISION ATELIER", "price": 37000, "shape": "aviator",  "icon": "✈️", "desc": "Авиатор с двойным ободом. Матовый чёрный.", "specs": "Материал: нержавеющая сталь · Цвет: матовый чёрный"},
        {"title": "Noir Aviator 03",  "brand": "NOIR VISION ATELIER", "price": 38000, "shape": "aviator",  "icon": "✈️", "desc": "Увеличенный силуэт авиатора для яркого образа.", "specs": "Материал: металл · Цвет: серебро/чёрный"},
        {"title": "Noir Aviator 04",  "brand": "STUDIO ECLIPSE",      "price": 42000, "shape": "aviator",  "icon": "✈️", "desc": "Авиатор с зеркальными линзами. Хромированные дужки.", "specs": "Материал: нержавеющая сталь · Цвет: хром · Зеркальные линзы"},
        {"title": "Noir Cat 01",      "brand": "STUDIO ECLIPSE",      "price": 35000, "shape": "cat",      "icon": "🐱", "desc": "Кошачий глаз с выраженным углом. Женственная форма.", "specs": "Материал: ацетат · Цвет: матовый чёрный"},
        {"title": "Noir Cat 02",      "brand": "STUDIO ECLIPSE",      "price": 36000, "shape": "cat",      "icon": "🐱", "desc": "Изящный кошачий глаз с металлическим декором на углах.", "specs": "Материал: ацетат + металл · Цвет: чёрный/золото"},
        {"title": "Noir Cat 03",      "brand": "NOIR VISION ATELIER", "price": 39000, "shape": "cat",      "icon": "🐱", "desc": "Увеличенный кошачий глаз. Драматичный силуэт.", "specs": "Материал: ацетат · Цвет: черепаховый · Ширина 54 мм"},
        {"title": "Noir Wayfarer 01", "brand": "OBSIDIAN FRAMES",     "price": 34000, "shape": "wayfarer", "icon": "🕶️", "desc": "Классический Вайфарер с массивным ацетатным ободом.", "specs": "Материал: ацетат · Цвет: чёрный · Трапециевидная форма"},
        {"title": "Noir Wayfarer 02", "brand": "OBSIDIAN FRAMES",     "price": 35000, "shape": "wayfarer", "icon": "🕶️", "desc": "Вайфарер с тонкой рамкой. Современная интерпретация классики.", "specs": "Материал: ацетат · Цвет: тёмно-коричневый"},
        {"title": "Noir Wayfarer 03", "brand": "STUDIO ECLIPSE",      "price": 31000, "shape": "wayfarer", "icon": "🕶️", "desc": "Вайфарер с ярким акцентом на дужках.", "specs": "Материал: ацетат · Цвет: белый/чёрный"},
        {"title": "Noir Heart 01",    "brand": "STUDIO ECLIPSE",      "price": 33000, "shape": "heart",    "icon": "🕶️", "desc": "Романтическая оправа в форме сердца.", "specs": "Материал: ацетат · Цвет: розовый кварц · Дужки 140 мм"},
    ],
    "lenses": [
        {"title": "Distance Standard",       "brand": "NOIR OPTICS",     "price": 2799,  "lens": "dal",   "icon": "🔬", "desc": "Базовые монофокальные линзы для дали.", "specs": "Индекс: 1.50 · УФ-фильтр · Антибликовое покрытие"},
        {"title": "Distance Slim 1.60",      "brand": "NOIR OPTICS",     "price": 3999,  "lens": "dal",   "icon": "🔬", "desc": "Тонкие линзы индекса 1.60 для умеренных диоптрий.", "specs": "Индекс: 1.60 · Лёгкие · Антибликовое покрытие"},
        {"title": "Distance Ultra Thin 1.67","brand": "NOIR OPTICS",     "price": 5499,  "lens": "dal",   "icon": "🔬", "desc": "Ультратонкие линзы 1.67 — минимальная толщина при высоких диоптриях.", "specs": "Индекс: 1.67 · Олеофобное покрытие · Лёгкие"},
        {"title": "Distance Hi-Index 1.74",  "brand": "OBSIDIAN OPTICS", "price": 7999,  "lens": "dal",   "icon": "🔬", "desc": "Премиальные линзы 1.74 для самых высоких рецептов.", "specs": "Индекс: 1.74 · Почти незаметны в оправе"},
        {"title": "Distance Aspheric 1.60",  "brand": "NOIR OPTICS",     "price": 4499,  "lens": "dal",   "icon": "🔬", "desc": "Асферические линзы 1.60 — минимум искажений.", "specs": "Асферический дизайн · Индекс 1.60 · УФ-фильтр"},
        {"title": "Reading Classic",         "brand": "NOIR OPTICS",     "price": 2499,  "lens": "bliz",  "icon": "🔬", "desc": "Классические линзы для близи — чтение, работа с документами.", "specs": "Монофокальные · Для близи · Антибликовое покрытие"},
        {"title": "Reading Slim 1.60",       "brand": "NOIR OPTICS",     "price": 3499,  "lens": "bliz",  "icon": "🔬", "desc": "Тонкие линзы для близи индекса 1.60.", "specs": "Индекс: 1.60 · Комфорт при длительном чтении"},
        {"title": "Reading Premium Wide",    "brand": "OBSIDIAN OPTICS", "price": 4999,  "lens": "bliz",  "icon": "🔬", "desc": "Линзы для близи с расширенным полем зрения.", "specs": "Широкое поле · Индекс 1.60 · Антиблик"},
        {"title": "Office Base",             "brand": "NOIR OPTICS",     "price": 3799,  "lens": "comp",  "icon": "💻", "desc": "Базовые офисные линзы для работы за ПК. 40–70 см.", "specs": "Оптимизированы для экрана · Антибликовые"},
        {"title": "Office Blue Block",       "brand": "NOIR OPTICS",     "price": 4799,  "lens": "comp",  "icon": "💻", "desc": "Офисные линзы с фильтром синего света.", "specs": "Blue Block · Снижают усталость глаз"},
        {"title": "Office Pro Ergo",         "brand": "OBSIDIAN OPTICS", "price": 6499,  "lens": "comp",  "icon": "💻", "desc": "Эргономичные офисные линзы для multi-мониторных setup.", "specs": "Широкая зона · Blue Block · Антиблик"},
        {"title": "Progressive Entry",       "brand": "NOIR OPTICS",     "price": 6999,  "lens": "prog",  "icon": "🔭", "desc": "Прогрессивные линзы начального уровня.", "specs": "3 зоны зрения · Удобная адаптация"},
        {"title": "Progressive Comfort",     "brand": "NOIR OPTICS",     "price": 8999,  "lens": "prog",  "icon": "🔭", "desc": "Улучшенный дизайн с расширенными боковыми зонами.", "specs": "Меньше периферийных искажений · FreeView"},
        {"title": "Progressive Premium",     "brand": "OBSIDIAN OPTICS", "price": 12999, "lens": "prog",  "icon": "🔭", "desc": "Премиальный прогрессивный дизайн.", "specs": "Минимум аберраций · Для всех дистанций"},
        {"title": "Progressive Sport",       "brand": "OBSIDIAN OPTICS", "price": 14999, "lens": "prog",  "icon": "🔭", "desc": "Прогрессивные линзы для активного образа жизни.", "specs": "Усиленная периферия · Ударопрочные"},
        {"title": "Drive Day",               "brand": "NOIR OPTICS",     "price": 4299,  "lens": "drive", "icon": "🚗", "desc": "Дневные линзы для вождения.", "specs": "Высокая контрастность · Поляризация"},
        {"title": "Drive Night",             "brand": "NOIR OPTICS",     "price": 5499,  "lens": "drive", "icon": "🌙", "desc": "Специальное покрытие для ночного вождения.", "specs": "Подавляет ореолы от фар встречных машин"},
        {"title": "Drive Polarized",         "brand": "OBSIDIAN OPTICS", "price": 6999,  "lens": "drive", "icon": "🚗", "desc": "Поляризационные линзы для вождения.", "specs": "Устраняют блики от асфальта"},
        {"title": "Photochromic Classic",    "brand": "NOIR CARE",       "price": 3999,  "lens": "photo", "icon": "🌤️", "desc": "Базовые фотохромные линзы.", "specs": "Прозрачные в помещении · Адаптируются за 30 сек"},
        {"title": "Photochromic Fast Adapt", "brand": "NOIR CARE",       "price": 5299,  "lens": "photo", "icon": "🌤️", "desc": "Ускоренная адаптация: затемнение за 20 сек.", "specs": "Просветление за 3 мин · Blue Block"},
        {"title": "Photochromic Drive",      "brand": "NOIR CARE",       "price": 6999,  "lens": "photo", "icon": "🌤️", "desc": "Фотохромные линзы для вождения. Затемняются в машине.", "specs": "Активация через лобовое стекло · UV400"},
        {"title": "Sun Classic",             "brand": "NOIR CARE",       "price": 2999,  "lens": "sun",   "icon": "☀️", "desc": "Базовые солнцезащитные линзы с UV400-защитой.", "specs": "Нейтральный серый оттенок · UV400"},
        {"title": "Sun Gradient Noir",       "brand": "STUDIO ECLIPSE",  "price": 4999,  "lens": "sun",   "icon": "☀️", "desc": "Градиентные линзы: тёмные сверху, светлые снизу.", "specs": "Стиль и функциональность · UV400"},
        {"title": "Sun Mirror Gold",         "brand": "STUDIO ECLIPSE",  "price": 5999,  "lens": "sun",   "icon": "☀️", "desc": "Зеркальные золотые солнцезащитные линзы.", "specs": "Зеркальное покрытие · UV400 · Стиль"},
        {"title": "Polarized Gray",          "brand": "NOIR CARE",       "price": 4499,  "lens": "polar", "icon": "🌊", "desc": "Серые поляризационные линзы.", "specs": "Устраняют блики от воды и дороги"},
        {"title": "Polarized Brown",         "brand": "NOIR CARE",       "price": 4999,  "lens": "polar", "icon": "🌊", "desc": "Коричневые поляризационные линзы — усиленный контраст.", "specs": "Поляризация · Усиление контраста · UV400"},
        {"title": "Polarized Mirror Blue",   "brand": "STUDIO ECLIPSE",  "price": 6499,  "lens": "polar", "icon": "🌊", "desc": "Синие зеркальные поляризационные линзы.", "specs": "Зеркальный эффект · Поляризация · UV400"},
        {"title": "Blue Light Basic",        "brand": "NOIR CARE",       "price": 1799,  "lens": "blue",  "icon": "📱", "desc": "Базовый фильтр синего света.", "specs": "Снижает нагрузку при работе за ПК"},
        {"title": "Blue Light Premium Slim", "brand": "STUDIO ECLIPSE",  "price": 5499,  "lens": "blue",  "icon": "📱", "desc": "Тонкие линзы 1.67 с фильтром синего света.", "specs": "Комфорт весь день · Индекс 1.67"},
        {"title": "Blue Light Gaming",       "brand": "OBSIDIAN OPTICS", "price": 6999,  "lens": "blue",  "icon": "🎮", "desc": "Линзы для геймеров с максимальной защитой синего.", "specs": "Макс.фильтр синего · Антиблик · Индекс 1.60"},
    ],
    "liquids": [
        {"title": "Clean Drops 10 мл",     "brand": "NOIR CARE", "price": 890,  "icon": "💧", "desc": "Увлажняющие капли для контактных линз. Снимают усталость.", "specs": "Объём: 10 мл · Состав: HA 0.1% · Гипоаллергенно"},
        {"title": "Clean Drops 15 мл",     "brand": "NOIR CARE", "price": 1190, "icon": "💧", "desc": "Увлажняющие капли увеличенного объёма.", "specs": "Объём: 15 мл · Состав: HA 0.1% · Гипоаллергенно"},
        {"title": "Clean Drops 20 мл",     "brand": "NOIR CARE", "price": 1590, "icon": "💧", "desc": "Увлажняющие капли макси-формата.", "specs": "Объём: 20 мл · Состав: HA 0.15% · Без консервантов"},
        {"title": "Pro Solution 100 мл",   "brand": "NOIR CARE", "price": 1490, "icon": "🧴", "desc": "Многофункциональный раствор для мягких линз.", "specs": "Объём: 100 мл · Дезинфекция · Хранение"},
        {"title": "Pro Solution 360 мл",   "brand": "NOIR CARE", "price": 2490, "icon": "🧴", "desc": "Многофункциональный раствор для мягких линз — увеличенный.", "specs": "Объём: 360 мл · Дезинфекция · Хранение"},
        {"title": "Pro Solution 500 мл",   "brand": "NOIR CARE", "price": 3290, "icon": "🧴", "desc": "Раствор экономичного объёма 500 мл.", "specs": "Объём: 500 мл · Дезинфекция · Хранение · 3 мес."},
        {"title": "Peroxide System",       "brand": "OBSIDIAN CARE", "price": 2990, "icon": "🧪", "desc": "Перекисная система очистки. Глубокая дезинфекция.", "specs": "Перекись 3% · Нейтрализация за 6ч · Комплект с контейнером"},
        {"title": "All-in-One Plus",       "brand": "NOIR CARE", "price": 1990, "icon": "🧴", "desc": "Многофункциональный раствор с увлажняющими компонентами.", "specs": "Объём: 250 мл · Глицерин · Комфорт на 20ч"},
        {"title": "Frame Cleaner Spray",   "brand": "NOIR CARE", "price": 990,  "icon": "🌀", "desc": "Спрей для очистки оправ и жёстких оптических линз.", "specs": "Объём: 50 мл · Не оставляет разводов"},
        {"title": "Ultra Clean Mist",      "brand": "NOIR CARE", "price": 3500, "icon": "✨", "desc": "Спрей для быстрой очистки и увлажнения.", "specs": "Объём: 30 мл · Микрофибра в комплекте"},
        {"title": "Anti-Fog Spray",        "brand": "NOIR CARE", "price": 790,  "icon": "🌀", "desc": "Спрей против запотевания линз.", "specs": "Объём: 20 мл · Эффект 48ч · Без спирта"},
        {"title": "Sensitive Solution",    "brand": "NOIR CARE", "price": 1890, "icon": "🧴", "desc": "Раствор для чувствительных глаз. Без консервантов.", "specs": "Объём: 200 мл · Без консервантов · Гипоаллергенно"},
        {"title": "Lens Case + Solution Set","brand": "NOIR CARE", "price": 2590, "icon": "🧴", "desc": "Набор: раствор 200 мл + контейнер для хранения.", "specs": "Раствор 200 мл · Контейнер 2 отделения · Зеркальце"},
    ],
    "cases": [
        {"title": "Stone Case 01",   "brand": "OBSIDIAN CASES", "price": 4500, "icon": "🗃️", "desc": "Жёсткий чехол тёмно-серого цвета. Защита от ударов.", "specs": "Материал: поликарбонат · Застёжка: кнопка · 165×70×40 мм"},
        {"title": "Stone Case 02",   "brand": "OBSIDIAN CASES", "price": 4900, "icon": "🗃️", "desc": "Матовый чёрный футляр с магнитной застёжкой.", "specs": "Материал: поликарбонат · Застёжка: магнит · Мягкая подкладка"},
        {"title": "Stone Case 03",   "brand": "OBSIDIAN CASES", "price": 3900, "icon": "🗃️", "desc": "Лёгкий поликарбонатный чехол базовой серии.", "specs": "Материал: поликарбонат · Застёжка: кнопка · Вес: 40 г"},
        {"title": "Leather Fold 01", "brand": "NOIR LEATHER",   "price": 5200, "icon": "👜", "desc": "Плоский кожаный чехол‑конверт. Минималистичный дизайн.", "specs": "Материал: натуральная кожа · Застёжка: кнопка · 170×75 мм"},
        {"title": "Leather Box 02",  "brand": "NOIR LEATHER",   "price": 5500, "icon": "👜", "desc": "Объёмный кожаный футляр с тиснением логотипа.", "specs": "Материал: натуральная кожа · Застёжка: металл · Вес: 95 г"},
        {"title": "Leather Box 03",  "brand": "NOIR LEATHER",   "price": 7200, "icon": "👜", "desc": "Премиальный кожаный футляр ручной работы.", "specs": "Материал: итальянская кожа · Ручная работа · 2 кармана"},
        {"title": "Concrete Box 01", "brand": "STUDIO ECLIPSE", "price": 5800, "icon": "🧱", "desc": "Дизайнерский чехол с бетонной текстурой. Ограниченная серия.", "specs": "Материал: полимер бетон-стиль · Магнит · 170×70×42 мм"},
        {"title": "Concrete Box 02", "brand": "STUDIO ECLIPSE", "price": 6300, "icon": "🧱", "desc": "Чехол с бетонной текстурой и металлическими вставками.", "specs": "Материал: полимер + металл · Магнит · 175×72×45 мм"},
        {"title": "Carbon Fiber Case", "brand": "OBSIDIAN CASES", "price": 8500, "icon": "⬛", "desc": "Чехол из карбона. Максимальная лёгкость и прочность.", "specs": "Материал: карбон · Магнит · Вес: 28 г · 170×70×38 мм"},
        {"title": "Travel Pouch",      "brand": "NOIR LEATHER", "price": 3200, "icon": "👝", "desc": "Мягкий чехол-мешочек для путешествий.", "specs": "Материал: микрофибра · Застёжка: шнурок · 180×80 мм"},
    ],
    "contact_lenses": [
        {"title": "Daily Aqua 30",         "brand": "NOIR CONTACT", "price": 1990, "lens": "daily", "icon": "💠", "desc": "Однодневные силикон-гидрогелевые линзы. Упаковка 30 шт.", "specs": "Dk/t: 100 · Влага: 56% · УФ-фильтр · 30 линз"},
        {"title": "Daily Aqua 90",         "brand": "NOIR CONTACT", "price": 4590, "lens": "daily", "icon": "💠", "desc": "Однодневные линзы. Упаковка 90 шт. на 3 месяца.", "specs": "Dk/t: 100 · Влага: 56% · УФ-фильтр · 90 линз"},
        {"title": "Daily Premium HD",      "brand": "OBSIDIAN CONTACT", "price": 3290, "lens": "daily", "icon": "💠", "desc": "Премиальные однодневные линзы с HD-оптикой.", "specs": "Dk/t: 120 · Сферический дизайн · УФ-фильтр · 30 линз"},
        {"title": "Weekly Comfort",        "brand": "NOIR CONTACT", "price": 1590, "lens": "weekly", "icon": "💠", "desc": "Линзы двухнедельной замены. Упаковка 6 шт.", "specs": "Dk/t: 90 · Замена: 2 недели · 6 линз · УФ-фильтр"},
        {"title": "Weekly Comfort 12",     "brand": "NOIR CONTACT", "price": 2790, "lens": "weekly", "icon": "💠", "desc": "Линзы двухнедельной замены. Упаковка 12 шт.", "specs": "Dk/t: 90 · Замена: 2 недели · 12 линз · УФ-фильтр"},
        {"title": "Monthly Classic",       "brand": "NOIR CONTACT", "price": 1290, "lens": "monthly", "icon": "💠", "desc": "Линзы ежемесячной замены. Упаковка 6 шт.", "specs": "Dk/t: 80 · Замена: 1 месяц · 6 линз"},
        {"title": "Monthly Classic 12",    "brand": "NOIR CONTACT", "price": 2290, "lens": "monthly", "icon": "💠", "desc": "Линзы ежемесячной замены. Упаковка 12 шт.", "specs": "Dk/t: 80 · Замена: 1 месяц · 12 линз"},
        {"title": "Monthly Premium",       "brand": "OBSIDIAN CONTACT", "price": 2490, "lens": "monthly", "icon": "💠", "desc": "Премиальные линзы месячной замены с высоким Dk/t.", "specs": "Dk/t: 130 · Влага: 48% · 6 линз · Точение"},
        {"title": "Toric Daily 30",        "brand": "NOIR CONTACT", "price": 3490, "lens": "toric", "icon": "💠", "desc": "Торические однодневные линзы для астигматизма.", "specs": "Dk/t: 90 · Астигматизм · 30 линз · Стабилизация"},
        {"title": "Toric Monthly 6",       "brand": "OBSIDIAN CONTACT", "price": 3990, "lens": "toric", "icon": "💠", "desc": "Торические линзы месячной замены для астигматизма.", "specs": "Dk/t: 100 · Замена: 1 месяц · 6 линз"},
        {"title": "Multifocal Daily 30",   "brand": "NOIR CONTACT", "price": 3890, "lens": "multi", "icon": "💠", "desc": "Мультифокальные однодневные линзы для пресбиопии.", "specs": "Dk/t: 85 · Пресбиопия · 30 линз · 3 зоны"},
        {"title": "Color Blue 30",         "brand": "NOIR CONTACT", "price": 2490, "lens": "color", "icon": "👁️", "desc": "Цветные линзы. Цвет: синий. Ежемесячная замена.", "specs": "Замена: 1 месяц · 2 линзы · Естественный оттенок"},
        {"title": "Color Green 30",        "brand": "NOIR CONTACT", "price": 2490, "lens": "color", "icon": "👁️", "desc": "Цветные линзы. Цвет: зелёный. Ежемесячная замена.", "specs": "Замена: 1 месяц · 2 линзы · Естественный оттенок"},
        {"title": "Color Hazel 30",        "brand": "STUDIO ECLIPSE", "price": 2990, "lens": "color", "icon": "👁️", "desc": "Цветные линзы. Цвет: ореховый. Ежемесячная замена.", "specs": "Замена: 1 месяц · 2 линзы · Диаметр 14.2 мм"},
    ],
}

def seed_products(db: Session):
    existing_titles = {p.title for p in db.query(models.Product.title).all()}

    product_id_map = {}
    for tab_name, product_list in SEED_DATA.items():
        ptype = tab_name if tab_name in ("frame", "lens", "contact_lens", "liquid", "case") else "accessory"
        if tab_name == "frames": ptype = "frame"
        elif tab_name == "lenses": ptype = "lens"
        elif tab_name == "contact_lenses": ptype = "contact_lens"
        elif tab_name == "liquids": ptype = "liquid"
        elif tab_name == "cases": ptype = "case"

        for item in product_list:
            if item["title"] in existing_titles:
                continue
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
        if not db.query(models.User).filter(models.User.role == models.UserRole.admin).first():
            admin = models.User(
                nickname="admin",
                phone="+71111111111",
                email="admin@noir.vision",
                hashed_password=bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode(),
                role=models.UserRole.admin,
                is_active=True,
            )
            db.add(admin)
            db.commit()
        if not db.query(models.User).filter(models.User.role == models.UserRole.doctor).first():
            doc = models.User(
                nickname="doctor",
                phone="+72222222222",
                email="doctor@noir.vision",
                hashed_password=bcrypt.hashpw(b"doctor123", bcrypt.gensalt()).decode(),
                role=models.UserRole.doctor,
                is_active=True,
            )
            db.add(doc)
            db.commit()
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
    log_action(db, "register", user_id=user.id, entity="user", entity_id=user.id,
               detail=f"Регистрация {body.nickname}", request=request)
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return schemas.TokenResponse(access_token=token, user=user)


@app.post("/api/auth/token", response_model=schemas.TokenResponse, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), request: Request = None,
           db: Session = Depends(get_db)):
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
    log_action(db, "login", user_id=user.id, entity="user", entity_id=user.id,
               detail=f"Вход {user.nickname or user.email}", request=request)
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return schemas.TokenResponse(access_token=token, user=user)


# ───── USERS ──────────────────────────────────────────

@app.get("/api/auth/me", response_model=schemas.UserOut, tags=["Auth"])
def get_me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.get("/api/auth/check-field", tags=["Auth"])
def check_field(field: str, value: str, db: Session = Depends(get_db)):
    if field not in ("nickname", "phone", "email"):
        raise HTTPException(400, "Поле должно быть nickname, phone или email")
    col = getattr(models.User, field, None)
    if not col:
        raise HTTPException(400, "Поле не найдено")
    exists = db.query(models.User).filter(col == value).first() is not None
    return {"field": field, "value": value, "available": not exists}


# ───── PRODUCTS ────────────────────────────────────────

TYPE_MAP = {
    "frames": "frame", "lenses": "lens", "liquids": "liquid", "cases": "case",
    "frame": "frame", "lens": "lens", "contact_lens": "contact_lens",
    "liquid": "liquid", "case": "case", "accessory": "accessory",
}

@app.get("/api/products", response_model=schemas.ProductPaginationResponse, tags=["Products"])
def list_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
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
    items = q.order_by(models.Product.id).offset((page - 1) * limit).limit(limit).all()

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

ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'}

@app.post("/admin/upload-image", tags=["Admin"])
def upload_image(file: UploadFile = File(...), current_user: models.User = Depends(require_admin)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Недопустимый формат: {ext}. Разрешены: {', '.join(ALLOWED_EXT)}")
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(file.file.read())
    url = f"/uploads/{filename}"
    return JSONResponse({"url": url, "filename": filename})

@app.post("/admin/products", response_model=schemas.ProductOut, tags=["Admin"])
def create_product(body: schemas.ProductCreate, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    data = body.dict()
    p = models.Product(**data)
    p.in_stock = data.get("stock", 0) > 0
    db.add(p); db.commit(); db.refresh(p)
    log_action(db, "create_product", user_id=current_user.id, entity="product",
               entity_id=p.id, detail=f"Создан товар {p.title}", request=request)
    return p


@app.put("/admin/products/{product_id}", response_model=schemas.ProductOut, tags=["Admin"])
def update_product(product_id: int, body: schemas.ProductCreate, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    data = body.dict()
    for k, v in data.items():
        setattr(p, k, v)
    p.in_stock = data.get("stock", 0) > 0
    db.commit(); db.refresh(p)
    log_action(db, "update_product", user_id=current_user.id, entity="product",
               entity_id=p.id, detail=f"Обновлён товар {p.title}", request=request)
    return p


@app.delete("/admin/products/{product_id}", tags=["Admin"])
def delete_product(product_id: int, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    title = p.title
    db.delete(p); db.commit()
    log_action(db, "delete_product", user_id=current_user.id, entity="product",
               entity_id=product_id, detail=f"Удалён товар {title}", request=request)
    return {"ok": True}


@app.post("/admin/reseed", tags=["Admin"])
def reseed_products(request: Request, db: Session = Depends(get_db),
                    current_user: models.User = Depends(require_admin)):
    db.query(models.CartItem).delete()
    db.query(models.OrderItem).delete()
    db.query(models.Image).delete()
    db.query(models.Attribute).delete()
    db.query(models.Product).delete()
    db.execute(text("ALTER SEQUENCE products_id_seq RESTART WITH 1"))
    db.execute(text("ALTER SEQUENCE attributes_id_seq RESTART WITH 1"))
    db.execute(text("ALTER SEQUENCE images_id_seq RESTART WITH 1"))
    db.commit()
    seed_products(db)
    log_action(db, "reseed", user_id=current_user.id, entity="product",
               detail="Перезаливка всех товаров", request=request)
    return {"ok": True, "message": "Товары перезалиты"}


@app.post("/admin/products/{product_id}/images", tags=["Admin"])
def upload_product_images(product_id: int, files: List[UploadFile] = File(...),
                          request: Request = None,
                          db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    uploaded = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            continue
        filename = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, "wb") as f:
            f.write(file.file.read())
        url = f"/uploads/{filename}"
        is_primary = db.query(models.Image).filter(models.Image.product_id == product_id).count() == 0
        img = models.Image(product_id=product_id, url=url, is_primary=is_primary)
        db.add(img); db.commit(); db.refresh(img)
        if is_primary or not p.image_url:
            p.image_url = url
            db.commit()
        uploaded.append({"id": img.id, "url": img.url, "is_primary": img.is_primary})
    log_action(db, "update_product", user_id=current_user.id, entity="product",
               entity_id=product_id, detail=f"Загружено {len(uploaded)} изображений", request=request)
    return {"uploaded": uploaded}


@app.delete("/admin/products/{product_id}/images/{image_id}", tags=["Admin"])
def delete_product_image(product_id: int, image_id: int,
                         db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    img = db.query(models.Image).filter(models.Image.id == image_id, models.Image.product_id == product_id).first()
    if not img:
        raise HTTPException(404, "Изображение не найдено")
    filename = os.path.basename(img.url)
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    was_primary = img.is_primary
    db.delete(img); db.commit()
    if was_primary:
        first = db.query(models.Image).filter(models.Image.product_id == product_id).first()
        if first:
            first.is_primary = True; db.commit()
    return {"ok": True}


@app.put("/admin/products/{product_id}/images/{image_id}/primary", tags=["Admin"])
def set_primary_image(product_id: int, image_id: int,
                      db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    img = db.query(models.Image).filter(models.Image.id == image_id, models.Image.product_id == product_id).first()
    if not img:
        raise HTTPException(404, "Изображение не найдено")
    db.query(models.Image).filter(models.Image.product_id == product_id).update({"is_primary": False})
    img.is_primary = True
    db.query(models.Product).filter(models.Product.id == product_id).update({"image_url": img.url})
    db.commit()
    return {"ok": True}


# ───── ORDERS ──────────────────────────────────────────

@app.post("/api/orders", response_model=schemas.OrderRead, tags=["Orders"])
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
    db.query(models.CartItem).filter(models.CartItem.user_id == current_user.id).delete()
    db.commit(); db.refresh(order)
    log_action(db, "create_order", user_id=current_user.id, entity="order",
               entity_id=order.id, detail=f"Заказ #{order.id} на сумму {total} ₽", request=request)
    return order


@app.get("/api/orders", response_model=List[schemas.OrderRead], tags=["Orders"])
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

@app.post("/api/appointments", response_model=schemas.AppointmentRead, tags=["Appointments"])
def create_appointment(body: schemas.AppointmentCreate, request: Request,
                       db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    apt = models.Appointment(user_id=current_user.id, date=body.date, time=body.time)
    db.add(apt); db.commit(); db.refresh(apt)
    log_action(db, "create_appointment", user_id=current_user.id, entity="appointment",
               entity_id=apt.id, detail=f"Запись на {body.date} в {body.time}", request=request)
    return apt

@app.get("/api/appointments", response_model=List[schemas.AppointmentRead], tags=["Appointments"])
def get_my_appointments(db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    return db.query(models.Appointment).filter(models.Appointment.user_id == current_user.id).all()


@app.get("/doctor/appointments", response_model=List[schemas.AppointmentWithUser], tags=["Doctor"])
def list_all_appointments(db: Session = Depends(get_db), _: models.User = Depends(require_doctor)):
    appts = db.query(models.Appointment).all()
    result = []
    for a in appts:
        user = db.query(models.User).get(a.user_id)
        result.append(schemas.AppointmentWithUser(
            id=a.id, user_id=a.user_id, date=a.date, time=a.time,
            status=a.status, created_at=a.created_at,
            user_phone=user.phone if user else None,
            user_name=((user.first_name or '') + ' ' + (user.last_name or '')).strip() or None,
            user_email=user.email if user else None,
        ))
    return result


@app.patch("/doctor/appointments/{appointment_id}", tags=["Doctor"])
def update_appointment(appointment_id: int, status: str, request: Request,
                       db: Session = Depends(get_db), current_user: models.User = Depends(require_doctor)):
    appo = db.query(models.Appointment).get(appointment_id)
    if not appo:
        raise HTTPException(404, "Запись не найдена")
    appo.status = status
    db.commit()
    return {"ok": True}

# ───── CART ──────────────────────────────────────────

@app.get("/api/cart", response_model=List[schemas.CartItemOut], tags=["Cart"])
def get_cart(db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    items = db.query(models.CartItem).filter(models.CartItem.user_id == current_user.id).all()
    result = []
    for item in items:
        p = item.product
        result.append(schemas.CartItemOut(
            product_id=p.id, quantity=item.quantity,
            title=p.title, price=p.price, icon=p.image_url or "🕶️"
        ))
    return result


@app.post("/api/cart/sync", tags=["Cart"])
def sync_cart(body: schemas.CartSyncIn, db: Session = Depends(get_db),
              current_user: models.User = Depends(require_user)):
    db.query(models.CartItem).filter(models.CartItem.user_id == current_user.id).delete()
    for item in body.items:
        db.add(models.CartItem(user_id=current_user.id, product_id=item.product_id, quantity=item.quantity))
    db.commit()
    return {"ok": True}


@app.get("/users", response_model=List[schemas.UserOut], tags=["Admin"])
def list_users(db: Session = Depends(get_db), _: models.User = Depends(require_admin)):
    return db.query(models.User).all()


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
    return _format_ai_response(result, db, request, user_id=None)


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


def _format_ai_response(result, db, request, user_id=None):
    face_shape = result["face_shape"]
    confidence = result["confidence"]
    measurements = result.get("measurements", {"face_ratio": 0, "jaw_ratio": 0, "forehead_ratio": 0})

    log_action(db, "face_analyze", user_id=user_id, entity="face",
               detail=f"Определена форма лица: {face_shape} ({(confidence*100):.0f}%)", request=request)

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
        all_scores=result.get("all_scores"),
    )


# ───── PAGES ───────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse, tags=["Admin"])
def admin_panel():
    html = open("admin_panel.html", encoding="utf-8").read()
    return HTMLResponse(content=html)


@app.get("/doctor", response_class=HTMLResponse, tags=["Doctor"])
def doctor_panel():
    html = open("doctor_panel.html", encoding="utf-8").read()
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
