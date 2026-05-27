from database import init_db, SessionLocal
from models import Product, User
import bcrypt

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def fill():
    init_db()
    db = SessionLocal()

    users = [
        User(phone="+71111111111", password_hash=hash_password("admin123"), role="admin"),
    User(phone="+72222222222", password_hash=hash_password("doctor123"), role="doctor"),
    User(phone="+73333333333", password_hash=hash_password("user123"), role="user"),
    ]
    db.add_all(users)

    frames = [
        Product(name="RB3025 Classic", brand="Ray-Ban", category="frame", shape="aviator",
                description="Классические авиаторы. Металлическая оправа, линзы G-15. Ширина 58 мм.", price=29000, in_stock=True),
        Product(name="GG0053O", brand="Gucci", category="frame", shape="cat",
                description="Кошачий глаз в ацетатной оправе. Итальянское производство. Ширина 52 мм.", price=45000, in_stock=True),
        Product(name="RB2140 Wayfarer", brand="Ray-Ban", category="frame", shape="wayfarer",
                description="Культовые вайфареры. Ацетатная трапециевидная оправа, ширина 50 мм.", price=32000, in_stock=True),
        Product(name="Lennon Round", brand="Lennon Collection", category="frame", shape="round",
                description="Круглые металлические оправы в стиле 70-х. Диаметр линзы 47 мм.", price=18000, in_stock=True),
        Product(name="OX3189 Beta", brand="Oakley", category="frame", shape="rect",
                description="Прямоугольная оправа из материала O-Matter. Спортивный дизайн, ширина 54 мм.", price=24000, in_stock=True),
        Product(name="PR 09YS", brand="Prada", category="frame", shape="oval",
                description="Овальная ацетатная оправа. Логотип Prada на дужках. Ширина 52 мм.", price=52000, in_stock=True),
        Product(name="Butterfly Air", brand="Silhouette", category="frame", shape="cat",
                description="Безободковая оправа-стрекоза. Сверхлёгкий титан, 1.8 г.", price=61000, in_stock=True),
    ]
    db.add_all(frames)

    lenses = [
        Product(name="ACUVUE Moist 1-Day", brand="Johnson & Johnson", category="lens",
                description="Однодневные контактные линзы LACREON. Диаметр 14.2, BC 8.5.", price=2800, in_stock=True),
        Product(name="Air Optix Plus HydraGlyde", brand="Alcon", category="lens",
                description="Месячные силикон-гидрогелевые линзы. Диаметр 14.2.", price=1900, in_stock=True),
        Product(name="Dailies Total1", brand="Alcon", category="lens",
                description="Однодневные линзы, содержание воды 80%. Для сухих глаз.", price=4200, in_stock=True),
    ]
    db.add_all(lenses)

    accessories = [
        Product(name="Кейс кожаный NOIR", brand="Noir Vision", category="case",
                description="Жёсткий кейс из натуральной кожи. Размер 165×70×40 мм.", price=3500, in_stock=True),
        Product(name="Раствор для линз 360ml", brand="ReNu", category="liquid",
                description="Многофункциональный раствор для контактных линз. Объём 360 мл.", price=890, in_stock=True),
    ]
    db.add_all(accessories)

    db.commit()
    db.close()
    print("✅ БД заполнена! Теперь запускай: uvicorn main:app --reload")

if __name__ == "__main__":
    fill()
