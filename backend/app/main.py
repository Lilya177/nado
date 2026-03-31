from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import models
import schemas
from database import SessionLocal, init_db
import bcrypt as _bcrypt

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


app = FastAPI(title="NOIR VISION API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.role == "user").first()
    if not user:
        user = models.User(
            phone="+70000000000",
            password_hash=pwd_context.hash("123456"),
            role="user"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

@app.on_event("startup")
def on_startup():
    init_db()

# ───── AUTH ─────

import bcrypt as _bcrypt

@app.post("/users/register", response_model=schemas.UserRead, tags=["Auth"])
def register(body: schemas.UserCreate, db: Session = Depends(get_db)):
    exists = db.query(models.User).filter(models.User.phone == body.phone).first()
    if exists:
        raise HTTPException(400, "Пользователь уже существует")
    hashed = _bcrypt.hashpw(body.password.encode(), _bcrypt.gensalt()).decode()
    user = models.User(
        phone=body.phone,
        password_hash=hashed,
        first_name=body.first_name,
        last_name=body.last_name,
        role="user"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get("/users", response_model=List[schemas.UserRead], tags=["Auth"])
def get_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()

# ───── PRODUCTS ─────

@app.get("/products", response_model=List[schemas.ProductRead], tags=["Products"])
def get_products(category: Optional[str] = Query(None), db: Session = Depends(get_db)):
    q = db.query(models.Product)
    if category:
        q = q.filter(models.Product.category == category)
    return q.all()

@app.get("/products/{product_id}", response_model=schemas.ProductRead, tags=["Products"])
def get_product(product_id: int, db: Session = Depends(get_db)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    return p

@app.post("/admin/products", response_model=schemas.ProductRead, tags=["Admin"])
def create_product(body: schemas.ProductCreate, db: Session = Depends(get_db)):
    p = models.Product(**body.dict())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p

@app.put("/admin/products/{product_id}", response_model=schemas.ProductRead, tags=["Admin"])
def update_product(product_id: int, body: schemas.ProductCreate, db: Session = Depends(get_db)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    for k, v in body.dict().items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p

@app.patch("/admin/products/{product_id}", response_model=schemas.ProductRead, tags=["Admin"])
def patch_product(product_id: int, body: schemas.ProductCreate, db: Session = Depends(get_db)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    for k, v in body.dict(exclude_unset=True).items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p

@app.delete("/admin/products/{product_id}", tags=["Admin"])
def delete_product(product_id: int, db: Session = Depends(get_db)):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    db.delete(p)
    db.commit()
    return {"ok": True}

# ───── ORDERS ─────

@app.post("/orders", response_model=schemas.OrderRead, tags=["Orders"])
def create_order(body: schemas.OrderCreate, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    order = models.Order(user_id=user.id, status="pending")
    total = 0
    for item in body.items:
        product = db.query(models.Product).get(item.product_id)
        if not product:
            raise HTTPException(400, f"Товар {item.product_id} не найден")
        oi = models.OrderItem(product_id=product.id, quantity=item.quantity, price=product.price)
        total += product.price * item.quantity
        order.items.append(oi)
    order.total_price = total
    db.add(order)
    db.commit()
    db.refresh(order)
    return order

@app.get("/orders", response_model=List[schemas.OrderRead], tags=["Orders"])
def get_my_orders(db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    return db.query(models.Order).filter(models.Order.user_id == user.id).all()

@app.get("/admin/orders", response_model=List[schemas.OrderRead], tags=["Admin"])
def list_all_orders(db: Session = Depends(get_db)):
    return db.query(models.Order).all()

@app.patch("/admin/orders/{order_id}", tags=["Admin"])
def update_order_status(order_id: int, status: str, db: Session = Depends(get_db)):
    order = db.query(models.Order).get(order_id)
    if not order:
        raise HTTPException(404, "Заказ не найден")
    order.status = status
    db.commit()
    return {"ok": True}

# ───── APPOINTMENTS ─────

@app.post("/appointments", response_model=schemas.AppointmentRead, tags=["Appointments"])
def create_appointment(body: schemas.AppointmentCreate, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    appo = models.Appointment(user_id=user.id, date=body.date, time=body.time, status="pending")
    db.add(appo)
    db.commit()
    db.refresh(appo)
    return appo

@app.get("/doctor/appointments", response_model=List[schemas.AppointmentRead], tags=["Doctor"])
def list_appointments(db: Session = Depends(get_db)):
    return db.query(models.Appointment).all()

@app.patch("/doctor/appointments/{appointment_id}", tags=["Doctor"])
def update_appointment(appointment_id: int, status: str, db: Session = Depends(get_db)):
    appo = db.query(models.Appointment).get(appointment_id)
    if not appo:
        raise HTTPException(404, "Запись не найдена")
    appo.status = status
    db.commit()
    return {"ok": True}

# ───── AI ─────

FACE_SHAPE_MAP = {
    "round":    {"recommended_shapes": ["Кошачий глаз", "Авиаторы", "Вайфареры"],  "shape_codes": ["cat", "aviator", "wayfarer"]},
    "oval":     {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],    "shape_codes": ["aviator", "round", "cat"]},
    "square":   {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],    "shape_codes": ["aviator", "round", "cat"]},
    "rect":     {"recommended_shapes": ["Авиаторы", "Круглые"],                     "shape_codes": ["aviator", "round"]},
    "triangle": {"recommended_shapes": ["Авиаторы", "Стрекозы", "Круглые"],        "shape_codes": ["aviator", "round"]},
    "heart":    {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],       "shape_codes": ["aviator", "wayfarer", "round"]},
}

@app.post("/ai/face-analyze", response_model=schemas.FaceAnalysisResult, tags=["AI"])
async def analyze_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    face_shape_key = "oval"
    meta = FACE_SHAPE_MAP[face_shape_key]
    products = db.query(models.Product).filter(
        models.Product.category == "frame",
        models.Product.shape.in_(meta["shape_codes"])
    ).limit(6).all()
    return schemas.FaceAnalysisResult(
        face_shape=face_shape_key,
        recommended_shapes=meta["recommended_shapes"],
        recommended_products=products
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
@app.post("/orders", response_model=schemas.OrderRead, tags=["Orders"])
def create_order(body: schemas.OrderCreate, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    order = models.Order(user_id=user.id)
    db.add(order)
    db.flush()
    total = 0
    for item in body.items:
        product = db.query(models.Product).get(item.product_id)
        if not product:
            raise HTTPException(404, f"Товар {item.product_id} не найден")
        oi = models.OrderItem(order_id=order.id, product_id=item.product_id, quantity=item.quantity, price=product.price)
        db.add(oi)
        total += product.price * item.quantity
    order.total_price = total
    db.commit()
    db.refresh(order)
    return order

@app.post("/appointments", response_model=schemas.AppointmentRead, tags=["Appointments"])
def create_appointment(body: schemas.AppointmentCreate, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    apt = models.Appointment(user_id=user.id, date=body.date, time=body.time)
    db.add(apt)
    db.commit()
    db.refresh(apt)
    return apt

@app.get("/appointments", response_model=list[schemas.AppointmentRead], tags=["Appointments"])
def get_appointments(db: Session = Depends(get_db)):
    return db.query(models.Appointment).all()

@app.get("/orders", response_model=list[schemas.OrderRead], tags=["Orders"])
def get_orders(db: Session = Depends(get_db)):
    return db.query(models.Order).all()
