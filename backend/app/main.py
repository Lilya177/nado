from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import bcrypt

import models
import schemas
from database import SessionLocal, init_db
from auth import create_access_token
from deps import get_db, get_current_user, require_user, require_doctor, require_admin
from journal import log_action
from face_analyzer import analyze_face_image, FACE_SHAPE_MAP

app = FastAPI(title="NOIR VISION API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ───── AUTH ──────────────────────────────────────────────────────────────────

@app.post("/users/register", response_model=schemas.UserRead, tags=["Auth"])
def register(body: schemas.UserCreate, request: Request, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.phone == body.phone).first():
        raise HTTPException(400, "Пользователь уже существует")
    hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    user = models.User(
        phone=body.phone,
        password_hash=hashed,
        first_name=body.first_name,
        last_name=body.last_name,
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log_action(db, action="register", user_id=user.id, entity="user", entity_id=user.id, request=request)
    return user


@app.post("/users/login", response_model=schemas.TokenResponse, tags=["Auth"])
def login(body: schemas.LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.phone == body.phone).first()
    if not user or not bcrypt.checkpw(body.password.encode(), user.password_hash.encode()):
        raise HTTPException(401, "Неверный телефон или пароль")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    log_action(db, action="login", user_id=user.id, entity="user", entity_id=user.id, request=request)
    return schemas.TokenResponse(access_token=token, user=user)


@app.get("/users/me", response_model=schemas.UserRead, tags=["Auth"])
def get_me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.get("/users", response_model=List[schemas.UserRead], tags=["Admin"])
def get_users(db: Session = Depends(get_db), _: models.User = Depends(require_admin)):
    return db.query(models.User).all()


# ───── PRODUCTS ──────────────────────────────────────────────────────────────

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
def create_product(
    body: schemas.ProductCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    p = models.Product(**body.dict())
    db.add(p)
    db.commit()
    db.refresh(p)
    log_action(db, action="create_product", user_id=current_user.id, entity="product", entity_id=p.id, request=request)
    return p


@app.put("/admin/products/{product_id}", response_model=schemas.ProductRead, tags=["Admin"])
def update_product(
    product_id: int,
    body: schemas.ProductCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    for k, v in body.dict().items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    log_action(db, action="update_product", user_id=current_user.id, entity="product", entity_id=p.id, request=request)
    return p


@app.patch("/admin/products/{product_id}", response_model=schemas.ProductRead, tags=["Admin"])
def patch_product(
    product_id: int,
    body: schemas.ProductCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    for k, v in body.dict(exclude_unset=True).items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    log_action(db, action="patch_product", user_id=current_user.id, entity="product", entity_id=p.id, request=request)
    return p


@app.delete("/admin/products/{product_id}", tags=["Admin"])
def delete_product(
    product_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    p = db.query(models.Product).get(product_id)
    if not p:
        raise HTTPException(404, "Товар не найден")
    db.delete(p)
    db.commit()
    log_action(db, action="delete_product", user_id=current_user.id, entity="product", entity_id=product_id, request=request)
    return {"ok": True}


# ───── ORDERS ────────────────────────────────────────────────────────────────

@app.post("/orders", response_model=schemas.OrderRead, tags=["Orders"])
def create_order(
    body: schemas.OrderCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_user),
):
    order = models.Order(user_id=current_user.id)
    db.add(order)
    db.flush()
    total = 0
    for item in body.items:
        product = db.query(models.Product).get(item.product_id)
        if not product:
            raise HTTPException(404, f"Товар {item.product_id} не найден")
        oi = models.OrderItem(
            order_id=order.id,
            product_id=item.product_id,
            quantity=item.quantity,
            price=product.price,
        )
        db.add(oi)
        total += product.price * item.quantity
    order.total_price = total
    db.commit()
    db.refresh(order)
    log_action(db, action="create_order", user_id=current_user.id, entity="order", entity_id=order.id,
               detail=f"total={total}", request=request)
    return order


@app.get("/orders", response_model=List[schemas.OrderRead], tags=["Orders"])
def get_my_orders(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_user),
):
    return db.query(models.Order).filter(models.Order.user_id == current_user.id).all()


@app.get("/admin/orders", response_model=List[schemas.OrderRead], tags=["Admin"])
def list_all_orders(db: Session = Depends(get_db), _: models.User = Depends(require_admin)):
    return db.query(models.Order).all()


@app.patch("/admin/orders/{order_id}", tags=["Admin"])
def update_order_status(
    order_id: int,
    status: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    order = db.query(models.Order).get(order_id)
    if not order:
        raise HTTPException(404, "Заказ не найден")
    order.status = status
    db.commit()
    log_action(db, action="update_order_status", user_id=current_user.id, entity="order",
               entity_id=order_id, detail=f"status={status}", request=request)
    return {"ok": True}


# ───── APPOINTMENTS ──────────────────────────────────────────────────────────

@app.post("/appointments", response_model=schemas.AppointmentRead, tags=["Appointments"])
def create_appointment(
    body: schemas.AppointmentCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_user),
):
    apt = models.Appointment(user_id=current_user.id, date=body.date, time=body.time)
    db.add(apt)
    db.commit()
    db.refresh(apt)
    log_action(db, action="create_appointment", user_id=current_user.id, entity="appointment",
               entity_id=apt.id, detail=f"{body.date} {body.time}", request=request)
    return apt


@app.get("/appointments", response_model=List[schemas.AppointmentRead], tags=["Appointments"])
def get_my_appointments(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_user),
):
    return db.query(models.Appointment).filter(models.Appointment.user_id == current_user.id).all()


@app.get("/doctor/appointments", response_model=List[schemas.AppointmentRead], tags=["Doctor"])
def list_all_appointments(db: Session = Depends(get_db), _: models.User = Depends(require_doctor)):
    return db.query(models.Appointment).all()


@app.patch("/doctor/appointments/{appointment_id}", tags=["Doctor"])
def update_appointment(
    appointment_id: int,
    status: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_doctor),
):
    appo = db.query(models.Appointment).get(appointment_id)
    if not appo:
        raise HTTPException(404, "Запись не найдена")
    appo.status = status
    db.commit()
    log_action(db, action="update_appointment", user_id=current_user.id, entity="appointment",
               entity_id=appointment_id, detail=f"status={status}", request=request)
    return {"ok": True}


# ───── JOURNAL ───────────────────────────────────────────────────────────────

@app.get("/admin/journal", response_model=List[schemas.JournalEntryRead], tags=["Admin"])
def get_journal(
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
    _: models.User = Depends(require_admin),
):
    return db.query(models.JournalEntry).order_by(models.JournalEntry.id.desc()).limit(limit).all()


# ───── AI ────────────────────────────────────────────────────────────────────

FACE_SHAPE_MAP = {
    "round":    {"recommended_shapes": ["Кошачий глаз", "Авиаторы", "Вайфареры"],  "shape_codes": ["cat", "aviator", "wayfarer"]},
    "oval":     {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],    "shape_codes": ["aviator", "round", "cat"]},
    "square":   {"recommended_shapes": ["Авиаторы", "Круглые", "Кошачий глаз"],    "shape_codes": ["aviator", "round", "cat"]},
    "rect":     {"recommended_shapes": ["Авиаторы", "Круглые"],                     "shape_codes": ["aviator", "round"]},
    "triangle": {"recommended_shapes": ["Авиаторы", "Стрекозы", "Круглые"],        "shape_codes": ["aviator", "round"]},
    "heart":    {"recommended_shapes": ["Авиаторы", "Вайфареры", "Круглые"],       "shape_codes": ["aviator", "wayfarer", "round"]},
    "diamond":  {"recommended_shapes": ["Авиаторы", "Овальные", "Кошачий глаз"],   "shape_codes": ["aviator", "oval", "cat"]},
}


@app.post("/ai/face-analyze", response_model=schemas.FaceAnalysisResult, tags=["AI"])
async def analyze_face(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    image_bytes = await file.read()
    result = analyze_face_image(image_bytes)

    if result.get("error"):
        raise HTTPException(422, detail=result["error"])

    face_shape = result["face_shape"]
    confidence = result["confidence"]
    measurements = result["measurements"]

    meta = FACE_SHAPE_MAP.get(face_shape, FACE_SHAPE_MAP["oval"])
    products = (
        db.query(models.Product)
        .filter(
            models.Product.category == "frame",
            models.Product.shape.in_(meta["shape_codes"]),
        )
        .limit(6)
        .all()
    )

    log_action(
        db,
        action="face_analyze",
        detail=f"shape={face_shape} confidence={confidence}",
        request=request,
    )

    return schemas.FaceAnalysisResult(
        face_shape=face_shape,
        confidence=confidence,
        measurements=schemas.FaceMeasurements(**measurements),
        recommended_shapes=meta["recommended_shapes"],
        recommended_products=products,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
