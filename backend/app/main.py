from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import bcrypt
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import models
import schemas
from database import SessionLocal, init_db
from auth import create_access_token
from deps import get_db, get_current_user, require_user, require_doctor, require_admin
from journal import log_action
from face_analyzer import analyze_face_image, FACE_SHAPE_MAP, analyze_multi_frames

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
    user = models.User(phone=body.phone, password_hash=hashed,
                       first_name=body.first_name, last_name=body.last_name, role="user")
    db.add(user)
    db.commit()
    db.refresh(user)
    log_action(db, action="register", user_id=user.id, entity="user", entity_id=user.id,
               detail=f"phone={body.phone}", request=request)
    return user

@app.post("/users/login", response_model=schemas.TokenResponse, tags=["Auth"])
def login(body: schemas.LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.phone == body.phone).first()
    if not user or not bcrypt.checkpw(body.password.encode(), user.password_hash.encode()):
        log_action(db, action="login_failed", detail=f"phone={body.phone}", request=request)
        raise HTTPException(401, "Неверный телефон или пароль")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    log_action(db, action="login", user_id=user.id, entity="user", entity_id=user.id,
               detail=f"role={user.role}", request=request)
    return schemas.TokenResponse(access_token=token, user=user)

@app.post("/users/logout", tags=["Auth"])
def logout(request: Request, db: Session = Depends(get_db),
           current_user: models.User = Depends(get_current_user)):
    log_action(db, action="logout", user_id=current_user.id, entity="user",
               entity_id=current_user.id, request=request)
    return {"ok": True}

@app.get("/users/me", response_model=schemas.UserRead, tags=["Auth"])
def get_me(current_user: models.User = Depends(get_current_user)):
    return current_user

@app.get("/users", response_model=List[schemas.UserRead], tags=["Admin"])
def get_users(db: Session = Depends(get_db), _: models.User = Depends(require_admin)):
    return db.query(models.User).all()

# ───── PRODUCTS ──────────────────────────────────────────────────────────────

@app.get("/products", response_model=List[schemas.ProductRead], tags=["Products"])
def get_products(category: Optional[str] = Query(None), db: Session = Depends(get_db),
                 request: Request = None):
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
def create_product(body: schemas.ProductCreate, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = models.Product(**body.dict())
    db.add(p); db.commit(); db.refresh(p)
    log_action(db, action="create_product", user_id=current_user.id, entity="product",
               entity_id=p.id, detail=f"name={p.name}", request=request)
    return p

@app.put("/admin/products/{product_id}", response_model=schemas.ProductRead, tags=["Admin"])
def update_product(product_id: int, body: schemas.ProductCreate, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = db.query(models.Product).get(product_id)
    if not p: raise HTTPException(404, "Товар не найден")
    for k, v in body.dict().items(): setattr(p, k, v)
    db.commit(); db.refresh(p)
    log_action(db, action="update_product", user_id=current_user.id, entity="product",
               entity_id=p.id, detail=f"name={p.name}", request=request)
    return p

@app.delete("/admin/products/{product_id}", tags=["Admin"])
def delete_product(product_id: int, request: Request,
                   db: Session = Depends(get_db), current_user: models.User = Depends(require_admin)):
    p = db.query(models.Product).get(product_id)
    if not p: raise HTTPException(404, "Товар не найден")
    db.delete(p); db.commit()
    log_action(db, action="delete_product", user_id=current_user.id, entity="product",
               entity_id=product_id, request=request)
    return {"ok": True}

# ───── ORDERS ────────────────────────────────────────────────────────────────

@app.post("/orders", response_model=schemas.OrderRead, tags=["Orders"])
def create_order(body: schemas.OrderCreate, request: Request,
                 db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    order = models.Order(user_id=current_user.id)
    db.add(order); db.flush()
    total = 0
    items_desc = []
    for item in body.items:
        product = db.query(models.Product).get(item.product_id)
        if not product: raise HTTPException(404, f"Товар {item.product_id} не найден")
        oi = models.OrderItem(order_id=order.id, product_id=item.product_id,
                              quantity=item.quantity, price=product.price)
        db.add(oi)
        total += product.price * item.quantity
        items_desc.append(f"{product.name}x{item.quantity}")
    order.total_price = total
    db.commit(); db.refresh(order)
    log_action(db, action="create_order", user_id=current_user.id, entity="order",
               entity_id=order.id, detail=f"total={total} items={','.join(items_desc)}", request=request)
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
    if not order: raise HTTPException(404, "Заказ не найден")
    old_status = order.status
    order.status = status; db.commit()
    log_action(db, action="update_order_status", user_id=current_user.id, entity="order",
               entity_id=order_id, detail=f"{old_status}->{status}", request=request)
    return {"ok": True}

# ───── APPOINTMENTS ──────────────────────────────────────────────────────────

@app.post("/appointments", response_model=schemas.AppointmentRead, tags=["Appointments"])
def create_appointment(body: schemas.AppointmentCreate, request: Request,
                       db: Session = Depends(get_db), current_user: models.User = Depends(require_user)):
    apt = models.Appointment(user_id=current_user.id, date=body.date, time=body.time)
    db.add(apt); db.commit(); db.refresh(apt)
    log_action(db, action="create_appointment", user_id=current_user.id, entity="appointment",
               entity_id=apt.id, detail=f"{body.date} {body.time}", request=request)
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
    if not appo: raise HTTPException(404, "Запись не найдена")
    old = appo.status; appo.status = status; db.commit()
    log_action(db, action="update_appointment", user_id=current_user.id, entity="appointment",
               entity_id=appointment_id, detail=f"{old}->{status}", request=request)
    return {"ok": True}

# ───── JOURNAL ───────────────────────────────────────────────────────────────

@app.get("/admin/journal", response_model=List[schemas.JournalEntryRead], tags=["Admin"])
def get_journal(limit: int = Query(100, le=500), action: Optional[str] = Query(None),
                db: Session = Depends(get_db), _: models.User = Depends(require_admin)):
    q = db.query(models.JournalEntry)
    if action:
        q = q.filter(models.JournalEntry.action == action)
    return q.order_by(models.JournalEntry.id.desc()).limit(limit).all()

# ───── AI ────────────────────────────────────────────────────────────────────

@app.post("/ai/face-analyze", response_model=schemas.FaceAnalysisResult, tags=["AI"])
async def analyze_face(request: Request, file: UploadFile = File(...),
                       db: Session = Depends(get_db)):
    """Анализ по ОДНОМУ фото (старый метод)"""
    image_bytes = await file.read()
    result = analyze_face_image(image_bytes)

    if result.get("error"):
        raise HTTPException(422, detail=result["error"])

    return _format_ai_response(result, db, request)

from face_analyzer import analyze_multi_frames

@app.post("/ai/analyze-video")
async def analyze_video(
    frame_front: UploadFile = File(...),
    frame_left:  UploadFile = File(None),
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
    """Вспомогательная функция для сборки финального ответа с товарами"""
    face_shape = result["face_shape"]
    confidence = result["confidence"]
    measurements = result.get("measurements", {"face_ratio": 0, "jaw_ratio": 0, "forehead_ratio": 0})

    meta = FACE_SHAPE_MAP.get(face_shape, FACE_SHAPE_MAP["oval"])
    products = (db.query(models.Product)
                .filter(models.Product.category == "frame",
                        models.Product.shape.in_(meta["shape_codes"]))
                .limit(6).all())

    log_action(db, action="face_analyze_video", 
               detail=f"shape={face_shape} confidence={confidence:.1%}", request=request)

    return schemas.FaceAnalysisResult(
    face_shape=face_shape, 
    confidence=confidence,
    yaw=result.get("yaw", 0),  # Добавь это поле
    # ... остальное
)
# ───── ADMIN PANEL ───────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse, tags=["Admin"])
def admin_panel():
    html = open("admin_panel.html", encoding="utf-8").read()
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/scan")
async def get_scan_page():
    return FileResponse("scan.html")