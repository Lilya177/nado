from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# ── Auth ──────────────────────────────────────────────

class UserCreate(BaseModel):
    phone: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserRead(BaseModel):
    id: int
    phone: str
    role: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    phone: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


# ── Products ──────────────────────────────────────────

class ProductBase(BaseModel):
    name: str
    brand: str
    category: str
    description: str
    price: float
    shape: Optional[str] = None
    image_url: Optional[str] = None


class ProductCreate(ProductBase):
    pass


class ProductRead(ProductBase):
    id: int
    in_stock: bool

    class Config:
        from_attributes = True


# ── Orders ────────────────────────────────────────────

class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int


class OrderItemRead(BaseModel):
    product: ProductRead
    quantity: int
    price: float

    class Config:
        from_attributes = True


class OrderCreate(BaseModel):
    items: List[OrderItemCreate]


class OrderRead(BaseModel):
    id: int
    total_price: float
    status: str
    created_at: Optional[datetime] = None
    items: List[OrderItemRead]

    class Config:
        from_attributes = True


# ── Appointments ──────────────────────────────────────

class AppointmentCreate(BaseModel):
    date: str
    time: str


class AppointmentRead(BaseModel):
    id: int
    date: str
    time: str
    status: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ── Journal ───────────────────────────────────────────

class JournalEntryRead(BaseModel):
    id: int
    user_id: Optional[int]
    action: str
    entity: Optional[str]
    entity_id: Optional[int]
    detail: Optional[str]
    ip_address: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ── AI ────────────────────────────────────────────────

class FaceMeasurements(BaseModel):
    face_ratio: float
    jaw_ratio: float
    forehead_ratio: float


class FaceAnalysisResult(BaseModel):
    face_shape: Optional[str]
    confidence: float                        # 0.0 – 1.0
    measurements: Optional[FaceMeasurements]
    recommended_shapes: List[str]
    recommended_products: List[ProductRead]
    error: Optional[str] = None
