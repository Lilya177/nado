from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime
import re


class UserCreate(BaseModel):
    nickname: str
    phone: str
    email: str
    password: str

    @field_validator("nickname")
    @classmethod
    def validate_nickname(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2 or len(v) > 30:
            raise ValueError("Никнейм должен быть от 2 до 30 символов")
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValueError("Никнейм может содержать только латиницу, цифры, дефис и подчёркивание")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        v = v.strip()
        digits = re.sub(r'\D', '', v)
        if not (digits.startswith('7') or digits.startswith('8')):
            raise ValueError("Номер телефона должен начинаться с +7 или 8")
        if len(digits) != 11:
            raise ValueError("Номер телефона должен содержать 11 цифр (например, +7 (999) 123-45-67)")
        return '+' + digits if digits.startswith('7') else '+7' + digits[1:]

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Некорректный формат email")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Пароль должен содержать минимум 8 символов")
        if not re.search(r'[A-Za-z]', v):
            raise ValueError("Пароль должен содержать хотя бы одну букву")
        if not re.search(r'\d', v):
            raise ValueError("Пароль должен содержать хотя бы одну цифру")
        return v


class UserOut(BaseModel):
    id: int
    nickname: str
    phone: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool
    role: str

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class AttributeRead(BaseModel):
    id: int
    key: str
    value: str

    class Config:
        from_attributes = True


class ImageRead(BaseModel):
    id: int
    url: str
    is_primary: bool

    class Config:
        from_attributes = True


class ProductCreate(BaseModel):
    title: str
    brand: str
    type: str
    description: str
    price: float
    shape: Optional[str] = None
    image_url: Optional[str] = None
    stock: int = 0


class ProductOut(BaseModel):
    id: int
    title: str
    brand: str
    type: str
    shape: Optional[str] = None
    description: str
    price: float
    stock: int
    in_stock: bool
    image_url: Optional[str] = None
    attributes: List[AttributeRead] = []
    images: List[ImageRead] = []

    class Config:
        from_attributes = True


class ProductPaginationResponse(BaseModel):
    items: List[ProductOut]
    total: int
    page: int
    limit: int


class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int


class OrderItemRead(BaseModel):
    product: ProductOut
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


class FaceMeasurements(BaseModel):
    face_ratio: float
    jaw_ratio: float
    forehead_ratio: float


class FaceAnalysisResult(BaseModel):
    face_shape: Optional[str]
    confidence: float
    measurements: Optional[FaceMeasurements]
    recommended_shapes: List[str]
    recommended_products: List[ProductOut]
    error: Optional[str] = None
