from pydantic import BaseModel
from typing import List, Optional

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
    class Config:
        from_attributes = True

class FaceAnalysisResult(BaseModel):
    face_shape: str
    recommended_shapes: List[str]
    recommended_products: List[ProductRead]
