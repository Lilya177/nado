from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    user = "user"
    doctor = "doctor"
    admin = "admin"


class ProductCategory(str, enum.Enum):
    frame = "frame"
    lens = "lens"
    case = "case"
    liquid = "liquid"


class OrderStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class AppointmentStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    phone = Column(String, unique=True, index=True)
    password_hash = Column(String)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    role = Column(String, default=UserRole.user)
    orders = relationship("Order", back_populates="user", foreign_keys="Order.user_id")
    appointments = relationship("Appointment", back_populates="user", foreign_keys="Appointment.user_id")
    journal_entries = relationship("JournalEntry", back_populates="user")


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    brand = Column(String)
    category = Column(String)
    shape = Column(String, nullable=True)
    description = Column(Text)
    price = Column(Float)
    in_stock = Column(Boolean, default=True)
    image_url = Column(String, nullable=True)


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    total_price = Column(Float, default=0)
    status = Column(String, default=OrderStatus.pending)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="orders", foreign_keys=[user_id])
    items = relationship("OrderItem", back_populates="order", cascade="all,delete")


class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer, default=1)
    price = Column(Float)
    order = relationship("Order", back_populates="items")
    product = relationship("Product")


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    date = Column(String)
    time = Column(String)
    status = Column(String, default=AppointmentStatus.pending)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="appointments", foreign_keys=[user_id])


class JournalEntry(Base):
    """Журнал действий пользователей."""
    __tablename__ = "journal"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable — для анонимных
    action = Column(String, nullable=False)       # напр. "create_order", "login", "face_analyze"
    entity = Column(String, nullable=True)        # напр. "order", "appointment"
    entity_id = Column(Integer, nullable=True)    # id объекта если есть
    detail = Column(Text, nullable=True)          # доп. инфо в свободной форме
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="journal_entries")
