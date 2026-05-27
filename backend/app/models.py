from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    client = "client"
    admin = "admin"


class ProductType(str, enum.Enum):
    frame = "frame"
    lens = "lens"
    liquid = "liquid"
    case = "case"
    accessory = "accessory"


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
    nickname = Column(String, unique=True, index=True)
    phone = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    role = Column(String, default=UserRole.client)
    orders = relationship("Order", back_populates="user", foreign_keys="Order.user_id")
    appointments = relationship("Appointment", back_populates="user", foreign_keys="Appointment.user_id")
    journal_entries = relationship("JournalEntry", back_populates="user")


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    title = Column(String, index=True)
    brand = Column(String)
    type = Column(String)
    shape = Column(String, nullable=True)
    description = Column(Text)
    price = Column(Float)
    stock = Column(Integer, default=0)
    in_stock = Column(Boolean, default=True)
    image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    attributes = relationship("Attribute", back_populates="product", cascade="all,delete")
    images = relationship("Image", back_populates="product", cascade="all,delete")


class Attribute(Base):
    __tablename__ = "attributes"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)

    product = relationship("Product", back_populates="attributes")


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    url = Column(String, nullable=False)
    is_primary = Column(Boolean, default=False)

    product = relationship("Product", back_populates="images")


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
    __tablename__ = "journal"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    entity = Column(String, nullable=True)
    entity_id = Column(Integer, nullable=True)
    detail = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="journal_entries")
