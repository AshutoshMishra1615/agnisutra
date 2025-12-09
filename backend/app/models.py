from sqlalchemy import Column, Integer, String, Float, DateTime, func, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="farmer")
    
    # Profile Fields
    profile_photo = Column(String, nullable=True)
    cover_photo = Column(String, nullable=True)
    is_deleted = Column(Integer, default=0) # 0: Active, 1: Deleted
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Location for Weather Forecasts
    city = Column(String, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)

    # IoT Configuration
    device_id = Column(String, nullable=True) # e.g., MAC Address of ESP32


class SensorLog(Base):
    __tablename__ = "sensor_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Soil Nutrients
    nitrogen = Column(Float, nullable=True)
    phosphorus = Column(Float, nullable=True)
    potassium = Column(Float, nullable=True)
    
    # Environmental Data
    moisture = Column(Float)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="sensor_logs")


class Field(Base):
    __tablename__ = "fields"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    crop = Column(String, nullable=False)
    area_acres = Column(Float, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="fields")


class YieldRecord(Base):
    __tablename__ = "yield_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    crop = Column(String, index=True)
    area_hectare = Column(Float)
    observed_yield = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="yields")
