from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class ChangePassword(BaseModel):
    current_password: str
    new_password: str


class UserUpdate(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str
    city: Optional[str] = None
    profile_photo: Optional[str] = None
    cover_photo: Optional[str] = None
    is_deleted: int
    created_at: datetime

    class Config:
        from_attributes = True


class SensorBase(BaseModel):
    moisture: float
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None


class SensorData(SensorBase):
    device_id: str


class SensorLogOut(SensorBase):
    id: int
    timestamp: datetime
    user_id: Optional[int] = None

    class Config:
        from_attributes = True


class YieldInput(BaseModel):
    nitrogen: float
    rainfall: float
    temp: float


class YieldOut(BaseModel):
    predicted_yield: float
    unit: str = "tons/hectare"


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class KrishiYieldInput(BaseModel):
    crop: str
    maturity_days: int = 120
    mean_temp_gs_C: float
    temp_flowering_C: float
    seasonal_rain_mm: float
    rain_flowering_mm: float
    humidity_mean_pct: float
    soil_pH: float =6.5
    clay_pct: float =30
    soil_N_status_kg_ha: float
    soil_P_status_kg_ha: float
    soil_K_status_kg_ha: float
    fert_N_kg_ha: float
    fert_P_kg_ha: float
    fert_K_kg_ha: float
    irrigation_events: int=3
    ndvi_flowering: float
    ndvi_peak: float
    ndvi_veg_slope: float
    soil_moisture_pct: float
   
    



class KrishiYieldOut(BaseModel):
    predicted_yield: float
    unit: str = "t/ha"
    alerts: list[str] = []
    benchmark_comparison: Optional[str] = None


class FertilizerRecommendationInput(BaseModel):
    crop: str
    target_yield: float
    soil_N: float
    soil_P: float
    soil_K: float
    temperature: float
    ph: float
    moisture: float


class FertilizerRecommendationOutput(BaseModel):
    recommended_N: float
    recommended_P: float
    recommended_K: float
    unit: str = "kg/ha"


class KrishiChatInput(BaseModel):
    session_id: str
    query: str
    yield_context: Optional[dict] = None
    language: str = "auto"


class KrishiChatOut(BaseModel):
    answer: str


class FieldCreate(BaseModel):
    name: str
    crop: str
    area_acres: float
    lat: float
    lon: float


class FieldOut(FieldCreate):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True
