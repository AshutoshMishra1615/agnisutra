from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str

    class Config:
        from_attributes = True


class SensorData(BaseModel):
    moisture: float
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None


class SensorLogOut(SensorData):
    id: int
    timestamp: datetime
    farm_id: Optional[int] = None

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
    variety_group: str = "early"
    maturity_days: int = 120
    base_yield_potential_t_ha: float = 2.5
    mean_temp_gs_C: float
    temp_flowering_C: float
    seasonal_rain_mm: float
    rain_flowering_mm: float
    humidity_mean_pct: float
    solar_MJ_m2_day: float
    soil_pH: float
    soil_oc_pct: float
    clay_pct: float
    soil_N_status_kg_ha: float
    soil_P_status_kg_ha: float
    soil_K_status_kg_ha: float
    soil_texture: str
    soil_depth_cm: float
    season_length_days: int
    plant_density_plants_m2: float
    irrigation_events: int
    herbicide_apps: int
    insecticide_apps: int
    fungicide_apps: int
    weed_pressure_index: float
    pest_pressure_index: float
    disease_pressure_index: float
    ndvi_early: float
    ndvi_flowering: float
    ndvi_peak: float
    ndvi_late: float
    ndvi_veg_slope: float
    seed_moisture_pct: float
    fert_N_kg_ha: float
    fert_P_kg_ha: float
    fert_K_kg_ha: float
    sowing_doy: int
    # Optional context
    district_display: Optional[str] = None
    pincode: Optional[str] = None


class KrishiYieldOut(BaseModel):
    predicted_yield: float
    unit: str = "t/ha"
    alerts: list[str] = []
    benchmark_comparison: Optional[str] = None


class KrishiChatInput(BaseModel):
    session_id: str
    query: str
    yield_context: Optional[dict] = None
    language: str = "auto"


class KrishiChatOut(BaseModel):
    answer: str
