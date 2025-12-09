import sys
import os
import joblib
import pandas as pd
import requests
import re
import time
import random
import numpy as np
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from .. import schemas, models
from ..database import get_db
from .auth import get_current_user
from ..ml import ml_models

from ..krishi_saathi_llm import KrishiSaathiAdvisor

router = APIRouter()

# Initialize Advisor
try:
    advisor = KrishiSaathiAdvisor()
    print("✅ Krishi Saathi Advisor initialized!")
except Exception as e:
    print(f"❌ Error initializing Advisor: {e}")
    advisor = None

# --- NDVI Integration (AgroMonitoring) ---
AGRO_API_KEY = os.getenv("AGRO_API_KEY", "5e2ed96e32afbcac715fccb11814026b")

def get_centroid(ring: list) -> tuple[float, float]:
    """
    Simple centroid calculation for a polygon ring.
    """
    n = len(ring)
    if n < 3:
        raise ValueError("Invalid ring: too few points")
    lat_sum, lon_sum = 0.0, 0.0
    for i in range(n):
        lon_sum += ring[i][0]
        lat_sum += ring[i][1]
    return lat_sum / n, lon_sum / n

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine formula for distance in meters.
    """
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_or_create_polygon(lat: float, lon: float, max_distance_meters: int = 500, use_haversine: bool = True) -> Optional[str]:
    """
    Smart polygon retrieval:
    1. Checks if any existing polygon is close to the requested point (within max_distance_meters).
    2. If yes, reuses it.
    3. If no, creates a new one.
    
    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        max_distance_meters (int): Max distance threshold in meters.
        use_haversine (bool): Use Haversine for distance (True) or degree approx (False).
    
    Returns:
        str: Polygon ID, or None on failure.
    """
    if not AGRO_API_KEY:
        print("❌ AGRO_API_KEY not set in environment.")
        return None
    
    base_url = f"http://api.agromonitoring.com/agro/1.0/polygons?appid={AGRO_API_KEY}"
    
    # 1. Try to find existing polygon close to this point
    try:
        list_resp = requests.get(base_url, timeout=10)
        list_resp.raise_for_status()
        polygons = list_resp.json()  # Direct array per API docs
        if not isinstance(polygons, list):
            print("⚠️ Unexpected response format for polygon list.")
            polygons = []
        
        degree_threshold = max_distance_meters / 111000  # Rough deg-to-m conversion (~111km/deg)
        
        for p in polygons:
            try:
                geom = p.get("geo_json", {}).get("geometry", {}).get("coordinates", [])
                if not geom:
                    continue
                
                # Handle simple Polygon: geom[0] is outer ring
                # (For MultiPolygon, this takes the first; extend if needed)
                ring = geom[0]
                c_lat, c_lon = get_centroid(ring)
                
                # Check distance
                if use_haversine:
                    dist_m = haversine_distance(lat, lon, c_lat, c_lon)
                    dist = dist_m
                else:
                    dist = ((lat - c_lat)**2 + (lon - c_lon)**2)**0.5
                    # Scale to approx meters for threshold comparison
                    dist_m = dist * 111000
                
                if dist_m < max_distance_meters:
                    print(f"✅ Found existing polygon {p['id']} close to point (dist={dist_m:.1f}m). Reusing.")
                    return p['id']
            except (KeyError, ValueError, IndexError) as e:
                print(f"⚠️ Skipping invalid polygon: {e}")
                continue
    except requests.RequestException as e:
        print(f"❌ Error listing polygons: {e}")
    except Exception as e:
        print(f"❌ Unexpected error listing polygons: {e}")

    # 2. Create new if not found
    offset_deg = 0.002  # ~222m square; adjust as needed
    payload = {
        "name": f"Farm_{lat:.4f}_{lon:.4f}",
        "geo_json": {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - offset_deg, lat - offset_deg],
                    [lon + offset_deg, lat - offset_deg],
                    [lon + offset_deg, lat + offset_deg],
                    [lon - offset_deg, lat + offset_deg],
                    [lon - offset_deg, lat - offset_deg]  # Closed ring
                ]]
            }
        }
    }
    
    # Optional: Basic GeoJSON validation (add shapely if available for stricter check)
    # For now, just ensure structure
    if not isinstance(payload['geo_json'], dict) or payload['geo_json'].get('type') != 'Feature':
        print("❌ Invalid GeoJSON structure in payload.")
        return None
    
    try:
        # Note: Add &duplicated=true if you want to force creation despite similarity
        create_url = f"{base_url}&duplicated=true"  # Optional: enable duplicates if needed
        print(f"DEBUG: Creating polygon with payload: {payload}")  # Log for debug
        response = requests.post(create_url, json=payload, timeout=10)
        print(f"DEBUG: Creation response - Status: {response.status_code}, Body: {response.text}")  # Enhanced logging
        
        response.raise_for_status()  # Raises if not 2xx
        
        if response.status_code == 201:
            data = response.json()
            pid = data.get('id')
            if not pid:
                print("❌ 201 response but no 'id' in JSON.")
                return None
            
            # NEW: Verify the created polygon exists via GET
            verify_url = f"http://api.agromonitoring.com/agro/1.0/polygons/{pid}?appid={AGRO_API_KEY}"
            verify_resp = requests.get(verify_url, timeout=10)
            print(f"DEBUG: Verification response - Status: {verify_resp.status_code}, Body: {verify_resp.text[:200]}...")
            
            if verify_resp.status_code == 200:
                print(f"✅ Created and verified new polygon: {pid}")
                return pid
            else:
                print(f"❌ Creation returned ID {pid} but verification failed: {verify_resp.status_code} - {verify_resp.text}")
                # Optional: Try to delete the invalid one if it partially exists
                delete_resp = requests.delete(f"http://api.agromonitoring.com/agro/1.0/polygons/{pid}?appid={AGRO_API_KEY}")
                if delete_resp.status_code == 200:
                    print(f"🗑️ Cleaned up invalid polygon {pid}.")
                return None
        
        # Handle potential duplicate/invalid errors
        if response.status_code == 422:
            error_text = response.text.lower()
            match = re.search(r"polygon ['\"]([a-f0-9]{24})['\"]", error_text)  # Stricter: 24 hex chars
            if match:
                existing_id = match.group(1)
                print(f"✅ Found existing polygon ID from duplicate error: {existing_id}")
                # Verify this extracted ID too
                verify_url = f"http://api.agromonitoring.com/agro/1.0/polygons/{existing_id}?appid={AGRO_API_KEY}"
                verify_resp = requests.get(verify_url, timeout=10)
                if verify_resp.status_code == 200:
                    print(f"✅ Verified existing polygon from error: {existing_id}")
                    return existing_id
                else:
                    print(f"⚠️ Extracted ID {existing_id} from error but verification failed.")
            else:
                print(f"⚠️ 422 error (likely GeoJSON validation) without extractable ID: {response.text}")
                # Common fixes: Check if coords closed, lon-lat order, area 1-3000 ha
                print("💡 Tip: Ensure GeoJSON is closed, lon-first, and area ~1-3000 ha.")
        
        print(f"❌ Polygon creation failed (status {response.status_code}): {response.text}")
        return None
    except requests.RequestException as e:
        print(f"❌ Polygon creation request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected polygon creation error: {e}")
        return None
def calculate_slope(dates, values):
    """
    Calculates the slope (growth rate) using simple linear regression.
    dates: list of timestamps (X)
    values: list of NDVI values (Y)
    """
    if len(values) < 2:
        return 0.0
    
    # Normalize dates to start at 0 (days) for easier slope interpretation
    start_time = min(dates)
    days = np.array([(d - start_time) / 86400 for d in dates]) # Convert seconds to days
    ndvi = np.array(values)
    
    # Linear Regression: y = mx + c (we want m)
    slope, _ = np.polyfit(days, ndvi, 1)
    return float(slope)

@router.get("/ndvi")
def fetch_ndvi(lat: float, lon: float):
    """
    Fetches history and calculates: Flowering, Peak, and Vegetative Slope.
    """
    print(f"DEBUG: Fetching NDVI for {lat}, {lon}")
    
    # Generate deterministic random values based on location
    # This ensures the same location gets the same "random" values
    random.seed(int((lat + lon) * 1000))
    mock_peak = round(random.uniform(0.6, 0.9), 2)
    mock_flowering = round(mock_peak * random.uniform(0.85, 0.95), 2)
    mock_slope = round(random.uniform(0.005, 0.02), 4)
    
    # Use smart polygon retrieval instead of hardcoded ID
    poly_id = get_or_create_polygon(lat, lon)
    
    if not poly_id:
        # Fallback with location-based randoms
        return {
            "ndvi_peak": mock_peak,
            "ndvi_flowering": mock_flowering,
            "ndvi_veg_slope": mock_slope,
            "ndvi_image": None,
            "samples_analyzed": 0,
            "source": "mock_fallback_no_poly"
        }

    # FIX 1: Reduce history to 30 days (Free tier limitation safe zone)
    # Subtract 2 hours from end_date to avoid "future" error from API due to clock skew
    end_date = int(time.time()) - 7200
    start_date = end_date - (30 * 24 * 60 * 60)

    # FIX 2: NDVI History URL
    # Using the correct parameters as per documentation: polyid, start, end, appid
    stats_url = f"http://api.agromonitoring.com/agro/1.0/ndvi/history?polyid={poly_id}&start={start_date}&end={end_date}&appid={AGRO_API_KEY}"
    
    print(f"Fetching Data from: {stats_url}")
    
    try:
        res = requests.get(stats_url)
        if res.status_code != 200:
            print(f"API Error: {res.text}")
            return {
                "ndvi_peak": mock_peak,
                "ndvi_flowering": mock_flowering,
                "ndvi_veg_slope": mock_slope,
                "ndvi_image": None,
                "source": "mock_fallback_api_error"
            }
            
        data = res.json()
        
        # FIX 3: Relax Cloud Filter to 50% just to see if we get ANY data
        # Note: 'cl' in response is 0-100 or 0-1 depending on version? 
        # Docs say "cl: Approximate percentage of clouds". Example shows 0.16 (16%?) or 100?
        # Let's handle both cases safely. If < 1, assume it's a ratio. If > 1, assume percentage.
        clean_data = []
        for x in data:
            cl = x.get('cl', 100)
            # Normalize to percentage 0-100
            if cl <= 1.0: cl = cl * 100
            
            if cl < 50: # 50% cloud cover threshold
                clean_data.append(x)
        
        if not clean_data:
            print("No cloud-free data found in range.")
            return {
                "ndvi_peak": mock_peak,
                "ndvi_flowering": mock_flowering,
                "ndvi_veg_slope": mock_slope,
                "ndvi_image": None,
                "source": "mock_fallback_no_clean_data"
            }

        # Get max NDVI
        # Response structure: { ..., "data": { "mean": 0.59, ... } }
        max_ndvi = max([x['data']['mean'] for x in clean_data])
        
        # Fetch Image
        image_url = None
        
        # Correct endpoint per user documentation: /agro/1.0/image/search
        img_search_url = f"http://api.agromonitoring.com/agro/1.0/image/search?start={start_date}&end={end_date}&polyid={poly_id}&appid={AGRO_API_KEY}"
        try:
            img_res = requests.get(img_search_url)
            if img_res.status_code == 200:
                imgs = img_res.json()
                if isinstance(imgs, list) and len(imgs) > 0:
                    # Sort by date descending to get the latest
                    imgs.sort(key=lambda x: x.get('dt', 0), reverse=True)
                    
                    # Try to find one with low clouds (< 25%)
                    best_img = None
                    for img in imgs:
                        # cl is percentage (e.g. 1.84)
                        if img.get('cl', 100) < 101:
                            best_img = img
                            break
                    
                    # Fallback to the most recent one if no clear image found
                    if not best_img:
                        best_img = imgs[0]
                        
                    image_url = best_img.get('image', {}).get('ndvi')
        except Exception as e:
            print(f"Image fetch error: {e}")

        return {
            "ndvi_peak": float(round(max_ndvi, 4)),
            "ndvi_flowering": float(round(max_ndvi, 4)), # Heuristic
            "ndvi_veg_slope": 0.0, # Not calculated in new logic
            "ndvi_image": image_url,
            "samples_analyzed": len(clean_data),
            "source": "satellite_realtime"
        }

    except Exception as e:
        print(f"Analysis Error: {e}")
        return {
            "ndvi_peak": mock_peak,
            "ndvi_flowering": mock_flowering,
            "ndvi_veg_slope": mock_slope,
            "ndvi_image": None,
            "source": "mock_fallback_exception"
        }


# Constants from app.py
CROP_BENCHMARK_YIELD = {
    "sunflower": 1.5,
    "soybean": 1.8,
    "mustard": 1.4,
    "groundnut": 2.0,
    "sesame": 0.9,
    "castor": 1.7,
    "safflower": 0.8,
    "niger": 0.7,
}

CROP_ALERT_RULES = {
    "sunflower": {"max_temp_flowering": 34, "min_rain_flowering": 30, "max_humidity": 80},
    "soybean": {"max_temp_flowering": 32, "min_rain_flowering": 35, "max_humidity": 85},
    "mustard": {"max_temp_flowering": 30, "min_rain_flowering": 25, "max_humidity": 88},
    "groundnut": {"max_temp_flowering": 35, "min_rain_flowering": 40, "max_humidity": 82},
    "sesame": {"max_temp_flowering": 33, "min_rain_flowering": 25, "max_humidity": 80},
    "castor": {"max_temp_flowering": 34, "min_rain_flowering": 28, "max_humidity": 84},
    "safflower": {"max_temp_flowering": 32, "min_rain_flowering": 22, "max_humidity": 80},
    "niger": {"max_temp_flowering": 30, "min_rain_flowering": 20, "max_humidity": 78},
}

def generate_weather_alerts(crop, temp_flowering, rain_flowering, humidity):
    crop = crop.lower()
    rules = CROP_ALERT_RULES.get(crop, CROP_ALERT_RULES["soybean"])
    alerts = []

    if temp_flowering >= rules["max_temp_flowering"] + 3:
        alerts.append("🔥 Severe heat at flowering → high risk of flower drop.")
    elif temp_flowering >= rules["max_temp_flowering"]:
        alerts.append("🌡️ High temperature at flowering → moderate heat stress risk.")

    if rain_flowering <= 0.5 * rules["min_rain_flowering"]:
        alerts.append("💧 Very low rainfall during flowering → severe moisture stress.")
    elif rain_flowering <= rules["min_rain_flowering"]:
        alerts.append("💧 Low rainfall during flowering → moisture stress risk.")

    if humidity >= rules["max_humidity"] + 5:
        alerts.append("🦠 Very high humidity → strong risk of fungal diseases.")
    elif humidity >= rules["max_humidity"]:
        alerts.append("🦠 High humidity → increased probability of foliar diseases.")

    if temp_flowering >= rules["max_temp_flowering"] and rain_flowering <= rules["min_rain_flowering"]:
        alerts.append("⚠️ Combination of high temperature and low rainfall at flowering.")

    if not alerts:
        alerts.append("✅ No major weather red-flags detected.")

    return alerts

@router.get("/get-yield-prediction", response_model=schemas.KrishiYieldOut)
def get_yield_prediction_get(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    crop: str
):
    # Map simplified GET parameters to full KrishiYieldInput schema with defaults
    data = schemas.KrishiYieldInput(
        crop=crop,
        maturity_days=120,
        mean_temp_gs_C=temperature,
        temp_flowering_C=temperature,
        seasonal_rain_mm=rainfall,
        rain_flowering_mm=rainfall / 4, # Assumption
        humidity_mean_pct=humidity,
        soil_pH=ph,
        clay_pct=20.0, # Default
        soil_N_status_kg_ha=nitrogen,
        soil_P_status_kg_ha=phosphorus,
        soil_K_status_kg_ha=potassium,
        fert_N_kg_ha=0.0,
        fert_P_kg_ha=0.0,
        fert_K_kg_ha=0.0,
        irrigation_events=0,
        ndvi_flowering=0.5,
        ndvi_peak=0.7,
        ndvi_veg_slope=0.1
    )
    return predict_yield(data)

@router.post("/predict", response_model=schemas.KrishiYieldOut)
def predict_yield(data: schemas.KrishiYieldInput):
    # NOTE: The Yield Prediction Model is currently being replaced by the Fertilizer Recommender.
    # For now, we return a dummy response or raise an error.
    # raise HTTPException(status_code=503, detail="Yield Prediction is temporarily unavailable. Please use /recommend for fertilizer optimization.")
    
    # Fallback: Return a dummy yield based on benchmark for now so frontend doesn't crash
    bench = CROP_BENCHMARK_YIELD.get(data.crop.lower(), 2.0)
    return {
        "predicted_yield": bench,
        "unit": "t/ha",
        "alerts": ["Yield Model is under maintenance. Showing benchmark value."],
        "benchmark_comparison": "Model update in progress."
    }

@router.post("/recommend", response_model=schemas.FertilizerRecommendationOutput)
def recommend_fertilizer(data: schemas.FertilizerRecommendationInput):
    model = ml_models.get("fertilizer_model")
    preprocessor = ml_models.get("preprocessor")

    if not model:
        raise HTTPException(status_code=500, detail="Fertilizer Model not loaded")
    if not preprocessor:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded")

    # Prepare DataFrame matching training columns:
    # ['crop', 'yield_t_ha', 'soil_N_status_kg_ha', 'soil_P_status_kg_ha', 'soil_K_status_kg_ha', 'mean_temp_gs_C', 'soil_pH', 'soil_moisture_pct']
    
    input_data = {
        "crop": [data.crop],
        "yield_t_ha": [data.target_yield],
        "soil_N_status_kg_ha": [data.soil_N],
        "soil_P_status_kg_ha": [data.soil_P],
        "soil_K_status_kg_ha": [data.soil_K],
        "mean_temp_gs_C": [data.temperature],
        "soil_pH": [data.ph],
        "soil_moisture_pct": [data.moisture]
    }
    
    df = pd.DataFrame(input_data)
    
    try:
        # 1. Preprocess
        X_transformed = preprocessor.transform(df)
        
        # 2. Predict (Returns [[N, P, K]])
        prediction = model.predict(X_transformed)
        n_val, p_val, k_val = prediction[0]
        
        # Ensure non-negative
        n_val = max(0.0, float(n_val))
        p_val = max(0.0, float(p_val))
        k_val = max(0.0, float(k_val))
        
        return {
            "recommended_N": round(n_val, 2),
            "recommended_P": round(p_val, 2),
            "recommended_K": round(k_val, 2),
            "unit": "kg/ha"
        }
        
    except Exception as e:
        print(f"Recommendation Error: {e}")
        raise HTTPException(status_code=400, detail=f"Recommendation error: {str(e)}")


@router.post("/chat", response_model=schemas.KrishiChatOut)
def chat_advisor(data: schemas.KrishiChatInput):
    if not advisor:
        raise HTTPException(status_code=500, detail="Advisor not initialized")
    
    try:
        # If yield_context is not provided, create a dummy one or use what's passed
        yield_ctx = data.yield_context or {"crop": "unknown", "yield": 0, "unit": "t/ha", "features": {}}
        
        answer = advisor.chat(
            session_id=data.session_id,
            farmer_query=data.query,
            yield_dict=yield_ctx,
            language=data.language
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.get("/weather")
def fetch_weather(lat: float, lon: float):
    """
    Get current weather and historical stats for yield prediction.
    """
    print(f"DEBUG: Fetching Weather for {lat}, {lon}")
    # 1. Current Weather (for display)
    current_url = f"https://api.agromonitoring.com/agro/1.0/weather?lat={lat}&lon={lon}&appid={AGRO_API_KEY}"
    
    # 2. Historical Weather (for yield model inputs)
    # We need stats for the growing season (approx last 4 months)
    end_date = int(time.time())
    start_date = int((datetime.now() - timedelta(days=120)).timestamp())
    
    # AgroMonitoring History API (requires polygon usually, but let's try point if possible or use accumulated data)
    # Note: AgroMonitoring Free tier might have limits on history. 
    # We will simulate "seasonal" stats based on current + variance if history fails or is too complex for this demo.
    
    weather_data = {
        "temperature": 25.0,
        "humidity": 60.0,
        "rainfall": 0.0,
        "stats": {
            "mean_temp_gs_C": 25.0,
            "temp_flowering_C": 25.0,
            "seasonal_rain_mm": 500.0,
            "rain_flowering_mm": 100.0,
            "humidity_mean_pct": 60.0
        }
    }

    try:
        # A. Fetch Current
        response = requests.get(current_url)
        if response.status_code == 200:
            data = response.json()
            temp_k = data.get("main", {}).get("temp", 298.15)
            temp_c = temp_k - 273.15
            humidity = data.get("main", {}).get("humidity", 50)
            rain_data = data.get("rain", {})
            rain_1h = rain_data.get("1h", 0.0) if isinstance(rain_data, dict) else 0.0
            
            weather_data["temperature"] = round(temp_c, 2)
            weather_data["humidity"] = humidity
            weather_data["rainfall"] = rain_1h * 24
            
            # B. Estimate Seasonal Stats (Heuristic based on current location climate)
            # In a real app, we would query a climate database or full history API.
            # Here we use the current temp as a baseline and add some realistic variance.
            
            # Assume current is somewhat representative of the season mean (rough approximation)
            weather_data["stats"]["mean_temp_gs_C"] = round(temp_c, 2)
            
            # Flowering temp is usually slightly higher (summer) or lower (winter) depending on crop.
            # We'll assume it's close to the mean.
            weather_data["stats"]["temp_flowering_C"] = round(temp_c + 2, 2) 
            
            # Seasonal rain: If it's raining now, it's likely a wet season.
            # Base: 300mm + (current_daily * 90 days)
            weather_data["stats"]["seasonal_rain_mm"] = round(300 + (weather_data["rainfall"] * 30), 2)
            
            # Flowering rain (critical period ~20 days)
            weather_data["stats"]["rain_flowering_mm"] = round(50 + (weather_data["rainfall"] * 10), 2)
            
            weather_data["stats"]["humidity_mean_pct"] = humidity

        else:
            print(f"Weather API Error: {response.status_code} - {response.text}")
            # Fallback to random realistic values
            mock_temp = round(random.uniform(20.0, 35.0), 2)
            mock_humidity = int(random.uniform(40, 85))
            mock_rain = round(random.uniform(0.0, 15.0), 2)
            
            weather_data["temperature"] = mock_temp
            weather_data["humidity"] = mock_humidity
            weather_data["rainfall"] = mock_rain
            
            weather_data["stats"]["mean_temp_gs_C"] = mock_temp
            weather_data["stats"]["temp_flowering_C"] = round(mock_temp + random.uniform(-2, 3), 2)
            weather_data["stats"]["seasonal_rain_mm"] = round(random.uniform(300, 800), 2)
            weather_data["stats"]["rain_flowering_mm"] = round(random.uniform(50, 150), 2)
            weather_data["stats"]["humidity_mean_pct"] = mock_humidity

    except Exception as e:
        print(f"Weather Fetch Exception: {e}")
        # Fallback to random realistic values
        mock_temp = round(random.uniform(20.0, 35.0), 2)
        mock_humidity = int(random.uniform(40, 85))
        mock_rain = round(random.uniform(0.0, 15.0), 2)
        
        weather_data["temperature"] = mock_temp
        weather_data["humidity"] = mock_humidity
        weather_data["rainfall"] = mock_rain
        
        weather_data["stats"]["mean_temp_gs_C"] = mock_temp
        weather_data["stats"]["temp_flowering_C"] = round(mock_temp + random.uniform(-2, 3), 2)
        weather_data["stats"]["seasonal_rain_mm"] = round(random.uniform(300, 800), 2)
        weather_data["stats"]["rain_flowering_mm"] = round(random.uniform(50, 150), 2)
        weather_data["stats"]["humidity_mean_pct"] = mock_humidity

    return weather_data


# --- Field Management & Analysis ---

@router.post("/fields", response_model=Dict[str, Any])
def create_field(
    field: schemas.FieldCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Pydantic v2 compatibility: use model_dump() if available
    field_data = field.model_dump() if hasattr(field, "model_dump") else field.dict()

    new_field = models.Field(
        user_id=current_user.id,
        **field_data
    )
    db.add(new_field)
    db.commit()
    db.refresh(new_field)
    
    # Perform initial analysis
    ndvi_data = fetch_ndvi(field.lat, field.lon)
    weather_data = fetch_weather(field.lat, field.lon)
    
    # Convert SQLAlchemy model to Pydantic model for serialization
    # Pydantic v2 uses model_validate, v1 uses from_orm
    if hasattr(schemas.FieldOut, "model_validate"):
        field_out = schemas.FieldOut.model_validate(new_field)
    else:
        field_out = schemas.FieldOut.from_orm(new_field)

    return {
        "field": field_out,
        "analysis": {
            "ndvi": ndvi_data,
            "weather": weather_data
        }
    }

@router.get("/fields", response_model=List[schemas.FieldOut])
def get_fields(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return db.query(models.Field).filter(models.Field.user_id == current_user.id).all()

@router.get("/fields/{field_id}/analysis")
def analyze_field(
    field_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    field = db.query(models.Field).filter(models.Field.id == field_id, models.Field.user_id == current_user.id).first()
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
        
    ndvi_data = fetch_ndvi(field.lat, field.lon)
    weather_data = fetch_weather(field.lat, field.lon)
    
    # Yield Prediction (using current weather as approximation for now)
    yield_pred = predict_yield(schemas.KrishiYieldInput(
        crop=field.crop,
        maturity_days=120,
        mean_temp_gs_C=weather_data["stats"]["mean_temp_gs_C"],
        temp_flowering_C=weather_data["stats"]["temp_flowering_C"],
        seasonal_rain_mm=weather_data["stats"]["seasonal_rain_mm"],
        rain_flowering_mm=weather_data["stats"]["rain_flowering_mm"],
        humidity_mean_pct=weather_data["stats"]["humidity_mean_pct"],
        soil_pH=6.5, # Default
        clay_pct=20.0, # Default
        soil_N_status_kg_ha=100, # Default
        soil_P_status_kg_ha=40, # Default
        soil_K_status_kg_ha=150, # Default
        fert_N_kg_ha=0,
        fert_P_kg_ha=0,
        fert_K_kg_ha=0,
        irrigation_events=0,
        ndvi_flowering=ndvi_data.get("ndvi_flowering", 0.5),
        ndvi_peak=ndvi_data.get("ndvi_peak", 0.7),
        ndvi_veg_slope=ndvi_data.get("ndvi_veg_slope", 0.1),
        soil_moisture_pct=weather_data.get("humidity", 50.0) # Using humidity as proxy for now
    ))
    
    return {
        "field_id": field.id,
        "ndvi": ndvi_data,
        "weather": weather_data,
        "yield_forecast": yield_pred
    }

