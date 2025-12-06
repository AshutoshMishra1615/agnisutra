import sys
import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from .. import schemas

# Add backend directory to sys.path to import from Sih_25272
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Sih_25272.krishi_saathi_llm import KrishiSaathiAdvisor

router = APIRouter()

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../Sih_25272/final_yield_model_CatBoost.pkl")
try:
    yield_model = joblib.load(MODEL_PATH)
    print("✅ CatBoost Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading CatBoost model: {e}")
    yield_model = None

# Initialize Advisor
try:
    advisor = KrishiSaathiAdvisor()
    print("✅ Krishi Saathi Advisor initialized!")
except Exception as e:
    print(f"❌ Error initializing Advisor: {e}")
    advisor = None

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

@router.post("/predict", response_model=schemas.KrishiYieldOut)
def predict_yield(data: schemas.KrishiYieldInput):
    if not yield_model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert input to DataFrame
    input_dict = data.dict()
    # Remove optional context fields that are not features
    context_keys = ["district_display", "pincode"]
    features = {k: v for k, v in input_dict.items() if k not in context_keys}
    
    # Add derived/extra fields if needed by model (app.py added flowering_day but it was ignored by model)
    # app.py did: df = df[st.session_state.model.feature_names_]
    
    df = pd.DataFrame([features])
    
    # Ensure categorical columns are strings
    for col in ["crop", "variety_group", "soil_texture"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    try:
        # Reorder columns to match model
        df = df[yield_model.feature_names_]
        prediction = float(yield_model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # Alerts
    alerts = generate_weather_alerts(
        data.crop, 
        data.temp_flowering_C, 
        data.rain_flowering_mm, 
        data.humidity_mean_pct
    )

    # Benchmark
    bench = CROP_BENCHMARK_YIELD.get(data.crop.lower())
    benchmark_msg = None
    if bench:
        ratio = prediction / bench
        benchmark_msg = f"Typical: {bench:.2f} t/ha. Your yield is {ratio*100:.1f}% of benchmark."

    return {
        "predicted_yield": round(prediction, 2),
        "unit": "t/ha",
        "alerts": alerts,
        "benchmark_comparison": benchmark_msg
    }

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
