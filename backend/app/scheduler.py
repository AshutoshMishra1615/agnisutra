import asyncio
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import User, SensorLog
from .manager import manager
import random
import json

# Mock Weather Service (Replace with real API call like OpenWeatherMap)
def get_weather_forecast(lat, lon):
    # Simulate API response
    conditions = ["Clear", "Rain", "Storm", "Cloudy", "Heatwave"]
    return {
        "condition": random.choice(conditions),
        "temp": random.uniform(20, 40),
        "rain_prob": random.uniform(0, 100)
    }

async def check_conditions_job():
    print("⏰ Running 30-minute Weather & Moisture Check...")
    db: Session = SessionLocal()
    try:
        users = db.query(User).all()
        for user in users:
            alerts = []
            
            # 1. Check Weather (if location exists)
            if user.lat and user.lon:
                weather = get_weather_forecast(user.lat, user.lon)
                
                if weather["condition"] == "Storm":
                    alerts.append(f"⚠️ STORM ALERT for {user.city}: Heavy rain expected. Delay irrigation.")
                elif weather["condition"] == "Heatwave":
                    alerts.append(f"🔥 HEATWAVE ALERT for {user.city}: High temps. Ensure sufficient irrigation.")
                elif weather["rain_prob"] > 80:
                    alerts.append(f"🌧️ RAIN FORECAST for {user.city}: 80% chance of rain. Skip irrigation today.")

            # 2. Check Latest Moisture Reading
            latest_log = db.query(SensorLog).filter(SensorLog.user_id == user.id).order_by(SensorLog.timestamp.desc()).first()
            
            if latest_log:
                if latest_log.moisture < 30:
                    alerts.append(f"💧 CRITICAL MOISTURE: Soil moisture is low ({latest_log.moisture}%). Turn on pump.")
                elif latest_log.moisture > 90:
                    alerts.append(f"💧 HIGH MOISTURE: Soil is saturated ({latest_log.moisture}%). Stop irrigation.")

            # 3. Send Alerts via WebSocket
            if alerts:
                alert_data = {
                    "user_id": user.id,
                    "messages": alerts
                }
                json_message = json.dumps(alert_data)
                print(f"🚀 Sending Alert to {user.name}: {json_message}")
                await manager.broadcast(json_message)
                    
    except Exception as e:
        print(f"❌ Scheduler Error: {e}")
    finally:
        db.close()
