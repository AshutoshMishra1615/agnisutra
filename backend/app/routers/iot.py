from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json

from .. import schemas, models
from ..database import get_db
from ..manager import manager
from .auth import get_current_user

router = APIRouter()


@router.get("/my-logs", response_model=List[schemas.SensorLogOut])
def get_my_logs(
    skip: int = 0, 
    limit: int = 100, 
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get sensor logs for the currently logged-in user."""
    logs = db.query(models.SensorLog)\
             .filter(models.SensorLog.user_id == current_user.id)\
             .order_by(models.SensorLog.timestamp.desc())\
             .offset(skip)\
             .limit(limit)\
             .all()
    return logs


@router.get("/latest", response_model=schemas.SensorLogOut)
def get_latest_log(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the most recent sensor reading for the logged-in user."""
    log = db.query(models.SensorLog)\
            .filter(models.SensorLog.user_id == current_user.id)\
            .order_by(models.SensorLog.timestamp.desc())\
            .first()
    if not log:
        raise HTTPException(status_code=404, detail="No sensor data found")
    return log


@router.post("/update", response_model=schemas.SensorLogOut)
async def update_sensor(data: schemas.SensorData, db: Session = Depends(get_db)):
    """Accept incoming IoT sensor payloads."""
    # Find ALL users by device_id
    users = db.query(models.User).filter(models.User.device_id == data.device_id).all()
    if not users:
        raise HTTPException(status_code=404, detail="Device ID not registered to any user")

    # Create a log for the first user to return as response (and save for all)
    first_log = None

    for user in users:
        sensor_log = models.SensorLog(
            user_id=user.id,
            moisture=data.moisture,
            nitrogen=data.nitrogen,
            phosphorus=data.phosphorus,
            potassium=data.potassium,
            temperature=data.temperature,
            humidity=data.humidity
        )
        db.add(sensor_log)
        
        if first_log is None:
            first_log = sensor_log

    db.commit()
    if first_log:
        db.refresh(first_log)

    # Real-time Alert Logic
    alerts = []
    if data.moisture is not None and data.moisture < 30:
        alerts.append(f"CRITICAL: Low soil moisture ({data.moisture}%)")
    
    if data.temperature is not None and data.temperature > 40:
        alerts.append(f"WARNING: High temperature ({data.temperature}°C)")

    if alerts:
        # Broadcast to ALL users associated with this device
        for user in users:
            message_payload = {
                "user_id": user.id,
                "messages": alerts,
                "timestamp": str(first_log.timestamp) if first_log else ""
            }
            await manager.broadcast(json.dumps(message_payload))

    return first_log


@router.post("/sensor", response_model=schemas.SensorLogOut)
def receive_sensor(data: schemas.SensorData, db: Session = Depends(get_db)):
    """Legacy endpoint - redirects to update logic (kept for backward compatibility if needed)."""
    sensor_log = models.SensorLog(
        user_id=data.user_id,
        moisture=data.moisture,
        nitrogen=data.nitrogen,
        phosphorus=data.phosphorus,
        potassium=data.potassium,
        temperature=data.temperature,
        humidity=data.humidity
    )
    db.add(sensor_log)
    db.commit()
    db.refresh(sensor_log)
    return sensor_log
