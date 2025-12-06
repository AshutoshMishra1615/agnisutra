from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import schemas, models
from ..database import get_db
from ..manager import manager

router = APIRouter()


@router.post("/update", response_model=schemas.SensorLogOut)
async def update_sensor(data: schemas.SensorData, db: Session = Depends(get_db)):
    """Accept incoming IoT sensor payloads and check for alerts."""
    # 1. Save to DB
    sensor_log = models.SensorLog(
        moisture=data.moisture,
        nitrogen=data.nitrogen,
        phosphorus=data.phosphorus,
        potassium=data.potassium
    )
    db.add(sensor_log)
    db.commit()
    db.refresh(sensor_log)

    # 2. Check Alert (Phase 6 will connect this to WebSocket)
    if data.moisture < 30:
        # WebSocket broadcast
        print(f"CRITICAL: Low moisture detected")
        await manager.broadcast(f"CRITICAL: Low moisture detected")

    return sensor_log


@router.post("/sensor", response_model=schemas.SensorLogOut)
def receive_sensor(data: schemas.SensorData, db: Session = Depends(get_db)):
    """Legacy endpoint - redirects to update logic (kept for backward compatibility if needed)."""
    # Reusing the logic from update_sensor would be better, but for now just duplicate simple save
    sensor_log = models.SensorLog(
        moisture=data.moisture,
        nitrogen=data.nitrogen,
        phosphorus=data.phosphorus,
        potassium=data.potassium
    )
    db.add(sensor_log)
    db.commit()
    db.refresh(sensor_log)
    return sensor_log
