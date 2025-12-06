from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from .. import models, schemas
from ..database import get_db

router = APIRouter()


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get aggregated statistics for the admin dashboard."""
    total_users = db.query(func.count(models.User.id)).scalar()
    total_yield_records = db.query(func.count(models.YieldRecord.id)).scalar()
    total_sensor_logs = db.query(func.count(models.SensorLog.id)).scalar()
    
    # Count unique farms (based on farm_id in sensor logs)
    total_farms = db.query(func.count(func.distinct(models.SensorLog.farm_id))).scalar()

    return {
        "total_users": total_users,
        "total_farms": total_farms,
        "total_yield_predictions": total_yield_records,
        "total_sensor_readings": total_sensor_logs
    }


@router.get("/map-data")
def get_map_data(db: Session = Depends(get_db)):
    """Return lat/longs of all farms joined with their latest sensor reading."""
    # Subquery to find the latest timestamp for each farm_id
    subquery = (
        db.query(
            models.SensorLog.farm_id,
            func.max(models.SensorLog.timestamp).label("max_timestamp")
        )
        .group_by(models.SensorLog.farm_id)
        .subquery()
    )

    # Join the main table with the subquery to get the full record for the latest timestamp
    latest_logs = (
        db.query(models.SensorLog)
        .join(
            subquery,
            (models.SensorLog.farm_id == subquery.c.farm_id) & 
            (models.SensorLog.timestamp == subquery.c.max_timestamp)
        )
        .all()
    )

    # Format the response
    map_data = []
    for log in latest_logs:
        if log.lat is not None and log.lon is not None:
            map_data.append({
                "farm_id": log.farm_id,
                "lat": log.lat,
                "lon": log.lon,
                "moisture": log.moisture,
                "timestamp": log.timestamp
            })
    
    return map_data
