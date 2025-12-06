from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import numpy as np

from .. import schemas, models
from ..database import get_db, engine
from .auth import get_current_user
from ..ml import ml_models

router = APIRouter()


@router.post("/predict", response_model=schemas.YieldOut)
def predict_yield(
    payload: schemas.YieldInput, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Predict yield based on input parameters using the loaded ML model."""
    model = ml_models.get("yield_model")
    
    if model:
        try:
            # Prepare input vector: [nitrogen, rainfall, temp]
            # Note: Ensure the order matches how the model was trained
            input_data = np.array([[payload.nitrogen, payload.rainfall, payload.temp]])
            prediction = model.predict(input_data)[0]
            return {"predicted_yield": round(float(prediction), 3)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    else:
        # Fallback logic if model is not loaded
        predicted_val = (payload.nitrogen * 0.05) + (payload.rainfall * 0.02) + (payload.temp * 0.1)
        return {"predicted_yield": round(predicted_val, 3)}
