from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("train", "xgb_traffic_predictor.pkl")
MAX_SIGNAL_DURATION = 90  # seconds

# Load model at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

app = FastAPI()

class PredictRequest(BaseModel):
    Hour: int = Field(..., ge=0, le=23)
    Day: int = Field(..., ge=1, le=31)
    DayOfWeek: int = Field(..., ge=0, le=6)
    Month: int = Field(..., ge=1, le=12)
    Vehicles: int = Field(..., ge=0)
    emergency_vehicle: bool = False

@app.post("/predict_signal_time")
async def predict_signal_time(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if req.emergency_vehicle:
        return {
            "predicted_vehicle_count": req.Vehicles,
            "signal_duration": MAX_SIGNAL_DURATION,
            "emergency_override": True
        }
    try:
        input_arr = np.array([[req.Hour, req.Day, req.DayOfWeek, req.Month, req.Vehicles]])
        pred = model.predict(input_arr)[0]
        pred_count = int(round(pred))
        # Simple logic: 2 seconds per vehicle, capped at MAX_SIGNAL_DURATION
        signal_time = min(MAX_SIGNAL_DURATION, max(10, int(pred_count * 2)))
        return {
            "predicted_vehicle_count": pred_count,
            "signal_duration": signal_time,
            "emergency_override": False
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
