import pytest
import joblib
import pandas as pd
import os
import numbers

MODEL_PATH = 'xgb_traffic_predictor.pkl'

# Sample test input
test_input = {
    'Hour': 10,
    'Day': 15,
    'DayOfWeek': 2,
    'Month': 6,
    'Vehicles': 25
}

# Expected signal time calculation logic
def calculate_signal_time(predicted_count, emergency=False):
    if emergency:
        return 90
    base = 10
    per_vehicle = 1
    return min(base + int(predicted_count) * per_vehicle, 90)

@pytest.fixture(scope="module")
def model():
    assert os.path.exists("xgb_traffic_predictor.pkl"), f"Model file not found: {MODEL_PATH}"
    return joblib.load("xgb_traffic_predictor.pkl")

def test_model_prediction(model):
    """Ensure model returns a valid float prediction"""
    input_df = pd.DataFrame([test_input])
    pred = model.predict(input_df)[0]
    
    assert isinstance(pred, numbers.Real), "Prediction is not a real number"
    assert 0 <= pred <= 200, f"Prediction {pred} out of expected range"
    print(f"✅ Model predicted: {pred:.2f}")

def test_signal_time_calculation(model):
    """Check signal time logic based on model output"""
    input_df = pd.DataFrame([test_input])
    pred = model.predict(input_df)[0]
    signal_time = calculate_signal_time(pred)

    assert 10 <= signal_time <= 90, "Signal time out of range"
    print(f"✅ Signal time for {pred:.2f} vehicles: {signal_time}s")

def test_emergency_override():
    """Check emergency override sets signal to max time"""
    signal_time = calculate_signal_time(30, emergency=True)
    assert signal_time == 90, "Emergency override failed"
    print(f"✅ Emergency override set signal to {signal_time}s")
