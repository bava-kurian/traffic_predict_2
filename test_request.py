import requests

url = "http://127.0.0.1:8000/predict_signal_time"
data = {
    "Hour": 14,
    "Day": 6,
    "DayOfWeek": 2,
    "Month": 11,
    "Vehicles": 20
}

response = requests.post(url, json=data)
print(response.json())
