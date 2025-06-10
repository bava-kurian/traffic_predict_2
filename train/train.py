import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from time import sleep

# Step 1: Load dataset
print("[10%] Loading dataset...")
df = pd.read_csv("traffic.csv")
sleep(0.5)

# Step 2: Preprocess datetime and engineer features
print("[25%] Preprocessing data...")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Month'] = df['DateTime'].dt.month
df = df.drop(columns=['ID', 'DateTime'])

# Step 3: Create prediction target (next hour vehicle count)
print("[40%] Creating prediction target...")
df['Target'] = df.groupby('Junction')['Vehicles'].shift(-1)
df.dropna(inplace=True)

# Step 4: Train-test split
print("[55%] Splitting train and test sets...")
X = df[['Hour', 'Day', 'DayOfWeek', 'Month', 'Vehicles']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
print("[70%] Training XGBoost model...")
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="rmse"
)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# Step 6: Save model
print("[85%] Saving model to 'xgb_traffic_predictor.pkl'...")
joblib.dump(model, "xgb_traffic_predictor.pkl")

# Step 7: Predict and evaluate
print("[95%] Evaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n✅ Model Accuracy (R² Score): {r2 * 100:.2f}%")
print(f"📉 Mean Squared Error (MSE): {mse:.2f}")

# Step 8: Plot training vs validation error
print("[100%] Plotting learning curves...")

results = model.evals_result()
epochs = len(results['validation_0']['rmse'])

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), results['validation_0']['rmse'], label='Train RMSE')
plt.plot(range(epochs), results['validation_1']['rmse'], label='Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('XGBoost Training vs Validation RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curve.png')
plt.show()
