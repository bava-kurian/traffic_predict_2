import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Step 1: Load dataset
print("[10%] Loading dataset...")
df = pd.read_csv("traffic.csv")

# Step 2: Preprocess datetime and engineer features
print("[25%] Preprocessing data...")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['hour'] = df['DateTime'].dt.hour
df['day'] = df['DateTime'].dt.day
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['month'] = df['DateTime'].dt.month

# Optional: Encode 'Junction' as categorical if needed
if 'Junction' in df.columns:
    df['Junction'] = df['Junction'].astype('category').cat.codes

# Drop unnecessary columns
df = df.drop(columns=['ID', 'DateTime'])

# Step 3: Create prediction target (next hour vehicle count)
print("[40%] Creating prediction target...")
df['Target'] = df.groupby('Junction')['Vehicles'].shift(-1)
df.dropna(inplace=True)

# Step 4: Define features and target variable
features = ['hour', 'day', 'day_of_week', 'month']
if 'Junction' in df.columns:
    features.append('Junction')
X = df[features]
y = df['Target']

# Step 5: Split dataset (80% train, 20% test)
print("[55%] Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
print("[70%] Training XGBRegressor model...")
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Save the Model
print("[85%] Saving model to 'xgb_traffic_predictor.pkl'...")
joblib.dump(model, "xgb_traffic_predictor.pkl")
print("Model saved successfully!")

# Step 8: Make Predictions
print("[95%] Evaluating model...")
y_pred = model.predict(X_test)

# Step 9: Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"\n✅ Model Accuracy (R² Score): {r2 * 100:.2f}%")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

# Step 10: Visualize Predictions
print("[100%] Plotting predictions...")
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual Vehicle Count', color='blue')
plt.plot(y_pred[:100], label='Predicted Vehicle Count', color='red', linestyle='dashed')
plt.xlabel('Sample Index')
plt.ylabel('Vehicle Count')
plt.title('Actual vs Predicted Vehicle Count (XGB)')
plt.legend()
plt.tight_layout()
plt.savefig("traffic_predictions_xgb.png")
print("Plot saved as traffic_predictions_xgb.png")