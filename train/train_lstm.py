import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define LSTM model
class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Custom Dataset
class TrafficDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
        
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.seq_length], self.y[idx+self.seq_length])

def train_lstm_model():
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
    features = ['hour', 'day', 'day_of_week', 'month', 'Vehicles']
    if 'Junction' in df.columns:
        features.append('Junction')
    X = df[features].values
    y = df['Target'].values

    # Normalize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Step 5: Prepare sequences for LSTM
    print("[55%] Preparing sequences for LSTM...")
    seq_length = 24  # Use 24 hours of data to predict next hour
    dataset = TrafficDataset(X_scaled, y_scaled, seq_length)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Step 6: Initialize and train the LSTM model
    print("[70%] Training LSTM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficLSTM(input_size=len(features), hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.float().to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Step 7: Save the Model
    print("[85%] Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': features
    }, "lstm_traffic_predictor.pth")
    print("Model saved successfully!")

    # Step 8: Evaluate Model
    print("[95%] Evaluating model...")
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.float().to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    # Inverse transform predictions and actuals
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)

    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(actuals, predictions)

    print(f"\n✅ Model Accuracy (R² Score): {r2 * 100:.2f}%")
    print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
    print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Step 9: Visualize Predictions
    print("[100%] Plotting predictions...")
    plt.figure(figsize=(10, 5))
    plt.plot(actuals[:100], label='Actual Vehicle Count', color='blue')
    plt.plot(predictions[:100], label='Predicted Vehicle Count', color='red', linestyle='dashed')
    plt.xlabel('Sample Index')
    plt.ylabel('Vehicle Count')
    plt.title('Actual vs Predicted Vehicle Count (LSTM)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("traffic_predictions_lstm.png")
    print("Plot saved as traffic_predictions_lstm.png")

    return model, scaler_X, scaler_y, features

if __name__ == "__main__":
    train_lstm_model() 