import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Load and Prepare Data here
swe_file = "/scratch/project/hackathon/data/SnowpackPredictionChallenge/input_data/swe_data/SWE_values_all.csv"
station_file = "/scratch/project/hackathon/data/SnowpackPredictionChallenge/input_data/swe_data/Station_Info.csv"

output_dir = "predictions_output"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
batch_size = 5000 
all_predictions = []

# Load SWE data and station info
df_swe = pd.read_csv(swe_file, nrows=10000)  # Load more data if needed #skiprows=lambda x: x % 5 != 0
df_stations = pd.read_csv(station_file)

# Convert date to datetime and extract useful features
df_swe['Date'] = pd.to_datetime(df_swe['Date'])
df_swe['year'] = df_swe['Date'].dt.year
df_swe['month'] = df_swe['Date'].dt.month
df_swe['day'] = df_swe['Date'].dt.day

# Merge with station info to get station names
df = df_swe.merge(df_stations, on=['Latitude', 'Longitude'], how='left')

# Feature selection here
features = ['Latitude', 'Longitude', 'year', 'month', 'day']  # Input features
target = 'SWE'  # Target variable

# Normalize features and target
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

df[features] = feature_scaler.fit_transform(df[features])
df[[target]] = target_scaler.fit_transform(df[[target]])

# Group data by stations to create sequences
sequence_length = 30  # Number of time steps

def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = target[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

# Group by station for time-series data
X, y = create_sequences(df[features].values, df[target].values, sequence_length)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset and DataLoader
class SWEDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_dataset = SWEDataset(X_train, y_train)
val_dataset = SWEDataset(X_val, y_val)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# LSTM Model here
class SWELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SWELSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Regression output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last output for prediction
        return out

# Model parameters
input_size = len(features)
hidden_size = 64
num_layers = 5

model = SWELSTM(input_size, hidden_size, num_layers)

# Loss and Optimizations
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        batch_y_scaled = batch_y.unsqueeze(1)  # Ensure correct shape
        loss = criterion(predictions, batch_y_scaled)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    lr_scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_X, val_y in val_dataloader:
            val_predictions = model(val_X)
            val_y_scaled = val_y.unsqueeze(1)
            val_loss += criterion(val_predictions, val_y_scaled).item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_dataloader):.4f}, Val Loss: {val_loss/len(val_dataloader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# Inference: Predicting SWE for New Data
with torch.no_grad():
    for start_idx in range(0, len(X), batch_size):
        end_idx = min(start_idx + batch_size, len(X))

        # Select batch
        batch_X = X[start_idx:end_idx]
        predictions = model(batch_X)
        predictions = target_scaler.inverse_transform(predictions.numpy())  # Inverse transform to original scale

        # Prepare DataFrame here
        df_batch = pd.DataFrame({
            'Date': df['Date'].iloc[start_idx:end_idx].values,
            'Station': df['Station'].iloc[start_idx:end_idx].values,
            'Latitude': df['Latitude'].iloc[start_idx:end_idx].values,
            'Longitude': df['Longitude'].iloc[start_idx:end_idx].values,
            'Predicted_SWE': predictions.flatten()
        })

        # Append to the CSV file
        csv_filename = os.path.join(output_dir, "predictions2.csv")
        df_batch.to_csv(csv_filename, mode='a', index=False, header=not os.path.exists(csv_filename))

        print(f"Appended {len(df_batch)} predictions to {csv_filename}")

print("All predictions saved successfully!")