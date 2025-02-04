# Team 5 Model
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your data
#df = pd.read_csv("test.csv", nrows = 100)
df = pd.read_csv("/scratch/project/hackathon/data/SnowpackPredictionChallenge/input_data/meteorological_data/Modified_Output_tmax.csv", nrows = 100)

# Convert date to datetime and extract useful features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Select features and target variable
features = ['lat', 'lon', 'year', 'month', 'day']
target = 'variable_value'

X = df[features].values  # Inputs
y = df[target].values    # Labels

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Ensure it's 2D

from torch.utils.data import Dataset, DataLoader

class TemperatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Dataset
dataset = TemperatureDataset(X_tensor, y_tensor)

# Create DataLoader (for batch processing)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import torch.nn as nn

class TempPredictor(nn.Module):
    def __init__(self, input_size):
        super(TempPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output 1 value (temperature)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
input_size = len(features)  # Number of input features
model = TempPredictor(input_size)

import torch.optim as optim

# Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 2000
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()  # Reset gradients
        predictions = model(batch_X)  # Forward pass
        loss = criterion(predictions, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():  # Disable gradient tracking for inference
    sample_input = X_tensor[:5]  # Take first 5 examples
    predictions = model(sample_input)
    print("Predicted Tmax:", predictions.numpy().flatten())

