from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"


def load_dataset():
    local_csv = Path(__file__).with_name("insurance.csv")
    if local_csv.exists():
        print(f"Loading dataset from {local_csv}...")
        return pd.read_csv(local_csv)

    print("Downloading dataset from public source...")
    try:
        return pd.read_csv(DATA_URL)
    except Exception as exc:
        raise FileNotFoundError(
            "Could not load the dataset. Place insurance.csv next to train.py or check network access."
        ) from exc


df = load_dataset()
print("Data loaded successfully!")

# 2. Preprocess Data
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
# Ensure consistent column ordering for the frontend
features = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
X = df[features].astype(np.float32).values
y = df['charges'].astype(np.float32).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Save the scaler for the web app
joblib.dump(scaler, 'scaler.pkl')

# 3. PyTorch Setup
class InsuranceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(InsuranceDataset(X_train, y_train), batch_size=32, shuffle=True)

class ActuarialPredictor(nn.Module):
    def __init__(self, input_dim):
        super(ActuarialPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.network(x)

model = ActuarialPredictor(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 4. Train Model
print("Training model...")
for epoch in range(150):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

# Save the trained model weights
torch.save(model.state_dict(), 'model.pth')
print("Training complete. Saved model.pth and scaler.pkl")