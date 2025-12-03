import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests


device = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


def train_model():
    print("Loading CSV...")
    df = pd.read_csv("data/coin_Bitcoin.csv", parse_dates=["Date"])
    df = df.set_index("Date")

    # features
    df["pct_return"] = df["Close"].pct_change()
    df["log_return"] = np.log1p(df["pct_return"])
    df = df.dropna()

    features = df[["Close", "log_return"]].values
    targets = df[["Close"]].values

    # split
    split = int(len(features) * 0.8)
    X_train_raw, y_train_raw = features[:split], targets[:split]

    # scalers
    feature_scaler = MinMaxScaler().fit(X_train_raw)
    target_scaler = MinMaxScaler().fit(y_train_raw)

    X_train = feature_scaler.transform(X_train_raw)
    y_train = target_scaler.transform(y_train_raw)

    # sequences
    seq_len = 100
    X_seq, y_seq = [], []

    for i in range(seq_len, len(X_train)):
        X_seq.append(X_train[i-seq_len:i])
        y_seq.append(y_train[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    train_ds = SequenceDataset(X_seq, y_seq)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # model
    model = LSTMRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    print("Training...")
    for epoch in range(1, 21):
        total = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch {epoch}/20 - Loss: {total/len(train_loader):.6f}")

    # save
    torch.save(model.state_dict(), "models/btc_lstm.pth")
    joblib.dump(feature_scaler, "models/feature_scaler.pkl")
    joblib.dump(target_scaler, "models/target_scaler.pkl")

    print("Model saved to models/ folder.")


if __name__ == "__main__":
    train_model()
