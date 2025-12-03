import torch
import numpy as np
import pandas as pd
import joblib

from train import LSTMRegressor, device


def predict_once():
    # load scalers + model
    feature_scaler = joblib.load("models/feature_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    model = LSTMRegressor().to(device)
    model.load_state_dict(torch.load("models/btc_lstm.pth", map_location=device))
    model.eval()

    # load CSV again for last 100 points
    df = pd.read_csv("data/coin_Bitcoin.csv", parse_dates=["Date"])
    df = df.set_index("Date")

    df["pct_return"] = df["Close"].pct_change()
    df["log_return"] = np.log1p(df["pct_return"])
    df = df.dropna()

    features = df[["Close", "log_return"]].values

    # last 100
    seq = features[-100:]
    seq_scaled = feature_scaler.transform(seq)

    X = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(X).cpu().numpy()
        pred = target_scaler.inverse_transform(pred_scaled)[0][0]

    print("Predicted Next BTC Price:", pred)


if __name__ == "__main__":
    predict_once()
