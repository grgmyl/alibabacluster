import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Paths / device

BASE_DIR = Path("/content/drive/MyDrive/project_5G")
FILE_PATH = BASE_DIR / "lstm_ready_aggregated.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


df = pd.read_csv(FILE_PATH)

print(df.head())
print(df.columns.tolist())

features = ["cpu_usage", "memory_usage"]
values = df[features].values.astype(np.float32)

print("Dataset size:", len(values))


def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def inverse_3d(arr, scaler):
    n, h, f = arr.shape
    flat = arr.reshape(-1, f)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(n, h, f)

def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def evaluate_metrics(y_true, y_pred):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mse_val = mean_squared_error(y_true_flat, y_pred_flat)
    mae_val = mean_absolute_error(y_true_flat, y_pred_flat)
    mape_val = mape(y_true_flat, y_pred_flat)

    return mse_val, mae_val, mape_val


# LSTM Model

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.2, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, horizon * input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1, self.horizon, self.input_size)


LOOKBACK = 24
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3

horizons = {
    "Short (H=5)": 5,
    "Long (H=10)": 10
}


# Train / Test split

n = len(values)
train_end = int(n * 0.80)

train_raw = values[:train_end]
test_raw = values[train_end:]

print("Train size:", len(train_raw))
print("Test size :", len(test_raw))

all_results = []

for name, horizon in horizons.items():
    print(f"\n{'='*50}")
    print(name)
    print(f"{'='*50}")

  
    # Scaling: 
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

   
    # sequences

    X_train, y_train = create_sequences(train_scaled, LOOKBACK, horizon)
    X_test, y_test = create_sequences(test_scaled, LOOKBACK, horizon)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

   
    # Tensors / loaders
  
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

 
    # Model / loss / optimizer
 
    model = LSTMModel(horizon=horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []

   
    # Training
  
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.6f}")

   
    # Predictions
  
    model.eval()
    with torch.no_grad():
        preds_test = model(X_test_t.to(device)).cpu().numpy()

    y_test_inv = inverse_3d(y_test, scaler)
    preds_test_inv = inverse_3d(preds_test, scaler)

  
    # Metrics
  
    mse_val, mae_val, mape_val = evaluate_metrics(y_test_inv, preds_test_inv)

    print("\nOverall metrics")
    print(f"MSE  : {mse_val:.6f}")
    print(f"MAE  : {mae_val:.6f}")
    print(f"MAPE : {mape_val:.2f}%")

    all_results.append({
        "Horizon": name,
        "Model": "LSTM",
        "MSE": mse_val,
        "MAE": mae_val,
        "MAPE (%)": mape_val
    })

    # per-feature metrics
    for i, feat in enumerate(features):
        feat_mse = mean_squared_error(
            y_test_inv[:, :, i].reshape(-1),
            preds_test_inv[:, :, i].reshape(-1)
        )
        feat_mae = mean_absolute_error(
            y_test_inv[:, :, i].reshape(-1),
            preds_test_inv[:, :, i].reshape(-1)
        )
        feat_mape = mape(
            y_test_inv[:, :, i].reshape(-1),
            preds_test_inv[:, :, i].reshape(-1)
        )

        print(f"\n{feat} metrics")
        print(f"  MSE  : {feat_mse:.6f}")
        print(f"  MAE  : {feat_mae:.6f}")
        print(f"  MAPE : {feat_mape:.2f}%")

   
    # Plot predictions
   
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f"LSTM Forecast — {name}", fontsize=13, fontweight='bold')

    labels = ["CPU Usage", "Memory Usage"]

    for i in range(2):
        axes[i].plot(y_test_inv[:80, 0, i], label="Actual", linewidth=1.3)
        axes[i].plot(preds_test_inv[:80, 0, i], "--", label="Predicted", linewidth=1.3)
        axes[i].set_title(labels[i])
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[1].set_xlabel("Test Sample Index")
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"lstm_forecast_{name.split()[0].lower()}.png", dpi=120)
    plt.show()

   
    # Plot
  
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.title(f"LSTM Training Curve — {name}", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"lstm_loss_{name.split()[0].lower()}.png", dpi=120)
    plt.show()


#results

results_df = pd.DataFrame(all_results)
print("\nFinal results:")
print(results)

results_df.to_csv(BASE_DIR / "lstm_results.csv", index=False)
