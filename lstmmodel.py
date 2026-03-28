import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).parent

# ─────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────
cols = ['id', 'timestamp', 'machine_id', 'cpu_usage', 'memory_usage']
df = pd.read_csv(BASE_DIR / "Node_0.csv", names=cols)

df = df.drop_duplicates()
df[['cpu_usage', 'memory_usage']] = df[['cpu_usage', 'memory_usage']].apply(pd.to_numeric, errors='coerce')
df = df.interpolate(method='linear').bfill().ffill()

for col in ['cpu_usage', 'memory_usage']:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df  = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

scaler = MinMaxScaler()
df[['cpu_usage', 'memory_usage']] = scaler.fit_transform(df[['cpu_usage', 'memory_usage']])

data   = df.groupby('timestamp')[['cpu_usage', 'memory_usage']].mean()
data   = data.reset_index(drop=True)
values = data.values  # shape: (T, 2)

print(f"Dataset size: {len(values)} timesteps")


# ─────────────────────────────────────────
# 2. CREATE SEQUENCES
# ─────────────────────────────────────────
def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────
# 3. LSTM MODEL
# ─────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc      = nn.Linear(64, horizon * 2)
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = out[:, -1, :]
        out     = self.fc(out)
        return out.view(-1, self.horizon, 2)


# ─────────────────────────────────────────
# 4. MAPE
# ─────────────────────────────────────────
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


# ─────────────────────────────────────────
# 5. RUN FOR SHORT AND LONG HORIZON
# ─────────────────────────────────────────
LOOKBACK = 10
EPOCHS   = 100     # more epochs = better learning
split    = int(len(values) * 0.8)
train    = values[:split]
test     = values[split:]

horizons = {'Short (H=5)': 5, 'Long (H=10)': 10}

for name, horizon in horizons.items():
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")

    X_train, y_train = create_sequences(train, LOOKBACK, horizon)
    X_test,  y_test  = create_sequences(test,  LOOKBACK, horizon)

    print(f"  Train sequences: {len(X_train)}, Test sequences: {len(X_test)}")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32)

    # DataLoader — trains in small batches, much faster and more stable
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model     = LSTMModel(horizon)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{EPOCHS}  |  Loss: {avg_loss:.6f}")

    # Predictions
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy()
    test_true = y_test.numpy()

    # Metrics
    mse_val  = mean_squared_error(test_true.reshape(-1), test_preds.reshape(-1))
    mae_val  = mean_absolute_error(test_true.reshape(-1), test_preds.reshape(-1))
    mape_val = mape(test_true.reshape(-1), test_preds.reshape(-1))

    print(f"\n  MSE  : {mse_val:.6f}")
    print(f"  MAE  : {mae_val:.6f}")
    print(f"  MAPE : {mape_val:.2f}%")

    # ── Plot: Actual vs Predicted ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f'LSTM — {name}', fontsize=13, fontweight='bold')

    labels = ['CPU Usage', 'Memory Usage']
    colors = [('blue', 'red'), ('orange', 'green')]

    for i in range(2):
        axes[i].plot(test_true[:100, 0, i],  color=colors[i][0], label='Actual',    linewidth=1.2)
        axes[i].plot(test_preds[:100, 0, i], color=colors[i][1], label='Predicted', linewidth=1.2, linestyle='--')
        axes[i].set_title(labels[i])
        axes[i].set_ylabel('Normalized Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[1].set_xlabel('Test Sample Index')
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"lstm_{name.split()[0].lower()}.png", dpi=100)
    plt.show()

    # ── Plot: Training loss ──
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, color='steelblue', linewidth=1.5)
    plt.title(f'Training Loss — {name}', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"lstm_loss_{name.split()[0].lower()}.png", dpi=100)
    plt.show()