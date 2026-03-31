import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/project_5G")
FILE_PATH = BASE_DIR / "lstm_ready_aggregated.csv"

df = pd.read_csv(FILE_PATH)
data = df[['cpu_usage', 'memory_usage']].values

def create_sequences(data, seq_len, horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+horizon])
    return np.array(X), np.array(y)

SEQ_LEN = 20

X_short, y_short = create_sequences(data, SEQ_LEN, 5)
X_long, y_long = create_sequences(data, SEQ_LEN, 15)

split = int(0.8 * len(X_short))

X_train_s, X_test_s = X_short[:split], X_short[split:]
y_train_s, y_test_s = y_short[:split], y_short[split:]

X_train_l, X_test_l = X_long[:split], X_long[split:]
y_train_l, y_test_l = y_long[:split], y_long[split:]

def build_gru(output_steps):
    model = Sequential([
        GRU(64, return_sequences=False, input_shape=(SEQ_LEN, 2)),
        Dense(output_steps * 2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model_short = build_gru(5)
model_short.fit(X_train_s, y_train_s.reshape(len(y_train_s), -1),
                epochs=20, batch_size=32, verbose=1)

model_long = build_gru(15)
model_long.fit(X_train_l, y_train_l.reshape(len(y_train_l), -1),
               epochs=20, batch_size=32, verbose=1)

pred_s = model_short.predict(X_test_s)
pred_l = model_long.predict(X_test_l)

y_test_s = y_test_s.reshape(len(y_test_s), -1)
y_test_l = y_test_l.reshape(len(y_test_l), -1)

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print("SHORT TERM:")
print("MSE:", mean_squared_error(y_test_s, pred_s))
print("MAE:", mean_absolute_error(y_test_s, pred_s))
print("MAPE:", MAPE(y_test_s, pred_s))

print("\nLONG TERM:")
print("MSE:", mean_squared_error(y_test_l, pred_l))
print("MAE:", mean_absolute_error(y_test_l, pred_l))
print("MAPE:", MAPE(y_test_l, pred_l))
