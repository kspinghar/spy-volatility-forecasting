"""
train_and_save.py
Mirror the AMI4100 SPY volatility notebook: download SPY, build forward-vol
supervised dataset, train CNN / LSTM / CNN-LSTM for horizons 5 and 10,
save all artefacts into ./artifacts/ for the Gradio Space.

Run once on the same machine that hosts the notebook venv. Re-run any time
SPY data should be refreshed for the demo's bundled snapshot.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ───────────────────────── constants (match the notebook) ─────────────────────────
TICKER = "SPY"
START_DATE = "1993-01-01"
END_DATE = None  # None = latest available
LOOKBACK = 120
HORIZONS = [5, 10]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85  # split point after train
EPOCHS = 50
BATCH = 64
SEED = 42

ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)

# ───────────────────────── data download ─────────────────────────
print(f"Downloading {TICKER} {START_DATE} → {END_DATE or 'today'}...")
raw = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
df = raw.loc[:, ["Close"]].dropna().copy()
if df.empty:
    raise RuntimeError("No SPY data downloaded. Check yfinance / network.")
print(f"  rows: {len(df)} | {df.index.min().date()} → {df.index.max().date()}")

# Bundle the snapshot so the Space can render dates without yfinance at runtime.
df_snapshot = df.reset_index()
df_snapshot.columns = ["Date", "Close"]
df_snapshot["Date"] = df_snapshot["Date"].dt.strftime("%Y-%m-%d")
df_snapshot.to_csv(ART / "spy_snapshot.csv", index=False)
print(f"  snapshot saved: {ART / 'spy_snapshot.csv'}")

# ───────────────────────── log returns + vol targets ─────────────────────────
df["log_close"] = np.log(df["Close"])
df["r"] = df["log_close"].diff()
df = df.dropna()
df["r2"] = df["r"] ** 2


def forward_realized_vol(r2_series, H):
    fwd_mean = r2_series.shift(-1).rolling(H).mean().shift(-(H - 1))
    return np.sqrt(fwd_mean)


for h in HORIZONS:
    df[f"vol_{h}"] = forward_realized_vol(df["r2"], h)
df = df.dropna()

# ───────────────────────── supervised dataset ─────────────────────────
r = df["r"].to_numpy()
vol_arrays = {h: df[f"vol_{h}"].to_numpy() for h in HORIZONS}

X, y_dict = [], {h: [] for h in HORIZONS}
for t in range(LOOKBACK, len(df)):
    X.append(r[t - LOOKBACK:t])
    for h in HORIZONS:
        y_dict[h].append(vol_arrays[h][t])
X = np.array(X, dtype=np.float32)[..., None]
y_dict = {h: np.array(y_dict[h], dtype=np.float32) for h in HORIZONS}
print(f"X shape: {X.shape}")

# ───────────────────────── train/val/test split (chronological) ─────────────────────────
n = len(X)
train_end = int(TRAIN_FRAC * n)
val_end = int(VAL_FRAC * n)
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = {}, {}, {}
for h in HORIZONS:
    y_train[h], y_val[h], y_test[h] = (
        y_dict[h][:train_end],
        y_dict[h][train_end:val_end],
        y_dict[h][val_end:],
    )
print(f"Train/Val/Test: {len(X_train)} / {len(X_val)} / {len(X_test)}")

# ───────────────────────── normalization (train stats only) ─────────────────────────
mean = float(X_train.mean())
std = float(X_train.std()) or 1.0
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std
print(f"norm mean={mean:.6f} std={std:.6f}")

# ───────────────────────── model builders ─────────────────────────
def build_cnn(L):
    m = keras.Sequential([
        layers.Input(shape=(L, 1)),
        layers.Conv1D(16, 5, activation="relu"),
        layers.MaxPool1D(2),
        layers.Conv1D(32, 5, activation="relu"),
        layers.MaxPool1D(2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                       keras.metrics.RootMeanSquaredError(name="rmse")])
    return m


def build_lstm(L, units=32, dropout=0.1):
    m = keras.Sequential([
        layers.Input(shape=(L, 1)),
        layers.LSTM(units, dropout=dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                       keras.metrics.RootMeanSquaredError(name="rmse")])
    return m


def build_cnn_lstm(L, cnn_filters=16, kernel_size=5, lstm_units=32, dropout=0.1):
    m = keras.Sequential([
        layers.Input(shape=(L, 1)),
        layers.Conv1D(cnn_filters, kernel_size, activation="relu"),
        layers.MaxPool1D(2),
        layers.Conv1D(cnn_filters * 2, kernel_size, activation="relu"),
        layers.MaxPool1D(2),
        layers.LSTM(lstm_units, dropout=dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                       keras.metrics.RootMeanSquaredError(name="rmse")])
    return m


BUILDERS = {"cnn": build_cnn, "lstm": build_lstm, "hybrid": build_cnn_lstm}


def baseline_persistence(X_norm, mean_, std_, h):
    """Predict forward vol as the realized vol of the last h returns in the input window."""
    raw = X_norm.squeeze(-1) * std_ + mean_
    last_h = raw[:, -h:]
    return np.sqrt((last_h ** 2).mean(axis=1))


def mae_rmse(y_true, y_pred):
    err = y_true - y_pred
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


# ───────────────────────── train all six models ─────────────────────────
metrics = {}
for name, builder in BUILDERS.items():
    metrics[name] = {}
    for h in HORIZONS:
        print(f"\n▶ Training {name} (horizon={h})...")
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        model = builder(LOOKBACK)
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model.fit(
            X_train, y_train[h],
            validation_data=(X_val, y_val[h]),
            epochs=EPOCHS, batch_size=BATCH,
            callbacks=[es], verbose=0,
        )
        y_pred = model.predict(X_test, verbose=0).reshape(-1)
        mae, rmse = mae_rmse(y_test[h], y_pred)
        out = ART / f"{name}_{h}.h5"
        model.save(out)
        metrics[name][h] = {"mae": mae, "rmse": rmse}
        print(f"  saved {out.name}  |  test MAE={mae:.6f}  RMSE={rmse:.6f}")

# Baseline metrics
baseline_metrics = {}
for h in HORIZONS:
    y_pred_base = baseline_persistence(X_test, mean, std, h)
    mae, rmse = mae_rmse(y_test[h], y_pred_base)
    baseline_metrics[h] = {"mae": mae, "rmse": rmse}
    print(f"\nbaseline (h={h}): MAE={mae:.6f}  RMSE={rmse:.6f}")

# ───────────────────────── save metadata ─────────────────────────
(ART / "norm.json").write_text(json.dumps({"mean": mean, "std": std}, indent=2))

meta = {
    "ticker": TICKER,
    "lookback": LOOKBACK,
    "horizons": HORIZONS,
    "train_frac": TRAIN_FRAC,
    "val_frac": VAL_FRAC,
    "data_start": df.index.min().strftime("%Y-%m-%d"),
    "data_end": df.index.max().strftime("%Y-%m-%d"),
    "n_samples": int(n),
    "train_end_date": df.index[LOOKBACK + train_end - 1].strftime("%Y-%m-%d"),
    "val_end_date": df.index[LOOKBACK + val_end - 1].strftime("%Y-%m-%d"),
}
(ART / "meta.json").write_text(json.dumps(meta, indent=2))

metrics_out = {
    "models": {name: {str(h): metrics[name][h] for h in HORIZONS} for name in BUILDERS},
    "baseline": {str(h): baseline_metrics[h] for h in HORIZONS},
}
(ART / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

# Sanity check on disk sizes
print("\nArtefact sizes:")
for path in sorted(ART.iterdir()):
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  {path.name}: {size_mb:.2f} MB")

print("\nDone.")
