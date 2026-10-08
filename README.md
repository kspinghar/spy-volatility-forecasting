---
title: SPY Volatility Forecasting
emoji: 📈
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# SPY Volatility Forecasting: CNN / LSTM / Hybrid

Interactive demo of three deep-learning models trained to predict forward realized volatility of the S&P 500 ETF (SPY) from its past daily returns. Pick a horizon (5 or 10 days), a model architecture, and a date, and the app shows the 120-day return window the model sees, its predicted forward volatility, the persistence baseline, and the actual realized volatility for comparison.

**Note:** This Space ships with a bundled SPY price snapshot taken at training time, so the "latest" date is frozen. Models are pre-trained; the Space does inference only.

## What this demo does

1. Pick a horizon: 5 days or 10 days of forward realized volatility.
2. Pick a model: CNN, LSTM, or CNN-LSTM hybrid.
3. Pick a date within the bundled SPY history.
4. The app builds the 120-day past-returns window ending at that date, normalizes it with training-set statistics, and produces a forward-volatility prediction. The persistence baseline (last observed value) and the actual realized volatility are shown alongside.

## Architecture

- **CNN:** two `Conv1D` + `MaxPool1D` blocks, `GlobalAveragePooling1D`, dense head.
- **LSTM:** single `LSTM(32)` with dropout, dense head.
- **CNN-LSTM hybrid:** two conv blocks feeding an LSTM, dense head.
- All trained on log returns, normalized with train-set mean/std, predicting forward realized volatility computed as the square-root of forward mean squared return over the horizon.

## Source code

https://github.com/kspinghar/spy-volatility-forecasting
