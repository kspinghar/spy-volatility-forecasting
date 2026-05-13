"""
app.py — Gradio demo for the SPY volatility forecasting Space.

Loads the six pre-trained .h5 models, the bundled SPY snapshot, and lets the
user pick a horizon, model, and date. Renders the 120-day return window the
model sees alongside its prediction, the persistence baseline, and (where
available) the actual realized volatility.
"""

import json
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

ART = Path(__file__).parent / "artifacts"

# ───────────────────────── load artefacts ─────────────────────────
print("Loading artefacts...")
norm = json.loads((ART / "norm.json").read_text())
meta = json.loads((ART / "meta.json").read_text())
metrics = json.loads((ART / "metrics.json").read_text())

MEAN, STD = norm["mean"], norm["std"]
LOOKBACK = meta["lookback"]
HORIZONS = meta["horizons"]
DATA_END = meta["data_end"]

snapshot = pd.read_csv(ART / "spy_snapshot.csv", parse_dates=["Date"])
snapshot = snapshot.sort_values("Date").reset_index(drop=True)
snapshot["log_close"] = np.log(snapshot["Close"])
snapshot["r"] = snapshot["log_close"].diff()
snapshot = snapshot.dropna().reset_index(drop=True)

# Pre-load the six models keyed (architecture, horizon).
MODELS = {}
for arch in ["cnn", "lstm", "hybrid"]:
    for h in HORIZONS:
        MODELS[(arch, h)] = load_model(ART / f"{arch}_{h}.h5", compile=False)
print(f"Loaded {len(MODELS)} models. SPY range: {snapshot['Date'].iloc[0].date()} → {snapshot['Date'].iloc[-1].date()}")

ARCH_LABELS = {"cnn": "CNN", "lstm": "LSTM", "hybrid": "CNN-LSTM Hybrid"}

# ───────────────────────── preset notable dates ─────────────────────────
# Hand-picked so the user can scrub through interesting vol regimes quickly.
_RAW_PRESETS = [
    ("Latest available", None),
    ("Calm bull (2017-06-15)", "2017-06-15"),
    ("Volpocalypse (2018-02-05)", "2018-02-05"),
    ("Q4 2018 selloff (2018-12-24)", "2018-12-24"),
    ("Pre-COVID (2020-02-14)", "2020-02-14"),
    ("COVID crash (2020-03-16)", "2020-03-16"),
    ("COVID rebound (2020-04-09)", "2020-04-09"),
    ("Inflation panic (2022-06-13)", "2022-06-13"),
    ("Banking jitters (2023-03-13)", "2023-03-13"),
    ("Aug 2024 unwind (2024-08-05)", "2024-08-05"),
    ("Lehman week (2008-09-19)", "2008-09-19"),
    ("Flash crash (2010-05-06)", "2010-05-06"),
]


def _resolve_date(label):
    for lab, d in _RAW_PRESETS:
        if lab == label:
            if d is None:
                # Latest valid date that has a 10-day forward window AND a 120-day input window
                last_idx = len(snapshot) - 1 - max(HORIZONS)
                return snapshot["Date"].iloc[last_idx].strftime("%Y-%m-%d")
            return d
    return label  # fallback: treat input itself as a date


PRESET_LABELS = [lab for lab, _ in _RAW_PRESETS]


# ───────────────────────── core inference ─────────────────────────
def _window_at(date_str):
    """Return the index t such that snapshot['Date'][t] is the chosen anchor date
    (last day of the 120-day input window). Raise ValueError if invalid."""
    target = pd.to_datetime(date_str)
    # Use searchsorted to find the closest <= target trading day.
    idx = snapshot["Date"].searchsorted(target, side="right") - 1
    if idx < LOOKBACK - 1:
        raise ValueError(
            f"Date {date_str} is too early — need at least {LOOKBACK} prior trading days."
        )
    if idx >= len(snapshot):
        raise ValueError(f"Date {date_str} is past the bundled snapshot.")
    return int(idx)


def _baseline_persistence(window_returns, h):
    """Predict forward vol as realized vol of the last h returns in the window."""
    last_h = window_returns[-h:]
    return float(np.sqrt(np.mean(last_h ** 2)))


def _actual_realized_vol(t_idx, h):
    """Forward realized vol from t+1..t+h. Returns None if not enough data ahead."""
    if t_idx + h >= len(snapshot):
        return None
    fwd = snapshot["r"].iloc[t_idx + 1: t_idx + 1 + h].to_numpy()
    return float(np.sqrt(np.mean(fwd ** 2)))


def _build_plot(t_idx, h, pred, baseline, actual, date_str, arch_label):
    """Render 2-panel matplotlib figure: price + return window with markers."""
    win = snapshot.iloc[t_idx - LOOKBACK + 1: t_idx + 1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    ax1.plot(win["Date"], win["Close"], linewidth=1.2, color="#6366f1")
    ax1.set_ylabel("SPY Close")
    ax1.set_title(f"{arch_label} · horizon={h}d · anchor={date_str}", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(win["Date"], win["r"], linewidth=0.8, color="#0ea5e9")
    ax2.axhline(0, color="#94a3b8", linewidth=0.5)
    ax2.set_ylabel("Log return")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


# ───────────────────────── predict entry point ─────────────────────────
def predict(date_label, horizon, arch_label):
    arch = {v: k for k, v in ARCH_LABELS.items()}[arch_label]
    h = int(horizon)
    date_str = _resolve_date(date_label)

    try:
        t_idx = _window_at(date_str)
    except ValueError as e:
        return f"### Error\n{e}", None

    anchor_date = snapshot["Date"].iloc[t_idx].strftime("%Y-%m-%d")

    # Build the (1, LOOKBACK, 1) input
    window_returns = snapshot["r"].iloc[t_idx - LOOKBACK + 1: t_idx + 1].to_numpy()
    X = ((window_returns - MEAN) / STD).astype(np.float32).reshape(1, LOOKBACK, 1)

    model = MODELS[(arch, h)]
    pred = float(model.predict(X, verbose=0).reshape(-1)[0])
    baseline = _baseline_persistence(window_returns, h)
    actual = _actual_realized_vol(t_idx, h)

    # Per-day vol → annualized % for human-readable display (×sqrt(252)×100)
    ann = lambda v: v * np.sqrt(252) * 100 if v is not None else None

    m = metrics["models"][arch][str(h)]
    b = metrics["baseline"][str(h)]

    summary = (
        f"### Forward {h}-day realized volatility — anchor {anchor_date}\n\n"
        f"**Model ({arch_label}):** {pred:.5f} per day &nbsp; · &nbsp; {ann(pred):.2f}% annualized\n\n"
        f"**Persistence baseline:** {baseline:.5f} per day &nbsp; · &nbsp; {ann(baseline):.2f}% annualized\n\n"
        + (
            f"**Actual realized:** {actual:.5f} per day &nbsp; · &nbsp; {ann(actual):.2f}% annualized\n\n"
            if actual is not None
            else f"*Actual realized vol not available — anchor too close to {DATA_END}.*\n\n"
        )
        + "---\n"
        f"**Test-set error** (model vs baseline):\n\n"
        f"- {arch_label}: MAE {m['mae']:.5f} · RMSE {m['rmse']:.5f}\n"
        f"- Baseline: MAE {b['mae']:.5f} · RMSE {b['rmse']:.5f}\n"
    )

    plot_path = _build_plot(t_idx, h, pred, baseline, actual, anchor_date, arch_label)
    return summary, plot_path


# ───────────────────────── Gradio UI ─────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="SPY Volatility Forecasting") as demo:
    gr.Markdown(
        "# SPY Volatility Forecasting — CNN / LSTM / Hybrid\n"
        "Pick a horizon, model, and a notable date. The app shows the 120-day SPY return window the model "
        "sees, then prints its forward-volatility prediction alongside the persistence baseline and the actual "
        "realized volatility for comparison.\n\n"
        "*Bundled snapshot — SPY data is frozen at training time. First load may take ~30s.*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            date_dd = gr.Dropdown(
                choices=PRESET_LABELS, value=PRESET_LABELS[0],
                label="Anchor date",
            )
            horizon_dd = gr.Dropdown(
                choices=[str(h) for h in HORIZONS], value=str(HORIZONS[0]),
                label="Horizon (days)",
            )
            arch_dd = gr.Dropdown(
                choices=list(ARCH_LABELS.values()), value=ARCH_LABELS["hybrid"],
                label="Model",
            )
            predict_btn = gr.Button("Predict", variant="primary")
        with gr.Column(scale=2):
            out_md = gr.Markdown()
            out_plot = gr.Image(label="120-day input window", type="filepath")

    predict_btn.click(
        predict,
        inputs=[date_dd, horizon_dd, arch_dd],
        outputs=[out_md, out_plot],
    )

if __name__ == "__main__":
    demo.launch(show_api=False)
