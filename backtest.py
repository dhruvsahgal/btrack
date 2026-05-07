"""
Kronos BTC/USDT Backtest
------------------------
Fetches historical 1h BTC/USDT candles from Binance, then walks forward in
time: at each step it uses the last LOOKBACK candles as context, asks Kronos
to predict the next PRED_LEN candles, and takes a long/short position based
on whether the mean predicted close is above or below the current close.

Usage:
    python backtest.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--step HOURS]

Requirements:
    pip install -r requirements.txt
    # Kronos model code must be on the Python path (clone the repo alongside):
    #   git clone https://github.com/shiyu-coder/Kronos  ../Kronos
"""

import argparse
import sys
import os
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Kronos path – adjust if you cloned the repo elsewhere
# ---------------------------------------------------------------------------
KRONOS_PATH = os.path.join(os.path.dirname(__file__), "..", "Kronos")
sys.path.insert(0, KRONOS_PATH)

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOOKBACK   = 360   # hours of context fed to the model (matches the live demo)
PRED_LEN   = 24    # hours to predict ahead
STEP       = 24    # hours between each prediction (walk-forward step)
SAMPLE_CNT = 10    # Monte-Carlo paths to average (higher = slower but smoother)
FEE        = 0.001 # 0.1 % taker fee per side

DEFAULT_START = "2024-01-01"
DEFAULT_END   = "2024-07-01"


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    since = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    until = int(datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    rows = []
    while since < until:
        batch = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 3_600_000  # next hour

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[df["timestamp"] < pd.Timestamp(end, tz="UTC")].reset_index(drop=True)
    print(f"Fetched {len(df)} candles  ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")
    return df


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
def load_predictor() -> KronosPredictor:
    print("Loading Kronos-mini …")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    return KronosPredictor(model, tokenizer, max_context=2048)


# ---------------------------------------------------------------------------
# 3. Walk-forward backtest
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, predictor: KronosPredictor) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per prediction step containing:
        timestamp, actual_close, pred_mean_close, signal,
        actual_return, strategy_return
    """
    records = []
    total_steps = (len(df) - LOOKBACK - PRED_LEN) // STEP + 1
    print(f"Running {total_steps} prediction steps …\n")

    for i, start_idx in enumerate(range(LOOKBACK, len(df) - PRED_LEN, STEP)):
        ctx   = df.iloc[start_idx - LOOKBACK : start_idx]
        future = df.iloc[start_idx : start_idx + PRED_LEN]

        x_df        = ctx[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        x_timestamp = ctx["timestamp"].reset_index(drop=True)
        y_timestamp = future["timestamp"].reset_index(drop=True)

        try:
            pred = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=PRED_LEN,
                T=1.0,
                top_p=0.9,
                sample_count=SAMPLE_CNT,
                verbose=False,
            )
            pred_mean_close = pred["close"].mean()
        except Exception as exc:
            print(f"  step {i}: prediction failed ({exc}), skipping")
            continue

        current_close = ctx["close"].iloc[-1]
        actual_close  = future["close"].iloc[-1]   # close at end of pred window

        signal = 1 if pred_mean_close > current_close else -1

        # Returns over the prediction window (entry at current_close, exit at actual_close)
        raw_return      = (actual_close - current_close) / current_close
        strategy_return = signal * raw_return - 2 * FEE  # fee on entry + exit

        records.append({
            "timestamp":       ctx["timestamp"].iloc[-1],
            "current_close":   current_close,
            "pred_mean_close": pred_mean_close,
            "actual_close":    actual_close,
            "signal":          signal,
            "actual_return":   raw_return,
            "strategy_return": strategy_return,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total_steps}]  {ctx['timestamp'].iloc[-1].date()}  "
                  f"signal={'LONG' if signal==1 else 'SHORT'}  "
                  f"pred={pred_mean_close:.1f}  actual={actual_close:.1f}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(results: pd.DataFrame) -> dict:
    n = len(results)
    correct = (results["signal"] * results["actual_return"] > 0).sum()
    dir_acc = correct / n

    cum_ret  = (1 + results["strategy_return"]).cumprod()
    total    = cum_ret.iloc[-1] - 1
    ann      = (1 + total) ** (8760 / (n * STEP)) - 1   # 8760 h/year
    vol      = results["strategy_return"].std() * np.sqrt(8760 / STEP)
    sharpe   = (ann - 0.05) / vol if vol > 0 else 0

    peak     = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd   = drawdown.min()

    bh_ret   = (results["actual_close"].iloc[-1] - results["current_close"].iloc[0]) / results["current_close"].iloc[0]

    return {
        "Steps":              n,
        "Directional Acc":    f"{dir_acc:.1%}",
        "Total Return":       f"{total:.2%}",
        "Ann. Return":        f"{ann:.2%}",
        "Ann. Volatility":    f"{vol:.2%}",
        "Sharpe (rf=5%)":     f"{sharpe:.2f}",
        "Max Drawdown":       f"{max_dd:.2%}",
        "Buy-and-Hold":       f"{bh_ret:.2%}",
    }


# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
def plot_results(results: pd.DataFrame, out_path: str = "backtest_results.png"):
    cum_strategy = (1 + results["strategy_return"]).cumprod()
    cum_bh       = (1 + results["actual_return"]).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Equity curves
    axes[0].plot(results["timestamp"], cum_strategy, label="Kronos L/S", linewidth=1.5)
    axes[0].plot(results["timestamp"], cum_bh,       label="Buy & Hold",  linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Kronos BTC/USDT Backtest")

    # Drawdown
    peak = cum_strategy.cummax()
    dd   = (cum_strategy - peak) / peak
    axes[1].fill_between(results["timestamp"], dd, 0, color="red", alpha=0.4, label="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Directional accuracy (rolling 20-step window)
    rolling_acc = (results["signal"] * results["actual_return"] > 0).rolling(20).mean()
    axes[2].plot(results["timestamp"], rolling_acc, color="green", linewidth=1.5)
    axes[2].axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Rolling Dir. Acc (20)")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kronos BTC/USDT backtest")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--step",  type=int, default=STEP, help="Walk-forward step in hours")
    args = parser.parse_args()

    # Fetch data
    df = fetch_ohlcv("BTC/USDT", args.start, args.end)

    # Load model
    predictor = load_predictor()

    # Run backtest
    results = run_backtest(df, predictor)
    if results.empty:
        print("No results – check your date range or model path.")
        return

    # Metrics
    metrics = compute_metrics(results)
    print("\n" + "=" * 40)
    print("BACKTEST RESULTS")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    # Save results CSV
    results.to_csv("backtest_results.csv", index=False)
    print("\nDetailed results saved → backtest_results.csv")

    # Plot
    plot_results(results)


if __name__ == "__main__":
    main()
