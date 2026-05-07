"""
btrack — BTC/USDT hourly data collector
Runs forever, fetching new 1h candles from Binance every hour.
Data is appended to /data/btc_1h.csv (Railway volume mount).

Environment variables (optional):
  DATA_DIR   — override mount path (default: /data)
  SYMBOL     — trading pair (default: BTC/USDT)
  INTERVAL   — candle interval (default: 1h)
"""

import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
SYMBOL   = os.getenv("SYMBOL", "BTC/USDT")
INTERVAL = os.getenv("INTERVAL", "1h")
CSV_PATH = DATA_DIR / "btc_1h.csv"

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def fetch_since(exchange: ccxt.Exchange, since_ms: int) -> pd.DataFrame:
    rows = []
    while True:
        batch = exchange.fetch_ohlcv(SYMBOL, INTERVAL, since=since_ms, limit=500)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 500:
            break
        since_ms = batch[-1][0] + 1
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def load_existing() -> pd.DataFrame:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    return pd.DataFrame(columns=COLUMNS)


def save(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)


def collect_once(exchange: ccxt.Exchange) -> int:
    existing = load_existing()

    if existing.empty:
        # Bootstrap: last 90 days
        since_ms = exchange.milliseconds() - 90 * 24 * 3_600_000
        log.info("No existing data — bootstrapping last 90 days")
    else:
        last_ts = existing["timestamp"].max()
        since_ms = int(last_ts.timestamp() * 1000) + 1
        log.info("Last candle: %s — fetching new candles since then", last_ts)

    new = fetch_since(exchange, since_ms)

    if new.empty:
        log.info("No new candles")
        return 0

    # Drop the last (incomplete) candle — it's still forming
    now_ms = exchange.milliseconds()
    interval_ms = exchange.parse_timeframe(INTERVAL) * 1000
    new = new[new["timestamp"].astype("int64") // 1_000_000 < now_ms - interval_ms]

    if new.empty:
        log.info("Only incomplete candle returned, skipping")
        return 0

    combined = pd.concat([existing, new], ignore_index=True)
    combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    combined.sort_values("timestamp", inplace=True)
    save(combined)

    log.info("Saved %d new candles  (total: %d)", len(new), len(combined))
    return len(new)


def main() -> None:
    log.info("btrack collector starting  symbol=%s  interval=%s  data=%s",
             SYMBOL, INTERVAL, CSV_PATH)

    exchange = ccxt.binance({"enableRateLimit": True})

    while True:
        try:
            collect_once(exchange)
        except Exception as exc:
            log.error("Collection failed: %s", exc)

        # Sleep until the top of the next hour
        now = datetime.now(timezone.utc)
        seconds_to_next_hour = 3600 - (now.minute * 60 + now.second)
        log.info("Next collection in %ds", seconds_to_next_hour)
        time.sleep(seconds_to_next_hour)


if __name__ == "__main__":
    main()
