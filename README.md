# btrack

Persistent BTC/USDT 1h OHLCV data collector + Kronos model backtest harness.

- **`collect.py`** — runs forever on Railway, fetching new candles from Binance every hour and appending to a mounted volume
- **`backtest.py`** — walk-forward backtest of the [Kronos](https://github.com/shiyu-coder/Kronos) foundation model against historical BTC/USDT data

---

## Deploy to Railway

### 1. Create a Railway project

Go to [railway.com](https://railway.com) → **New Project** → **Deploy from GitHub repo** → select `dhruvsahgal/btrack`.

Railway will auto-detect `railway.toml` and use it for build/start config.

### 2. Add a persistent volume

The collector writes to `/data/btc_1h.csv`. You need a volume mounted there so data survives redeploys.

In your Railway project:
1. Right-click the canvas (or `⌘K`) → **Add Volume**
2. Select your `btrack` service
3. Set **Mount Path** to `/data`
4. Click **Deploy**

### 3. (Optional) Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `/data` | Override the volume mount path |
| `SYMBOL` | `BTC/USDT` | Trading pair to collect |
| `INTERVAL` | `1h` | Candle interval |

Set these in Railway → your service → **Variables** tab if you want to change them.

### 4. Watch it run

Railway → your service → **Logs** tab. On first boot you'll see it bootstrapping the last 90 days of candles, then it will sleep until the top of the next hour and collect from there.

```
2026-05-07T03:00:00Z  INFO  btrack collector starting  symbol=BTC/USDT  interval=1h  data=/data/btc_1h.csv
2026-05-07T03:00:01Z  INFO  No existing data — bootstrapping last 90 days
2026-05-07T03:00:12Z  INFO  Saved 2160 new candles  (total: 2160)
2026-05-07T03:00:12Z  INFO  Next collection in 3588s
```

### 5. Auto-deploy on push

Railway auto-deploys every time you push to the linked branch. No action needed.

---

## Run the Kronos backtest locally

The backtest requires the Kronos model code. Clone it alongside this repo:

```bash
git clone https://github.com/shiyu-coder/Kronos  ../Kronos
pip install -r requirements.txt
python backtest.py --start 2024-01-01 --end 2024-07-01
```

Results are saved to `backtest_results.csv` and `backtest_results.png`.

### Backtest parameters

Edit the constants at the top of `backtest.py`:

| Param | Default | Description |
|---|---|---|
| `LOOKBACK` | 360h | Context window fed to Kronos |
| `PRED_LEN` | 24h | Prediction horizon |
| `STEP` | 24h | Walk-forward step size |
| `SAMPLE_CNT` | 10 | Monte-Carlo paths per prediction |
| `FEE` | 0.001 | Taker fee per side (0.1%) |

---

## References

- [Kronos GitHub](https://github.com/shiyu-coder/Kronos) — foundation model for financial K-lines
- [Kronos live demo](https://shiyu-coder.github.io/Kronos-demo/) — live BTC/USDT 24h forecast
- [Kronos paper (arXiv)](https://arxiv.org/abs/2508.02739) — AAAI 2026
- [Railway volumes](https://docs.railway.com/volumes) — persistent storage docs
- [ccxt](https://github.com/ccxt/ccxt) — Binance data via unified exchange API
