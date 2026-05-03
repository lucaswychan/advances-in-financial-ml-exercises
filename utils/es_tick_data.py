"""
es_tick_data.py

Downloads E-mini S&P 500 (ES) tick data from Databento and saves to Parquet.
Subsequent runs load from disk — the API is never called twice for the same data.

Directory layout:
    data/
        ESH5_trades.parquet
        ESM5_trades.parquet
        ESU5_trades.parquet
        ESZ5_trades.parquet
"""

import os
import databento as db
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY   = os.environ.get("DATABENTO_API_KEY", "YOUR_API_KEY")
DATA_DIR  = "data"
DATASET   = "GLBX.MDP3"
SCHEMA    = "trades"          # individual trade ticks: price, size, ts_event

# Define each contract's active window.
# Rule of thumb: from the day after the prior expiry to a few days past
# the current expiry (so roll-window overlap is captured for the ETF trick).
# CME quarterly expiry: 3rd Friday of Mar/Jun/Sep/Dec.
CONTRACTS = [
    {"symbol": "ESH6", "start": "2026-02-16", "end": "2026-03-25"},  # ~5 weeks, covers Mar expiry + roll overlap
    {"symbol": "ESM6", "start": "2026-03-17", "end": "2026-05-02"},  # ~7 weeks, overlap starts before H6 expiry
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)

def parquet_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_{SCHEMA}.parquet")

def cost_check(client: db.Historical, symbol: str, start: str, end: str) -> int:
    """Return record count (proxy for cost) before pulling data."""
    return client.metadata.get_record_count(
        dataset=DATASET,
        symbols=[symbol],
        schema=SCHEMA,
        stype_in="raw_symbol",
        start=start,
        end=end,
    )

def fetch_and_save(client: db.Historical, symbol: str, start: str, end: str) -> None:
    path = parquet_path(symbol)

    if os.path.exists(path):
        print(f"[SKIP]  {symbol} already on disk → {path}")
        return

    n = cost_check(client, symbol, start, end)
    print(f"[FETCH] {symbol}  {start} → {end}  ({n:,} ticks) ...")

    data = client.timeseries.get_range(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=[symbol],
        start=start,
        end=end,
        stype_in="raw_symbol",
    )

    # Save with:
    #   pretty_ts=True  → ts_event as human-readable datetime string
    #   price_type="fixed" → prices as integers (multiply by 1e-9 to get float)
    #   map_symbols=True   → symbol column added for clarity
    data.to_parquet(
        path,
        pretty_ts=True,
        price_type="fixed",
        map_symbols=True,
    )
    print(f"[SAVED] {symbol} → {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = db.Historical(API_KEY)

    for contract in CONTRACTS:
        fetch_and_save(client, **contract)

    print("\nAll contracts downloaded. Loading from disk for verification ...")

    dfs = []
    for contract in CONTRACTS:
        path = parquet_path(contract["symbol"])
        df = pd.read_parquet(path, columns=["ts_event", "symbol", "price", "size"])
        print(f"  {contract['symbol']}: {len(df):>10,} ticks | "
              f"{df['ts_event'].min()} → {df['ts_event'].max()}")
        dfs.append(df)

    # Concatenated series across all contracts — use this as input for
    # bar construction + ETF trick roll logic
    full_series = pd.concat(dfs).sort_values("ts_event").reset_index(drop=True)
    print(f"\nTotal ticks across all contracts: {len(full_series):,}")
    return full_series

if __name__ == "__main__":
    full_series = main()
