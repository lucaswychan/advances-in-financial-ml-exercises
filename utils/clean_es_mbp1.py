#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BAD_TS_RECV = 8
MAYBE_BAD_BOOK = 4
BAD_DATA_FLAGS = BAD_TS_RECV | MAYBE_BAD_BOOK

ES_TICK_SIZE = 0.25

PRICE_COLUMNS = [
    "price",
    "bid_px_00",
    "ask_px_00",
]

INTEGER_COLUMNS = [
    "size",
    "flags",
    "sequence",
    "depth",
    "bid_sz_00",
    "ask_sz_00",
    "bid_ct_00",
    "ask_ct_00",
]


def clean_mbp1(
    input_file: Path,
    output_dir: Path,
    symbol: str | None = None,
    rth_only: bool = False,
) -> None:
    df = pd.read_csv(input_file)
    input_rows = len(df)

    required = {
        "ts_recv",
        "ts_event",
        "action",
        "side",
        "price",
        "size",
        "flags",
        "sequence",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "bid_ct_00",
        "ask_ct_00",
        "symbol",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # 1. Parse timestamps
    # ------------------------------------------------------------------
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True, errors="coerce")
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")

    # ------------------------------------------------------------------
    # 2. Normalize numeric columns
    # ------------------------------------------------------------------
    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in INTEGER_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["flags"] = df["flags"].fillna(0).astype("uint8")

    # ------------------------------------------------------------------
    # 3. Basic validity filters
    # ------------------------------------------------------------------
    df = df.dropna(subset=["ts_recv", "ts_event", "symbol"])

    if symbol is not None:
        df = df[df["symbol"] == symbol]

    # Keep known MBP-1 event types.
    df = df[df["action"].isin(["A", "C", "M", "R", "T"])]
    df = df[df["side"].isin(["A", "B", "N"])]

    # Remove rows with known timestamp or book-integrity problems.
    bad_quality = (df["flags"] & BAD_DATA_FLAGS) != 0
    bad_quality_rows = int(bad_quality.sum())
    df = df[~bad_quality].copy()

    # Preserve original order as the final tie breaker.
    df["_original_order"] = np.arange(len(df))

    # Databento is delivered in receive-time order. Sequence helps order
    # messages sharing the same receive timestamp.
    df = df.sort_values(
        ["ts_recv", "sequence", "_original_order"],
        kind="stable",
    )

    # Do not call drop_duplicates(): separate trades can legitimately have
    # identical timestamps, prices, and sizes.

    # ------------------------------------------------------------------
    # 4. Keep rows with a usable top of book
    # ------------------------------------------------------------------
    valid_book = (
        df["bid_px_00"].notna()
        & df["ask_px_00"].notna()
        & (df["bid_px_00"] > 0)
        & (df["ask_px_00"] > 0)
        & (df["ask_px_00"] >= df["bid_px_00"])
        & df["bid_sz_00"].notna()
        & df["ask_sz_00"].notna()
        & (df["bid_sz_00"] > 0)
        & (df["ask_sz_00"] > 0)
    )

    invalid_book_rows = int((~valid_book).sum())
    df = df[valid_book].copy()

    # ------------------------------------------------------------------
    # 5. Optionally restrict to the U.S. cash-equity session
    # ------------------------------------------------------------------
    if rth_only:
        ny_time = df["ts_event"].dt.tz_convert("America/New_York")

        minutes_after_midnight = (
            ny_time.dt.hour * 60
            + ny_time.dt.minute
            + ny_time.dt.second / 60
        )

        # 09:30 <= time < 16:00 New York time.
        df = df[
            (minutes_after_midnight >= 9 * 60 + 30)
            & (minutes_after_midnight < 16 * 60)
        ].copy()

    # ------------------------------------------------------------------
    # 6. Standard top-of-book features
    # ------------------------------------------------------------------
    df["spread"] = df["ask_px_00"] - df["bid_px_00"]
    df["spread_ticks"] = df["spread"] / ES_TICK_SIZE
    df["midpoint"] = (df["bid_px_00"] + df["ask_px_00"]) / 2

    total_depth = df["bid_sz_00"] + df["ask_sz_00"]

    df["book_imbalance"] = (
        (df["bid_sz_00"] - df["ask_sz_00"]) / total_depth
    )

    # A size-weighted estimate of the next-price direction.
    df["microprice"] = (
        df["ask_px_00"] * df["bid_sz_00"]
        + df["bid_px_00"] * df["ask_sz_00"]
    ) / total_depth

    df["microprice_minus_mid"] = df["microprice"] - df["midpoint"]

    # Number of resting orders at the BBO.
    total_orders = df["bid_ct_00"] + df["ask_ct_00"]
    df["order_count_imbalance"] = np.where(
        total_orders > 0,
        (df["bid_ct_00"] - df["ask_ct_00"]) / total_orders,
        np.nan,
    )

    # Approximate exchange-message sending time.
    if "ts_in_delta" in df.columns:
        delta_ns = pd.to_numeric(df["ts_in_delta"], errors="coerce")
        df["ts_send"] = df["ts_recv"] - pd.to_timedelta(delta_ns, unit="ns")
        df["feed_latency_ns"] = delta_ns

    # ------------------------------------------------------------------
    # 7. Extract and enrich trade rows
    # ------------------------------------------------------------------
    trades = df[
        (df["action"] == "T")
        & df["price"].notna()
        & (df["price"] > 0)
        & df["size"].notna()
        & (df["size"] > 0)
    ].copy()

    # B = buy aggressor; A = sell aggressor.
    trades["trade_sign"] = trades["side"].map(
        {"B": 1, "A": -1, "N": 0}
    ).astype("int8")

    trades["signed_size"] = trades["trade_sign"] * trades["size"]

    # Trade price relative to the contemporaneous midpoint.
    trades["price_minus_mid"] = trades["price"] - trades["midpoint"]
    trades["price_minus_mid_ticks"] = (
        trades["price_minus_mid"] / ES_TICK_SIZE
    )

    # $50 per full ES index point.
    trades["trade_notional"] = trades["price"] * trades["size"] * 50

    # ------------------------------------------------------------------
    # 8. Mark changes in the displayed BBO
    # ------------------------------------------------------------------
    book_columns = [
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "bid_ct_00",
        "ask_ct_00",
    ]

    changed = df.groupby("symbol", sort=False)[book_columns].transform(
        lambda values: values.ne(values.shift())
    )

    df["bbo_changed"] = changed.any(axis=1)

    # One row per distinct BBO state. This is useful for quote-duration studies.
    quotes = df[df["bbo_changed"]].copy()

    # ------------------------------------------------------------------
    # 9. Save compact research files
    # ------------------------------------------------------------------
    df = df.drop(columns="_original_order")
    trades = trades.drop(columns="_original_order")
    quotes = quotes.drop(columns="_original_order")

    output_dir.mkdir(parents=True, exist_ok=True)

    events_file = output_dir / "clean_events.parquet"
    trades_file = output_dir / "trades.parquet"
    quotes_file = output_dir / "quote_changes.parquet"

    df.to_parquet(events_file, index=False)
    trades.to_parquet(trades_file, index=False)
    quotes.to_parquet(quotes_file, index=False)

    print(f"Input rows:          {input_rows:,}")
    print(f"Bad-quality rows:    {bad_quality_rows:,}")
    print(f"Invalid-book rows:   {invalid_book_rows:,}")
    print(f"Clean event rows:    {len(df):,}")
    print(f"Trade rows:          {len(trades):,}")
    print(f"Quote-change rows:   {len(quotes):,}")
    print()
    print(f"Events: {events_file}")
    print(f"Trades: {trades_file}")
    print(f"Quotes: {quotes_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Databento MBP-1 ES futures data."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cleaned"),
    )
    parser.add_argument(
        "--symbol",
        help="Optional individual contract, for example ESZ3.",
    )
    parser.add_argument(
        "--rth-only",
        action="store_true",
        help="Keep only 09:30-16:00 America/New_York.",
    )

    args = parser.parse_args()

    clean_mbp1(
        input_file=args.input_file,
        output_dir=args.output_dir,
        symbol=args.symbol,
        rth_only=args.rth_only,
    )


if __name__ == "__main__":
    main()