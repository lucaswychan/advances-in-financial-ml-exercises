#!/usr/bin/env python3
"""Build the local, notebook-ready datasets without redistributing vendor data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

IVE_RAW_FILE = RAW_DIR / "IVE_tickbidask.txt"
IVE_OUTPUT_FILE = PROCESSED_DIR / "clean_IVE_tickbidask.parquet"
ES_RAW_FILE = RAW_DIR / "ESZ3_mbp1_2023-11-01.parquet"
ES_EVENTS_FILE = PROCESSED_DIR / "emini_sp500_futures_clean_events.parquet"
ES_TRADES_FILE = PROCESSED_DIR / "emini_sp500_futures_trades.parquet"
ES_QUOTES_FILE = PROCESSED_DIR / "emini_sp500_futures_quote_changes.parquet"

KIBOT_COLUMNS = ["day", "time", "price", "bid", "ask", "vol"]
MBP1_REQUIRED_COLUMNS = {
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
PRICE_COLUMNS = ["price", "bid_px_00", "ask_px_00"]
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

BAD_TS_RECV = 8
MAYBE_BAD_BOOK = 4
BAD_DATA_FLAGS = BAD_TS_RECV | MAYBE_BAD_BOOK
ES_TICK_SIZE = 0.25


def _resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = path.suffixes
    if path.suffix in {".parq", ".parquet"}:
        df = pd.read_parquet(path)
        if df.index.name and df.index.name not in df.columns:
            df = df.reset_index()
        return df
    if suffixes[-2:] in [[".parquet", ".zst"], [".parq", ".zst"]]:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index, compression="zstd")


def prepare_kibot_ive(
    input_file: Path | str = IVE_RAW_FILE,
    output_file: Path | str = IVE_OUTPUT_FILE,
    mad_threshold: float = 3.0,
) -> dict[str, Any]:
    """Aggregate Kibot IVE ticks by timestamp and remove price MAD outliers."""
    input_path = _resolve(input_file)
    output_path = _resolve(output_file)
    if not input_path.exists():
        compressed = input_path.with_suffix(input_path.suffix + ".zst")
        if compressed.exists():
            input_path = compressed
        else:
            raise FileNotFoundError(
                f"Missing Kibot input: {input_path}. See DATA.md for acquisition steps."
            )

    ticks = pd.read_csv(
        input_path,
        header=None,
        names=KIBOT_COLUMNS,
        compression="infer",
    )
    input_rows = len(ticks)
    ticks["date"] = pd.to_datetime(
        ticks["day"].astype(str) + ticks["time"].astype(str),
        format="%m/%d/%Y%H:%M:%S",
        errors="coerce",
    )
    for column in ["price", "bid", "ask", "vol"]:
        ticks[column] = pd.to_numeric(ticks[column], errors="coerce")
    ticks = ticks.dropna(subset=["date", "price", "bid", "ask", "vol"])
    ticks = ticks[ticks["vol"] > 0].copy()
    ticks["dollar_vol"] = ticks["price"] * ticks["vol"]

    grouped = ticks.groupby("date", sort=True, observed=True)
    result = grouped.agg(
        bid=("bid", "last"),
        ask=("ask", "last"),
        vol=("vol", "sum"),
        dollar_vol=("dollar_vol", "sum"),
        trade_count=("price", "size"),
        price_high=("price", "max"),
        price_low=("price", "min"),
    )
    result.insert(0, "price", result["dollar_vol"] / result["vol"])

    median = result["price"].median()
    median_abs_deviation = (result["price"] - median).abs().median()
    if pd.notna(median_abs_deviation) and median_abs_deviation > 0:
        modified_z = (
            0.6745 * (result["price"] - median).abs() / median_abs_deviation
        )
        result = result.loc[modified_z <= mad_threshold]

    _write_parquet(result, output_path, index=True)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": input_rows,
        "output_rows": len(result),
    }


def clean_mbp1(
    input_file: Path | str,
    output_dir: Path | str = PROCESSED_DIR,
    symbol: str | None = None,
    rth_only: bool = False,
    include_quotes: bool = False,
) -> dict[str, Any]:
    """Validate and enrich Databento MBP-1 data, then write compact Parquet."""
    input_path = _resolve(input_file)
    output_path = _resolve(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Databento input: {input_path}")

    df = _read_table(input_path)
    input_rows = len(df)
    missing = MBP1_REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required MBP-1 columns: {sorted(missing)}")

    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True, errors="coerce")
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in INTEGER_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["flags"] = df["flags"].fillna(0).astype("uint8")
    df["action"] = df["action"].astype(str)
    df["side"] = df["side"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df = df.dropna(subset=["ts_recv", "ts_event", "symbol"])
    if symbol is not None:
        df = df[df["symbol"] == symbol]
    df = df[df["action"].isin(["A", "C", "M", "R", "T"])]
    df = df[df["side"].isin(["A", "B", "N"])]

    bad_quality = (df["flags"] & BAD_DATA_FLAGS) != 0
    bad_quality_rows = int(bad_quality.sum())
    df = df[~bad_quality].copy()
    df["_original_order"] = np.arange(len(df))
    df = df.sort_values(
        ["ts_recv", "sequence", "_original_order"],
        kind="stable",
    )

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

    if rth_only:
        ny_time = df["ts_event"].dt.tz_convert("America/New_York")
        minute = ny_time.dt.hour * 60 + ny_time.dt.minute
        df = df[(minute >= 9 * 60 + 30) & (minute < 16 * 60)].copy()

    df["spread"] = df["ask_px_00"] - df["bid_px_00"]
    df["spread_ticks"] = df["spread"] / ES_TICK_SIZE
    df["midpoint"] = (df["bid_px_00"] + df["ask_px_00"]) / 2
    total_depth = df["bid_sz_00"] + df["ask_sz_00"]
    df["book_imbalance"] = (
        (df["bid_sz_00"] - df["ask_sz_00"]) / total_depth
    )
    df["microprice"] = (
        df["ask_px_00"] * df["bid_sz_00"]
        + df["bid_px_00"] * df["ask_sz_00"]
    ) / total_depth
    df["microprice_minus_mid"] = df["microprice"] - df["midpoint"]
    total_orders = df["bid_ct_00"] + df["ask_ct_00"]
    df["order_count_imbalance"] = np.where(
        total_orders > 0,
        (df["bid_ct_00"] - df["ask_ct_00"]) / total_orders,
        np.nan,
    )

    if "ts_in_delta" in df.columns:
        delta_ns = pd.to_numeric(df["ts_in_delta"], errors="coerce")
        df["ts_send"] = df["ts_recv"] - pd.to_timedelta(delta_ns, unit="ns")
        df["feed_latency_ns"] = delta_ns

    trades = df[
        (df["action"] == "T")
        & df["price"].notna()
        & (df["price"] > 0)
        & df["size"].notna()
        & (df["size"] > 0)
    ].copy()
    trades["trade_sign"] = (
        trades["side"].map({"B": 1, "A": -1, "N": 0}).astype("int8")
    )
    trades["signed_size"] = trades["trade_sign"] * trades["size"]
    trades["price_minus_mid"] = trades["price"] - trades["midpoint"]
    trades["price_minus_mid_ticks"] = (
        trades["price_minus_mid"] / ES_TICK_SIZE
    )
    trades["trade_notional"] = trades["price"] * trades["size"] * 50

    quotes = None
    if include_quotes:
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
        quotes = df[changed.any(axis=1)].copy()

    df = df.drop(columns="_original_order")
    trades = trades.drop(columns="_original_order")
    output_path.mkdir(parents=True, exist_ok=True)
    events_file = output_path / ES_EVENTS_FILE.name
    trades_file = output_path / ES_TRADES_FILE.name
    _write_parquet(df, events_file)
    _write_parquet(trades, trades_file)

    result = {
        "input_rows": input_rows,
        "bad_quality_rows": bad_quality_rows,
        "invalid_book_rows": invalid_book_rows,
        "event_rows": len(df),
        "trade_rows": len(trades),
        "events": str(events_file),
        "trades": str(trades_file),
    }
    if quotes is not None:
        quotes = quotes.drop(columns="_original_order")
        quotes_file = output_path / ES_QUOTES_FILE.name
        _write_parquet(quotes, quotes_file)
        result["quote_rows"] = len(quotes)
        result["quotes"] = str(quotes_file)
    return result


def download_databento_mbp1(
    output_file: Path | str = ES_RAW_FILE,
    symbol: str = "ESZ3",
    start: str = "2023-11-01T14:30:00Z",
    end: str = "2023-11-01T21:00:00Z",
    max_cost: float = 5.0,
) -> Path:
    """Download one historical ES session after checking its estimated cost."""
    try:
        import databento as db
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to use Databento.") from exc

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    output_path = _resolve(output_file)
    request = {
        "dataset": "GLBX.MDP3",
        "symbols": [symbol],
        "stype_in": "raw_symbol",
        "schema": "mbp-1",
        "start": start,
        "end": end,
    }
    client = db.Historical(api_key)
    estimated_cost = client.metadata.get_cost(**request)
    if estimated_cost > max_cost:
        raise RuntimeError(
            f"Estimated cost ${estimated_cost:.2f} exceeds ${max_cost:.2f} limit."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = client.timeseries.get_range(**request)
    data.to_parquet(
        str(output_path),
        pretty_ts=True,
        price_type="float",
        map_symbols=True,
    )
    return output_path


def _parquet_time_range(parquet_file: Any, column: str) -> str | None:
    if column not in parquet_file.schema.names:
        return None
    column_index = parquet_file.schema.names.index(column)
    minima = []
    maxima = []
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        statistics = (
            parquet_file.metadata.row_group(row_group_index)
            .column(column_index)
            .statistics
        )
        if statistics and statistics.has_min_max:
            minima.append(statistics.min)
            maxima.append(statistics.max)
    if not minima:
        return None
    return f"{min(minima)} to {max(maxima)}"


def _describe_parquet(
    path: Path,
    required_columns: set[str],
    time_column: str,
) -> tuple[bool, str]:
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        columns = set(parquet_file.schema_arrow.names)
        missing = required_columns.difference(columns)
        time_range = _parquet_time_range(parquet_file, time_column)
        details = (
            f"OK {path.relative_to(PROJECT_ROOT)}: "
            f"{metadata.num_rows:,} rows, {metadata.num_columns} columns, "
            f"{path.stat().st_size / 1024**2:.1f} MiB"
        )
        if time_range:
            details += f", {time_column} {time_range}"
        if missing:
            return False, f"INVALID {path.relative_to(PROJECT_ROOT)}: missing {sorted(missing)}"
        return True, details
    except Exception as exc:
        return False, f"INVALID {path.relative_to(PROJECT_ROOT)}: {exc}"


def _describe_csv(
    path: Path,
    required_columns: set[str],
    time_column: str,
) -> tuple[bool, str]:
    try:
        df = pd.read_csv(path)
        missing = required_columns.difference(df.columns)
        if missing:
            return False, f"INVALID {path.relative_to(PROJECT_ROOT)}: missing {sorted(missing)}"
        timestamps = pd.to_datetime(df[time_column], errors="coerce").dropna()
        time_range = (
            f", {time_column} {timestamps.min()} to {timestamps.max()}"
            if not timestamps.empty
            else ""
        )
        return (
            True,
            f"OK {path.relative_to(PROJECT_ROOT)}: {len(df):,} rows, "
            f"{len(df.columns)} columns, {path.stat().st_size / 1024:.1f} KiB"
            f"{time_range}",
        )
    except Exception as exc:
        return False, f"INVALID {path.relative_to(PROJECT_ROOT)}: {exc}"


def check_data() -> bool:
    """Print the readiness of every external input and notebook-ready dataset."""
    required = {
        "IVE notebook data": (
            IVE_OUTPUT_FILE,
            {"price", "bid", "ask", "vol", "dollar_vol"},
            "date",
        ),
        "ES events": (
            ES_EVENTS_FILE,
            {"ts_event", "bid_px_00", "ask_px_00", "book_imbalance"},
            "ts_event",
        ),
        "ES trades": (
            ES_TRADES_FILE,
            {"ts_event", "price", "size", "trade_sign"},
            "ts_event",
        ),
        "SPX daily data": (
            RAW_DIR / "spx.csv",
            {"Date", "Price"},
            "Date",
        ),
        "Euro Stoxx daily data": (
            RAW_DIR / "eurostoxx.csv",
            {"Date", "Price"},
            "Date",
        ),
        "EUR/USD daily data": (
            RAW_DIR / "eur_usd.csv",
            {"Date", "Price"},
            "Date",
        ),
    }
    ready = True
    for label, (path, columns, time_column) in required.items():
        if path.exists():
            if path.suffix in {".parq", ".parquet"}:
                valid, message = _describe_parquet(path, columns, time_column)
            else:
                valid, message = _describe_csv(path, columns, time_column)
            ready &= valid
            print(message)
        else:
            ready = False
            print(f"MISSING {label}: {path.relative_to(PROJECT_ROOT)}")
    if not ready:
        print("See DATA.md for licensed-data acquisition and preparation steps.")
    return ready


def _print_result(result: dict[str, Any]) -> None:
    for key, value in result.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    kibot = subparsers.add_parser("kibot", help="Prepare licensed Kibot IVE data")
    kibot.add_argument("--input", type=Path, default=IVE_RAW_FILE)
    kibot.add_argument("--output", type=Path, default=IVE_OUTPUT_FILE)
    kibot.add_argument("--mad-threshold", type=float, default=3.0)

    databento = subparsers.add_parser(
        "databento",
        help="Download or prepare licensed Databento ES data",
    )
    databento.add_argument("--input", type=Path)
    databento.add_argument("--raw-output", type=Path, default=ES_RAW_FILE)
    databento.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    databento.add_argument("--symbol", default="ESZ3")
    databento.add_argument("--start", default="2023-11-01T14:30:00Z")
    databento.add_argument("--end", default="2023-11-01T21:00:00Z")
    databento.add_argument("--max-cost", type=float, default=5.0)
    databento.add_argument("--rth-only", action="store_true")
    databento.add_argument("--include-quotes", action="store_true")

    subparsers.add_parser("check", help="Check whether notebook data is ready")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "kibot":
        _print_result(
            prepare_kibot_ive(
                input_file=args.input,
                output_file=args.output,
                mad_threshold=args.mad_threshold,
            )
        )
        return 0
    if args.command == "databento":
        input_file = args.input
        if input_file is None:
            input_file = download_databento_mbp1(
                output_file=args.raw_output,
                symbol=args.symbol,
                start=args.start,
                end=args.end,
                max_cost=args.max_cost,
            )
        _print_result(
            clean_mbp1(
                input_file=input_file,
                output_dir=args.output_dir,
                symbol=args.symbol,
                rth_only=args.rth_only,
                include_quotes=args.include_quotes,
            )
        )
        return 0
    return 0 if check_data() else 1


if __name__ == "__main__":
    raise SystemExit(main())
