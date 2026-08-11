"""Stable repository data paths for scripts and notebooks."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

IVE_TICKS = PROCESSED_DIR / "clean_IVE_tickbidask.parquet"
ES_EVENTS = PROCESSED_DIR / "emini_sp500_futures_clean_events.parquet"
ES_TRADES = PROCESSED_DIR / "emini_sp500_futures_trades.parquet"
SPX_DAILY = RAW_DIR / "spx.csv"
EUROSTOXX_DAILY = RAW_DIR / "eurostoxx.csv"
EUR_USD_DAILY = RAW_DIR / "eur_usd.csv"


def artifact_path(name: str) -> Path:
    """Return an ignored output path, creating its directory when needed."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / name
