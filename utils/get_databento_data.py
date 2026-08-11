"""Compatibility entry point for the unified Databento data pipeline."""

from dotenv import load_dotenv

try:
    from utils.prepare_data import main
except ModuleNotFoundError:
    from prepare_data import main


if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main(["databento"]))