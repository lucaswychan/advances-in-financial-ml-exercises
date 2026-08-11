#!/usr/bin/env python3
"""Compatibility wrapper for the unified MBP-1 cleaning pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from utils.prepare_data import clean_mbp1
except ModuleNotFoundError:
    from prepare_data import clean_mbp1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--symbol")
    parser.add_argument("--rth-only", action="store_true")
    parser.add_argument("--include-quotes", action="store_true")
    args = parser.parse_args()

    result = clean_mbp1(
        input_file=args.input_file,
        output_dir=args.output_dir,
        symbol=args.symbol,
        rth_only=args.rth_only,
        include_quotes=args.include_quotes,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
