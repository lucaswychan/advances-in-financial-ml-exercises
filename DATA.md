# Data setup

The notebooks use market data from third-party providers. Those records are
not included in Git because their licenses do not grant this repository public
redistribution rights. The repository contains a reproducible preparation
pipeline instead.

## Directory layout

```text
data/
├── raw/          # Local vendor downloads; ignored by Git
├── processed/    # Notebook-ready Parquet; ignored by Git
└── artifacts/    # Regenerable notebook CSV outputs; ignored by Git
```

`data/manifest.json` records the expected files, producers, and license status.

## Chapters 2–18: Kibot IVE data

1. Review the [Kibot license agreement](https://www.kibot.com/License_agreement.aspx).
   It permits private use but prohibits redistribution.
2. Obtain the IVE tick-with-bid/ask sample directly from
   [Kibot](https://www.kibot.com/free_historical_data.aspx).
3. Save it as either:
   - `data/raw/IVE_tickbidask.txt`, or
   - `data/raw/IVE_tickbidask.txt.zst`
4. Build the shared notebook dataset:

   ```bash
   python utils/prepare_data.py kibot
   ```

The output is `data/processed/clean_IVE_tickbidask.parquet`.

Chapter 2 Q3 also expects user-supplied daily files at:

- `data/raw/spx.csv`
- `data/raw/eurostoxx.csv`
- `data/raw/eur_usd.csv`

Their provenance has not been verified, so they remain local-only.

## Chapter 19: Databento CME data

1. Confirm that your Databento/CME entitlement permits your intended use.
2. Copy `.env.example` to `.env` and add your Databento API key:

   ```text
   DATABENTO_API_KEY=your-key
   ```

3. Run the pipeline:

   ```bash
   python utils/prepare_data.py databento
   ```

The command estimates the charge before downloading and aborts above $5 by
default. It downloads ESZ3 MBP-1 for the 2023-11-01 regular session to
compressed Parquet, then creates:

- `data/processed/emini_sp500_futures_clean_events.parquet`
- `data/processed/emini_sp500_futures_trades.parquet`

To prepare an existing CSV or Parquet download without another API request:

```bash
python utils/prepare_data.py databento \
  --input data/raw/your_mbp1_file.parquet
```

Quote-change data is not needed by Chapter 19. Generate it only when required:

```bash
python utils/prepare_data.py databento --include-quotes
```

## Validate before opening notebooks

```bash
python utils/prepare_data.py check
```

The command reports each required path, row count, schema width, and file size.
Notebook paths come from `utils/data_paths.py`, so they do not depend on the
directory from which Jupyter was launched.

## Git policy

Never force-add files under `data/raw`, `data/processed`, or `data/artifacts`.
Only `data/README.md` and `data/manifest.json` are intended to be tracked.
Generated `applied_pt_sl.csv` and `trend_follow_signals.csv` files are written
under `data/artifacts`.
