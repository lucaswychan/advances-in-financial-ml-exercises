# Local data directory

Market data in this project is licensed by its original provider and is not
redistributed with the repository. Only this guide and `manifest.json` are
tracked.

The standard local layout is:

```text
data/
├── raw/          # Original or compressed vendor downloads
├── processed/    # Notebook-ready Parquet files
└── artifacts/    # Regenerable notebook outputs
```

See the repository-level `DATA.md` for acquisition and preparation commands.
Run `python utils/prepare_data.py check` to see which files are ready.
