# Advances in Financial Machine Learning — Exercises

Personal exercise solutions and implementations following the book [*Advances in Financial Machine Learning*](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) by Marcos López de Prado.

Each chapter directory contains Jupyter notebooks that work through the end-of-chapter questions and apply the concepts to real tick data (will be uploaded once I finish all the chapters).

---

## Repository Structure

```
.
├── chapter-2-Financial_Data_Structures/   # Bar sampling: tick, volume, dollar, imbalance bars
├── chapter-3-Labeling/                    # Triple-barrier labeling, CUSUM filter, trend following
├── chapter-4-Sample_Weights/             # Uniqueness, sample weights, time-decay
├── chapter-5-Fractionally_Differentiated_Features/  # Fractional differentiation (FFD)
├── chapter-6-Ensemble_Methods/           # Bagging, random forests, feature importance
├── chapter-7_Cross_Validation_in_Finance/# Purged K-Fold CV, embargo, combinatorial CV
├── utils/                                # Shared helper modules (see below)
├── data/                                 # Raw / processed data files
└── requirements.txt
```

### `utils/` modules

| Module | Description |
|---|---|
| `sampling_bars.py` | Tick, volume, dollar, and imbalance bar construction |
| `labeling.py` | Triple-barrier method, daily volatility, CUSUM filter |
| `sampling_features.py` | Sample uniqueness and sample-weight computation |
| `frac_diff.py` | Fixed-width window and expanding-window FFD |
| `cv.py` | `PurgedKFold` — leak-free cross-validation splitter |
| `multiprocess.py` | Pandas-friendly multiprocessing wrapper |
| `es_tick_data.py` | Databento data loader for E-mini S&P 500 tick data |

---

## Getting Started

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/advances-in-financial-ml-exercises.git
   cd advances-in-financial-ml-exercises
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the notebooks**

   Launch Jupyter and open any chapter notebook:

   ```bash
   jupyter notebook
   ```

---

## Chapter Progress

| Chapter | Topic | Status |
|---|---|---|
| 2 | Financial Data Structures | Done |
| 3 | Labeling | Done |
| 4 | Sample Weights | Done |
| 5 | Fractionally Differentiated Features | Done |
| 6 | Ensemble Methods | Done |
| 7 | Cross-Validation in Finance | Done |
| 8 | Feature Importance | Done |

---

## TODO

- [ ] Add Chapter 8 — Feature Importance
- [ ] Add Chapter 9 — Hyper-Parameter Tuning
- [ ] Add Chapter 17 — Structural Breaks
- [ ] Add Chapter 20 — Backtesting on Synthetic Data
- [ ] Refactor `utils/` into a proper installable package with `pyproject.toml`
- [ ] Add a unified data pipeline script so all notebooks share one pre-built dataset

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.