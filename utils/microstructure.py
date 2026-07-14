import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS


def tick_rule(df: pd.DataFrame, price_col: str) -> pd.Series:
    price_diff = df[price_col].diff()
    signs = pd.Series(np.sign(price_diff), index=df.index)

    # A zero tick inherits the most recent non-zero tick direction.
    # The first observation has no preceding price, so leave it neutral.
    return signs.replace(0, np.nan).ffill().fillna(0).astype("int8")


def roll_model(price: pd.Series) -> pd.Series:
    """Estimate Roll's effective half-spread and efficient-price variance."""
    price = pd.Series(price, dtype="float64").dropna()
    price_diff = price.diff().dropna()
    lagged_diff = price_diff.shift(1)
    aligned = pd.concat([price_diff, lagged_diff], axis=1).dropna()

    price_diff_var = price_diff.var()
    cov = (
        aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        if len(aligned) > 1
        else np.nan
    )

    c = np.sqrt(max(0.0, -cov)) if pd.notna(cov) else np.nan
    sigma_u_2 = (
        price_diff_var + 2 * cov
        if pd.notna(price_diff_var) and pd.notna(cov)
        else np.nan
    )

    return pd.Series(
        {
            "price_diff_var": price_diff_var,
            "price_diff_lag_cov": cov,
            "c": c,
            "spread": 2 * c if pd.notna(c) else np.nan,
            "sigma_u_2": sigma_u_2,
        }
    )


def high_low_volatility(high: pd.Series, low: pd.Series) -> float:
    """Parkinson high-low volatility estimator."""
    high_low = pd.concat(
        [
            pd.Series(high, dtype="float64"),
            pd.Series(low, dtype="float64"),
        ],
        axis=1,
    ).dropna()
    high_low = high_low[(high_low.iloc[:, 0] > 0) & (high_low.iloc[:, 1] > 0)]
    if high_low.empty:
        return np.nan

    log_range = np.log(high_low.iloc[:, 0] / high_low.iloc[:, 1])
    return np.sqrt((log_range**2).mean() / (4 * np.log(2)))


def close_to_close_volatility(close: pd.Series) -> float:
    """Standard deviation of close-to-close log returns."""
    close = pd.Series(close, dtype="float64").dropna()
    close = close[close > 0]
    log_returns = np.log(close).diff().dropna()
    return log_returns.std(ddof=1)


def make_time_bars(
    df: pd.DataFrame,
    frequency: str,
    price_col: str = "price",
    volume_col: str | None = None,
    dollar_col: str | None = None,
) -> pd.DataFrame:
    """Build OHLC bars from a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("make_time_bars requires a DatetimeIndex.")
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    data = df.sort_index()
    aggregations = {
        "price_open": (price_col, "first"),
        "price_high": (price_col, "max"),
        "price_low": (price_col, "min"),
        "price": (price_col, "last"),
    }
    if volume_col is not None:
        if volume_col not in df.columns:
            raise ValueError(f"Missing volume column: {volume_col}")
        aggregations["volume"] = (volume_col, "sum")
    if dollar_col is not None:
        if dollar_col not in df.columns:
            raise ValueError(f"Missing dollar column: {dollar_col}")
        aggregations["dollar_vol"] = (dollar_col, "sum")

    return data.resample(frequency).agg(**aggregations).dropna(subset=["price"])


def make_dollar_bars(
    df: pd.DataFrame,
    dollar_threshold: float,
    price_col: str = "price",
    dollar_col: str = "dollar_vol",
) -> pd.DataFrame:
    """Build OHLC bars whose rows each contain roughly dollar_threshold notional."""
    if dollar_threshold <= 0:
        raise ValueError("dollar_threshold must be positive.")
    missing = {price_col, dollar_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df[[price_col, dollar_col]].dropna().copy()
    data = data[(data[price_col] > 0) & (data[dollar_col] > 0)]
    if data.empty:
        return pd.DataFrame(
            columns=[
                "bar_id",
                "price_open",
                "price_high",
                "price_low",
                "price",
                "dollar_vol",
                "tick_count",
            ]
        )

    data = data.sort_index()
    bar_ids = []
    bar_id = 0
    bar_dollars = 0.0
    for dollar_value in data[dollar_col].to_numpy():
        bar_dollars += dollar_value
        bar_ids.append(bar_id)
        if bar_dollars >= dollar_threshold:
            bar_id += 1
            bar_dollars = 0.0
    data["_bar_id"] = bar_ids

    grouped = data.groupby("_bar_id", sort=True)
    bars = grouped.agg(
        price_open=(price_col, "first"),
        price_high=(price_col, "max"),
        price_low=(price_col, "min"),
        price=(price_col, "last"),
        dollar_vol=(dollar_col, "sum"),
        tick_count=(price_col, "size"),
    )
    bars.insert(0, "bar_id", bars.index.to_numpy())
    bars.index.name = "bar_id"

    if isinstance(data.index, pd.DatetimeIndex):
        bars["start_time"] = grouped.apply(lambda group: group.index[0])
        bars["end_time"] = grouped.apply(lambda group: group.index[-1])
        bars = bars.set_index("end_time", drop=False)

    return bars


def corwin_schultz_spread(
    df: pd.DataFrame,
    sl: int,
    high_col: str,
    low_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if sl < 1:
        raise ValueError("sl must be at least 1.")

    high_price = df[high_col]
    low_price = df[low_col]

    # Calculate beta
    inner_term = np.log(high_price / low_price) ** 2
    inner_term = pd.Series(inner_term, index=df.index)
    # rolling sum of the inner term
    beta = inner_term.rolling(window=2).sum()
    # rolling mean of the inner term
    beta = beta.rolling(window=sl).mean()

    # Calculate gamma
    rolling_high_max = high_price.rolling(window=2).max()
    rolling_low_min = low_price.rolling(window=2).min()
    gamma = np.log(rolling_high_max / rolling_low_min) ** 2
    gamma = pd.Series(gamma, index=df.index).dropna()

    # Calculate alpha
    denominator = 3 - np.sqrt(8)
    alpha = (
        (np.sqrt(2) - 1) * np.sqrt(beta) / denominator
        - np.sqrt(gamma / denominator)
    )
    alpha = alpha.mask(alpha < 0, 0).dropna()

    # Get the spread
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    start_time = pd.Series(df.index[:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1).dropna()
    spread.columns = ["spread", "start_time"]

    return spread, alpha, beta, gamma


def corwin_schultz_implied_volatility(
    beta: pd.Series,
    gamma: pd.Series,
) -> pd.Series:
    k2 = np.sqrt(8 / np.pi)
    denominator = 3 - np.sqrt(8)
    sigma = (
        (1 / np.sqrt(2) - 1) * np.sqrt(beta) / (k2 * denominator)
        + np.sqrt(gamma / (k2**2 * denominator))
    )
    sigma = sigma.mask(sigma < 0, 0).dropna()

    return sigma


def kyle_lambda(
    returns: pd.Series,
    aggressor_flags: pd.Series,
    trade_volumes: pd.Series,
) -> pd.Series:
    X = np.array(aggressor_flags * trade_volumes)
    y = np.array(returns)
    model = OLS(y, X)
    results = model.fit()

    return results.params[0]

def amihud_lambda(
    abs_log_returns: pd.Series,
    dollar_vol: pd.Series,
) -> pd.Series:
    X = np.array(dollar_vol)
    y = np.array(abs_log_returns)
    model = OLS(y, X)
    results = model.fit()

    return results.params[0]