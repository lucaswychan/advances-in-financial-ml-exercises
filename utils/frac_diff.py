import pandas as pd
import numpy as np

def get_weights(d, size):
    weights = [1.]
    for k in range(1, size):
        weights.append(-weights[-1] * (d - k + 1) / k)

    # weights shape: (size, 1)
    return np.array(weights[::-1]).reshape(-1, 1)


def expanding_window_fracdiff(series, d, tau=1e-2):
    # weights shape: (len(series), 1)
    weights = get_weights(d, len(series))
    
    cumsum_weights = np.cumsum(np.abs(weights))
    normalized_cumsum_weights = cumsum_weights / cumsum_weights[-1]
    num_skip = sum(normalized_cumsum_weights > tau)
    
    df = {}
    
    for col in series.columns:
        series_col = series[[col]].ffill().dropna()
        df_ = pd.Series(index=series.index)

        for i in range(num_skip, len(series)):
            curr_val = series_col.index[i]
            if not np.isfinite(series.loc[curr_val, col]):
                continue # exclude NAs
            weights_subset = weights[-(i + 1):, :].T
            series_subset = series_col.loc[:curr_val]
            df_[curr_val] = np.dot(weights_subset, series_subset)[0, 0] # \sum_{j=0}^{i} w_{i-j} * y_{j}

        df[col] = df_.copy(deep=True)

    df = pd.concat(df, axis=1)
    return df


def get_weights_ffd(d, threshold=1e-5):
    weights = [1.]
    k = 1
    
    curr_w = float("inf")
    
    while abs(curr_w) >= threshold:
        curr_w = -weights[-1] * (d - k + 1) / k
        weights.append(curr_w)
        k += 1

    return np.array(weights[::-1]).reshape(-1, 1)


def frac_diff_ffd(series, d, tau=1e-5):
    weights = get_weights_ffd(d, tau).ravel()
    width = len(weights) - 1
    n_rows = len(series.index)

    out = {}

    for col in series.columns:
        original = series[col].to_numpy(dtype=float)
        ffilled = series[col].ffill().to_numpy(dtype=float)

        finite_ffilled = np.isfinite(ffilled)
        if not finite_ffilled.any():
            out[col] = np.full(n_rows, np.nan, dtype=float)
            continue

        # Match old behavior: effectively drop leading NaNs before windowing.
        start = int(np.argmax(finite_ffilled))
        sub = ffilled[start:]

        result = np.full(n_rows, np.nan, dtype=float)
        if len(sub) <= width:
            out[col] = result
            continue

        conv = np.convolve(sub, weights, mode="valid")
        right_positions = np.arange(start + width, n_rows)

        # Keep outputs only where the original right-edge value is finite.
        finite_right = np.isfinite(original[right_positions])
        result[right_positions[finite_right]] = conv[finite_right]
        out[col] = result

    return pd.DataFrame(out, index=series.index)