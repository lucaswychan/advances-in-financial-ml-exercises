import numpy as np


def _as_finite_returns(returns):
    returns = np.asarray(returns, dtype=float).ravel()
    return returns[np.isfinite(returns)]


def _validate_positive_int(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _to_symbols(encoded_returns):
    if isinstance(encoded_returns, str):
        return np.asarray(list(encoded_returns)), True

    encoded_returns = np.asarray(encoded_returns).ravel()
    if encoded_returns.size == 0:
        raise ValueError("encoded_returns must contain at least one symbol")
    return encoded_returns, False


def _sliding_words(encoded_returns, w):
    w = _validate_positive_int(w, "w")
    encoded_returns, source_is_string = _to_symbols(encoded_returns)
    if encoded_returns.size < w:
        raise ValueError("w cannot be greater than the encoded message length")

    return np.lib.stride_tricks.sliding_window_view(encoded_returns, w), source_is_string


def _word_counts(encoded_returns, w, return_words=True):
    windows, source_is_string = _sliding_words(encoded_returns, w)
    if windows.dtype != object:
        contiguous = np.ascontiguousarray(windows)
        packed_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
        packed = contiguous.view(packed_dtype).ravel()
        if return_words:
            _, unique_idx, counts = np.unique(
                packed,
                return_index=True,
                return_counts=True,
            )
            words = contiguous[unique_idx]
        else:
            _, counts = np.unique(packed, return_counts=True)
            words = None
        return words, counts, source_is_string

    counts_by_word = {}
    for word in windows:
        key = tuple(word.tolist())
        counts_by_word[key] = counts_by_word.get(key, 0) + 1

    words = list(counts_by_word) if return_words else None
    counts = np.fromiter(counts_by_word.values(), dtype=int)
    return words, counts, source_is_string


def binary_encoding(returns):
    """Encode non-zero returns by sign: 1 for positive, 0 for negative."""
    returns = _as_finite_returns(returns)
    returns = returns[returns != 0]
    return np.where(returns > 0, 1, 0)


def quantile_encoding(returns, n_quantiles=10):
    """Encode returns by the in-sample quantile bucket they belong to."""
    n_quantiles = _validate_positive_int(n_quantiles, "n_quantiles")
    returns = _as_finite_returns(returns)
    if returns.size == 0:
        raise ValueError("returns must contain at least one finite value")

    boundaries = np.quantile(
        returns,
        np.linspace(0, 1, n_quantiles + 1)[1:-1],
    )
    return np.searchsorted(boundaries, returns, side="right")


def sigma_encoding(returns, sigma):
    """Encode returns in equal-width buckets of size sigma from min(returns)."""
    returns = _as_finite_returns(returns)
    if returns.size == 0:
        raise ValueError("returns must contain at least one finite value")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")

    min_r = returns.min()
    max_r = returns.max()
    data_range = max_r - min_r
    if data_range == 0:
        return np.zeros(returns.size, dtype=int)

    n_codes = int(np.ceil(data_range / sigma))
    codes = np.floor((returns - min_r) / sigma).astype(int)
    return np.clip(codes, 0, n_codes - 1)


def _pmf(encoded_returns, w):
    """Estimate the PMF of overlapping words of length w."""
    words, counts, source_is_string = _word_counts(encoded_returns, w)
    probabilities = counts / counts.sum()
    if source_is_string:
        keys = ("".join(word.tolist()) for word in words)
    else:
        keys = (tuple(np.atleast_1d(word).tolist()) for word in words)
    return dict(zip(keys, probabilities))


def plug_in_entropy_estimation(encoded_returns, w):
    _, counts, _ = _word_counts(encoded_returns, w, return_words=False)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities)) / w


def _common_prefix_lengths(matches):
    mismatches = ~matches
    has_mismatch = mismatches.any(axis=1)
    first_mismatch = np.argmax(mismatches, axis=1)
    return np.where(has_mismatch, first_mismatch, matches.shape[1])


def _match_length(encoded_returns, i, n, return_sub_seq=False):
    # Maximum matched length + 1, with overlap.
    previous_windows = np.lib.stride_tricks.sliding_window_view(
        encoded_returns[i - n:i + n - 1],
        n,
    )
    current_window = encoded_returns[i:i + n]
    common_lengths = _common_prefix_lengths(previous_windows == current_window)
    max_length = int(common_lengths.max())
    if return_sub_seq:
        return max_length + 1, current_window[:max_length]
    return max_length + 1, None


def _match_length_from_windows(windows, i, n, return_sub_seq=False):
    previous_windows = windows[i - n:i]
    current_window = windows[i]
    common_lengths = _common_prefix_lengths(previous_windows == current_window)
    max_length = int(common_lengths.max())
    if return_sub_seq:
        return max_length + 1, current_window[:max_length]
    return max_length + 1, None


def _format_sub_seq(sub_seq, source_is_string):
    if source_is_string:
        return "".join(sub_seq.tolist())
    return tuple(sub_seq.tolist())


def kontoyiannis_entropy_estimation(encoded_returns, w=None, return_sub_seq=False):
    encoded_returns, source_is_string = _to_symbols(encoded_returns)
    msg_len = encoded_returns.size
    if msg_len < 2:
        raise ValueError("encoded_returns must contain at least two symbols")

    res = {"num": 0, "sum": 0, "sub_seq": []}
    if w is None:
        window = None
        points = range(1, msg_len // 2 + 1)
    else:
        w = _validate_positive_int(w, "w")
        window = min(w, msg_len // 2)
        points = range(window, msg_len - window + 1)
        windows = np.lib.stride_tricks.sliding_window_view(encoded_returns, window)

    for i in points:
        if window is None:
            length, sub_seq = _match_length(
                encoded_returns,
                i,
                i,
                return_sub_seq=return_sub_seq,
            )
            res["sum"] += np.log2(i + 1) / length
        else:
            length, sub_seq = _match_length_from_windows(
                windows,
                i,
                window,
                return_sub_seq=return_sub_seq,
            )
            res["sum"] += np.log2(window + 1) / length

        res["num"] += 1
        if return_sub_seq:
            res["sub_seq"].append(_format_sub_seq(sub_seq, source_is_string))

    if res["num"] == 0:
        raise ValueError("encoded_returns is too short for entropy estimation")

    res["entropy"] = res["sum"] / res["num"]
    alphabet_size = np.unique(encoded_returns).size
    max_entropy = np.log2(alphabet_size) if alphabet_size > 1 else 0
    res["residual"] = 1 - res["entropy"] / max_entropy if max_entropy > 0 else 0
    return res
