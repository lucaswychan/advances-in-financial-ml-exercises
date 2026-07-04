import pandas as pd
import numpy as np


def bde_recursive_residual_cusum(
    y,
    X,
    initial=None,
    add_intercept=True,
    conf_level=0.95,
):
    """
    Brown-Durbin-Evans CUSUM test on recursive residuals.

    y: pd.Series or array-like
    X: pd.DataFrame or array-like, excluding intercept unless add_intercept=True
    initial: number of observations used for the initial OLS fit
    conf_level: one of 0.90, 0.95, 0.99
    """
    crit = {
        0.90: 0.850,
        0.95: 0.948,
        0.99: 1.143,
    }
    if conf_level not in crit:
        raise ValueError("conf_level must be one of 0.90, 0.95, 0.99")

    y_has_index = isinstance(y, pd.Series)
    X_has_index = isinstance(X, (pd.Series, pd.DataFrame))

    X = pd.DataFrame(X).astype(float)
    if y_has_index:
        y = pd.Series(y)
        if not X_has_index and len(X) == len(y):
            X.index = y.index
    elif len(y) == len(X):
        y = pd.Series(y, index=X.index)
    else:
        y = pd.Series(y)

    target_col = "__bde_target__"
    while target_col in X.columns:
        target_col = f"_{target_col}"

    y = y.rename(target_col).astype(float)
    data = pd.concat([y, X], axis=1).dropna()
    y_arr = data[target_col].to_numpy()
    X_arr = data.drop(columns=target_col).to_numpy()

    if add_intercept:
        X_arr = np.column_stack([np.ones(len(X_arr)), X_arr])

    nobs, nvars = X_arr.shape

    if initial is None:
        initial = nvars

    if initial < nvars:
        raise ValueError("initial must be at least the number of regressors")

    while initial < nobs and np.linalg.matrix_rank(X_arr[:initial]) < nvars:
        initial += 1

    if initial >= nobs - 1:
        raise ValueError("Could not find a full-rank initial design matrix")

    X0 = X_arr[:initial]
    y0 = y_arr[:initial]

    P = np.linalg.inv(X0.T @ X0)
    beta = P @ X0.T @ y0

    recursive_resids = []

    for t in range(initial, nobs):
        x_t = X_arr[t]
        y_t = y_arr[t]

        pred_err = y_t - x_t @ beta
        leverage_factor = 1.0 + x_t @ P @ x_t

        w_t = pred_err / np.sqrt(leverage_factor)
        recursive_resids.append(w_t)

        # Recursive least squares update for future forecasts.
        gain = P @ x_t / leverage_factor
        beta = beta + gain * pred_err
        P = P - np.outer(gain, x_t @ P)
        P = (P + P.T) / 2.0

    recursive_resids = np.asarray(recursive_resids)

    sigma_hat = recursive_resids.std(ddof=1)
    if not np.isfinite(sigma_hat) or sigma_hat <= 0:
        raise ValueError("Recursive residual standard deviation is not positive")

    standardized = recursive_resids / sigma_hat

    m = len(standardized)
    cusum = np.r_[0.0, np.cumsum(standardized)]

    a = crit[conf_level]
    step = np.arange(m + 1)
    upper = a * np.sqrt(m) + 2 * a * step / np.sqrt(m)
    lower = -upper

    result = pd.DataFrame(
        {
            "recursive_resid": np.r_[np.nan, recursive_resids],
            "std_recursive_resid": np.r_[np.nan, standardized],
            "cusum": cusum,
            "lower": lower,
            "upper": upper,
        },
        index=data.index[initial - 1:],
    )

    crossed = (result["cusum"] < result["lower"]) | (result["cusum"] > result["upper"])
    result["crossed"] = crossed

    result.attrs["reject"] = bool(crossed.any())
    result.attrs["first_crossing"] = crossed[crossed].index[0] if crossed.any() else None
    result.attrs["initial_observations"] = initial
    result.attrs["nobs"] = nobs
    result.attrs["n_recursive_residuals"] = m
    result.attrs["sigma_hat"] = sigma_hat
    result.attrs["recursive_resid_mean"] = float(recursive_resids.mean())

    return result


def chow_type_dickey_fuller_test(y, min_fraction=0.15):
    """
    Chow-type Dickey-Fuller explosiveness test over candidate break dates.

    The test starts from a first-order autoregressive process:

        y_t = rho * y_{t-1} + eps_t

    Under the null hypothesis, rho = 1, so y follows a random walk and
    Delta y_t = eps_t. Under the explosive alternative, the process starts
    as a random walk and switches after an unknown break date b into
    rho > 1. Subtracting y_{t-1} from both sides after the break gives:

        Delta y_t = (rho - 1) * y_{t-1} + eps_t

    Let delta = rho - 1 and let D_t[b] be a break dummy equal to 0 before
    b and 1 from b onward. For each candidate break b, estimate:

        Delta y_t = delta * y_{t-1} * D_t[b] + eps_t

    For a fixed candidate break b, define x_t = y_{t-1} * D_t[b]. This is
    a one-regressor OLS model with no intercept:

        Delta y_t = delta * x_t + eps_t

    OLS chooses delta to minimize the residual sum of squares:

        RSS(delta) = sum_t (Delta y_t - delta * x_t)^2
                   = sum_t Delta y_t^2
                     - 2 * delta * sum_t x_t * Delta y_t
                     + delta^2 * sum_t x_t^2

    Differentiating with respect to delta and setting the derivative to 0:

        d RSS / d delta = -2 * sum_t x_t * Delta y_t
                          + 2 * delta * sum_t x_t^2 = 0

    Therefore:

        delta_hat = sum_t x_t * Delta y_t / sum_t x_t^2

    Because x_t = 0 before the candidate break and x_t = y_{t-1} after the
    candidate break, this equals:

        delta_hat = sum_after_break y_{t-1} * Delta y_t
                    / sum_after_break y_{t-1}^2

    The DFC statistic for a candidate break is delta_hat / se(delta_hat).
    The SDFC statistic is the maximum DFC over all candidate breaks inside
    the trimmed interval [min_fraction, 1 - min_fraction].

    Parameters
    ----------
    y : pd.Series or array-like
        Price or log-price level series. Do not pass returns; the function
        internally computes Delta y_t.
    min_fraction : float, default 0.15
        Trim fraction for candidate break dates. Candidate breaks are tried
        only between min_fraction and 1 - min_fraction of the sample.

    Returns
    -------
    pd.DataFrame
        One row per candidate break date with columns break_loc, tau, delta,
        delta_se, dfc, and selected. Summary values are stored in attrs:
        sdfc, break_date, break_loc, min_fraction, nobs, and n_candidates.

    Notes
    -----
    This implementation does not report a p-value because SDFC critical
    values are non-standard. Use the statistic and selected break date for
    the exercise mechanics unless appropriate simulated/tabulated critical
    values are added separately.
    """
    if not 0 < min_fraction < 0.5:
        raise ValueError("min_fraction must be between 0 and 0.5")

    y = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    y_arr = y.to_numpy()
    nobs = len(y_arr)

    if nobs < 4:
        raise ValueError("y must contain at least four non-null observations")

    dy = np.diff(y_arr)
    y_lag = y_arr[:-1]
    if not np.isfinite(dy).all() or not np.isfinite(y_lag).all():
        raise ValueError("y must contain only finite observations after dropping nulls")

    if dy.std(ddof=0) <= 0:
        raise ValueError("First differences must have positive variance")

    min_loc = max(1, int(np.ceil(min_fraction * nobs)))
    max_loc = min(nobs - 1, int(np.floor((1 - min_fraction) * nobs)))
    if min_loc > max_loc:
        raise ValueError("No candidate break dates remain after applying trim")

    break_locs = np.arange(min_loc, max_loc + 1)
    m = len(dy)
    if m <= 1:
        raise ValueError("Not enough observations to estimate the regression")

    # Precompute the OLS cross-products without the break dummy. The dummy is
    # applied below by suffix sums: for a candidate break, only rows from the
    # break onward are summed, which is equivalent to multiplying earlier rows
    # by D_t[b] = 0.
    zdy = y_lag * dy
    z2 = y_lag**2
    suffix_zdy = np.r_[np.cumsum(zdy[::-1])[::-1], 0.0]
    suffix_z2 = np.r_[np.cumsum(z2[::-1])[::-1], 0.0]

    # dy[j] = y[j + 1] - y[j], so a break at level index k first affects
    # diff row k - 1. The suffix sum from diff_locs therefore implements
    # sum_t D_t[b] * y_{t-1} * Delta y_t and sum_t (D_t[b] * y_{t-1})^2.
    diff_locs = break_locs - 1
    numerator = suffix_zdy[diff_locs]
    denominator = suffix_z2[diff_locs]
    valid = denominator > 0
    if not np.any(valid):
        raise ValueError("No candidate break dates have positive regressor variance")

    delta = np.full(len(break_locs), np.nan)
    delta[valid] = numerator[valid] / denominator[valid]

    # For no-intercept one-regressor OLS:
    # RSS = sum(Delta y_t^2) - (sum(x_t * Delta y_t)^2 / sum(x_t^2)).
    # Pre-break observations still contribute to RSS through Delta y_t^2
    # because their regressor x_t is zero, so fitted values are zero there.
    total_dy2 = np.sum(dy**2)
    rss = np.full(len(break_locs), np.nan)
    rss[valid] = total_dy2 - numerator[valid] ** 2 / denominator[valid]
    rss = np.maximum(rss, 0.0)

    sigma2 = rss / (m - 1)
    delta_se = np.full(len(break_locs), np.nan)
    delta_se[valid] = np.sqrt(sigma2[valid] / denominator[valid])

    dfc = np.full(len(break_locs), np.nan)
    se_valid = valid & np.isfinite(delta_se) & (delta_se > 0)
    dfc[se_valid] = delta[se_valid] / delta_se[se_valid]

    finite_dfc = np.isfinite(dfc)
    if not np.any(finite_dfc):
        raise ValueError("No finite DFC statistics could be estimated")

    selected_pos = int(np.nanargmax(dfc))
    selected = np.zeros(len(break_locs), dtype=bool)
    selected[selected_pos] = True

    result = pd.DataFrame(
        {
            "break_loc": break_locs,
            "tau": break_locs / nobs,
            "delta": delta,
            "delta_se": delta_se,
            "dfc": dfc,
            "selected": selected,
        },
        index=y.index.to_numpy()[break_locs],
    )

    result.attrs["sdfc"] = float(dfc[selected_pos])
    result.attrs["break_date"] = result.index[selected_pos]
    result.attrs["break_loc"] = int(break_locs[selected_pos])
    result.attrs["min_fraction"] = min_fraction
    result.attrs["nobs"] = nobs
    result.attrs["n_candidates"] = int(len(break_locs))

    return result


def supremum_adf_test(y, min_length, lags, add_intercept=True, q=0.95, v=0.025):
    """
    Supremum, Quantile, and Conditional Augmented Dickey-Fuller statistics.

    The book definition fixes a right endpoint t, estimates the ADF regression

        Delta y_t = alpha + beta * y_{t-1}
                    + sum_i gamma_i * Delta y_{t-i} + eps_t

    over every valid backward-expanding start point t0, then takes

        SADF_t = sup_{t0 in [1, t - tau]} ADF_{t0,t}
               = sup_{t0 in [1, t - tau]} beta_hat_{t0,t}
                                            / se(beta_hat_{t0,t})

    This is a right-tail test with H0: beta <= 0 and H1: beta > 0, so the
    supremum is over the raw beta t-statistic, not its absolute value.

    Quantile ADF
    ------------
    SADF is sensitive to the single most extreme value in the endpoint set

        s_t = {ADF_{t0,t} | t0 in [1, t - tau]}.

    QADF replaces that maximum with a high quantile of the same set:

        QADF_{t,q} = Q[s_t, q],

    where Q[., q] is the empirical q quantile. A local dispersion measure is

        QDOT_{t,q,v} = QADF_{t,q+v} - QADF_{t,q-v},

    with 0 < v <= min(q, 1 - q). When q = 1 and v is not defined, QADF is the
    same as SADF.

    Conditional ADF
    ---------------
    CADF summarizes the right tail of s_t above QADF_{t,q}. Let f[x] be the
    empirical probability distribution over s_t and

        K = integral_{QADF_{t,q}}^infinity f[x] dx.

    The conditional center and dispersion are

        CADF_{t,q} = K^{-1} integral_{QADF_{t,q}}^infinity x f[x] dx

        CDOT_{t,q} = sqrt(
            K^{-1} integral_{QADF_{t,q}}^infinity
            (x - CADF_{t,q})^2 f[x] dx
        ).

    In this implementation the integrals are empirical averages over ADF
    t-statistics greater than or equal to QADF_{t,q}.

    Implementation map
    ------------------
    After dropping nulls, the function constructs one regression row per usable
    time point:

        z_t = Delta y_t
        x_t = [1, y_{t-1}, Delta y_{t-1}, ..., Delta y_{t-L}]

    where the intercept column is omitted when add_intercept=False. A candidate
    window is a contiguous block of these regression rows, from start s to end
    e. The endpoint e represents the fixed right side t in SADF_t; the candidate
    start s represents t0; min_length is tau, the minimum number of regression
    rows allowed in a window.

    OLS derivation used by the code
    -------------------------------
    For a fixed candidate window, stack the dependent variables in vector z and
    regressors in matrix X. OLS minimizes

        RSS(b) = (z - Xb)'(z - Xb)
               = z'z - 2 b'X'z + b'X'Xb.

    Differentiating and setting the first-order condition to zero gives

        d RSS / d b = -2 X'z + 2 X'Xb = 0
        X'X beta_hat = X'z
        beta_hat = (X'X)^{-1} X'z,

    when X'X is full rank. With n observations and k regressors,

        sigma_hat^2 = RSS(beta_hat) / (n - k)
        Var(beta_hat | X) = sigma_hat^2 (X'X)^{-1}
        se(beta_hat_j) = sqrt(sigma_hat^2 * (X'X)^{-1}_{j,j}).

    The ADF statistic for this window is therefore

        beta_hat_y_lag / se(beta_hat_y_lag).

    Fast exact window updates
    -------------------------
    The naive implementation would refit OLS for every (start, endpoint) pair.
    This function computes cumulative cross-products once:

        prefix_xx[r] = sum_{i < r} x_i x_i'
        prefix_xy[r] = sum_{i < r} x_i z_i
        prefix_yy[r] = sum_{i < r} z_i^2.

    For any window [s, e],

        X'X = prefix_xx[e + 1] - prefix_xx[s]
        X'z = prefix_xy[e + 1] - prefix_xy[s]
        z'z = prefix_yy[e + 1] - prefix_yy[s].

    The statistic is still the exact OLS statistic for each window; the prefix
    sums only avoid rebuilding and multiplying each window's X matrix.

    Parameters
    ----------
    y : pd.Series or array-like
        Price or log-price level series. Do not pass returns; the function
        internally computes Delta y_t.
    min_length : int
        Minimum number of usable ADF regression rows in each candidate window,
        after differencing and lag construction.
    lags : int
        Number of lagged first-difference terms to include.
    add_intercept : bool, default True
        Whether to include alpha in the ADF regression.
    q : float, default 0.95
        Quantile of s_t used for QADF and CADF. Use q=1 with v=None to recover
        the SADF value as QADF.
    v : float or None, default 0.025
        Half-width around q used for the QADF dispersion measure
        QADF_{t,q+v} - QADF_{t,q-v}. Set v=None when the dispersion measure is
        not defined, for example q=1.

    Returns
    -------
    pd.DataFrame
        One row per evaluated endpoint. SADF columns are:

        sadf
            Supremum ADF t-statistic for this endpoint.
        start, start_loc, end_loc
            Start date, start row, and endpoint row of the maximizing window.
        beta, beta_se, nobs
            Coefficient, standard error, and observation count for the
            maximizing ADF window.

        QADF and CADF columns are:

        qadf
            Empirical q quantile of all finite ADF t-statistics ending at this
            endpoint.
        qadf_lower, qadf_upper, qadf_dispersion
            Quantiles q-v and q+v, and their difference. These are NaN when
            v=None.
        cadf, cadf_dispersion
            Empirical conditional mean and standard deviation of ADF
            t-statistics greater than or equal to qadf.
        n_adf_windows, tail_count, tail_probability
            Number of finite endpoint ADF windows, number in the right tail,
            and their ratio.
        sadf_cadf_z
            (sadf - cadf) / cadf_dispersion.

        Summary values are stored in attrs: lags, min_length, nobs,
        n_endpoints, input_nobs, add_intercept, q, and v.

    Notes
    -----
    SADF, QADF, and CADF critical values are non-standard, so this function
    reports statistics but no p-values.
    """
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)):
        raise TypeError("lags must be an integer")
    if lags < 0:
        raise ValueError("lags must be non-negative")

    if isinstance(min_length, bool) or not isinstance(min_length, (int, np.integer)):
        raise TypeError("min_length must be an integer")
    if min_length <= 0:
        raise ValueError("min_length must be positive")

    if not isinstance(q, (int, float, np.integer, np.floating)):
        raise TypeError("q must be numeric")
    q = float(q)
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")

    if v is not None:
        if not isinstance(v, (int, float, np.integer, np.floating)):
            raise TypeError("v must be numeric or None")
        v = float(v)
        if not 0 < v <= min(q, 1 - q):
            raise ValueError("v must satisfy 0 < v <= min(q, 1 - q), or be None")

    y = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    input_nobs = len(y)
    if input_nobs < lags + 3:
        raise ValueError("y is too short for the requested lag order")

    # Build the exact ADF regression design from the displayed equation.
    dy = y.diff()
    data = pd.DataFrame({"dy": dy, "y_lag": y.shift(1)})
    for lag in range(1, lags + 1):
        data[f"dy_lag_{lag}"] = dy.shift(lag)
    data = data.dropna()

    if data.empty:
        raise ValueError("No usable ADF regression rows remain after lagging")

    z = data["dy"].to_numpy(dtype=float)
    regressors = data.drop(columns="dy").to_numpy(dtype=float)
    y_lag_col = 0
    if add_intercept:
        X = np.column_stack([np.ones(len(regressors)), regressors])
        beta_col = y_lag_col + 1
    else:
        X = regressors
        beta_col = y_lag_col

    nobs, nvars = X.shape
    if nobs < min_length:
        raise ValueError("Not enough usable observations for min_length")
    if min_length <= nvars:
        raise ValueError("min_length must exceed the number of regressors")
    if z.std(ddof=0) <= 0:
        raise ValueError("First differences must have positive variance")
    if not np.isfinite(X).all() or not np.isfinite(z).all():
        raise ValueError("ADF regression data must be finite")

    # Cross-products for each single row; cumulative sums below turn these into
    # X'X, X'z, and z'z for any start/end window in O(1) slicing time.
    xx = X[:, :, None] * X[:, None, :]
    xy = X * z[:, None]
    yy = z * z

    prefix_xx = np.concatenate(
        [np.zeros((1, nvars, nvars)), np.cumsum(xx, axis=0)],
        axis=0,
    )
    prefix_xy = np.concatenate(
        [np.zeros((1, nvars)), np.cumsum(xy, axis=0)],
        axis=0,
    )
    prefix_yy = np.r_[0.0, np.cumsum(yy)]

    # The first endpoint is the first row that can close a tau-sized window.
    endpoint_locs = np.arange(min_length - 1, nobs)
    sadf = np.full(len(endpoint_locs), np.nan)
    selected_start = np.full(len(endpoint_locs), -1)
    selected_beta = np.full(len(endpoint_locs), np.nan)
    selected_beta_se = np.full(len(endpoint_locs), np.nan)
    selected_nobs = np.full(len(endpoint_locs), np.nan)
    qadf = np.full(len(endpoint_locs), np.nan)
    qadf_lower = np.full(len(endpoint_locs), np.nan)
    qadf_upper = np.full(len(endpoint_locs), np.nan)
    qadf_dispersion = np.full(len(endpoint_locs), np.nan)
    cadf = np.full(len(endpoint_locs), np.nan)
    cadf_dispersion = np.full(len(endpoint_locs), np.nan)
    n_adf_windows = np.zeros(len(endpoint_locs), dtype=int)
    tail_count = np.zeros(len(endpoint_locs), dtype=int)

    eps = np.finfo(float).eps

    for row_pos, end_loc in enumerate(endpoint_locs):
        # For this endpoint, try all backward-expanding start points whose
        # windows contain at least min_length regression rows.
        start_locs = np.arange(0, end_loc - min_length + 2)
        window_nobs = end_loc - start_locs + 1

        # Window cross-products from prefix differences:
        # sum_{s <= i <= e}(.) = prefix[e + 1] - prefix[s].
        xtx = prefix_xx[end_loc + 1] - prefix_xx[start_locs]
        xty = prefix_xy[end_loc + 1] - prefix_xy[start_locs]
        yty = prefix_yy[end_loc + 1] - prefix_yy[start_locs]

        det = np.linalg.det(xtx)
        scale = np.linalg.norm(xtx, ord=np.inf, axis=(1, 2))
        rank_threshold = eps * np.maximum(scale, 1.0) ** nvars
        full_rank = np.isfinite(det) & (np.abs(det) > rank_threshold)
        valid = (window_nobs > nvars) & full_rank
        if not np.any(valid):
            print("Warning: No valid ADF windows found for endpoint", end_loc)
            continue

        beta = np.full((len(start_locs), nvars), np.nan)
        inv_xtx = np.full_like(xtx, np.nan)

        try:
            # beta_hat = (X'X)^{-1} X'z, computed by solve for numerical
            # stability instead of explicitly multiplying by the inverse.
            beta[valid] = np.linalg.solve(xtx[valid], xty[valid][..., None]).squeeze(-1)
            inv_xtx[valid] = np.linalg.inv(xtx[valid])
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix for endpoint", end_loc)
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            # RSS = z'z - beta_hat' X'z follows from the OLS normal equations.
            rss = yty - np.einsum("ij,ij->i", beta, xty)
            rss = np.maximum(rss, 0.0)
            sigma2 = rss / (window_nobs - nvars)
            beta_se = np.sqrt(sigma2 * inv_xtx[:, beta_col, beta_col])
            adf_stat = beta[:, beta_col] / beta_se

        finite_stat = valid & np.isfinite(adf_stat)
        if not np.any(finite_stat):
            print("Warning: No finite ADF statistics for endpoint", end_loc)
            continue

        endpoint_stats = adf_stat[finite_stat]
        n_adf_windows[row_pos] = len(endpoint_stats)

        # SADF_t is the supremum over starts for this fixed endpoint.
        best_pos = int(np.nanargmax(np.where(finite_stat, adf_stat, np.nan)))
        sadf[row_pos] = adf_stat[best_pos]
        selected_start[row_pos] = start_locs[best_pos]
        selected_beta[row_pos] = beta[best_pos, beta_col]
        selected_beta_se[row_pos] = beta_se[best_pos]
        selected_nobs[row_pos] = window_nobs[best_pos]

        qadf[row_pos] = np.nanquantile(endpoint_stats, q)
        if v is not None:
            qadf_lower[row_pos] = np.nanquantile(endpoint_stats, q - v)
            qadf_upper[row_pos] = np.nanquantile(endpoint_stats, q + v)
            qadf_dispersion[row_pos] = qadf_upper[row_pos] - qadf_lower[row_pos]

        tail = endpoint_stats[endpoint_stats >= qadf[row_pos]]
        tail_count[row_pos] = len(tail)
        cadf[row_pos] = tail.mean()
        cadf_dispersion[row_pos] = tail.std(ddof=0)

    finite_sadf = np.isfinite(sadf)
    if not np.any(finite_sadf):
        raise ValueError("No finite SADF statistics could be estimated")

    starts = np.full(len(endpoint_locs), np.nan, dtype=object)
    valid_starts = selected_start >= 0
    starts[valid_starts] = data.index.to_numpy()[selected_start[valid_starts]]

    result = pd.DataFrame(
        {
            "sadf": sadf,
            "start": starts,
            "start_loc": selected_start,
            "end_loc": endpoint_locs,
            "beta": selected_beta,
            "beta_se": selected_beta_se,
            "nobs": selected_nobs,
            "qadf": qadf,
            "qadf_lower": qadf_lower,
            "qadf_upper": qadf_upper,
            "qadf_dispersion": qadf_dispersion,
            "cadf": cadf,
            "cadf_dispersion": cadf_dispersion,
            "n_adf_windows": n_adf_windows,
            "tail_count": tail_count,
        },
        index=data.index.to_numpy()[endpoint_locs],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result["tail_probability"] = result["tail_count"] / result["n_adf_windows"]
        result["sadf_cadf_z"] = (result["sadf"] - result["cadf"]) / result["cadf_dispersion"]

    result.attrs["lags"] = int(lags)
    result.attrs["min_length"] = int(min_length)
    result.attrs["nobs"] = int(nobs)
    result.attrs["n_endpoints"] = int(len(result))
    result.attrs["input_nobs"] = int(input_nobs)
    result.attrs["add_intercept"] = bool(add_intercept)
    result.attrs["q"] = q
    result.attrs["v"] = v

    return result


def sub_super_martingale_test(y, min_length, phi=1.0, trends="all"):
    """
    Sub- and super-martingale trend explosiveness statistics.

    For each fixed endpoint, this test fits the selected trend specifications
    over all backward-expanding start points with at least min_length
    observations and reports the largest penalized absolute t-statistic:

        SMT_t = sup_{t0} |beta_hat_{t0,t} / se(beta_hat_{t0,t})|
                / (t - t0)^phi.

    Supported trend specifications are:

        poly1: y_t = alpha + gamma * t + beta * t^2 + eps_t
        poly2: log(y_t) = alpha + gamma * t + beta * t^2 + eps_t
        exp: log(y_t) = alpha + beta * t + eps_t
        power: log(y_t) = alpha + beta * log(t) + eps_t

    Equation and code map
    ---------------------
    The code uses 0-based NumPy locations internally, but constructs the model
    time variable as t = 1, ..., T to match the notation above. For a fixed
    endpoint location e, the endpoint date is y.index[e]. For every valid start
    location s, the regression window is the inclusive slice [s, e], and:

        window_nobs = e - s + 1
        elapsed = e - s

    Therefore elapsed is the code version of the book's (t - t0) denominator.

    For each trend the function builds a dependent vector z and design matrix X:

        poly1: z = y,      X = [1, t, t^2],    beta_col = t^2
        poly2: z = log(y), X = [1, t, t^2],    beta_col = t^2
        exp:   z = log(y), X = [1, t],         beta_col = t
        power: z = log(y), X = [1, log(t)],    beta_col = log(t)

    For every candidate window [s, e], the code estimates the OLS model

        z = X b + eps

    using the standard cross-product formulas:

        beta_hat = (X'X)^(-1) X'z
        RSS = z'z - beta_hat' X'z
        sigma_hat^2 = RSS / (n - k)
        se(beta_hat_j) = sqrt(sigma_hat^2 * (X'X)^(-1)_{j,j})
        t_stat = beta_hat_j / se(beta_hat_j)

    The endpoint SMT statistic is then:

        max_s abs(t_stat_{s,e}) / elapsed_{s,e}^phi.

    The absolute value implements the sub- and super-martingale idea: both
    upward explosive growth and downward explosive collapse are counted as
    large departures. The elapsed penalty implements the sample-length
    correction described in the book.

    Parameters
    ----------
    y : pd.Series or array-like
        Positive level series when any log-based trend is requested.
    min_length : int
        Minimum number of observations in each candidate trend window.
    phi : float, default 1.0
        Sample-length penalty exponent in [0, 1].
    trends : "all", str, or iterable of str, default "all"
        Trend specification names to estimate.

    Returns
    -------
    pd.DataFrame
        One row per evaluated endpoint, with trend-prefixed columns:
        smt, start, start_loc, end_loc, beta, beta_se, t_stat, nobs, elapsed.
        Summary values are stored in attrs: min_length, phi, trends, nobs,
        and input_nobs.

    Notes
    -----
    SMT critical values are non-standard, so this function reports statistics
    but no p-values.
    """
    # These are the four trend specifications named in AFML section 17.4.3.
    valid_trends = ("poly1", "poly2", "exp", "power")

    # min_length is the minimum regression window length, tau in the book text.
    if isinstance(min_length, bool) or not isinstance(min_length, (int, np.integer)):
        raise TypeError("min_length must be an integer")
    if min_length <= 0:
        raise ValueError("min_length must be positive")

    # phi is the exponent in the denominator (t - t0)^phi and is restricted to
    # the interval suggested by the text.
    if isinstance(phi, bool) or not isinstance(phi, (int, float, np.integer, np.floating)):
        raise TypeError("phi must be numeric")
    phi = float(phi)
    if not 0 <= phi <= 1:
        raise ValueError("phi must be in [0, 1]")

    # Normalize the trend selector so the main loop can always iterate over a
    # list of explicit specification names.
    if isinstance(trends, str):
        if trends == "all":
            selected_trends = list(valid_trends)
        else:
            selected_trends = [trends]
    else:
        try:
            selected_trends = list(trends)
        except TypeError as exc:
            raise TypeError("trends must be 'all', a string, or an iterable") from exc

    if not selected_trends:
        raise ValueError("At least one trend must be selected")

    # Validate names and preserve user order while removing duplicates. This
    # keeps the output column order predictable.
    deduped_trends = []
    for trend in selected_trends:
        if trend not in valid_trends:
            raise ValueError(
                "trends must contain only 'poly1', 'poly2', 'exp', or 'power'"
            )
        if trend not in deduped_trends:
            deduped_trends.append(trend)
    selected_trends = deduped_trends

    # Convert the input to the cleaned level series y_t used in the equations.
    y = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    input_nobs = len(y)
    y_arr = y.to_numpy(dtype=float)
    nobs = len(y_arr)

    if nobs < min_length:
        raise ValueError("Not enough usable observations for min_length")

    # poly2, exp, and power all use log(y_t), so the level series must be
    # strictly positive whenever any of those specifications is requested.
    log_trends = {"poly2", "exp", "power"}
    needs_log_y = any(trend in log_trends for trend in selected_trends)
    if needs_log_y and np.any(y_arr <= 0):
        raise ValueError("y must be positive for log-based SMT trends")

    # The book writes t = 1, ..., T. NumPy row locations are 0-based, but the
    # regressor values are 1-based to match the displayed trend equations.
    t = np.arange(1, nobs + 1, dtype=float)
    log_y = np.log(y_arr) if needs_log_y else None
    # The first endpoint is the earliest row that can close a tau-sized window.
    endpoint_locs = np.arange(min_length - 1, nobs)
    trend_frames = []

    for trend in selected_trends:
        # Build z and X for the selected trend, and record which coefficient is
        # beta in H0: beta = 0, H1: beta != 0.
        if trend == "poly1":
            # SM-Poly1: y_t = alpha + gamma*t + beta*t^2 + eps_t.
            z = y_arr
            X = np.column_stack([np.ones(nobs), t, t**2])
            beta_col = 2
        elif trend == "poly2":
            # SM-Poly2: log(y_t) = alpha + gamma*t + beta*t^2 + eps_t.
            z = log_y
            X = np.column_stack([np.ones(nobs), t, t**2])
            beta_col = 2
        elif trend == "exp":
            # SM-Exp: y_t = alpha*exp(beta*t) + eps_t is estimated as
            # log(y_t) = log(alpha) + beta*t + transformed error.
            z = log_y
            X = np.column_stack([np.ones(nobs), t])
            beta_col = 1
        else:
            # SM-Power: y_t = alpha*t^beta + eps_t is estimated as
            # log(y_t) = log(alpha) + beta*log(t) + transformed error.
            z = log_y
            X = np.column_stack([np.ones(nobs), np.log(t)])
            beta_col = 1

        # k in the OLS formulas is the number of regressors. Each window must
        # have n > k so sigma_hat^2 = RSS / (n - k) is defined.
        nvars = X.shape[1]
        if min_length <= nvars:
            raise ValueError(
                f"min_length must exceed the number of regressors for {trend}"
            )
        if z.std(ddof=0) <= 0:
            raise ValueError(f"Dependent variable for {trend} must have positive variance")
        if not np.isfinite(X).all() or not np.isfinite(z).all():
            raise ValueError(f"SMT regression data for {trend} must be finite")

        # Per-row cross-products. Prefix sums below let any window [s, e] be
        # retrieved as prefix[e + 1] - prefix[s], avoiding repeated matrix
        # construction for every candidate start.
        xx = X[:, :, None] * X[:, None, :]
        xy = X * z[:, None]
        yy = z * z

        # prefix_xx[r] = sum_{i < r} x_i x_i'
        # prefix_xy[r] = sum_{i < r} x_i z_i
        # prefix_yy[r] = sum_{i < r} z_i^2
        prefix_xx = np.concatenate(
            [np.zeros((1, nvars, nvars)), np.cumsum(xx, axis=0)],
            axis=0,
        )
        prefix_xy = np.concatenate(
            [np.zeros((1, nvars)), np.cumsum(xy, axis=0)],
            axis=0,
        )
        prefix_yy = np.r_[0.0, np.cumsum(yy)]

        # These arrays hold, for each endpoint e, the start s that maximizes
        # abs(t_stat_{s,e}) / (e - s)^phi and the diagnostics for that window.
        smt = np.full(len(endpoint_locs), np.nan)
        selected_start = np.full(len(endpoint_locs), -1)
        selected_beta = np.full(len(endpoint_locs), np.nan)
        selected_beta_se = np.full(len(endpoint_locs), np.nan)
        selected_t_stat = np.full(len(endpoint_locs), np.nan)
        selected_nobs = np.full(len(endpoint_locs), np.nan)
        selected_elapsed = np.full(len(endpoint_locs), np.nan)

        for row_pos, end_loc in enumerate(endpoint_locs):
            # For endpoint e, valid starts are s = 0, ..., e - tau + 1. This is
            # the backward-expanding start-point set t0 in [1, t - tau].
            start_locs = np.arange(0, end_loc - min_length + 2)
            # window_nobs is inclusive n = e - s + 1 for the OLS degrees of
            # freedom. elapsed is e - s, the denominator's t - t0.
            window_nobs = end_loc - start_locs + 1
            elapsed = end_loc - start_locs

            # Window cross-products:
            # X'X = sum_{s <= i <= e} x_i x_i'
            # X'z = sum_{s <= i <= e} x_i z_i
            # z'z = sum_{s <= i <= e} z_i^2
            xtx = prefix_xx[end_loc + 1] - prefix_xx[start_locs]
            xty = prefix_xy[end_loc + 1] - prefix_xy[start_locs]
            yty = prefix_yy[end_loc + 1] - prefix_yy[start_locs]

            # Keep only windows with finite cross-products, enough degrees of
            # freedom, and positive elapsed time for the penalty denominator.
            finite_window = (
                np.isfinite(xtx).all(axis=(1, 2))
                & np.isfinite(xty).all(axis=1)
                & np.isfinite(yty)
            )
            valid = (window_nobs > nvars) & finite_window & (elapsed > 0)
            if not np.any(valid):
                continue

            beta = np.full((len(start_locs), nvars), np.nan)
            inv_xtx = np.full_like(xtx, np.nan)

            try:
                # beta_hat = (X'X)^(-1) X'z. solve is numerically preferable
                # to explicitly multiplying by the inverse.
                beta[valid] = np.linalg.solve(
                    xtx[valid],
                    xty[valid][..., None],
                ).squeeze(-1)
                # The inverse is still needed for Var(beta_hat | X).
                inv_xtx[valid] = np.linalg.inv(xtx[valid])
            except np.linalg.LinAlgError:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                # At the OLS solution, RSS = z'z - beta_hat' X'z.
                rss = yty - np.einsum("ij,ij->i", beta, xty)
                rss = np.maximum(rss, 0.0)
                # sigma_hat^2 = RSS / (n - k).
                sigma2 = rss / (window_nobs - nvars)
                # se(beta_hat_j) = sqrt(sigma_hat^2 * (X'X)^(-1)_{j,j}).
                beta_se = np.sqrt(sigma2 * inv_xtx[:, beta_col, beta_col])
                # This is beta_hat_{t0,t} / sigma_hat_{beta,t0,t}.
                t_stat = beta[:, beta_col] / beta_se
                # SMT applies the absolute value and the sample-length penalty.
                smt_stat = np.abs(t_stat) / (elapsed.astype(float) ** phi)

            # Drop singular, zero-variance, or otherwise undefined candidates.
            finite_stat = valid & np.isfinite(smt_stat)
            if not np.any(finite_stat):
                continue

            # sup over t0: choose the start giving the largest finite SMT value
            # for this fixed endpoint.
            best_pos = int(np.nanargmax(np.where(finite_stat, smt_stat, np.nan)))
            smt[row_pos] = smt_stat[best_pos]
            selected_start[row_pos] = start_locs[best_pos]
            selected_beta[row_pos] = beta[best_pos, beta_col]
            selected_beta_se[row_pos] = beta_se[best_pos]
            selected_t_stat[row_pos] = t_stat[best_pos]
            selected_nobs[row_pos] = window_nobs[best_pos]
            selected_elapsed[row_pos] = elapsed[best_pos]

        finite_smt = np.isfinite(smt)
        if not np.any(finite_smt):
            raise ValueError(f"No finite SMT statistics could be estimated for {trend}")

        # Convert the winning start row s back to the original y index so users
        # can inspect the selected window without translating row locations.
        starts = np.full(len(endpoint_locs), np.nan, dtype=object)
        valid_starts = selected_start >= 0
        starts[valid_starts] = y.index.to_numpy()[selected_start[valid_starts]]

        # Use trend-prefixed columns because the returned DataFrame can contain
        # several different SMT specifications side by side.
        trend_frames.append(
            pd.DataFrame(
                {
                    f"{trend}_smt": smt,
                    f"{trend}_start": starts,
                    f"{trend}_start_loc": selected_start,
                    f"{trend}_end_loc": endpoint_locs,
                    f"{trend}_beta": selected_beta,
                    f"{trend}_beta_se": selected_beta_se,
                    f"{trend}_t_stat": selected_t_stat,
                    f"{trend}_nobs": selected_nobs,
                    f"{trend}_elapsed": selected_elapsed,
                },
                index=y.index.to_numpy()[endpoint_locs],
            )
        )

    # Combine the requested specifications on the same endpoint index.
    result = pd.concat(trend_frames, axis=1)
    # attrs keep the test configuration with the result, matching the pattern
    # used by the other structural-break helpers in this module.
    result.attrs["min_length"] = int(min_length)
    result.attrs["phi"] = phi
    result.attrs["trends"] = tuple(selected_trends)
    result.attrs["nobs"] = int(nobs)
    result.attrs["input_nobs"] = int(input_nobs)

    return result


def csw_cusum_test(y, b_alpha=4.6):
    """
    Chu-Stinchcombe-White CUSUM test on levels.

    y: pd.Series or array-like log-price series
    b_alpha: critical-value constant; 4.6 is the one-sided 5% value
    """
    if b_alpha <= 0:
        raise ValueError("b_alpha must be positive")

    y = pd.Series(y).astype(float).dropna()
    y_arr = y.to_numpy()
    nobs = len(y_arr)

    if nobs < 2:
        raise ValueError("y must contain at least two non-null observations")

    dy = np.diff(y_arr)
    sigma_hat = np.r_[np.nan, np.sqrt(np.cumsum(dy**2) / np.arange(1, nobs))]

    if not np.any(np.isfinite(sigma_hat) & (sigma_hat > 0)):
        raise ValueError("CSW volatility estimate is not positive")

    elapsed = np.subtract.outer(np.arange(nobs), np.arange(nobs)).astype(float)
    elapsed[elapsed <= 0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = (y_arr[:, None] - y_arr[None, :]) / (
            sigma_hat[:, None] * np.sqrt(elapsed)
        )
        critical_values = np.sqrt(b_alpha + np.log(elapsed))

    statistic[~np.isfinite(statistic)] = np.nan

    finite_stat = np.isfinite(statistic)
    has_stat = finite_stat.any(axis=1)
    statistic_filled = np.where(finite_stat, statistic, -np.inf)
    reference_loc = np.argmax(statistic_filled, axis=1)

    csw = np.full(nobs, np.nan)
    critical = np.full(nobs, np.nan)
    reference = np.full(nobs, np.nan, dtype=object)
    elapsed_at_reference = np.full(nobs, np.nan)

    rows = np.flatnonzero(has_stat)
    refs = reference_loc[rows]

    csw[rows] = statistic[rows, refs]
    critical[rows] = critical_values[rows, refs]
    reference[rows] = y.index.to_numpy()[refs]
    elapsed_at_reference[rows] = elapsed[rows, refs]

    result = pd.DataFrame(
        {
            "sigma_hat": sigma_hat,
            "csw": csw,
            "critical": critical,
            "reference": reference,
            "elapsed": elapsed_at_reference,
        },
        index=y.index,
    )

    result["crossed"] = result["csw"] > result["critical"]

    crossed = result["crossed"]
    result.attrs["reject"] = bool(crossed.any())
    result.attrs["first_crossing"] = crossed[crossed].index[0] if crossed.any() else None
    result.attrs["b_alpha"] = b_alpha
    result.attrs["nobs"] = nobs

    return result
