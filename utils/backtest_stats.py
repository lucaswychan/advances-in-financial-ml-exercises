import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

# Euler-Mascheroni constant
EULER_MASCHERONI = 0.57721566490153286060651209008240243104215933593992

def get_hhi(returns):
    returns = pd.Series(returns).dropna()
    num_bars = returns.shape[0]
    if num_bars <= 2:
        return np.nan

    sum_returns = returns.sum()
    if np.isclose(sum_returns, 0):
        return np.nan

    weights = returns / sum_returns
    w_sum_square = np.sum(weights ** 2)
    hhi = (w_sum_square - 1 / num_bars) / (1 - 1 / num_bars)
    return hhi

def compute_drawdown_and_time_under_water(pnl_series, as_dollars=False):
    """
    Compute the series of drawdowns and the associated time under water (TuW).
    
    Parameters:
        pnl_series (pd.Series): Cumulative PnL or price series indexed by datetime.
        as_dollars (bool): If True, drawdown is in dollar terms. If False, drawdown is in relative terms.

    Returns:
        drawdowns (pd.Series): Drawdown values, indexed by the start of each drawdown period.
        time_under_water (pd.Series): Time under water for each drawdown, in years, indexed by drawdown starts.
    """
    pnl_series = pnl_series.dropna().sort_index()
    df = pnl_series.to_frame('pnl')
    df['hwm'] = pnl_series.expanding().max()  # Running High Water Mark
    
    # For each new HWM, find the minimum PnL before the next HWM is achieved
    grouped_min = df.groupby('hwm').min().reset_index()
    grouped_min.columns = ['hwm', 'min_pnl']
    # The index of grouped_min should be the index at which each new hwm is reached
    grouped_min.index = df['hwm'].drop_duplicates(keep='first').index
    
    # Only consider periods where there is a drawdown
    drawdown_periods = grouped_min[grouped_min['hwm'] > grouped_min['min_pnl']]
    
    if as_dollars:
        drawdowns = drawdown_periods['hwm'] - drawdown_periods['min_pnl']
    else:
        drawdowns = 1 - drawdown_periods['min_pnl'] / drawdown_periods['hwm']
    
    # Compute time under water from each drawdown start to the next high-water mark.
    hwm_starts = grouped_min.index
    tuw = {}
    for start in drawdown_periods.index:
        pos = hwm_starts.get_loc(start)
        if pos + 1 < len(hwm_starts):
            end = hwm_starts[pos + 1]
        else:
            end = df.index[-1]
        tuw[start] = (end - start) / pd.to_timedelta(365, unit='D')
    time_under_water = pd.Series(tuw, dtype=float)
    
    return drawdowns, time_under_water

def get_psr(sharpe_ratio, r_skewness, r_kurtosis, T, target_sr=0.015):
    psr_inner_term = (sharpe_ratio - target_sr) * np.sqrt(T - 1) / np.sqrt(1.0 - r_skewness * sharpe_ratio + (r_kurtosis - 1) / 4.0 * sharpe_ratio ** 2)
    psr = norm.cdf(psr_inner_term)

    return psr

def get_dsr_s_star(var_trial_sr, num_trial):
    return np.sqrt(var_trial_sr) * ((1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / num_trial) + EULER_MASCHERONI * norm.ppf(1 - 1.0 / (num_trial*np.exp(1))))

def get_statistics(
    prices,
    obs_per_year,
    rf_rate=0.015,
    target_sr=1.1,
    dsr_var_trial_sr=0.5,
    dsr_num_trial=100,
):
    prices = prices.dropna().sort_index()
    returns = prices.pct_change().dropna()
    T = returns.shape[0]
    dd, tuw = compute_drawdown_and_time_under_water(prices)
    
    sr = returns.mean() / returns.std()
    annualized_sr = sr * np.sqrt(obs_per_year)
    r_skewness = skew(returns)
    r_kurtosis = kurtosis(returns, fisher=False)
    target_sr_per_obs = target_sr / np.sqrt(obs_per_year)
    psr = get_psr(sr, r_skewness, r_kurtosis, T, target_sr_per_obs)
    
    dsr_s_star_annualized = get_dsr_s_star(dsr_var_trial_sr, dsr_num_trial)
    dsr_s_star = dsr_s_star_annualized / np.sqrt(obs_per_year)
    dsr = get_psr(sr, r_skewness, r_kurtosis, T, dsr_s_star)
    rf_rate_per_obs = (1 + rf_rate) ** (1 / obs_per_year) - 1

    return {
        "hhi_positive": get_hhi(returns[returns > 0]),
        "hhi_negative": get_hhi(returns[returns < 0]),
        "hhi_time_between_bars": get_hhi(returns.groupby(pd.Grouper(freq='ME')).count()),
        "dd_95": dd.quantile(0.95),
        "tuw_95": tuw.quantile(0.95),
        "annualized_return": returns.mean() * obs_per_year,
        "average_return_from_hits": returns[returns > 0].mean(),
        "average_return_from_misses": returns[returns < 0].mean(),
        "annualized_sr": annualized_sr,
        "information_ratio": (returns.mean() - rf_rate_per_obs) / returns.std() * np.sqrt(obs_per_year),
        "psr": psr,
        "dsr": dsr,
        "dsr_s_star": dsr_s_star,
        "dsr_s_star_annualized": dsr_s_star_annualized,
        "target_sr_per_observation": target_sr_per_obs,
    }
