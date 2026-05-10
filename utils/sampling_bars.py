import numpy as np
import pandas as pd

def tick_bar(df, tick_size=1):
    indices = np.arange(0, len(df), tick_size)
    return df.iloc[indices]

def volume_bar(df, volume_size=100):
    curr_vol = 0
    tick_indices = []
    
    for i in range(len(df)):
        curr_vol += df.iloc[i]['vol']
        if curr_vol >= volume_size:
            tick_indices.append(i)
            curr_vol = 0

    return df.iloc[tick_indices]

def dollar_bar(df, dollar_size=100000):
    if dollar_size <= 0:
        return df.iloc[np.arange(len(df))]

    dollar_values = df['dollar_vol'].to_numpy()
    dollar_cumsum = np.cumsum(dollar_values)
    tick_indices = []

    # Dollar volume is non-negative, so the cumulative sum can be searched directly.
    curr_idx = -1
    while curr_idx < len(dollar_cumsum) - 1:
        prev_dollar = 0 if curr_idx == -1 else dollar_cumsum[curr_idx]
        curr_idx = np.searchsorted(
            dollar_cumsum,
            prev_dollar + dollar_size,
            side='left',
        )
        if curr_idx == len(dollar_cumsum):
            break
        tick_indices.append(curr_idx)

    return df.iloc[tick_indices]

def ewma_last(values, window):
    """Return the latest EWMA value as a scalar."""
    window = max(1, int(round(window)))
    return pd.Series(values).ewm(span=window, adjust=False).mean().iloc[-1]

def dollar_imbalance_bar(
    df,
    expected_num_ticks_init=100,
    num_prev_bars=3,
    dollar_col='dollar_vol',
    exp_num_ticks_constraints=None,
):
    """Sample dollar imbalance bars using EWMA estimates of E[T] and E[b_t v_t]."""
    if exp_num_ticks_constraints is None:
        exp_num_ticks_constraints = (
            max(1, expected_num_ticks_init / 2),
            expected_num_ticks_init * 10,
        )
    min_exp_num_ticks, max_exp_num_ticks = exp_num_ticks_constraints
    
    theta_t = 0
    num_ticks = 0
    expected_num_ticks = float(expected_num_ticks_init)
    expected_imbalance = None
    tick_rule = 1
    tick_indices = [] # for storing indices for sampling
    imbalance_arr = [] # b_t * v_t values used to estimate expected dollar imbalance
    bar_lengths = [] # realized T values used to estimate E[T]
    prices = df['price'].to_numpy()
    dollar_values = df[dollar_col].to_numpy()
    
    for i in range(1, len(df)):
        num_ticks += 1
        
        # Tick rule: update on price changes, otherwise carry forward the previous sign.
        diff = prices[i] - prices[i - 1]
        if abs(diff) > 1e-5:
            tick_rule = np.sign(diff)
        
        imbalance = tick_rule * dollar_values[i]
        theta_t += imbalance
        imbalance_arr.append(imbalance)
        
        # Bootstrap the expected imbalance before testing the first bar threshold.
        if expected_imbalance is None:
            if len(imbalance_arr) < expected_num_ticks_init:
                continue
            expected_imbalance = ewma_last(
                imbalance_arr[-expected_num_ticks_init:],
                window=expected_num_ticks_init,
            )
        if np.isclose(expected_imbalance, 0):
            continue
        
        threshold = expected_num_ticks * abs(expected_imbalance)
        if abs(theta_t) >= threshold:
            tick_indices.append(i)
            bar_lengths.append(num_ticks)
            theta_t = 0
            num_ticks = 0
            
            recent_lengths = bar_lengths[-num_prev_bars:]
            expected_num_ticks = ewma_last(recent_lengths, window=len(recent_lengths))
            expected_num_ticks = float(
                np.clip(expected_num_ticks, min_exp_num_ticks, max_exp_num_ticks)
            )
            imbalance_window = int(round(num_prev_bars * expected_num_ticks))
            expected_imbalance = ewma_last(
                imbalance_arr[-imbalance_window:],
                window=imbalance_window,
            )
    
    return df.iloc[tick_indices]