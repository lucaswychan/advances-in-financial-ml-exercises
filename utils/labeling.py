"""
For detail usage, please refer to chapter 3 notebooks.
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
from utils.multiprocess import mpPandasObj


def get_daily_vol(close, span0=100):
    """Compute exponentially weighted daily return volatility."""
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]

    prev_pos = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    valid = prev_pos > 0

    current_pos = np.arange(len(close))[valid]
    prev_pos = prev_pos[valid] - 1

    returns = pd.Series(
        close.iloc[current_pos].to_numpy() / close.iloc[prev_pos].to_numpy() - 1,
        index=close.index[current_pos],
        name="daily_vol",
    )
    return returns.ewm(span=span0).std().dropna()


def symmetric_cusum_filter(close, threshold):
    """Apply a symmetric CUSUM filter to log returns."""
    close = close.sort_index()
    log_ret = np.log(close).diff().dropna()

    if isinstance(threshold, pd.DataFrame):
        if threshold.shape[1] != 1:
            raise ValueError("threshold DataFrame must have exactly one column")
        threshold = threshold.iloc[:, 0]

    if isinstance(threshold, pd.Series):
        threshold_values = threshold.sort_index().reindex(log_ret.index).ffill().to_numpy()
    else:
        threshold_values = np.full(log_ret.shape[0], threshold)

    s_pos, s_neg = 0.0, 0.0
    t_events = []
    for t, ret, h in zip(log_ret.index, log_ret.to_numpy(), threshold_values):
        if pd.isna(h) or h <= 0:
            continue

        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if s_pos > h:
            s_pos = 0.0
            t_events.append(t)
        elif s_neg < -h:
            s_neg = 0.0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)


def get_vertical_bars(close, t_events, num_days=1):
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]]) # NaNs at the end
    return t1


# TRIPLE-BARRIER LABELING METHOD
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    """
    For example, from applied_pt_sl.csv:

    2010-05-06 14:46:25,2010-05-07 14:50:01,2010-05-06 14:48:37,2010-05-06 14:46:27
    For the event starting at 2010-05-06 14:46:25:

    vertical barrier: 2010-05-07 14:50:01
    stop loss touched: 2010-05-06 14:48:37
    profit taking touched: 2010-05-06 14:46:27
    
    Since pt happens first, the event should end at 2010-05-06 14:46:27.
    """
    """Apply stop loss on t1."""
    events_ = events.copy().loc[molecule]
    out = events_[['t1']].copy(deep=True)

    # upper limit (profit taking)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['target']
    else:
        pt = pd.Series(index=events.index) # NaNs
    
    # lower limit (stop loss)
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['target']
    else:
        sl = pd.Series(index=events.index) #  NaNs
    
    for loc, t1_val in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1_val] # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] # path linear returns
        out.loc[loc, 'sl'] = df0[df0<sl[loc]].index.min() # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0>pt[loc]].index.min() # earliest profit take

    
    # The output from this function is a pandas dataframe containing 
    # the timestamps (if any) at which each barrier was touched.
    return out


def get_events(close, t_events, pt_sl, target, min_ret, num_threads=1, t1=None, side=None):
    # Aligns target to t_events
    # Only events whose target is larger than min_ret are kept.
    target = target.loc[t_events]
    target = target[target > min_ret]
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=t_events)
    
    if side is None:
        # default long side
        side_, pt_sl_ = pd.Series(1.0, index=target.index), [pt_sl[0], pt_sl[0]]
    else:
        side_, pt_sl_ = side.loc[target.index], pt_sl[:2]
    
    events = pd.concat({'t1': t1, 'target': target, 'side': side_}, axis=1).dropna(subset=['target'])
    barriers = mpPandasObj(func=apply_pt_sl_on_t1, pdObj=('molecule', events.index), numThreads=num_threads, close=close, events=events, pt_sl=pt_sl_)
    
    barriers.to_csv('applied_pt_sl.csv')
    # Replaces events['t1'] with the earliest touched barrier
    # However, we don't know which barrier is touched
    barriers = barriers.dropna(how='all')
    events['t1'] = barriers.min(axis=1)
    events['barrier'] = barriers.idxmin(axis=1)


    if side is None:
        events = events.drop('side', axis=1)
    
    return events


def get_bins(events, close):
    events_ = events.dropna(subset=['t1'])

    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling


    out['bin'] = events_['barrier'].map({
        'pt': 1,
        'sl': -1,
        't1': 0,
    })
    
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
        out['side'] = events['side']

    return out


def drop_labels(events, mitPct=0.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > mitPct or df0.shape[0] < 3:
            break
        print("Dropped label", df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]
    return events