import pandas as pd

def cusum_filter(series, threshold):
    """Return integer positions where the signed CUSUM exceeds +/- threshold."""
    s_pos, s_neg = 0, 0
    event_positions = []

    for position, value in enumerate(series):
        if pd.isna(value):
            continue

        s_pos = max(0, s_pos + value)
        s_neg = min(0, s_neg + value)

        if s_pos > threshold:
            event_positions.append(position)
            s_pos = 0
        elif s_neg < -threshold:
            event_positions.append(position)
            s_neg = 0

    return event_positions