import numpy as np
import pandas as pd


def run_autotune(
    df,
    time_column,
    calculate_metrics,
    minutes_range=range(1, 15),
    threshold_range=np.arange(0.40, 0.801, 0.01),
    progress_callback=None,
):
    best_result = None
    best_edge = None
    total_minutes = len(list(minutes_range))

    for minutes_index, minutes_value in enumerate(minutes_range, start=1):
        if progress_callback:
            progress_callback(minutes_index, total_minutes, f"Evaluating minutes_after_open={minutes_value}")
        for threshold_value in threshold_range:
            strike, avg_entry, win_rate, total_count = calculate_metrics(
                df,
                time_column,
                minutes_value,
                round(float(threshold_value), 2),
            )
            if total_count == 0 or pd.isna(win_rate) or pd.isna(strike):
                continue
            edge = strike - win_rate
            if best_edge is None or edge > best_edge:
                best_edge = edge
                best_result = {
                    "minutes_after_open": minutes_value,
                    "entry_threshold": round(float(threshold_value), 2),
                    "strike_rate": strike,
                    "win_rate_needed": win_rate,
                    "edge": edge,
                }

    return best_result
