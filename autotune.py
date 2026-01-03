import numpy as np
import pandas as pd


def run_autotune(
    df,
    time_column,
    calculate_metrics,
    minutes_range=range(5, 14),
    entry_threshold_range=np.arange(0.60, 0.801, 0.02),
    hold_until_close_threshold_range=np.arange(0.60, 0.801, 0.02),
    progress_callback=None,
):
    best_result = None
    best_edge = None
    minutes_values = list(minutes_range)
    entry_values = list(entry_threshold_range)
    hold_values = list(hold_until_close_threshold_range)
    total_steps = len(minutes_values) * len(entry_values) * len(hold_values)
    completed_steps = 0

    for minutes_value in minutes_values:
        for entry_value in entry_values:
            for hold_value in hold_values:
                if hold_value < entry_value:
                    completed_steps += 1
                    continue
                completed_steps += 1
                if progress_callback:
                    progress_callback(
                        completed_steps,
                        total_steps,
                        (
                            "Evaluating "
                            f"minutes_after_open={minutes_value}, "
                            f"entry_threshold={entry_value:.2f}, "
                            f"hold_until_close_threshold={hold_value:.2f}"
                        ),
                    )
                strike, _, _, _, win_rate, total_count = calculate_metrics(
                    df,
                    time_column,
                    minutes_value,
                    round(float(entry_value), 2),
                    round(float(hold_value), 2),
                )
                if total_count == 0 or pd.isna(win_rate) or pd.isna(strike):
                    continue
                edge = strike - win_rate
                if best_edge is None or edge > best_edge:
                    best_edge = edge
                    best_result = {
                        "minutes_after_open": minutes_value,
                        "entry_threshold": round(float(entry_value), 2),
                        "hold_until_close_threshold": round(float(hold_value), 2),
                        "strike_rate": strike,
                        "win_rate_needed": win_rate,
                        "edge": edge,
                    }

    return best_result
