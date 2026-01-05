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
    objective="edge",
):
    best_result = None
    best_score = None
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
                metrics = calculate_metrics(
                    df,
                    time_column,
                    minutes_value,
                    round(float(entry_value), 2),
                    round(float(hold_value), 2),
                )
                if isinstance(metrics, dict):
                    strike = metrics.get("strike_rate", metrics.get("strike"))
                    win_rate = metrics.get("win_rate_needed", metrics.get("win_rate"))
                    total_count = metrics.get(
                        "total_count",
                        metrics.get("sample_size", metrics.get("count")),
                    )
                    expectancy = metrics.get("expectancy")
                    expected_pnl = metrics.get("expected_pnl")
                    if (
                        expected_pnl is None
                        and expectancy is not None
                        and total_count not in {0, None}
                        and not pd.isna(total_count)
                    ):
                        expected_pnl = expectancy * total_count
                else:
                    metrics_values = list(metrics)
                    if len(metrics_values) < 3:
                        continue
                    strike = metrics_values[0]
                    win_rate = metrics_values[-2]
                    total_count = metrics_values[-1]
                    expectancy = None
                    expected_pnl = None
                if (
                    total_count in {0, None}
                    or pd.isna(total_count)
                    or pd.isna(win_rate)
                    or pd.isna(strike)
                ):
                    continue
                edge = strike - win_rate
                if objective == "expected_pnl":
                    score = expected_pnl
                else:
                    score = edge
                if score is None or pd.isna(score):
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_result = {
                        "minutes_after_open": minutes_value,
                        "entry_threshold": round(float(entry_value), 2),
                        "hold_until_close_threshold": round(float(hold_value), 2),
                        "strike_rate": strike,
                        "win_rate_needed": win_rate,
                        "edge": edge,
                        "expectancy": expectancy,
                    }

    return best_result
