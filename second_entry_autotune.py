import numpy as np
import pandas as pd

from second_entry_processing import calculate_market_trade_records_with_second_entry

TIME_FORMAT = "%d/%m/%Y %H:%M:%S"


def _split_trade_records(trade_records):
    total_records = len(trade_records)
    if total_records == 0:
        return [], []
    if total_records > 2000:
        windowed_records = trade_records[-2000:]
        autotune_records = windowed_records[:1000]
        strike_records = windowed_records[1000:]
    else:
        split_point = total_records // 2
        autotune_records = trade_records[:split_point]
        strike_records = trade_records[split_point:]
    return autotune_records, strike_records


def _calculate_strike_rate_metrics(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_threshold,
    second_entry_mode,
    history_segment="autotune",
    time_format=TIME_FORMAT,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    trade_records = calculate_market_trade_records_with_second_entry(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        time_format,
        second_entry_threshold,
        second_entry_mode,
        precomputed_groups=precomputed_groups,
        precomputed_target_order=precomputed_target_order,
    )

    autotune_records, strike_records = _split_trade_records(trade_records)
    if history_segment == "autotune":
        segment_records = autotune_records
    else:
        segment_records = strike_records

    total_count = len(segment_records)
    trade_records = [record for record in segment_records if record["outcome"] in {"Win", "Lose", "Tie"}]
    trade_count = len(trade_records)
    wins = sum(1 for record in trade_records if record["outcome"] == "Win")
    strike_rate = (wins / trade_count * 100) if trade_count else np.nan
    entry_prices = [record["entry_price"] for record in trade_records if record["entry_price"] is not None]
    if entry_prices:
        avg_entry_price = sum(entry_prices) / len(entry_prices)
        gain = 1 - avg_entry_price
        loss = 1.0
        win_rate_needed = loss / (gain + loss) * 100
    else:
        win_rate_needed = np.nan

    return {
        "strike_rate": strike_rate,
        "win_rate_needed": win_rate_needed,
        "total_count": total_count,
    }


def run_second_entry_autotune(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_threshold_range=np.arange(0.60, 0.801, 0.02),
    modes=("additive", "sole"),
    time_format=TIME_FORMAT,
    progress_callback=None,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    second_entry_values = list(second_entry_threshold_range)
    mode_list = [mode for mode in modes if mode]
    total_steps = len(second_entry_values) * len(mode_list)
    completed_steps = 0
    results = {}

    for mode in mode_list:
        best_edge = None
        best_result = None
        for second_entry_value in second_entry_values:
            completed_steps += 1
            if progress_callback:
                progress_callback(
                    completed_steps,
                    total_steps,
                    (
                        "Evaluating "
                        f"second_entry_threshold={float(second_entry_value):.2f}, "
                        f"mode={mode}"
                    ),
                )
            metrics = _calculate_strike_rate_metrics(
                df,
                time_column,
                minutes_after_open,
                entry_threshold,
                hold_until_close_threshold,
                round(float(second_entry_value), 2),
                mode,
                time_format=time_format,
                precomputed_groups=precomputed_groups,
                precomputed_target_order=precomputed_target_order,
            )
            strike_rate = metrics["strike_rate"]
            win_rate_needed = metrics["win_rate_needed"]
            total_count = metrics["total_count"]
            if (
                total_count in {0, None}
                or pd.isna(total_count)
                or pd.isna(win_rate_needed)
                or pd.isna(strike_rate)
            ):
                continue
            edge = strike_rate - win_rate_needed
            if best_edge is None or edge > best_edge:
                best_edge = edge
                best_result = {
                    "minutes_after_open": minutes_after_open,
                    "entry_threshold": round(float(entry_threshold), 2),
                    "hold_until_close_threshold": round(float(hold_until_close_threshold), 2),
                    "second_entry_threshold": round(float(second_entry_value), 2),
                    "second_entry_mode": mode,
                    "strike_rate": strike_rate,
                    "win_rate_needed": win_rate_needed,
                    "edge": edge,
                    "total_count": total_count,
                }
        results[mode] = best_result

    return results
