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


def _select_history_segment(trade_records, history_segment):
    autotune_records, strike_records = _split_trade_records(trade_records)
    if history_segment == "autotune":
        return autotune_records
    return strike_records


def _calculate_trade_pnl_usd(record, trade_value_usd):
    position_multiplier = record.get("position_multiplier", 1)
    outcome = record.get("outcome")
    if outcome == "Win":
        return (
            (record["exit_price"] - record["entry_price"])
            * trade_value_usd
            * position_multiplier
        )
    return -trade_value_usd * position_multiplier


def _calculate_strike_rate_metrics(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_threshold,
    second_entry_mode,
    trade_value_usd=1.0,
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
        use_cache=False,
        precomputed_groups=precomputed_groups,
        precomputed_target_order=precomputed_target_order,
    )

    segment_records = _select_history_segment(trade_records, history_segment)

    total_count = len(segment_records)
    trade_records = [record for record in segment_records if record["outcome"] in {"Win", "Lose", "Tie"}]
    trade_count = len(trade_records)
    wins = sum(1 for record in trade_records if record["outcome"] == "Win")
    strike_rate = (wins / trade_count * 100) if trade_count else np.nan
    pnl_values = [
        _calculate_trade_pnl_usd(record, trade_value_usd)
        for record in trade_records
        if record.get("entry_price") is not None
        and record.get("exit_price") is not None
        and not pd.isna(record.get("entry_price"))
        and not pd.isna(record.get("exit_price"))
    ]
    win_pnl_values = [pnl for pnl in pnl_values if pnl > 0]
    loss_pnl_values = [abs(pnl) for pnl in pnl_values if pnl <= 0]
    avg_win = (sum(win_pnl_values) / len(win_pnl_values)) if win_pnl_values else np.nan
    avg_loss = (sum(loss_pnl_values) / len(loss_pnl_values)) if loss_pnl_values else np.nan
    if not pd.isna(avg_win) and not pd.isna(avg_loss) and avg_win > 0 and avg_loss > 0:
        win_rate_needed = avg_loss / (avg_win + avg_loss) * 100
    else:
        win_rate_needed = np.nan

    return {
        "strike_rate": strike_rate,
        "win_rate_needed": win_rate_needed,
        "total_count": total_count,
    }


def _summarize_trade_records(trade_records):
    closed_records = [
        record
        for record in trade_records
        if record["outcome"] in {"Win", "Lose", "Tie"}
        and record.get("entry_price") is not None
        and record.get("exit_price") is not None
        and not pd.isna(record.get("entry_price"))
        and not pd.isna(record.get("exit_price"))
    ]
    trade_count = len(closed_records)
    wins = sum(1 for record in closed_records if record.get("outcome") == "Win")
    win_rate = (wins / trade_count * 100) if trade_count else np.nan
    pnl_values = [
        record["exit_price"] - record["entry_price"]
        for record in closed_records
    ]
    expectancy = (sum(pnl_values) / len(pnl_values)) if pnl_values else np.nan
    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "expectancy": expectancy,
    }


def run_second_entry_autotune(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_threshold_range=np.arange(0.40, 0.701, 0.02),
    modes=("additive", "sole"),
    time_format=TIME_FORMAT,
    summary_segment="strike",
    trade_value_usd=1.0,
    progress_callback=None,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    second_entry_values = list(second_entry_threshold_range)
    normalized_modes = [
        str(mode).strip().lower()
        for mode in modes
        if mode and str(mode).strip().lower() != "off"
    ]
    mode_list = normalized_modes
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
                        f"second_entry_threshold={float(second_entry_value):.3f}, "
                        f"mode={mode}"
                    ),
                )
            metrics = _calculate_strike_rate_metrics(
                df,
                time_column,
                minutes_after_open,
                entry_threshold,
                hold_until_close_threshold,
                round(float(second_entry_value), 3),
                mode,
                trade_value_usd=trade_value_usd,
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
                    "second_entry_threshold": round(float(second_entry_value), 3),
                    "second_entry_mode": mode,
                    "strike_rate": strike_rate,
                    "win_rate_needed": win_rate_needed,
                    "edge": edge,
                    "total_count": total_count,
                }
        if best_result:
            trade_records = calculate_market_trade_records_with_second_entry(
                df,
                time_column,
                minutes_after_open,
                entry_threshold,
                hold_until_close_threshold,
                time_format,
                best_result["second_entry_threshold"],
                mode,
                use_cache=False,
                precomputed_groups=precomputed_groups,
                precomputed_target_order=precomputed_target_order,
            )
            segment_records = _select_history_segment(trade_records, summary_segment)
            summary = _summarize_trade_records(segment_records)
            best_result.update(summary)
        results[mode] = best_result

    return results
