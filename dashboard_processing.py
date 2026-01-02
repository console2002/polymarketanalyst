import numpy as np
import pandas as pd


def align_market_open(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return pd.NaT
    return pd.Timestamp(timestamp).floor("15min")


def _last_non_zero(series):
    cleaned = series.replace(0, np.nan).dropna()
    if cleaned.empty:
        return np.nan
    return cleaned.iloc[-1]


def _get_close_prices(market_group, time_column, close_window_points=6):
    market_group = market_group.sort_values(time_column)
    if market_group.empty:
        return np.nan, np.nan

    window_points = max(close_window_points, int(len(market_group) * 0.1))
    tail_group = market_group.tail(window_points)

    def _median_non_zero(series):
        cleaned = series.replace(0, np.nan).dropna()
        if cleaned.empty:
            return np.nan
        return cleaned.median()

    close_up = _median_non_zero(tail_group["UpPrice"])
    close_down = _median_non_zero(tail_group["DownPrice"])

    if pd.isna(close_up):
        close_up = _last_non_zero(market_group["UpPrice"])
    if pd.isna(close_down):
        close_down = _last_non_zero(market_group["DownPrice"])

    return close_up, close_down


def _find_threshold_crossing(series, threshold):
    above = series >= threshold
    crossings = above & ~above.shift(fill_value=False)
    if crossings.any():
        return crossings[crossings].index[0]
    return None


def calculate_market_trade_records(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    time_format,
    target_order=None,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    if (df is None or df.empty) and not precomputed_groups:
        return []

    if precomputed_groups is None:
        df = df.copy()
        if "TargetTime_dt" not in df.columns:
            df["TargetTime_dt"] = pd.to_datetime(df["TargetTime"], format=time_format, errors="coerce")

    minutes_threshold = pd.Timedelta(minutes=int(minutes_after_open))
    probability_threshold = float(entry_threshold)
    hold_threshold = float(hold_until_close_threshold)

    if precomputed_groups is None and target_order is None:
        target_order = df["TargetTime_dt"].dropna().drop_duplicates().tolist()

    if precomputed_groups is not None:
        target_order = precomputed_target_order or target_order or list(precomputed_groups.keys())

    target_indices = {target: idx for idx, target in enumerate(target_order)}
    last_index = len(target_order) - 1
    records = []

    for target_time in target_order:
        if precomputed_groups is None:
            market_group = df[df["TargetTime_dt"] == target_time].sort_values(time_column)
        else:
            market_group = precomputed_groups.get(target_time)
            if market_group is None or market_group.empty:
                continue
            if not market_group[time_column].is_monotonic_increasing:
                market_group = market_group.sort_values(time_column)
        if market_group.empty:
            continue

        market_open = align_market_open(market_group[time_column].min())
        open_threshold_time = market_open + minutes_threshold
        eligible = market_group[market_group[time_column] >= open_threshold_time].copy()

        expected_side = None
        entry_time = None
        entry_price = None
        if not eligible.empty:
            up_cross_index = _find_threshold_crossing(eligible["UpPrice"], probability_threshold)
            down_cross_index = _find_threshold_crossing(eligible["DownPrice"], probability_threshold)
            candidates = []
            if up_cross_index is not None:
                candidates.append(
                    ("Up", eligible.loc[up_cross_index, time_column], eligible.loc[up_cross_index, "UpPrice"])
                )
            if down_cross_index is not None:
                candidates.append(
                    ("Down", eligible.loc[down_cross_index, time_column], eligible.loc[down_cross_index, "DownPrice"])
                )
            if candidates:
                expected_side, entry_time, entry_price = min(candidates, key=lambda item: item[1])

        market_end_time = market_open + pd.Timedelta(minutes=15)
        market_close_time = market_group[time_column].iloc[-1]
        target_index = target_indices.get(target_time)
        market_closed = (
            (target_index is not None and target_index < last_index)
            or market_close_time >= market_end_time
        )

        close_up, close_down = _get_close_prices(market_group, time_column)
        exit_time = None
        exit_price = None
        exit_price_market = None
        exit_reason = None
        if expected_side and entry_price is not None and not pd.isna(entry_price):
            if entry_price >= hold_threshold:
                exit_time = market_close_time
                exit_reason = "held_to_close"
            else:
                side_column = "UpPrice" if expected_side == "Up" else "DownPrice"
                eligible_after_entry = eligible[eligible[time_column] >= entry_time]
                exit_cross_index = _find_threshold_crossing(eligible_after_entry[side_column], hold_threshold)
                if exit_cross_index is not None:
                    exit_time = eligible_after_entry.loc[exit_cross_index, time_column]
                    exit_price = eligible_after_entry.loc[exit_cross_index, side_column]
                    exit_price_market = exit_price
                    exit_reason = "threshold"
                else:
                    exit_time = market_close_time
                    exit_reason = "held_to_close"

            if exit_time == market_close_time:
                exit_price = close_up if expected_side == "Up" else close_down
                exit_price_market = exit_price

        outcome = None
        if market_closed:
            if expected_side:
                if exit_reason == "threshold":
                    outcome = "Win"
                else:
                    if pd.isna(close_up) or pd.isna(close_down):
                        outcome = "N/A"
                    elif close_up == close_down:
                        outcome = "Tie"
                    elif expected_side == "Up":
                        outcome = "Win" if close_up > close_down else "Lose"
                    else:
                        outcome = "Win" if close_down > close_up else "Lose"
        else:
            outcome = "Pending"

        if market_closed and exit_reason == "held_to_close" and outcome in {"Win", "Lose"}:
            exit_price = 1.0 if outcome == "Win" else 0.0

        records.append(
            {
                "target_time_dt": target_time,
                "target_time": market_group["TargetTime"].iloc[0] if "TargetTime" in market_group.columns else None,
                "market_open": market_open,
                "open_threshold_time": open_threshold_time,
                "market_close_time": market_close_time,
                "expected_side": expected_side,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_price_market": exit_price_market,
                "exit_reason": exit_reason,
                "outcome": outcome,
                "close_up": close_up,
                "close_down": close_down,
                "market_closed": market_closed,
            }
        )

    return records
