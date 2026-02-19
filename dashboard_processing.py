import numpy as np
import pandas as pd

MARKET_WINDOW_MINUTES = 15


def align_market_open(timestamp, market_window_minutes=MARKET_WINDOW_MINUTES):
    if timestamp is None or pd.isna(timestamp):
        return pd.NaT
    return pd.Timestamp(timestamp).floor(f"{int(market_window_minutes)}min")


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


def _resolve_market_winner(market_group, close_up, close_down, tie_tolerance=0.005):
    def _clean_value(value):
        if value is None or pd.isna(value):
            return np.nan
        return float(value)

    def _last_non_zero_value(column_name):
        if column_name not in market_group.columns:
            return np.nan
        return _last_non_zero(market_group[column_name])

    close_up = _clean_value(close_up)
    close_down = _clean_value(close_down)

    # 1) Strong terminal price signal: when one side is effectively resolved.
    if not pd.isna(close_up) and not pd.isna(close_down):
        if close_up >= 0.99 and close_down <= 0.01:
            return "Up", "price_resolved"
        if close_down >= 0.99 and close_up <= 0.01:
            return "Down", "price_resolved"

        if abs(close_up - close_down) > tie_tolerance:
            return ("Up", "price") if close_up > close_down else ("Down", "price")

    # 2) If terminal prices are tied/noisy, use terminal displayed depth as tie-breaker.
    close_up_vol = _clean_value(_last_non_zero_value("UpVol"))
    close_down_vol = _clean_value(_last_non_zero_value("DownVol"))
    if not pd.isna(close_up_vol) or not pd.isna(close_down_vol):
        up_has_depth = not pd.isna(close_up_vol) and close_up_vol > 0
        down_has_depth = not pd.isna(close_down_vol) and close_down_vol > 0
        if up_has_depth and not down_has_depth:
            return "Up", "volume_presence"
        if down_has_depth and not up_has_depth:
            return "Down", "volume_presence"
        if up_has_depth and down_has_depth and close_up_vol != close_down_vol:
            return ("Up", "volume") if close_up_vol > close_down_vol else ("Down", "volume")

    # 3) Fallback to most recent observable price.
    last_up = _clean_value(_last_non_zero_value("UpPrice"))
    last_down = _clean_value(_last_non_zero_value("DownPrice"))
    if not pd.isna(last_up) and not pd.isna(last_down):
        if abs(last_up - last_down) > tie_tolerance:
            return ("Up", "last_price") if last_up > last_down else ("Down", "last_price")

    # 4) Resolve end-of-window feed collapse where both sides print near-zero at expiry.
    # Walk backward to find the latest decisive price snapshot before the collapse.
    collapse_threshold = 0.05
    if {"UpPrice", "DownPrice"}.issubset(market_group.columns):
        recent = market_group[["UpPrice", "DownPrice"]].dropna()
        if not recent.empty:
            for _, row in recent.iloc[::-1].iterrows():
                up_price = _clean_value(row["UpPrice"])
                down_price = _clean_value(row["DownPrice"])
                if pd.isna(up_price) or pd.isna(down_price):
                    continue
                if up_price <= collapse_threshold and down_price <= collapse_threshold:
                    continue
                if abs(up_price - down_price) > tie_tolerance:
                    return ("Up", "pre_collapse_price") if up_price > down_price else ("Down", "pre_collapse_price")
                break

    return None, "indeterminate"


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
    market_window_minutes=MARKET_WINDOW_MINUTES,
):
    if (df is None or df.empty) and not precomputed_groups:
        return []

    if precomputed_groups is None:
        df = df.copy()
        if "TargetTime_dt" not in df.columns:
            df["TargetTime_dt"] = pd.to_datetime(df["TargetTime"], format=time_format, errors="coerce")

    # Preserve sub-minute offsets (e.g., 2.5 minutes on 5m cadence backtests); int() truncation
    # delays eligibility and can shift threshold-crossing detection by a full sample.
    minutes_threshold = pd.Timedelta(minutes=float(minutes_after_open))
    probability_threshold = float(entry_threshold)

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

        market_open = align_market_open(market_group[time_column].min(), market_window_minutes)
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

        market_end_time = market_open + pd.Timedelta(minutes=market_window_minutes)
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
            exit_time = market_close_time
            exit_reason = "held_to_close"
            exit_price = close_up if expected_side == "Up" else close_down
            exit_price_market = exit_price

        winning_side, winning_side_method = _resolve_market_winner(market_group, close_up, close_down)

        outcome = None
        if market_closed:
            if expected_side:
                if exit_reason == "threshold":
                    outcome = "Win"
                else:
                    if winning_side is None:
                        outcome = "N/A"
                    else:
                        outcome = "Win" if expected_side == winning_side else "Lose"
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
                "winning_side": winning_side,
                "winning_side_method": winning_side_method,
                "market_closed": market_closed,
            }
        )

    return records
