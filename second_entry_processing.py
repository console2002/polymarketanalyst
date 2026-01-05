import hashlib
import json
import os

import numpy as np
import pandas as pd

from dashboard_processing import (
    _find_threshold_crossing,
    _get_close_prices,
    align_market_open,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "second_entry")
CACHE_SCHEMA_VERSION = 3
HOLD_EXIT_THRESHOLD = 0.99


def _find_pullback_crossing(series, threshold):
    below = series <= threshold
    crossings = below & ~below.shift(fill_value=False)
    if crossings.any():
        return crossings[crossings].index[0]
    return None


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _hash_dataframe_signature(df, time_column):
    if df is None or df.empty:
        return {"shape": (0, 0), "hash": 0}
    columns = [col for col in [time_column, "UpPrice", "DownPrice", "TargetTime", "TargetTime_dt"] if col in df.columns]
    if not columns:
        return {"shape": df.shape, "hash": 0}
    hashed = pd.util.hash_pandas_object(df[columns], index=True)
    return {"shape": df.shape, "hash": int(hashed.sum())}


def _get_data_signature(df, time_column, precomputed_groups, precomputed_target_order):
    if df is not None:
        signature = getattr(df, "attrs", {}).get("data_signature")
        if signature is not None:
            return signature
        return _hash_dataframe_signature(df, time_column)
    if precomputed_groups is not None:
        target_order = precomputed_target_order or list(precomputed_groups.keys())
        target_signature = [str(target) for target in target_order]
        return {"precomputed_targets": target_signature, "group_count": len(precomputed_groups)}
    return "empty"


def _build_second_entry_cache_key(
    data_signature,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_threshold,
    second_entry_mode,
):
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "data_signature": data_signature,
        "entry_threshold": float(entry_threshold),
        "hold_until_close_threshold": float(hold_until_close_threshold),
        "second_entry_threshold": float(second_entry_threshold),
        "second_entry_mode": str(second_entry_mode),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_second_entry_cache_path(cache_key):
    return os.path.join(CACHE_DIR, f"{cache_key}.csv")


def _load_second_entry_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    try:
        cached_df = pd.read_csv(cache_path)
    except Exception:
        return None
    datetime_columns = [
        "target_time_dt",
        "market_open",
        "open_threshold_time",
        "market_close_time",
        "trigger_time",
        "second_entry_time",
        "entry_time",
        "exit_time",
    ]
    for column in datetime_columns:
        if column in cached_df.columns:
            cached_df[column] = pd.to_datetime(cached_df[column], errors="coerce")
    return cached_df.to_dict("records")


def _write_second_entry_cache(cache_path, trade_records):
    _ensure_cache_dir()
    cached_df = pd.DataFrame(trade_records)
    cached_df.to_csv(cache_path, index=False)


def calculate_market_trade_records_with_second_entry(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    time_format,
    second_entry_threshold,
    second_entry_mode,
    use_cache=True,
    target_order=None,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    if (df is None or df.empty) and not precomputed_groups:
        return []

    cache_path = None
    if use_cache:
        data_signature = _get_data_signature(df, time_column, precomputed_groups, precomputed_target_order)
        cache_key = _build_second_entry_cache_key(
            data_signature,
            entry_threshold,
            hold_until_close_threshold,
            second_entry_threshold,
            second_entry_mode,
        )
        cache_path = _get_second_entry_cache_path(cache_key)
        cached_records = _load_second_entry_cache(cache_path)
        if cached_records is not None:
            return cached_records

    if precomputed_groups is None:
        df = df.copy()
        if "TargetTime_dt" not in df.columns:
            df["TargetTime_dt"] = pd.to_datetime(df["TargetTime"], format=time_format, errors="coerce")

    minutes_threshold = pd.Timedelta(minutes=int(minutes_after_open))
    probability_threshold = float(entry_threshold)
    hold_threshold = float(hold_until_close_threshold)
    second_threshold = float(second_entry_threshold)
    entry_mode = (second_entry_mode or "off").lower()

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
        trigger_time = None
        trigger_price = None
        trigger_threshold = probability_threshold
        if not eligible.empty:
            up_cross_index = _find_threshold_crossing(eligible["UpPrice"], trigger_threshold)
            down_cross_index = _find_threshold_crossing(eligible["DownPrice"], trigger_threshold)
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
                expected_side, trigger_time, trigger_price = min(candidates, key=lambda item: item[1])

        market_end_time = market_open + pd.Timedelta(minutes=15)
        market_close_time = market_group[time_column].iloc[-1]
        target_index = target_indices.get(target_time)
        market_closed = (
            (target_index is not None and target_index < last_index)
            or market_close_time >= market_end_time
        )

        entry_time = None
        entry_price = None
        second_entry_time = None
        second_entry_price = None
        second_entry_taken = False
        trade_executed = False
        position_multiplier = 1
        if expected_side and trigger_time is not None:
            side_column = "UpPrice" if expected_side == "Up" else "DownPrice"
            if entry_mode == "off":
                entry_time = trigger_time
                entry_price = trigger_price
                trade_executed = True
            else:
                eligible_after_trigger = eligible[eligible[time_column] >= trigger_time]
                if not second_entry_taken:
                    pullback_index = _find_pullback_crossing(eligible_after_trigger[side_column], second_threshold)
                    if pullback_index is not None:
                        second_entry_time = eligible_after_trigger.loc[pullback_index, time_column]
                        second_entry_price = eligible_after_trigger.loc[pullback_index, side_column]
                        if second_entry_price is not None and not pd.isna(second_entry_price):
                            second_entry_taken = True

                if entry_mode == "sole":
                    if second_entry_taken:
                        entry_time = second_entry_time
                        entry_price = second_entry_price
                    trade_executed = entry_time is not None and entry_price is not None
                elif entry_mode == "additive":
                    entry_time = trigger_time
                    entry_price = trigger_price
                    trade_executed = entry_time is not None and entry_price is not None

        close_up, close_down = _get_close_prices(market_group, time_column)
        exit_time = None
        exit_price = None
        exit_price_market = None
        exit_reason = None
        entry_valid = (
            trade_executed
            and entry_time is not None
            and entry_price is not None
            and not pd.isna(entry_price)
        )
        if entry_valid and entry_price > 0:
            side_column = "UpPrice" if expected_side == "Up" else "DownPrice"
            eligible_after_entry = eligible[eligible[time_column] >= entry_time]
            if entry_price >= hold_threshold:
                exit_cross_index = _find_threshold_crossing(eligible_after_entry[side_column], HOLD_EXIT_THRESHOLD)
                if exit_cross_index is not None:
                    exit_time = eligible_after_entry.loc[exit_cross_index, time_column]
                    exit_price = eligible_after_entry.loc[exit_cross_index, side_column]
                    exit_price_market = exit_price
                    exit_reason = "threshold"
                else:
                    exit_time = market_close_time
                    exit_reason = "held_to_close"
            else:
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

        if entry_mode == "additive" and second_entry_taken:
            if second_entry_time is None or exit_time is None or second_entry_time > exit_time:
                second_entry_taken = False
                second_entry_time = None
                second_entry_price = None
            elif trigger_price is not None and second_entry_price is not None:
                entry_price = np.mean([trigger_price, second_entry_price])
                position_multiplier = 2

        outcome = None
        if market_closed:
            if trade_executed and expected_side:
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
            if trade_executed:
                outcome = "Pending"

        if market_closed and exit_reason == "held_to_close" and outcome in {"Win", "Lose"}:
            exit_price = 1.0 if outcome == "Win" else 0.0

        if entry_valid and entry_price > 0:
            records.append(
                {
                    "target_time_dt": target_time,
                    "target_time": market_group["TargetTime"].iloc[0] if "TargetTime" in market_group.columns else None,
                    "market_open": market_open,
                    "open_threshold_time": open_threshold_time,
                    "market_close_time": market_close_time,
                    "expected_side": expected_side,
                    "trigger_time": trigger_time,
                    "trigger_price": trigger_price,
                    "second_entry_time": second_entry_time,
                    "second_entry_price": second_entry_price,
                    "entry_mode": entry_mode,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "position_multiplier": position_multiplier,
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

    if cache_path:
        _write_second_entry_cache(cache_path, records)
    return records


def calculate_market_trade_records(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    time_format,
    second_entry_threshold,
    second_entry_mode,
    use_cache=True,
    target_order=None,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    return calculate_market_trade_records_with_second_entry(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        time_format,
        second_entry_threshold,
        second_entry_mode,
        use_cache=use_cache,
        target_order=target_order,
        precomputed_groups=precomputed_groups,
        precomputed_target_order=precomputed_target_order,
    )
