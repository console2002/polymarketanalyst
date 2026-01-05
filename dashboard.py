import hashlib
import json
import numpy as np
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re
from autotune import run_autotune
from second_entry_autotune import run_second_entry_autotune
from dashboard_metrics import (
    build_trade_pnl_records,
    summarize_drawdowns,
    summarize_profit_loss,
)
from dashboard_processing import align_market_open, calculate_market_trade_records
from second_entry_processing import calculate_market_trade_records_with_second_entry


# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATE_FORMAT = "%d%m%Y"
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache", "second_entry")
CACHE_SCHEMA_VERSION = 2


def add_vline_all_rows(fig, x, **kwargs):
    grid_ref = getattr(fig, "_grid_ref", None)
    row_count = len(grid_ref) if grid_ref else 1
    for row in range(1, row_count + 1):
        fig.add_vline(x=x, row=row, col=1, **kwargs)


def _parse_date_from_filename(filename):
    name = os.path.splitext(filename)[0]
    patterns = [
        (r"(?P<date>\d{2}\d{2}\d{4})", "%d%m%Y"),
        (r"(?P<date>\d{4}\d{2}\d{2})", "%Y%m%d"),
        (r"(?P<date>\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
        (r"(?P<date>\d{2}-\d{2}-\d{4})", "%d-%m-%Y"),
    ]
    for pattern, date_format in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        try:
            return datetime.datetime.strptime(match.group("date"), date_format).date()
        except ValueError:
            continue
    return None


def _get_available_data_files():
    files_by_date = {}
    for filename in os.listdir(SCRIPT_DIR):
        if not filename.endswith(".csv"):
            continue
        file_date = _parse_date_from_filename(filename)
        if file_date:
            files_by_date[file_date] = os.path.join(SCRIPT_DIR, filename)
    legacy_path = os.path.join(SCRIPT_DIR, "market_data.csv")
    if not os.path.exists(legacy_path):
        legacy_path = None
    return files_by_date, legacy_path


def _resolve_data_file(selected_date, files_by_date, legacy_path):
    if selected_date and selected_date in files_by_date:
        return files_by_date[selected_date], selected_date
    if files_by_date:
        latest_date = max(files_by_date)
        return files_by_date[latest_date], latest_date
    return legacy_path, None


st.set_page_config(page_title="Polymarket 8020 Monitor", layout="wide")

st.sidebar.header("Analysis Controls")
lookback_period = st.sidebar.number_input(
    "Lookback period (markets)",
    min_value=1,
    max_value=20,
    value=1,
    step=1,
    help="Number of markets to display in the window, including the current one.",
)
minutes_after_open = st.sidebar.number_input(
    "Minutes after open",
    min_value=1,
    max_value=60,
    value=5,
    step=1,
)
entry_threshold = st.sidebar.number_input(
    "Entry threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.01,
    format="%.2f",
)
hold_until_close_threshold = st.sidebar.number_input(
    "Hold Until Close Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.01,
    format="%.2f",
)
second_entry_mode = st.sidebar.selectbox(
    "Second entry mode",
    options=("Off", "Additive", "Sole"),
    index=0,
    key="second_entry_mode",
    help="Enable pullback-based second entry processing for new trades.",
)
second_entry_threshold = st.sidebar.number_input(
    "Second entry threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.01,
    format="%.2f",
    key="second_entry_threshold",
    help="Threshold used when second entry mode is Additive or Sole.",
    disabled=second_entry_mode == "Off",
)
resample_interval = st.sidebar.selectbox(
    "Resample interval",
    options=("1s", "5s", "15s", "30s", "60s", "all"),
    index=("1s", "5s", "15s", "30s", "60s", "all").index("5s"),
)
show_markers = st.sidebar.checkbox("Show markers", value=True)
refresh_interval_seconds = st.sidebar.number_input(
    "Auto-refresh interval (seconds)",
    min_value=1,
    max_value=60,
    value=60,
    step=1,
    help="Controls the sleep duration for the auto-refresh loop.",
)
trade_value_usd = st.sidebar.number_input(
    "Trade value (USD)",
    min_value=0.0,
    value=5.0,
    step=0.5,
    format="%.2f",
    help="USD value applied to each trade when calculating profit/loss.",
)
test_balance_start = st.sidebar.number_input(
    "Test balance start",
    min_value=0.0,
    value=1000.0,
    step=100.0,
    format="%.2f",
    help="Starting balance used for equity curve and drawdown calculations.",
)
time_axis = st.sidebar.selectbox(
    "Chart time axis",
    options=("Polymarket Time (ET)", "UK Time"),
    index=0,
    help="Switch the chart between Polymarket (ET) and UK timestamps.",
)

def _normalize_outcome(value, fallback_map):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if "yes" in normalized or "up" in normalized:
        return "Up"
    if "no" in normalized or "down" in normalized:
        return "Down"
    return fallback_map.get(value)


def _reshape_new_style_csv(df):
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["timestamp_et"], format=TIME_FORMAT, errors="coerce")
    df["Timestamp_UK"] = pd.to_datetime(df["timestamp_uk"], format=TIME_FORMAT, errors="coerce")
    df["TargetTime"] = df["target_time_uk"]
    unique_outcomes = [value for value in df["outcome"].dropna().unique()]
    fallback_map = {}
    if len(unique_outcomes) >= 2:
        fallback_map = {
            unique_outcomes[0]: "Up",
            unique_outcomes[1]: "Down",
        }
    df["side"] = df["outcome"].apply(lambda value: _normalize_outcome(value, fallback_map))
    df = df[df["side"].isin(["Up", "Down"])]
    df["best_ask"] = pd.to_numeric(df["best_ask"], errors="coerce")
    df["best_ask_size"] = pd.to_numeric(df["best_ask_size"], errors="coerce")
    base_cols = ["Timestamp", "Timestamp_UK", "TargetTime"]
    price_table = df.pivot_table(index=base_cols, columns="side", values="best_ask", aggfunc="last")
    volume_table = df.pivot_table(index=base_cols, columns="side", values="best_ask_size", aggfunc="last")
    wide = pd.DataFrame(index=price_table.index)
    wide["UpPrice"] = price_table.get("Up")
    wide["DownPrice"] = price_table.get("Down")
    wide["UpVol"] = volume_table.get("Up")
    wide["DownVol"] = volume_table.get("Down")
    wide = wide.reset_index()
    return wide


def _get_file_signature(data_file):
    try:
        stat_result = os.stat(data_file)
    except FileNotFoundError:
        return None
    return data_file, stat_result.st_mtime, stat_result.st_size


@st.cache_data(show_spinner=False)
def _load_data_file_cached(data_file, modified_time, file_size):
    df = pd.read_csv(data_file)
    if "timestamp_et" in df.columns:
        df = _reshape_new_style_csv(df)
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=TIME_FORMAT, errors="coerce")
        if "Timestamp_UK" in df.columns:
            df["Timestamp_UK"] = pd.to_datetime(df["Timestamp_UK"], format=TIME_FORMAT, errors="coerce")
    for column in ("UpPrice", "DownPrice"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("UpVol", "DownVol"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def _load_data_file(data_file):
    signature = _get_file_signature(data_file)
    if signature is None:
        raise FileNotFoundError(data_file)
    df = _load_data_file_cached(*signature)
    df.attrs["data_signature"] = signature
    return df


def load_data(selected_date, files_by_date, legacy_path):
    try:
        data_file, resolved_date = _resolve_data_file(selected_date, files_by_date, legacy_path)
        if not data_file:
            return None, None
        df = _load_data_file(data_file)
        return df, resolved_date
    except FileNotFoundError:
        return None, None
    except Exception as e:  # Catch other potential errors during loading/parsing
        st.error(f"Error loading data: {e}")
        return None, None


@st.cache_data(show_spinner=False)
def _load_all_data_cached(file_signatures):
    data_frames = []
    for data_file, modified_time, file_size in file_signatures:
        data_frames.append(_load_data_file_cached(data_file, modified_time, file_size))
    if not data_frames:
        return None
    return pd.concat(data_frames, ignore_index=True)


def load_all_data(files_by_date, legacy_path):
    file_signatures = []
    for _, data_file in sorted(files_by_date.items()):
        signature = _get_file_signature(data_file)
        if signature is not None:
            file_signatures.append(signature)
    if legacy_path:
        signature = _get_file_signature(legacy_path)
        if signature is not None:
            file_signatures.append(signature)
    if not file_signatures:
        return None
    df = _load_all_data_cached(tuple(file_signatures))
    if df is not None:
        df.attrs["data_signature"] = tuple(file_signatures)
    return df


def _ensure_second_entry_cache_dir():
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
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_threshold,
    second_entry_mode,
):
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "data_signature": data_signature,
        "minutes_after_open": int(minutes_after_open),
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
    _ensure_second_entry_cache_dir()
    cached_df = pd.DataFrame(trade_records)
    cached_df.to_csv(cache_path, index=False)
    _prune_second_entry_cache_dir(keep_paths={cache_path})


def _prune_second_entry_cache_dir(keep_paths=None, max_entries=5):
    if not os.path.isdir(CACHE_DIR):
        return
    keep_paths = {os.path.abspath(path) for path in (keep_paths or set())}
    cache_files = []
    for filename in os.listdir(CACHE_DIR):
        if not filename.endswith(".csv"):
            continue
        cache_path = os.path.join(CACHE_DIR, filename)
        if os.path.abspath(cache_path) in keep_paths:
            continue
        try:
            modified_time = os.path.getmtime(cache_path)
        except OSError:
            continue
        cache_files.append((modified_time, cache_path))
    allowed_other = max(0, max_entries - len(keep_paths))
    if len(cache_files) <= allowed_other:
        return
    cache_files.sort()
    for _, cache_path in cache_files[: len(cache_files) - allowed_other]:
        try:
            os.remove(cache_path)
        except OSError:
            continue

# Top-row controls
files_by_date, legacy_path = _get_available_data_files()
available_dates = sorted(files_by_date)
latest_available_date = max(available_dates) if available_dates else None
min_available_date = min(available_dates) if available_dates else None


def _resample_market_data(df, time_column, interval):
    if df.empty or not interval or interval == "all":
        return df
    resampled_groups = []
    for target_time, group in df.groupby("TargetTime", sort=False):
        if group.empty:
            continue
        group = group.sort_values(time_column)
        resampled = group.set_index(time_column).resample(interval).agg(
            {
                "UpPrice": "last",
                "DownPrice": "last",
                "UpVol": "sum",
                "DownVol": "sum",
            }
        )
        resampled["TargetTime"] = target_time
        if "TargetTime_dt" in group.columns:
            resampled["TargetTime_dt"] = group["TargetTime_dt"].iloc[0]
        resampled = resampled.reset_index()
        resampled_groups.append(resampled)
    if not resampled_groups:
        return df
    return pd.concat(resampled_groups, ignore_index=True)

def _format_metric(value, formatter):
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return formatter(value)
    except (TypeError, ValueError):
        return "N/A"


def _build_second_entry_autotune_panel(results):
    if not results:
        return []
    rows = []
    for mode_label in ("additive", "sole"):
        result = results.get(mode_label)
        if not result:
            continue
        rows.append(
            {
                "Mode": mode_label.title(),
                "Best 2nd Threshold": _format_metric(
                    result.get("second_entry_threshold"),
                    lambda v: f"{v:.3f}",
                ),
                "Strike Rate": _format_metric(
                    result.get("strike_rate"),
                    lambda v: f"{v:.2f}%",
                ),
                "Win Rate Needed": _format_metric(
                    result.get("win_rate_needed"),
                    lambda v: f"{v:.2f}%",
                ),
                "Edge": _format_metric(
                    result.get("edge"),
                    lambda v: f"{v:+.2f}%",
                ),
                "Trades": _format_metric(
                    result.get("trade_count"),
                    lambda v: f"{int(v)}",
                ),
                "Expectancy": _format_metric(
                    result.get("expectancy"),
                    lambda v: f"{v:.4f}",
                ),
            }
        )
    return rows

def _normalize_second_entry_mode(mode):
    if not mode:
        return "off"
    return str(mode).strip().lower()


def _calculate_trade_records(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    target_order=None,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    normalized_mode = _normalize_second_entry_mode(second_entry_mode)
    data_signature = _get_data_signature(df, time_column, precomputed_groups, precomputed_target_order)
    cache_key = _build_second_entry_cache_key(
        data_signature,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_threshold,
        normalized_mode,
    )
    cache_path = _get_second_entry_cache_path(cache_key)
    cached_records = _load_second_entry_cache(cache_path)
    if cached_records is not None:
        return cached_records

    if normalized_mode == "off":
        trade_records = calculate_market_trade_records(
            df,
            time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            TIME_FORMAT,
            target_order=target_order,
            precomputed_groups=precomputed_groups,
            precomputed_target_order=precomputed_target_order,
        )
    else:
        trade_records = calculate_market_trade_records_with_second_entry(
            df,
            time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            TIME_FORMAT,
            second_entry_threshold,
            normalized_mode,
            target_order=target_order,
            precomputed_groups=precomputed_groups,
            precomputed_target_order=precomputed_target_order,
        )

    _write_second_entry_cache(cache_path, trade_records)
    return trade_records


def _get_cached_trade_records(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    allow_compute,
    target_order=None,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    normalized_mode = _normalize_second_entry_mode(second_entry_mode)
    data_signature = _get_data_signature(df, time_column, precomputed_groups, precomputed_target_order)
    cache_key = _build_second_entry_cache_key(
        data_signature,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_threshold,
        normalized_mode,
    )
    cache_path = _get_second_entry_cache_path(cache_key)
    cached_records = _load_second_entry_cache(cache_path)
    if cached_records is not None or not allow_compute:
        return cached_records
    return _calculate_trade_records(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        normalized_mode,
        second_entry_threshold,
        target_order=target_order,
        precomputed_groups=precomputed_groups,
        precomputed_target_order=precomputed_target_order,
    )


def _summarize_trade_record_metrics(trade_records, trade_value_usd):
    if not trade_records:
        return {
            "trade_count": np.nan,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "edge": np.nan,
        }
    closed_records = [
        record
        for record in trade_records
        if record.get("outcome") in {"Win", "Lose", "Tie"}
        and record.get("entry_price") is not None
        and record.get("exit_price") is not None
        and not pd.isna(record.get("entry_price"))
        and not pd.isna(record.get("exit_price"))
    ]
    trade_count = len(closed_records)
    wins = sum(1 for record in closed_records if record.get("outcome") == "Win")
    win_rate = (wins / trade_count * 100) if trade_count else np.nan
    entry_prices = [record["entry_price"] for record in closed_records]
    if entry_prices:
        avg_entry_price = sum(entry_prices) / len(entry_prices)
        gain = 1 - avg_entry_price
        loss = 1.0
        win_rate_needed = loss / (gain + loss) * 100
    else:
        win_rate_needed = np.nan
    edge = win_rate - win_rate_needed if not pd.isna(win_rate) and not pd.isna(win_rate_needed) else np.nan
    pnl_values = [
        (record["exit_price"] - record["entry_price"])
        * trade_value_usd
        * record.get("position_multiplier", 1)
        for record in closed_records
    ]
    expectancy = (sum(pnl_values) / len(pnl_values)) if pnl_values else np.nan
    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "edge": edge,
    }



def _split_trade_records(trade_records):
    total_records = len(trade_records)
    if total_records >= 2000:
        windowed_records = trade_records[-2000:]
        autotune_records = windowed_records[:1000]
        strike_records = windowed_records[1000:]
    else:
        split_point = total_records // 2
        autotune_records = trade_records[:split_point]
        strike_records = trade_records[split_point:]
    return autotune_records, strike_records


def _build_market_groups(df, time_column):
    if df is None or df.empty:
        return {}, []
    if "TargetTime_dt" not in df.columns:
        df = df.copy()
        df["TargetTime_dt"] = pd.to_datetime(df["TargetTime"], format=TIME_FORMAT, errors="coerce")
    target_order = df["TargetTime_dt"].dropna().drop_duplicates().tolist()
    groups = {}
    for target_time, group in df.groupby("TargetTime_dt", sort=False):
        if group.empty:
            continue
        groups[target_time] = group.sort_values(time_column)
    return groups, target_order


def _calculate_strike_rate_metrics(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    history_segment="strike",
    precomputed_groups=None,
    precomputed_target_order=None,
):
    trade_records = _calculate_trade_records(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
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
        min_entry_price = min(entry_prices)
        max_entry_price = max(entry_prices)
        gain = 1 - avg_entry_price
        loss = 1.0
        win_rate_needed = loss / (gain + loss) * 100
    else:
        avg_entry_price = np.nan
        min_entry_price = np.nan
        max_entry_price = np.nan
        win_rate_needed = np.nan
    return strike_rate, avg_entry_price, min_entry_price, max_entry_price, win_rate_needed, total_count


def _calculate_window_summary(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    trade_value_usd,
):
    summary_rows = []
    loss_targets = []
    trade_records = _calculate_trade_records(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
    )

    for record in trade_records:
        if record["expected_side"] is None or record["entry_price"] is None:
            continue
        if record["exit_price"] is None or record["exit_reason"] is None:
            continue
        if not record["market_closed"] or record["outcome"] == "Pending":
            continue

        target_time = record["target_time_dt"]
        market_group = df[df["TargetTime_dt"] == target_time].sort_values(time_column)
        if market_group.empty:
            continue

        pnl_usd = None
        if (
            record["entry_price"] is not None
            and record["exit_price"] is not None
            and not pd.isna(record["entry_price"])
            and not pd.isna(record["exit_price"])
        ):
            pnl_usd = (
                (record["exit_price"] - record["entry_price"])
                * trade_value_usd
                * record.get("position_multiplier", 1)
            )
        exit_price_display = record.get("exit_price_market", record["exit_price"])

        summary_rows.append(
            {
                "TargetTime": market_group["TargetTime"].iloc[0],
                "Market Open": record["market_open"],
                "First Crossing Side": record["expected_side"] or "None",
                "Crossing Time": record["entry_time"],
                "Entry Price": record["entry_price"],
                "Exit Time": record["exit_time"],
                "Exit Price": exit_price_display,
                "Exit Reason": record["exit_reason"],
                "P/L (USD)": pnl_usd,
                "Outcome": record["outcome"],
                "Final UpPrice": record["close_up"],
                "Final DownPrice": record["close_down"],
            }
        )
        if pnl_usd is not None and not pd.isna(pnl_usd) and pnl_usd < 0:
            loss_targets.append(target_time)

    latest_loss_target = max(loss_targets) if loss_targets else None
    return summary_rows, latest_loss_target


def _find_latest_loss_target(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    trade_value_usd,
):
    _, latest_loss_target = _calculate_window_summary(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
        trade_value_usd,
    )
    return latest_loss_target


def _initialize_strike_rate_state(
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
):
    if "last_market_open" not in st.session_state:
        st.session_state.last_market_open = pd.NaT
    if "strike_rate" not in st.session_state:
        st.session_state.strike_rate = np.nan
    if "strike_rate_initialized" not in st.session_state:
        st.session_state.strike_rate_initialized = False
    if "last_minutes_after_open" not in st.session_state:
        st.session_state.last_minutes_after_open = minutes_after_open
    if "last_entry_threshold" not in st.session_state:
        st.session_state.last_entry_threshold = entry_threshold
    if "last_hold_until_close_threshold" not in st.session_state:
        st.session_state.last_hold_until_close_threshold = hold_until_close_threshold
    if "last_second_entry_mode" not in st.session_state:
        st.session_state.last_second_entry_mode = _normalize_second_entry_mode(second_entry_mode)
    if "last_second_entry_threshold" not in st.session_state:
        st.session_state.last_second_entry_threshold = second_entry_threshold
    if "autotune_result" not in st.session_state:
        st.session_state.autotune_result = None
    if "autotune_message" not in st.session_state:
        st.session_state.autotune_message = None
    if "second_entry_autotune_result" not in st.session_state:
        st.session_state.second_entry_autotune_result = None
    if "second_entry_autotune_message" not in st.session_state:
        st.session_state.second_entry_autotune_message = None
    if "second_entry_autotune_panel" not in st.session_state:
        st.session_state.second_entry_autotune_panel = None
    if "strike_sample_size" not in st.session_state:
        st.session_state.strike_sample_size = None
    if "autotune_sample_size" not in st.session_state:
        st.session_state.autotune_sample_size = None


def _should_recalculate_strike_rate(
    current_open,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
):
    should_recalculate = not st.session_state.strike_rate_initialized
    last_market_open = st.session_state.last_market_open
    if pd.isna(current_open):
        return False
    if pd.isna(last_market_open):
        should_recalculate = True
    else:
        should_recalculate = current_open > last_market_open

    minutes_after_open_changed = (
        minutes_after_open != st.session_state.last_minutes_after_open
    )
    entry_threshold_changed = (
        entry_threshold != st.session_state.last_entry_threshold
    )
    hold_until_close_threshold_changed = (
        hold_until_close_threshold != st.session_state.last_hold_until_close_threshold
    )
    second_entry_mode_changed = (
        _normalize_second_entry_mode(second_entry_mode)
        != st.session_state.last_second_entry_mode
    )
    second_entry_threshold_changed = (
        second_entry_threshold != st.session_state.last_second_entry_threshold
    )
    if (
        minutes_after_open_changed
        or entry_threshold_changed
        or hold_until_close_threshold_changed
        or second_entry_mode_changed
        or second_entry_threshold_changed
    ):
        should_recalculate = True
    return should_recalculate


def _update_strike_rate_state(
    history_df,
    history_time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    current_open,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    _initialize_strike_rate_state(
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
    )
    should_recalculate = _should_recalculate_strike_rate(
        current_open,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
    )
    if should_recalculate:
        (
            strike_rate,
            avg_entry_price,
            min_entry_price,
            max_entry_price,
            win_rate_needed,
            strike_sample_size,
        ) = _calculate_strike_rate_metrics(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            second_entry_mode,
            second_entry_threshold,
            history_segment="strike",
            precomputed_groups=precomputed_groups,
            precomputed_target_order=precomputed_target_order,
        )
        _, _, _, _, _, autotune_sample_size = _calculate_strike_rate_metrics(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            second_entry_mode,
            second_entry_threshold,
            history_segment="autotune",
            precomputed_groups=precomputed_groups,
            precomputed_target_order=precomputed_target_order,
        )
        st.session_state.strike_rate = strike_rate
        st.session_state.avg_entry_price = avg_entry_price
        st.session_state.min_entry_price = min_entry_price
        st.session_state.max_entry_price = max_entry_price
        st.session_state.win_rate_needed = win_rate_needed
        st.session_state.strike_sample_size = strike_sample_size
        st.session_state.autotune_sample_size = autotune_sample_size
        st.session_state.last_market_open = current_open
        st.session_state.strike_rate_initialized = True
        st.session_state.last_minutes_after_open = minutes_after_open
        st.session_state.last_entry_threshold = entry_threshold
        st.session_state.last_hold_until_close_threshold = hold_until_close_threshold
        st.session_state.last_second_entry_mode = _normalize_second_entry_mode(second_entry_mode)
        st.session_state.last_second_entry_threshold = second_entry_threshold


def _initialize_window_summary_state(
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
):
    if "window_summary_minutes_after_open" not in st.session_state:
        st.session_state.window_summary_minutes_after_open = minutes_after_open
    if "window_summary_entry_threshold" not in st.session_state:
        st.session_state.window_summary_entry_threshold = entry_threshold
    if "window_summary_hold_until_close_threshold" not in st.session_state:
        st.session_state.window_summary_hold_until_close_threshold = hold_until_close_threshold
    if "window_summary_second_entry_mode" not in st.session_state:
        st.session_state.window_summary_second_entry_mode = _normalize_second_entry_mode(second_entry_mode)
    if "window_summary_second_entry_threshold" not in st.session_state:
        st.session_state.window_summary_second_entry_threshold = second_entry_threshold
    if "window_summary_rows" not in st.session_state:
        st.session_state.window_summary_rows = []
    if "window_summary_last_updated" not in st.session_state:
        st.session_state.window_summary_last_updated = pd.NaT
    if "window_summary_last_loss_target" not in st.session_state:
        st.session_state.window_summary_last_loss_target = None
    if "window_summary_last_market_open" not in st.session_state:
        st.session_state.window_summary_last_market_open = pd.NaT


def _update_window_summary_state(
    history_df,
    history_time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    trade_value_usd,
    current_open,
):
    _initialize_window_summary_state(
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
    )
    minutes_after_open_changed = (
        minutes_after_open != st.session_state.window_summary_minutes_after_open
    )
    entry_threshold_changed = (
        entry_threshold != st.session_state.window_summary_entry_threshold
    )
    hold_until_close_threshold_changed = (
        hold_until_close_threshold != st.session_state.window_summary_hold_until_close_threshold
    )
    second_entry_mode_changed = (
        _normalize_second_entry_mode(second_entry_mode)
        != st.session_state.window_summary_second_entry_mode
    )
    second_entry_threshold_changed = (
        second_entry_threshold != st.session_state.window_summary_second_entry_threshold
    )
    recalculate_window_summary = (
        minutes_after_open_changed
        or entry_threshold_changed
        or hold_until_close_threshold_changed
        or second_entry_mode_changed
        or second_entry_threshold_changed
        or not st.session_state.window_summary_rows
    )
    if (
        not recalculate_window_summary
        and pd.notna(current_open)
        and (
            pd.isna(st.session_state.window_summary_last_market_open)
            or current_open > st.session_state.window_summary_last_market_open
        )
    ):
        recalculate_window_summary = True

    if not recalculate_window_summary and history_df is not None and not history_df.empty:
        latest_loss_target = _find_latest_loss_target(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            second_entry_mode,
            second_entry_threshold,
            trade_value_usd,
        )
        if latest_loss_target is not None:
            last_loss_target = st.session_state.window_summary_last_loss_target
            new_loss_seen = last_loss_target is None or latest_loss_target > last_loss_target
            if new_loss_seen:
                last_updated = st.session_state.window_summary_last_updated
                now = pd.Timestamp.utcnow()
                if pd.isna(last_updated) or now - last_updated >= pd.Timedelta(minutes=15):
                    recalculate_window_summary = True

    if recalculate_window_summary and history_df is not None and not history_df.empty:
        summary_rows, latest_loss_target = _calculate_window_summary(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            second_entry_mode,
            second_entry_threshold,
            trade_value_usd,
        )
        st.session_state.window_summary_rows = summary_rows
        st.session_state.window_summary_last_loss_target = latest_loss_target
        st.session_state.window_summary_last_updated = pd.Timestamp.utcnow()
        st.session_state.window_summary_minutes_after_open = minutes_after_open
        st.session_state.window_summary_entry_threshold = entry_threshold
        st.session_state.window_summary_hold_until_close_threshold = hold_until_close_threshold
        st.session_state.window_summary_second_entry_mode = _normalize_second_entry_mode(second_entry_mode)
        st.session_state.window_summary_second_entry_threshold = second_entry_threshold
        st.session_state.window_summary_last_market_open = current_open


def _initialize_summary_refresh_state(
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    trade_value_usd,
    test_balance_start,
):
    if "last_summary_updated" not in st.session_state:
        st.session_state.last_summary_updated = pd.NaT
    if "last_summary_market_open" not in st.session_state:
        st.session_state.last_summary_market_open = pd.NaT
    if "summary_minutes_after_open" not in st.session_state:
        st.session_state.summary_minutes_after_open = minutes_after_open
    if "summary_entry_threshold" not in st.session_state:
        st.session_state.summary_entry_threshold = entry_threshold
    if "summary_hold_until_close_threshold" not in st.session_state:
        st.session_state.summary_hold_until_close_threshold = hold_until_close_threshold
    if "summary_second_entry_mode" not in st.session_state:
        st.session_state.summary_second_entry_mode = _normalize_second_entry_mode(second_entry_mode)
    if "summary_second_entry_threshold" not in st.session_state:
        st.session_state.summary_second_entry_threshold = second_entry_threshold
    if "summary_trade_value_usd" not in st.session_state:
        st.session_state.summary_trade_value_usd = trade_value_usd
    if "summary_test_balance_start" not in st.session_state:
        st.session_state.summary_test_balance_start = test_balance_start
    if "profit_loss_summary" not in st.session_state:
        st.session_state.profit_loss_summary = None
    if "drawdown_summary" not in st.session_state:
        st.session_state.drawdown_summary = None


def _should_recalculate_summary(
    current_open,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    trade_value_usd,
    test_balance_start,
):
    should_recalculate = pd.isna(st.session_state.last_summary_updated)
    if pd.notna(current_open):
        last_market_open = st.session_state.last_summary_market_open
        if pd.isna(last_market_open) or current_open > last_market_open:
            should_recalculate = True
    if (
        minutes_after_open != st.session_state.summary_minutes_after_open
        or entry_threshold != st.session_state.summary_entry_threshold
        or hold_until_close_threshold != st.session_state.summary_hold_until_close_threshold
        or _normalize_second_entry_mode(second_entry_mode)
        != st.session_state.summary_second_entry_mode
        or second_entry_threshold != st.session_state.summary_second_entry_threshold
        or trade_value_usd != st.session_state.summary_trade_value_usd
        or test_balance_start != st.session_state.summary_test_balance_start
    ):
        should_recalculate = True
    if not should_recalculate:
        last_updated = st.session_state.last_summary_updated
        now = pd.Timestamp.utcnow()
        if pd.isna(last_updated) or now - last_updated >= pd.Timedelta(minutes=15):
            should_recalculate = True
    return should_recalculate


def prepare_probability_window(
    df,
    time_column,
    lookback_period,
    resample_interval,
    jump_container,
):
    if 'window_offset' not in st.session_state:
        st.session_state.window_offset = 0

    window_size = int(lookback_period)
    target_times = df['TargetTime_dt'].dropna().drop_duplicates().tolist()
    total_markets = len(target_times)
    max_offset = max(0, total_markets - window_size)
    if st.session_state.window_offset > max_offset:
        st.session_state.window_offset = max_offset

    jump_default = df['TargetTime_dt'].max()
    if pd.isna(jump_default):
        jump_default = df[time_column].max()

    jump_time = jump_container.datetime_input(
        "Jump to time",
        value=jump_default,
        help=f"Jump to the {window_size}-market window that includes this time.",
    )
    if jump_container.button("Jump", key="window_jump_button") and total_markets:
        eligible_times = [t for t in target_times if t and t <= jump_time]
        if eligible_times:
            target_index = target_times.index(eligible_times[-1])
        else:
            target_index = 0
        st.session_state.window_offset = max(0, total_markets - (target_index + 1))

    if total_markets:
        window_end = total_markets - st.session_state.window_offset
        window_start = max(0, window_end - window_size)
        active_targets = target_times[window_start:window_end]
        df_window = df[df['TargetTime_dt'].isin(active_targets)]
    else:
        df_window = df

    if df_window.empty:
        st.warning("No data available for the selected window.")
        st.stop()

    df_window = _resample_market_data(df_window, time_column, resample_interval)

    if df_window.empty:
        st.warning("No data available after resampling.")
        st.stop()

    latest = df_window.iloc[-1]
    return {
        "df_window": df_window,
        "latest": latest,
        "max_offset": max_offset,
        "total_markets": total_markets,
    }

def build_market_summary_table(df_window, latest, time_column):
    latest_timestamp = df_window[time_column].max()
    market_rows = df_window[df_window['TargetTime'] == latest['TargetTime']]
    market_start_time = market_rows[time_column].min()
    market_open_time = align_market_open(market_start_time)
    if pd.isna(market_start_time):
        countdown_display = "N/A"
    else:
        market_end_time = market_open_time + pd.Timedelta(minutes=15)
        remaining_seconds = int((market_end_time - latest_timestamp).total_seconds())
        remaining_seconds = max(0, remaining_seconds)
        minutes_left = remaining_seconds // 60
        seconds_left = remaining_seconds % 60
        countdown_display = f"{minutes_left:02d}:{seconds_left:02d}"
    return pd.DataFrame(
        [
            {"Metric": "Minutes Left (MM:SS)", "Value": countdown_display},
            {
                "Metric": "Yes (Up) Cost",
                "Value": _format_metric(latest.get("UpPrice"), lambda v: f"${v:.2f}"),
            },
            {
                "Metric": "No (Down) Cost",
                "Value": _format_metric(latest.get("DownPrice"), lambda v: f"${v:.2f}"),
            },
        ]
    ).set_index("Metric")


def render_probability_history(
    df,
    chart_data,
    time_column,
    show_markers,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
):
    df_window = chart_data["df_window"]
    max_offset = chart_data["max_offset"]
    latest = chart_data["latest"]

    # Process data for charts (add gaps between different markets)
    df_chart = df_window.copy().sort_values(time_column)
    df_chart['group'] = (df_chart['TargetTime'] != df_chart['TargetTime'].shift()).cumsum()

    segments = []
    for _, group in df_chart.groupby('group'):
        segments.append(group)
        # Add gap row
        gap_row = group.iloc[[-1]].copy()
        gap_row[time_column] += pd.Timedelta(seconds=1)
        # Set values to NaN to break the line
        for col in ['UpPrice', 'DownPrice', 'UpVol', 'DownVol']:
            gap_row[col] = np.nan
        segments.append(gap_row)

    df_chart = pd.concat(segments).reset_index(drop=True)

    # Initialize zoom mode
    if 'zoom_mode' not in st.session_state:
        st.session_state.zoom_mode = None

    # Zoom Controls
    col_z1, col_z2 = st.columns([1, 10])
    with col_z1:
        if st.button("Reset Zoom", key='reset_zoom_button'):
            st.session_state.zoom_mode = None
    with col_z2:
        if st.button("Zoom Last 15m", key='zoom_15m_button'):
            st.session_state.zoom_mode = 'last_15m'

    # Calculate range based on mode
    current_range = None
    if st.session_state.zoom_mode == 'last_15m':
        end_time = df_window[time_column].max()
        start_time = end_time - pd.Timedelta(minutes=15)
        current_range = [start_time, end_time]

    trace_mode = "lines+markers" if show_markers else "lines"
    colors = {
        "up": "rgba(34, 139, 34, 0.75)",
        "down": "rgba(220, 20, 60, 0.65)",
    }

    # Create Subplots with shared x-axis
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{}]],
        subplot_titles=(
            "Probability History",
        ),
    )

    # Probability Chart (Row 1)
    fig.add_trace(
        go.Scatter(
            x=df_chart[time_column],
            y=df_chart['UpPrice'],
            name="Yes (Up)",
            line=dict(color=colors["up"], width=2, shape="spline", smoothing=1.1),
            connectgaps=True,
            mode=trace_mode,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_chart[time_column],
            y=df_chart['DownPrice'],
            name="No (Down)",
            line=dict(color=colors["down"], dash='dash', width=2, shape="spline", smoothing=1.1),
            connectgaps=True,
            mode=trace_mode,
        ),
        row=1,
        col=1,
    )

    ordered_targets = df_window["TargetTime_dt"].dropna().drop_duplicates().tolist()
    full_target_dt_order = df["TargetTime_dt"].dropna().drop_duplicates().tolist()
    trade_records = _calculate_trade_records(
        df_window,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
        target_order=full_target_dt_order,
    )
    trade_record_map = {record["target_time_dt"]: record for record in trade_records}
    entry_times = []
    entry_prices = []
    second_entry_times = []
    second_entry_prices = []
    exit_times = []
    exit_prices = []
    held_times = []
    held_prices = []

    for target_time in ordered_targets:
        market_group = df_window[df_window["TargetTime_dt"] == target_time].sort_values(time_column)
        if market_group.empty:
            continue
        record = trade_record_map.get(target_time)
        if record is None:
            continue
        open_threshold_time = record["open_threshold_time"]
        if pd.notna(open_threshold_time):
            add_vline_all_rows(
                fig,
                open_threshold_time,
                line_width=1,
                line_dash="solid",
                line_color="rgba(200, 200, 200, 0.4)",
            )

        if record["entry_time"] is not None and record["entry_price"] is not None:
            if record.get("entry_mode") == "additive" and record.get("trigger_price") is not None:
                entry_times.append(record["entry_time"])
                entry_prices.append(record["trigger_price"])
                if record.get("second_entry_time") is not None and record.get("second_entry_price") is not None:
                    second_entry_times.append(record["second_entry_time"])
                    second_entry_prices.append(record["second_entry_price"])
            else:
                entry_times.append(record["entry_time"])
                entry_prices.append(record["entry_price"])

        exit_price_display = record.get("exit_price_market", record["exit_price"])
        if record["exit_time"] is not None and exit_price_display is not None and not pd.isna(exit_price_display):
            if record["exit_reason"] == "threshold":
                exit_times.append(record["exit_time"])
                exit_prices.append(exit_price_display)
            elif record["exit_reason"] == "held_to_close":
                held_times.append(record["exit_time"])
                held_prices.append(exit_price_display)

        if record["outcome"] in {"Win", "Lose", "Tie"}:
            if record["exit_reason"] == "threshold":
                outcome_text = f"{record['outcome']} (exit)"
            else:
                outcome_text = f"{record['outcome']} (close)"

            outcome_color = "#808080" if record["outcome"] == "Tie" else (
                "#00AA00" if record["outcome"] == "Win" else "#FF0000"
            )
            fig.add_annotation(
                x=record["exit_time"] or record["market_close_time"],
                y=1.03,
                text=outcome_text,
                showarrow=False,
                font=dict(color=outcome_color, size=16),
                row=1,
                col=1,
            )

    if entry_times:
        fig.add_trace(
            go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode="markers+text",
                marker=dict(color="#1E90FF", size=9),
                text=["entry"] * len(entry_times),
                textposition="top center",
                textfont=dict(size=10, color="#1E90FF"),
                name="Entry",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if exit_times:
        fig.add_trace(
            go.Scatter(
                x=exit_times,
                y=exit_prices,
                mode="markers+text",
                marker=dict(color="#6A5ACD", size=9),
                text=["exit"] * len(exit_times),
                textposition="top center",
                textfont=dict(size=10, color="#6A5ACD"),
                name="Exit",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if second_entry_times:
        fig.add_trace(
            go.Scatter(
                x=second_entry_times,
                y=second_entry_prices,
                mode="markers+text",
                marker=dict(color="#FFA500", size=8),
                text=["2nd entry"] * len(second_entry_times),
                textposition="top center",
                textfont=dict(size=10, color="#FFA500"),
                name="Second Entry",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if held_times:
        fig.add_trace(
            go.Scatter(
                x=held_times,
                y=held_prices,
                mode="markers+text",
                marker=dict(color="#808080", size=9),
                text=["held to close"] * len(held_times),
                textposition="top center",
                textfont=dict(size=10, color="#808080"),
                name="Held to close",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Add vertical lines for market transitions to both plots
    # Identify where TargetTime changes
    transitions = df_window.loc[df_window['TargetTime'].shift() != df_window['TargetTime'], time_column].iloc[1:]

    for t in transitions:
        add_vline_all_rows(fig, t, line_width=1, line_dash="dot", line_color="gray")

    # Update Layout
    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis=dict(title="Probability", range=[0, 1.05]),
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
    )
    # Explicitly set range for the chart x-axis.
    if current_range:
        fig.update_xaxes(range=current_range, row=1, col=1)

    # Enable crosshair (spike lines) across both subplots
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='dash')
    st.plotly_chart(fig, width='stretch', config={'scrollZoom': True})

    def _handle_window_back(offset_limit):
        st.session_state.window_offset = min(offset_limit, st.session_state.window_offset + 1)

    def _handle_window_forward():
        st.session_state.window_offset = max(0, st.session_state.window_offset - 1)

    def _handle_window_latest():
        st.session_state.window_offset = 0

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    with nav_col1:
        st.button(
            "Back",
            key="window_back_button",
            disabled=st.session_state.window_offset >= max_offset,
            on_click=_handle_window_back,
            args=(max_offset,),
        )
    with nav_col2:
        st.button(
            "Forward",
            key="window_forward_button",
            disabled=st.session_state.window_offset <= 0,
            on_click=_handle_window_forward,
        )
    with nav_col3:
        st.button(
            "Latest",
            key="window_latest_button",
            disabled=st.session_state.window_offset == 0,
            on_click=_handle_window_latest,
        )

    return {
        "df_window": df_window,
        "latest": latest,
        "max_offset": max_offset,
        "total_markets": chart_data["total_markets"],
    }


def compute_summary_state(
    history_df,
    history_time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    second_entry_mode,
    second_entry_threshold,
    trade_value_usd,
    test_balance_start,
    summary_reference_time,
    today_start_time,
    current_open,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    _initialize_strike_rate_state(
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
    )
    _update_strike_rate_state(
        history_df,
        history_time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
        current_open,
        precomputed_groups=precomputed_groups,
        precomputed_target_order=precomputed_target_order,
    )

    strike_rate = st.session_state.strike_rate
    avg_entry_price = st.session_state.get("avg_entry_price", np.nan)
    min_entry_price = st.session_state.get("min_entry_price", np.nan)
    max_entry_price = st.session_state.get("max_entry_price", np.nan)
    win_rate_needed = st.session_state.get("win_rate_needed", np.nan)
    strike_sample_size = st.session_state.get("strike_sample_size")
    autotune_sample_size = st.session_state.get("autotune_sample_size")

    _update_window_summary_state(
        history_df,
        history_time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
        trade_value_usd,
        current_open,
    )

    _initialize_summary_refresh_state(
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
        trade_value_usd,
        test_balance_start,
    )
    recalculate_summary = _should_recalculate_summary(
        current_open,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
        trade_value_usd,
        test_balance_start,
    )
    profit_loss_summary = st.session_state.profit_loss_summary
    drawdown_summary = st.session_state.drawdown_summary
    if recalculate_summary and history_df is not None and not history_df.empty:
        profit_loss_trade_records = _calculate_trade_records(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            second_entry_mode,
            second_entry_threshold,
        )
        closed_trades = build_trade_pnl_records(profit_loss_trade_records, trade_value_usd)
        profit_loss_summary = summarize_profit_loss(
            closed_trades,
            reference_time=summary_reference_time,
            today_start_time=today_start_time,
        )
        drawdown_summary = summarize_drawdowns(
            closed_trades,
            reference_time=summary_reference_time,
            test_balance_start=test_balance_start,
            today_start_time=today_start_time,
        )
        st.session_state.profit_loss_summary = profit_loss_summary
        st.session_state.drawdown_summary = drawdown_summary
        st.session_state.last_summary_updated = pd.Timestamp.utcnow()
        st.session_state.last_summary_market_open = current_open
        st.session_state.summary_minutes_after_open = minutes_after_open
        st.session_state.summary_entry_threshold = entry_threshold
        st.session_state.summary_hold_until_close_threshold = hold_until_close_threshold
        st.session_state.summary_second_entry_mode = _normalize_second_entry_mode(second_entry_mode)
        st.session_state.summary_second_entry_threshold = second_entry_threshold
        st.session_state.summary_trade_value_usd = trade_value_usd
        st.session_state.summary_test_balance_start = test_balance_start

    profit_loss_summary = profit_loss_summary or {}
    drawdown_summary = drawdown_summary or {}
    second_entry_records = _get_cached_trade_records(
        history_df,
        history_time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
        allow_compute=True,
        precomputed_groups=precomputed_groups,
        precomputed_target_order=precomputed_target_order,
    )
    baseline_records = _get_cached_trade_records(
        history_df,
        history_time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        "off",
        second_entry_threshold,
        allow_compute=False,
        precomputed_groups=precomputed_groups,
        precomputed_target_order=precomputed_target_order,
    )
    second_entry_metrics = _summarize_trade_record_metrics(second_entry_records, trade_value_usd)
    baseline_metrics = _summarize_trade_record_metrics(baseline_records, trade_value_usd)
    return {
        "strike_rate": strike_rate,
        "avg_entry_price": avg_entry_price,
        "min_entry_price": min_entry_price,
        "max_entry_price": max_entry_price,
        "win_rate_needed": win_rate_needed,
        "strike_sample_size": strike_sample_size,
        "autotune_sample_size": autotune_sample_size,
        "profit_loss_summary": profit_loss_summary,
        "drawdown_summary": drawdown_summary,
        "second_entry_metrics": second_entry_metrics,
        "baseline_metrics": baseline_metrics,
    }


def render_strike_rate_section(
    summary_state,
    history_df,
    history_time_column,
    second_entry_mode,
    second_entry_threshold,
    precomputed_groups=None,
    precomputed_target_order=None,
):
    strike_rate = summary_state["strike_rate"]
    avg_entry_price = summary_state["avg_entry_price"]
    min_entry_price = summary_state["min_entry_price"]
    max_entry_price = summary_state["max_entry_price"]
    win_rate_needed = summary_state["win_rate_needed"]
    strike_sample_size = summary_state["strike_sample_size"]
    autotune_sample_size = summary_state["autotune_sample_size"]

    gauge_value = 50 if pd.isna(strike_rate) else strike_rate
    gauge_value = max(50, min(100, gauge_value))
    win_rate_needed_pct = 50 if pd.isna(win_rate_needed) else win_rate_needed
    win_rate_needed_pct = max(50, min(100, win_rate_needed_pct))
    green_end = win_rate_needed_pct
    red_start = win_rate_needed_pct
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Strike Rate"},
            gauge={
                "shape": "angular",
                "axis": {"range": [50, 100]},
                "bar": {"color": "rgba(0, 0, 0, 0)"},
                "steps": [
                    {"range": [50, green_end], "color": "red"},
                    {"range": [red_start, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": gauge_value,
                },
            },
        )
    )
    gauge_fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(gauge_fig, width='stretch', config={'displayModeBar': False})
    if not pd.isna(avg_entry_price):
        average_entry_display = f"{avg_entry_price:.2f}"
        if not pd.isna(min_entry_price) and not pd.isna(max_entry_price):
            average_entry_display = (
                f"{average_entry_display} (L {min_entry_price:.2f}, H {max_entry_price:.2f})"
            )
    else:
        average_entry_display = "N/A"
    win_rate_display = f"{win_rate_needed:.2f}%" if not pd.isna(win_rate_needed) else "N/A"
    if not pd.isna(strike_rate) and not pd.isna(win_rate_needed):
        edge_value = strike_rate - win_rate_needed
        edge_display = f"{edge_value:+.2f}%"
    else:
        edge_display = "N/A"
    if strike_sample_size is not None and autotune_sample_size is not None:
        st.caption(
            f"Samples: autotune={autotune_sample_size}, strike rate={strike_sample_size}"
        )
    metrics_container = st.container()
    with metrics_container:
        metrics_table = pd.DataFrame(
            {
                "Metric": ["Average Entry", "Win Rate Needed", "Edge"],
                "Value": [
                    average_entry_display,
                    win_rate_display,
                    edge_display,
                ],
            }
        )
        st.table(metrics_table)
    autotune_clicked = st.button(
        "Autotune",
        key="autotune_button",
        use_container_width=True,
    )
    if autotune_clicked:
        progress_container = st.empty()
        status_container = st.status("Autotuning…", expanded=True)
        progress_bar = progress_container.progress(0)

        def _progress_callback(current_step, total_steps, message):
            progress_bar.progress(current_step / total_steps)
            status_container.write(message)

        with status_container:
            def _autotune_metrics(df, column, minutes, threshold, hold_threshold):
                return _calculate_strike_rate_metrics(
                    df,
                    column,
                    minutes,
                    threshold,
                    hold_threshold,
                    second_entry_mode,
                    second_entry_threshold,
                    history_segment="autotune",
                    precomputed_groups=precomputed_groups,
                    precomputed_target_order=precomputed_target_order,
                )

            best_result = run_autotune(
                history_df,
                history_time_column,
                _autotune_metrics,
                progress_callback=_progress_callback,
            )
        progress_container.empty()
        status_container.update(state="complete", label="Autotune complete")
        if best_result:
            st.session_state.autotune_result = best_result
            st.session_state.autotune_message = None
        else:
            st.session_state.autotune_result = None
            st.session_state.autotune_message = "No viable data for autotune"
    if st.session_state.autotune_result:
        result = st.session_state.autotune_result
        st.caption(
            "Best: "
            f"minutes_after_open={result['minutes_after_open']}, "
            f"entry_threshold={result['entry_threshold']:.2f}, "
            f"hold_until_close_threshold={result['hold_until_close_threshold']:.2f}, "
            f"strike_rate={result['strike_rate']:.2f}%, "
            f"win_rate_needed={result['win_rate_needed']:.2f}%, "
            f"edge={result['edge']:.2f}%"
        )
    elif st.session_state.autotune_message:
        st.caption(st.session_state.autotune_message)
    second_entry_autotune_clicked = st.button(
        "Run Second-Entry Autotune",
        key="second_entry_autotune_button",
        use_container_width=True,
    )
    if second_entry_autotune_clicked:
        normalized_second_entry_mode = _normalize_second_entry_mode(second_entry_mode)
        if normalized_second_entry_mode == "off":
            st.session_state.second_entry_autotune_result = None
            st.session_state.second_entry_autotune_panel = None
            st.session_state.second_entry_autotune_message = None
        else:
            second_entry_progress_container = st.empty()
            second_entry_status_container = st.status("Second-entry autotuning…", expanded=True)
            second_entry_progress_bar = second_entry_progress_container.progress(0)

            def _second_entry_progress_callback(current_step, total_steps, message):
                second_entry_progress_bar.progress(current_step / total_steps)
                second_entry_status_container.write(message)

            with second_entry_status_container:
                second_entry_results = run_second_entry_autotune(
                    history_df,
                    history_time_column,
                    minutes_after_open,
                    entry_threshold,
                    hold_until_close_threshold,
                    modes=(normalized_second_entry_mode,),
                    progress_callback=_second_entry_progress_callback,
                    precomputed_groups=precomputed_groups,
                    precomputed_target_order=precomputed_target_order,
                )
            second_entry_progress_container.empty()
            second_entry_status_container.update(
                state="complete",
                label="Second-entry autotune complete",
            )
            if second_entry_results and any(second_entry_results.values()):
                st.session_state.second_entry_autotune_result = second_entry_results
                st.session_state.second_entry_autotune_panel = _build_second_entry_autotune_panel(
                    second_entry_results
                )
                st.session_state.second_entry_autotune_message = None
            else:
                st.session_state.second_entry_autotune_result = None
                st.session_state.second_entry_autotune_panel = None
                st.session_state.second_entry_autotune_message = "No viable data for second-entry autotune"
    if st.session_state.second_entry_autotune_panel:
        st.markdown("Second-entry autotune results")
        panel_df = pd.DataFrame(st.session_state.second_entry_autotune_panel)
        st.table(panel_df)
    elif st.session_state.second_entry_autotune_message:
        st.caption(st.session_state.second_entry_autotune_message)


def render_profit_loss_section(summary_state):
    profit_loss_summary = summary_state["profit_loss_summary"] or {}
    drawdown_summary = summary_state["drawdown_summary"] or {}
    pnl_table = pd.DataFrame(
        [
            {
                "Period": "Today",
                "P/L (USD)": _format_metric(
                    profit_loss_summary.get("today"),
                    lambda v: f"${v:,.2f}",
                ),
                "Max Drawdown %": _format_metric(
                    drawdown_summary.get("today"),
                    lambda v: f"{v * 100:.2f}%",
                ),
            },
            {
                "Period": "7-day rolling",
                "P/L (USD)": _format_metric(
                    profit_loss_summary.get("week_to_date"),
                    lambda v: f"${v:,.2f}",
                ),
                "Max Drawdown %": _format_metric(
                    drawdown_summary.get("week_to_date"),
                    lambda v: f"{v * 100:.2f}%",
                ),
            },
            {
                "Period": "30-day rolling",
                "P/L (USD)": _format_metric(
                    profit_loss_summary.get("month_to_date"),
                    lambda v: f"${v:,.2f}",
                ),
                "Max Drawdown %": _format_metric(
                    drawdown_summary.get("month_to_date"),
                    lambda v: f"{v * 100:.2f}%",
                ),
            },
            {
                "Period": "All Time",
                "P/L (USD)": _format_metric(
                    profit_loss_summary.get("all_time"),
                    lambda v: f"${v:,.2f}",
                ),
                "Max Drawdown %": _format_metric(
                    None,
                    lambda v: f"{v * 100:.2f}%",
                ),
            },
        ]
    ).set_index("Period")
    st.dataframe(pnl_table, width="stretch")


def render_second_entry_summary(summary_state):
    second_entry_metrics = summary_state.get("second_entry_metrics") or {}
    baseline_metrics = summary_state.get("baseline_metrics") or {}
    edge_delta = np.nan
    if not pd.isna(second_entry_metrics.get("edge", np.nan)) and not pd.isna(
        baseline_metrics.get("edge", np.nan)
    ):
        edge_delta = second_entry_metrics["edge"] - baseline_metrics["edge"]
    metrics_table = pd.DataFrame(
        [
            {
                "Metric": "Trade Count",
                "Value": _format_metric(
                    second_entry_metrics.get("trade_count"),
                    lambda v: f"{int(v)}",
                ),
            },
            {
                "Metric": "Win Rate",
                "Value": _format_metric(
                    second_entry_metrics.get("win_rate"),
                    lambda v: f"{v:.2f}%",
                ),
            },
            {
                "Metric": "Expectancy (USD)",
                "Value": _format_metric(
                    second_entry_metrics.get("expectancy"),
                    lambda v: f"${v:,.2f}",
                ),
            },
            {
                "Metric": "Edge Δ vs Baseline",
                "Value": _format_metric(
                    edge_delta,
                    lambda v: f"{v:+.2f}%",
                ),
            },
        ]
    ).set_index("Metric")
    st.subheader("Second-Entry Summary")
    st.dataframe(metrics_table, width="stretch")

def render_dashboard():
    if available_dates:
        selected_date = st.sidebar.date_input(
            "Data date",
            value=latest_available_date,
            min_value=min_available_date,
            max_value=latest_available_date,
            help="Select a historical data file by date. Defaults to the latest available file.",
        )
    else:
        selected_date = None
        st.sidebar.caption("No dated CSV files found; using legacy data file if available.")

    jump_container = st.sidebar.container()

    progress_container = st.empty()
    status_container = st.status("Loading dashboard data…", expanded=True)
    progress_bar = progress_container.progress(0, text="Loading data files…")

    df, resolved_date = load_data(selected_date, files_by_date, legacy_path)
    progress_bar.progress(0.25, text="Loaded selected data file.")
    history_df = load_all_data(files_by_date, legacy_path)
    progress_bar.progress(0.45, text="Loaded historical data.")
    if history_df is None or history_df.empty:
        history_df = df
    if selected_date and resolved_date and selected_date != resolved_date:
        st.info(
            f"No data file found for {selected_date.strftime('%Y-%m-%d')}. "
            f"Showing {resolved_date.strftime('%Y-%m-%d')} instead."
        )

    if df is not None and not df.empty:
        time_column = "Timestamp" if time_axis == "Polymarket Time (ET)" else "Timestamp_UK"
        if time_column not in df.columns:
            st.warning("UK timestamps are not available in this data file.")
            time_column = "Timestamp"

        df = df.sort_values(time_column)
        df['TargetTime_dt'] = pd.to_datetime(df['TargetTime'], format=TIME_FORMAT, errors='coerce')
        if history_df is not None and not history_df.empty:
            history_time_column = time_column if time_column in history_df.columns else "Timestamp"
            history_df = history_df.sort_values(history_time_column)
            history_df["TargetTime_dt"] = pd.to_datetime(
                history_df["TargetTime"], format=TIME_FORMAT, errors="coerce"
            )
        else:
            history_time_column = time_column

        history_latest_timestamp = (
            history_df[history_time_column].max()
            if history_df is not None and not history_df.empty
            else df[time_column].max()
        )
        summary_reference_time = df[time_column].max()
        if pd.isna(summary_reference_time):
            summary_reference_time = history_latest_timestamp
        today_start_time = df[time_column].min()
        if pd.isna(today_start_time):
            today_start_time = None
        current_open = align_market_open(history_latest_timestamp)
        history_market_groups, history_target_order = _build_market_groups(
            history_df,
            history_time_column,
        )

        probability_window = prepare_probability_window(
            df,
            time_column,
            lookback_period,
            resample_interval,
            jump_container,
        )
        progress_bar.progress(0.65, text="Prepared market window.")

        summary_state = compute_summary_state(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            second_entry_mode,
            second_entry_threshold,
            trade_value_usd,
            test_balance_start,
            summary_reference_time,
            today_start_time,
            current_open,
            precomputed_groups=history_market_groups,
            precomputed_target_order=history_target_order,
        )
        progress_bar.progress(0.85, text="Calculated summary metrics.")

        header_cols = st.columns([2.2, 3, 2.2])
        with header_cols[0]:
            st.title("Polymarket 8020 Monitor")
            st.button("Refresh Data", key="refresh_data_button", use_container_width=True)
            st.checkbox(
                "Auto-refresh",
                key="auto_refresh",
                value=st.session_state.get("auto_refresh", False),
            )
        with header_cols[1]:
            render_strike_rate_section(
                summary_state,
                history_df,
                history_time_column,
                second_entry_mode,
                second_entry_threshold,
                precomputed_groups=history_market_groups,
                precomputed_target_order=history_target_order,
            )
        with header_cols[2]:
            market_summary_table = build_market_summary_table(
                probability_window["df_window"],
                probability_window["latest"],
                time_column,
            )
            st.dataframe(market_summary_table, width="stretch")
            render_profit_loss_section(summary_state)
            render_second_entry_summary(summary_state)

        probability_renderer = render_probability_history
        if st.session_state.get("auto_refresh", False):
            probability_renderer = st.fragment(run_every=refresh_interval_seconds)(render_probability_history)
        chart_result = probability_renderer(
            df,
            probability_window,
            time_column,
            show_markers,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            second_entry_mode,
            second_entry_threshold,
        )

        with st.expander("Window summary"):
            summary_df = pd.DataFrame(st.session_state.window_summary_rows)
            st.dataframe(summary_df, width='stretch')

        st.caption(f"Last updated: {chart_result['latest']['Timestamp']}")
        progress_bar.progress(1.0, text="Dashboard ready.")
        status_container.update(state="complete", label="Dashboard ready")
        progress_container.empty()

    else:
        st.warning("No data found yet. Please ensure data_logger.py is running.")
        progress_bar.progress(1.0, text="No data available.")
        status_container.update(state="complete", label="No data available")
        progress_container.empty()


render_dashboard()
