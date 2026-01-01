import numpy as np
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re
from autotune import run_autotune


# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATE_FORMAT = "%d%m%Y"


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


st.set_page_config(page_title="Polymarket BTC Monitor", layout="wide")

st.sidebar.header("Analysis Controls")
lookback_period = st.sidebar.number_input(
    "Lookback period (markets)",
    min_value=1,
    max_value=20,
    value=4,
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
resample_interval = st.sidebar.selectbox(
    "Resample interval",
    options=("1s", "5s", "15s", "30s", "60s", "all"),
    index=1,
)
liquidity_aggregation = st.sidebar.selectbox(
    "Liquidity aggregation",
    options=("sum", "mean"),
    index=0,
)
liquidity_y_scale = st.sidebar.selectbox(
    "Liquidity y-scale",
    options=("linear", "log"),
    index=0,
)
liquidity_bar_mode = st.sidebar.selectbox(
    "Liquidity bar mode",
    options=("group", "stack"),
    index=0,
)
show_markers = st.sidebar.checkbox("Show markers", value=True)
momentum_window_seconds = st.sidebar.number_input(
    "Rolling window seconds for momentum",
    min_value=1,
    max_value=600,
    value=60,
    step=5,
)
refresh_interval_seconds = st.sidebar.number_input(
    "Auto-refresh interval (seconds)",
    min_value=1,
    max_value=60,
    value=1,
    step=1,
    help="Controls the sleep duration for the auto-refresh loop.",
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


def _load_data_file(data_file):
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


def load_all_data(files_by_date, legacy_path):
    data_frames = []
    for _, data_file in sorted(files_by_date.items()):
        try:
            df = _load_data_file(data_file)
        except FileNotFoundError:
            continue
        data_frames.append(df)
    if legacy_path:
        try:
            data_frames.append(_load_data_file(legacy_path))
        except FileNotFoundError:
            pass
    if not data_frames:
        return None
    return pd.concat(data_frames, ignore_index=True)

# Top-row controls
files_by_date, legacy_path = _get_available_data_files()
available_dates = sorted(files_by_date)
latest_available_date = max(available_dates) if available_dates else None
min_available_date = min(available_dates) if available_dates else None

col_top1, col_top2 = st.columns(2)
with col_top1:
    st.button('Refresh Data', key='refresh_data_button', width='stretch')
    auto_refresh = st.checkbox("Auto-refresh", value=True)
with col_top2:
    if available_dates:
        selected_date = st.date_input(
            "Data date",
            value=latest_available_date,
            min_value=min_available_date,
            max_value=latest_available_date,
            help="Select a historical data file by date. Defaults to the latest available file.",
        )
    else:
        selected_date = None
        st.caption("No dated CSV files found; using legacy data file if available.")


def _resample_market_data(df, time_column, interval, liquidity_aggregation):
    if df.empty or not interval or interval == "all":
        return df
    volume_agg = "sum" if liquidity_aggregation == "sum" else "mean"
    resampled_groups = []
    for target_time, group in df.groupby("TargetTime", sort=False):
        if group.empty:
            continue
        group = group.sort_values(time_column)
        resampled = group.set_index(time_column).resample(interval).agg(
            {
                "UpPrice": "last",
                "DownPrice": "last",
                "UpVol": volume_agg,
                "DownVol": volume_agg,
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


def _last_non_zero(series):
    cleaned = series.replace(0, np.nan).dropna()
    if cleaned.empty:
        return np.nan
    return cleaned.iloc[-1]

def _infer_interval_seconds(timestamp_series):
    cleaned = timestamp_series.dropna().sort_values()
    diffs = cleaned.diff().dropna()
    if diffs.empty:
        return 1
    median_seconds = diffs.median().total_seconds()
    if np.isnan(median_seconds) or median_seconds <= 0:
        return 1
    return max(1, int(round(median_seconds)))


def _align_market_open(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return pd.NaT
    return pd.Timestamp(timestamp).floor("15min")


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


def _calculate_market_trade_records(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    target_order=None,
):
    if df is None or df.empty:
        return []

    df = df.copy()
    if "TargetTime_dt" not in df.columns:
        df["TargetTime_dt"] = pd.to_datetime(df["TargetTime"], format=TIME_FORMAT, errors="coerce")

    minutes_threshold = pd.Timedelta(minutes=int(minutes_after_open))
    probability_threshold = float(entry_threshold)
    hold_threshold = float(hold_until_close_threshold)

    if target_order is None:
        target_order = df["TargetTime_dt"].dropna().drop_duplicates().tolist()

    target_indices = {target: idx for idx, target in enumerate(target_order)}
    last_index = len(target_order) - 1
    records = []

    for target_time in target_order:
        market_group = df[df["TargetTime_dt"] == target_time].sort_values(time_column)
        if market_group.empty:
            continue

        market_open = _align_market_open(market_group[time_column].min())
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
                    exit_reason = "threshold"
                else:
                    exit_time = market_close_time
                    exit_reason = "held_to_close"

            if exit_time == market_close_time:
                exit_price = close_up if expected_side == "Up" else close_down

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
                "exit_reason": exit_reason,
                "outcome": outcome,
                "close_up": close_up,
                "close_down": close_down,
                "market_closed": market_closed,
            }
        )

    return records


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


def _calculate_strike_rate_metrics(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    history_segment="strike",
):
    trade_records = _calculate_market_trade_records(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
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
        avg_entry_price = np.nan
        win_rate_needed = np.nan
    return strike_rate, avg_entry_price, win_rate_needed, total_count


def _calculate_window_summary(
    df,
    time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
):
    summary_rows = []
    loss_targets = []
    trade_records = _calculate_market_trade_records(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
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
            pnl_usd = record["exit_price"] - record["entry_price"]

        summary_rows.append(
            {
                "TargetTime": market_group["TargetTime"].iloc[0],
                "Market Open": record["market_open"],
                "First Crossing Side": record["expected_side"] or "None",
                "Crossing Time": record["entry_time"],
                "Entry Price": record["entry_price"],
                "Exit Time": record["exit_time"],
                "Exit Price": record["exit_price"],
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
):
    _, latest_loss_target = _calculate_window_summary(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
    )
    return latest_loss_target


def _initialize_strike_rate_state(minutes_after_open, entry_threshold, hold_until_close_threshold):
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
    if "autotune_result" not in st.session_state:
        st.session_state.autotune_result = None
    if "autotune_message" not in st.session_state:
        st.session_state.autotune_message = None
    if "strike_sample_size" not in st.session_state:
        st.session_state.strike_sample_size = None
    if "autotune_sample_size" not in st.session_state:
        st.session_state.autotune_sample_size = None


def _should_recalculate_strike_rate(
    current_open,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
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
    if (
        minutes_after_open_changed
        or entry_threshold_changed
        or hold_until_close_threshold_changed
    ):
        should_recalculate = True
    return should_recalculate


def _update_strike_rate_state(
    history_df,
    history_time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
    current_open,
):
    _initialize_strike_rate_state(minutes_after_open, entry_threshold, hold_until_close_threshold)
    should_recalculate = _should_recalculate_strike_rate(
        current_open,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
    )
    if should_recalculate:
        strike_rate, avg_entry_price, win_rate_needed, strike_sample_size = _calculate_strike_rate_metrics(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            history_segment="strike",
        )
        _, _, _, autotune_sample_size = _calculate_strike_rate_metrics(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            history_segment="autotune",
        )
        st.session_state.strike_rate = strike_rate
        st.session_state.avg_entry_price = avg_entry_price
        st.session_state.win_rate_needed = win_rate_needed
        st.session_state.strike_sample_size = strike_sample_size
        st.session_state.autotune_sample_size = autotune_sample_size
        st.session_state.last_market_open = current_open
        st.session_state.strike_rate_initialized = True
        st.session_state.last_minutes_after_open = minutes_after_open
        st.session_state.last_entry_threshold = entry_threshold
        st.session_state.last_hold_until_close_threshold = hold_until_close_threshold


def _initialize_window_summary_state(minutes_after_open, entry_threshold, hold_until_close_threshold):
    if "window_summary_minutes_after_open" not in st.session_state:
        st.session_state.window_summary_minutes_after_open = minutes_after_open
    if "window_summary_entry_threshold" not in st.session_state:
        st.session_state.window_summary_entry_threshold = entry_threshold
    if "window_summary_hold_until_close_threshold" not in st.session_state:
        st.session_state.window_summary_hold_until_close_threshold = hold_until_close_threshold
    if "window_summary_rows" not in st.session_state:
        st.session_state.window_summary_rows = []
    if "window_summary_last_updated" not in st.session_state:
        st.session_state.window_summary_last_updated = pd.NaT
    if "window_summary_last_loss_target" not in st.session_state:
        st.session_state.window_summary_last_loss_target = None


def _update_window_summary_state(
    history_df,
    history_time_column,
    minutes_after_open,
    entry_threshold,
    hold_until_close_threshold,
):
    _initialize_window_summary_state(minutes_after_open, entry_threshold, hold_until_close_threshold)
    minutes_after_open_changed = (
        minutes_after_open != st.session_state.window_summary_minutes_after_open
    )
    entry_threshold_changed = (
        entry_threshold != st.session_state.window_summary_entry_threshold
    )
    hold_until_close_threshold_changed = (
        hold_until_close_threshold != st.session_state.window_summary_hold_until_close_threshold
    )
    recalculate_window_summary = (
        minutes_after_open_changed
        or entry_threshold_changed
        or hold_until_close_threshold_changed
        or not st.session_state.window_summary_rows
    )

    if not recalculate_window_summary and history_df is not None and not history_df.empty:
        latest_loss_target = _find_latest_loss_target(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
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
        )
        st.session_state.window_summary_rows = summary_rows
        st.session_state.window_summary_last_loss_target = latest_loss_target
        st.session_state.window_summary_last_updated = pd.Timestamp.utcnow()
        st.session_state.window_summary_minutes_after_open = minutes_after_open
        st.session_state.window_summary_entry_threshold = entry_threshold
        st.session_state.window_summary_hold_until_close_threshold = hold_until_close_threshold

def render_dashboard():
    df, resolved_date = load_data(selected_date, files_by_date, legacy_path)
    history_df = load_all_data(files_by_date, legacy_path)
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

        header_col, jump_col = st.columns([3, 2])
        with header_col:
            st.title("Polymarket 15m BTC Monitor")
        with jump_col:
            jump_time = st.datetime_input(
                "Jump to time",
                value=jump_default,
                help=f"Jump to the {window_size}-market window that includes this time.",
            )
            if st.button("Jump", key="window_jump_button") and total_markets:
                eligible_times = [t for t in target_times if t and t <= jump_time]
                if eligible_times:
                    target_index = target_times.index(eligible_times[-1])
                else:
                    target_index = 0
                st.session_state.window_offset = max(0, total_markets - (target_index + 1))

        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
        with col_nav1:
            if st.button("Back", key="window_back_button", disabled=st.session_state.window_offset >= max_offset):
                st.session_state.window_offset = min(max_offset, st.session_state.window_offset + 1)
        with col_nav2:
            if st.button("Forward", key="window_forward_button", disabled=st.session_state.window_offset <= 0):
                st.session_state.window_offset = max(0, st.session_state.window_offset - 1)
        with col_nav3:
            if st.button("Latest", key="window_latest_button", disabled=st.session_state.window_offset == 0):
                st.session_state.window_offset = 0

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

        df_window = _resample_market_data(df_window, time_column, resample_interval, liquidity_aggregation)

        if df_window.empty:
            st.warning("No data available after resampling.")
            st.stop()

        # Get latest from windowed data
        latest = df_window.iloc[-1]

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

        df_chart["total_liq"] = df_chart["UpVol"] + df_chart["DownVol"]
        df_chart["liq_imbalance"] = np.where(
            df_chart["total_liq"] > 0,
            (df_chart["UpVol"] - df_chart["DownVol"]) / df_chart["total_liq"],
            np.nan,
        )

        df_chart["liq_imbalance_volume"] = df_chart["UpVol"] - df_chart["DownVol"]
        df_chart["liq_share"] = np.where(
            df_chart["total_liq"] > 0,
            df_chart["UpVol"] / df_chart["total_liq"],
            np.nan,
        )

        if resample_interval == "all":
            interval_seconds = _infer_interval_seconds(df_chart[time_column])
        else:
            interval_seconds = pd.Timedelta(resample_interval).total_seconds()
        momentum_window_points = max(1, int(momentum_window_seconds / interval_seconds))

        df_chart["UpPrice_Momentum"] = (
            df_chart.groupby("group")["UpPrice"]
            .transform(lambda x: x.diff().rolling(momentum_window_points, min_periods=1).mean())
        )

        latest_timestamp = df_window[time_column].max()
        history_latest_timestamp = (
            history_df[history_time_column].max()
            if history_df is not None and not history_df.empty
            else latest_timestamp
        )
        current_open = _align_market_open(history_latest_timestamp)
        _initialize_strike_rate_state(minutes_after_open, entry_threshold, hold_until_close_threshold)

        latest_up_vol = latest.get("UpVol")
        latest_down_vol = latest.get("DownVol")
        total_liquidity = latest_up_vol + latest_down_vol
        liquidity_imbalance = np.nan
        if pd.notna(total_liquidity) and total_liquidity > 0:
            liquidity_imbalance = (latest_up_vol - latest_down_vol) / total_liquidity

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            market_rows = df_window[df_window['TargetTime'] == latest['TargetTime']]
            market_start_time = market_rows[time_column].min()
            market_open_time = _align_market_open(market_start_time)
            if pd.isna(market_start_time):
                countdown_display = "N/A"
            else:
                market_end_time = market_open_time + pd.Timedelta(minutes=15)
                remaining_seconds = int((market_end_time - latest_timestamp).total_seconds())
                remaining_seconds = max(0, remaining_seconds)
                minutes_left = remaining_seconds // 60
                seconds_left = remaining_seconds % 60
                countdown_display = f"{minutes_left:02d}:{seconds_left:02d}"
            st.metric("Minutes Left (MM:SS)", countdown_display)
        with col2:
            st.metric(
                "Total Liquidity",
                _format_metric(total_liquidity, lambda v: f"{v:,.2f}"),
            )
        with col3:
            st.metric(
                "Liquidity Imbalance",
                _format_metric(liquidity_imbalance, lambda v: f"{v:.2%}"),
            )
        with col4:
            st.metric(
                "Yes (Up) Cost",
                _format_metric(latest.get("UpPrice"), lambda v: f"${v:.2f}"),
            )
        with col5:
            st.metric(
                "No (Down) Cost",
                _format_metric(latest.get("DownPrice"), lambda v: f"${v:.2f}"),
            )

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
            "up_liquidity": "rgba(34, 139, 34, 0.9)",
            "down_liquidity": "rgba(220, 20, 60, 0.85)",
            "imbalance": "rgba(128, 0, 128, 0.55)",
            "share": "rgba(255, 165, 0, 0.85)",
            "momentum": "rgba(30, 144, 255, 0.7)",
        }

        # Create Subplots with shared x-axis
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            specs=[[{}], [{}], [{"secondary_y": True}]],
            subplot_titles=(
                "Probability History",
                "Liquidity (available size at quoted price)",
                "Liquidity Imbalance / Share",
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

        liquidity_hover = (
            "Time: %{x}<br>"
            "Up liq: %{customdata[0]:,.2f}<br>"
            "Down liq: %{customdata[1]:,.2f}<br>"
            "Total liq: %{customdata[2]:,.2f}<br>"
            "Imbalance: %{customdata[3]:.2%}"
            "<extra></extra>"
        )

        liquidity_customdata = np.column_stack((
            df_chart["UpVol"],
            df_chart["DownVol"],
            df_chart["total_liq"],
            df_chart["liq_imbalance"],
        ))

        # Liquidity Chart (Row 2)
        fig.add_trace(
            go.Bar(
                x=df_chart[time_column],
                y=df_chart['UpVol'],
                name="Yes (Up) Liquidity",
                marker_color=colors["up_liquidity"],
                customdata=liquidity_customdata,
                hovertemplate=liquidity_hover,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df_chart[time_column],
                y=df_chart['DownVol'],
                name="No (Down) Liquidity",
                marker_color=colors["down_liquidity"],
                customdata=liquidity_customdata,
                hovertemplate=liquidity_hover,
            ),
            row=2,
            col=1,
        )

        # Liquidity Imbalance/Share Chart (Row 3)
        fig.add_trace(
            go.Bar(
                x=df_chart[time_column],
                y=df_chart["liq_imbalance_volume"],
                name="Liquidity Imbalance (Up - Down)",
                marker_color=colors["imbalance"],
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df_chart[time_column],
                y=df_chart["liq_share"],
                name="Up Liquidity Share",
                line=dict(color=colors["share"], width=2),
                mode=trace_mode,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=df_chart[time_column],
                y=df_chart["UpPrice_Momentum"],
                name="Up Price Momentum",
                line=dict(color=colors["momentum"], width=2, dash="dot"),
                mode=trace_mode,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

        ordered_targets = df_window["TargetTime_dt"].dropna().drop_duplicates().tolist()
        full_target_dt_order = df["TargetTime_dt"].dropna().drop_duplicates().tolist()
        trade_records = _calculate_market_trade_records(
            df_window,
            time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            target_order=full_target_dt_order,
        )
        trade_record_map = {record["target_time_dt"]: record for record in trade_records}
        entry_times = []
        entry_prices = []
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
                entry_times.append(record["entry_time"])
                entry_prices.append(record["entry_price"])

            if record["exit_time"] is not None and record["exit_price"] is not None:
                if record["exit_reason"] == "threshold":
                    exit_times.append(record["exit_time"])
                    exit_prices.append(record["exit_price"])
                elif record["exit_reason"] == "held_to_close":
                    held_times.append(record["exit_time"])
                    held_prices.append(record["exit_price"])

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
            height=1000,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Time",
            yaxis=dict(title="Probability", range=[0, 1.05]),
            yaxis2=dict(title="Liquidity (available size at quoted price)", type=liquidity_y_scale),
            yaxis3=dict(title="Liquidity Imbalance (Up - Down)", zeroline=True),
            yaxis4=dict(title="Up Liquidity Share / Momentum"),
            barmode=liquidity_bar_mode,
            xaxis=dict(rangeslider=dict(visible=False), type="date"),
            xaxis2=dict(rangeslider=dict(visible=False), type="date"),
            xaxis3=dict(rangeslider=dict(visible=True), type="date")
        )
        # Explicitly set range for xaxis1 and xaxis2 (main and shared x-axes)
        if current_range:
            fig.update_xaxes(range=current_range, row=1, col=1)
            fig.update_xaxes(range=current_range, row=2, col=1)
            fig.update_xaxes(range=current_range, row=3, col=1)

        # Enable crosshair (spike lines) across both subplots
        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='dash')
        probability_threshold = float(entry_threshold)
        minutes_threshold = pd.Timedelta(minutes=int(minutes_after_open))
        _update_strike_rate_state(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
            current_open,
        )

        strike_rate = st.session_state.strike_rate
        avg_entry_price = st.session_state.get("avg_entry_price", np.nan)
        win_rate_needed = st.session_state.get("win_rate_needed", np.nan)
        strike_sample_size = st.session_state.get("strike_sample_size")
        autotune_sample_size = st.session_state.get("autotune_sample_size")

        chart_col, gauge_col = st.columns([4, 1])
        with chart_col:
            st.plotly_chart(fig, width='stretch', config={'scrollZoom': True})
        with gauge_col:
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
            average_entry_display = f"{avg_entry_price:.2f}" if not pd.isna(avg_entry_price) else "N/A"
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
                st.metric("Average Entry", average_entry_display)
                st.metric("Win Rate Needed", win_rate_display)
                st.metric("Edge", edge_display)
            autotune_clicked = st.button(
                "Autotune",
                key="autotune_button",
                use_container_width=True,
            )
            if autotune_clicked:
                progress_container = st.empty()
                status_container = st.status("Autotuning…", expanded=True)
                progress_bar = progress_container.progress(0)
                best_result = None

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
                            history_segment="autotune",
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

        _update_window_summary_state(
            history_df,
            history_time_column,
            minutes_after_open,
            entry_threshold,
            hold_until_close_threshold,
        )

        with st.expander("Window summary"):
            summary_df = pd.DataFrame(st.session_state.window_summary_rows)
            st.dataframe(summary_df, width='stretch')

        st.caption(f"Last updated: {latest['Timestamp']}")

    else:
        st.title("Polymarket 15m BTC Monitor")
        st.warning("No data found yet. Please ensure data_logger.py is running.")


if auto_refresh:
    render_dashboard = st.fragment(run_every=refresh_interval_seconds)(render_dashboard)

render_dashboard()
