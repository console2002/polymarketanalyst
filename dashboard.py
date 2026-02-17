import hashlib
import html
import json
import logging
import numpy as np
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import re
import uuid
from autotune import run_coarse_autotune
from dashboard_metrics import (
    build_trade_pnl_records,
    summarize_drawdowns,
    summarize_profit_loss,
)
from dashboard_processing import MARKET_WINDOW_MINUTES, align_market_open, calculate_market_trade_records
from second_entry_processing import calculate_market_trade_records_with_second_entry


# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATE_FORMAT = "%d%m%Y"
CADENCE_OPTIONS = {
    "15min": 15,
    "5min": 5,
}
DEFAULT_CADENCE_KEY = "15min"
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache", "second_entry")
CACHE_SCHEMA_VERSION = 2
COARSE_AUTOTUNE_COLUMNS = [
    "run_id",
    "minutes_after_open",
    "entry_threshold",
    "hold_until_close_threshold",
    "second_entry_threshold",
    "second_entry_mode",
    "strike_rate",
    "win_rate_needed",
    "edge",
    "expectancy",
    "expected_pnl",
    "total_count",
]


def _get_cadence_autotune_config(cadence_key):
    base_config = {
        "15min": {
            "market_window_minutes": 15,
            "minutes_after_open_min": 5,
            "minutes_after_open_max": 12,
            "minutes_after_open_default": 5,
            "minutes_after_open_step": 1,
            "minutes_after_open_help": "15min cadence: choose a whole minute from 5 to 12 minutes after market open.",
            "coarse_minutes_min": 5,
            "coarse_minutes_max": 12,
            "coarse_minutes_default": (5, 12),
            "coarse_minutes_step": 2,
            "coarse_minutes_format": "%d",
        },
        "5min": {
            "market_window_minutes": 5,
            "minutes_after_open_min": 0.5,
            "minutes_after_open_max": 4.5,
            "minutes_after_open_default": 2.0,
            "minutes_after_open_step": 1 / 12,
            "minutes_after_open_help": "5min cadence: choose 5-second increments from 0:30 to 4:30 after market open.",
            "coarse_minutes_min": 0.5,
            "coarse_minutes_max": 4.5,
            "coarse_minutes_default": (0.5, 4.5),
            "coarse_minutes_step": 1 / 12,
            "coarse_minutes_format": "%.3f",
        },
    }
    config = base_config.get(cadence_key, base_config[DEFAULT_CADENCE_KEY]).copy()
    config.update(
        {
            "minutes_display_format": "%g",
            "seconds_display_format": "%d",
            "minutes_display_label": "{value} min",
            "seconds_display_label": "{value} sec",
            "coarse_entry_step": 0.05,
            "coarse_entry_bounds": (0.52, 0.80),
            "coarse_hold_step": 0.05,
            "coarse_hold_bounds": (0.52, 0.85),
            "coarse_second_entry_step": 0.05,
            "coarse_second_entry_bounds": (0.40, 0.80),
        }
    )
    return config


def _format_minutes_as_clock(minutes_value):
    total_seconds = int(round(float(minutes_value) * 60))
    minute_component, second_component = divmod(total_seconds, 60)
    return f"{minute_component}:{second_component:02d}"


def _format_minutes_for_ui(minutes_value, cadence_key):
    if cadence_key == "5min":
        return _format_minutes_as_clock(minutes_value)
    return f"{float(minutes_value):g}"


def _humanize_autotune_progress_message(message, cadence_key):
    if cadence_key != "5min" or not message:
        return message

    def _replace_minutes(match):
        minutes_value = match.group(1)
        return f"minutes_after_open={_format_minutes_for_ui(minutes_value, cadence_key)}"

    return re.sub(r"minutes_after_open=([0-9]*\.?[0-9]+)", _replace_minutes, message)


def _append_optimization_log(message, log_placeholder=None):
    if not message:
        return
    if "optimization_log_lines" not in st.session_state:
        st.session_state.optimization_log_lines = []
    st.session_state.optimization_log_lines.append(str(message))
    if log_placeholder is not None:
        _render_optimization_log_window(log_placeholder)


def _render_optimization_log_window(log_placeholder):
    if "optimization_log_lines" not in st.session_state:
        st.session_state.optimization_log_lines = []
    log_lines = st.session_state.optimization_log_lines
    if log_lines:
        log_content = "<br>".join(html.escape(line) for line in log_lines)
    else:
        log_content = "Optimization logs will appear here."
    log_placeholder.markdown(
        (
            "<div style=\"height: 220px; overflow-y: auto; border: 1px solid #d9d9d9; "
            "border-radius: 0.5rem; padding: 0.5rem; background-color: #fafafa; "
            "font-family: monospace; font-size: 0.85rem;\">"
            f"{log_content}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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


def _get_available_data_files_for_cadence(cadence_key):
    files_by_date = {}
    cadence_dir = os.path.join(SCRIPT_DIR, "data", cadence_key)
    if os.path.isdir(cadence_dir):
        for filename in os.listdir(cadence_dir):
            if not filename.endswith(".csv"):
                continue
            if filename.startswith("coarse_autotune_"):
                continue
            file_date = _parse_date_from_filename(filename)
            if file_date:
                files_by_date[file_date] = os.path.join(cadence_dir, filename)

    legacy_path = None
    if not files_by_date:
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


def _detect_market_cadence_minutes(df):
    if df is None or df.empty or "TargetTime" not in df.columns:
        return None
    target_times = pd.to_datetime(df["TargetTime"], format=TIME_FORMAT, errors="coerce").dropna()
    if target_times.empty:
        return None
    unique_targets = pd.Series(target_times.unique()).sort_values()
    if len(unique_targets) < 2:
        return None
    diffs = unique_targets.diff().dropna()
    if diffs.empty:
        return None
    diff_minutes = (diffs.dt.total_seconds() / 60).round().astype(int)
    diff_minutes = diff_minutes[diff_minutes > 0]
    if diff_minutes.empty:
        return None
    return int(diff_minutes.mode().iloc[0])


def _is_compatible_cadence(df, expected_cadence_minutes):
    detected_cadence = _detect_market_cadence_minutes(df)
    if detected_cadence is None:
        return True, detected_cadence
    return detected_cadence == int(expected_cadence_minutes), detected_cadence


def _get_file_signature(data_file):
    try:
        stat_result = os.stat(data_file)
    except FileNotFoundError:
        return None
    return data_file, stat_result.st_mtime, stat_result.st_size


@st.cache_data(show_spinner=False)
def _load_data_file_cached(data_file, modified_time, file_size, expected_cadence_minutes):
    df = pd.read_csv(data_file)
    if "timestamp_et" in df.columns:
        df = _reshape_new_style_csv(df)
    else:
        timestamp_column = None
        if "Timestamp" in df.columns:
            timestamp_column = "Timestamp"
        else:
            for candidate in ("timestamp", "time", "date", "datetime"):
                match = next((col for col in df.columns if col.lower() == candidate), None)
                if match:
                    timestamp_column = match
                    break
        if timestamp_column is None:
            raise ValueError(
                f"Missing timestamp columns in {os.path.basename(data_file)} "
                f"(found: {', '.join(df.columns)})"
            )
        df["Timestamp"] = pd.to_datetime(df[timestamp_column], format=TIME_FORMAT, errors="coerce")
        if "Timestamp_UK" in df.columns:
            df["Timestamp_UK"] = pd.to_datetime(df["Timestamp_UK"], format=TIME_FORMAT, errors="coerce")
    for column in ("UpPrice", "DownPrice"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("UpVol", "DownVol"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    cadence_ok, detected_cadence = _is_compatible_cadence(df, expected_cadence_minutes)
    df.attrs["detected_cadence_minutes"] = detected_cadence
    df.attrs["cadence_compatible"] = cadence_ok
    return df


def _load_data_file(data_file, expected_cadence_minutes):
    signature = _get_file_signature(data_file)
    if signature is None:
        raise FileNotFoundError(data_file)
    df = _load_data_file_cached(*signature, expected_cadence_minutes)
    df.attrs["data_signature"] = signature
    return df


def load_data(selected_date, files_by_date, legacy_path, expected_cadence_minutes, cadence_key):
    warnings = []
    try:
        data_file, resolved_date = _resolve_data_file(selected_date, files_by_date, legacy_path)
        if not data_file:
            return None, None, warnings
        df = _load_data_file(data_file, expected_cadence_minutes)
        cadence_ok = bool(df.attrs.get("cadence_compatible", True))
        detected_cadence = df.attrs.get("detected_cadence_minutes")
        if not cadence_ok:
            warning = (
                f"Skipping {os.path.basename(data_file)}: detected {detected_cadence}m cadence, "
                f"dashboard currently supports selected {expected_cadence_minutes}m cadence "
                f"for this view (selected cadence: {cadence_key})."
            )
            warnings.append(warning)
            logging.warning(warning)
            return None, resolved_date, warnings
        return df, resolved_date, warnings
    except FileNotFoundError:
        return None, None, warnings
    except Exception as e:  # Catch other potential errors during loading/parsing
        st.error(f"Error loading data: {e}")
        return None, None, warnings


@st.cache_data(show_spinner=False)
def _load_all_data_cached(file_signatures, expected_cadence_minutes):
    data_frames = []
    warnings = []
    for data_file, modified_time, file_size in file_signatures:
        try:
            loaded_df = _load_data_file_cached(
                data_file,
                modified_time,
                file_size,
                expected_cadence_minutes,
            )
        except (KeyError, ValueError) as exc:
            warning = f"Skipping data file {data_file}: {exc}"
            logging.warning(warning)
            warnings.append(warning)
            continue

        cadence_ok = bool(loaded_df.attrs.get("cadence_compatible", True))
        detected_cadence = loaded_df.attrs.get("detected_cadence_minutes")
        if not cadence_ok:
            warning = (
                f"Skipping {os.path.basename(data_file)}: detected {detected_cadence}m cadence, "
                f"dashboard currently supports selected {expected_cadence_minutes}m cadence "
                f"for this view."
            )
            logging.warning(warning)
            warnings.append(warning)
            continue

        data_frames.append(loaded_df)
    if not data_frames:
        return None, tuple(warnings)
    return pd.concat(data_frames, ignore_index=True), tuple(warnings)


def load_all_data(files_by_date, legacy_path, expected_cadence_minutes, cadence_key):
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
        return None, []
    df, warnings = _load_all_data_cached(tuple(file_signatures), expected_cadence_minutes)
    warnings = [f"[{cadence_key}] {warning}" for warning in warnings]
    if df is not None:
        df.attrs["data_signature"] = tuple(file_signatures)
    return df, list(warnings)


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
    market_window_minutes,
):
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "data_signature": data_signature,
        "minutes_after_open": float(minutes_after_open),
        "entry_threshold": float(entry_threshold),
        "hold_until_close_threshold": float(hold_until_close_threshold),
        "second_entry_threshold": float(second_entry_threshold),
        "second_entry_mode": str(second_entry_mode),
        "market_window_minutes": int(market_window_minutes),
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


def _default_coarse_autotune_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"coarse_autotune_{timestamp}.csv"


def _resolve_results_path(path_value):
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(SCRIPT_DIR, path_value)


def _prepare_coarse_results_df(results):
    if isinstance(results, pd.DataFrame):
        df = results.copy()
    else:
        df = pd.DataFrame(results)
    for column in COARSE_AUTOTUNE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df = df[COARSE_AUTOTUNE_COLUMNS]
    numeric_columns = [
        "minutes_after_open",
        "entry_threshold",
        "hold_until_close_threshold",
        "second_entry_threshold",
        "strike_rate",
        "win_rate_needed",
        "edge",
        "expectancy",
        "expected_pnl",
        "total_count",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["second_entry_mode"] = df["second_entry_mode"].astype(str)
    df["run_id"] = df["run_id"].astype(str)
    return df


def _select_best_coarse_result(results_df, objective):
    if results_df is None or results_df.empty:
        return None
    score_column = "expected_pnl" if objective == "expected_pnl" else "edge"
    filtered_df = results_df.dropna(subset=[score_column, "total_count"])
    filtered_df = filtered_df[filtered_df["total_count"].fillna(0) > 0]
    if filtered_df.empty:
        return None
    best_row = filtered_df.loc[filtered_df[score_column].idxmax()]
    return best_row.to_dict()


def _format_optimization_candidate_summary(result, selected_cadence):
    if not result:
        return "N/A"
    minutes_value_display = _format_minutes_for_ui(result.get("minutes_after_open"), selected_cadence)
    entry_threshold = result.get("entry_threshold")
    hold_threshold = result.get("hold_until_close_threshold")
    second_entry_threshold = result.get("second_entry_threshold")
    expected_pnl = result.get("expected_pnl")
    total_count = result.get("total_count")
    second_entry_mode = result.get("second_entry_mode", "off")
    entry_threshold_display = f"{entry_threshold:.2f}" if pd.notna(entry_threshold) else "N/A"
    hold_threshold_display = f"{hold_threshold:.2f}" if pd.notna(hold_threshold) else "N/A"
    second_entry_threshold_display = (
        f"{second_entry_threshold:.2f}" if pd.notna(second_entry_threshold) else "N/A"
    )
    expected_pnl_display = f"{expected_pnl:.2f}" if pd.notna(expected_pnl) else "N/A"
    samples_display = int(total_count) if pd.notna(total_count) else 0
    return (
        f"minutes_after_open={minutes_value_display}, "
        f"entry_threshold={entry_threshold_display}, "
        f"hold_until_close_threshold={hold_threshold_display}, "
        f"second_entry_threshold={second_entry_threshold_display}, "
        f"second_entry_mode={second_entry_mode}, "
        f"expected_pnl={expected_pnl_display}, "
        f"samples={samples_display}"
    )

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
    market_window_minutes=MARKET_WINDOW_MINUTES,
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
        market_window_minutes,
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
            market_window_minutes=market_window_minutes,
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
            market_window_minutes=market_window_minutes,
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
    market_window_minutes=MARKET_WINDOW_MINUTES,
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
        market_window_minutes,
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
        market_window_minutes=market_window_minutes,
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
        if record.get("outcome") in {"Win", "Lose"}
        and record.get("entry_price") is not None
        and record.get("exit_price") is not None
        and not pd.isna(record.get("entry_price"))
        and not pd.isna(record.get("exit_price"))
    ]
    trade_count = len(closed_records)
    wins = sum(1 for record in closed_records if record.get("outcome") == "Win")
    win_rate = (wins / trade_count * 100) if trade_count else np.nan
    pnl_values = [
        _calculate_trade_pnl_usd(record, trade_value_usd)
        for record in closed_records
    ]
    win_pnl_values = [pnl for pnl in pnl_values if pnl > 0]
    loss_pnl_values = [abs(pnl) for pnl in pnl_values if pnl <= 0]
    avg_win = (sum(win_pnl_values) / len(win_pnl_values)) if win_pnl_values else np.nan
    avg_loss = (sum(loss_pnl_values) / len(loss_pnl_values)) if loss_pnl_values else np.nan
    if not pd.isna(avg_win) and not pd.isna(avg_loss) and avg_win > 0 and avg_loss > 0:
        win_rate_needed = avg_loss / (avg_win + avg_loss) * 100
    else:
        win_rate_needed = np.nan
    edge = win_rate - win_rate_needed if not pd.isna(win_rate) and not pd.isna(win_rate_needed) else np.nan
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
    trade_value_usd,
    history_segment="strike",
    precomputed_groups=None,
    precomputed_target_order=None,
    return_dict=False,
    market_window_minutes=MARKET_WINDOW_MINUTES,
):
    trade_records = _calculate_trade_records(
        df,
        time_column,
        minutes_after_open,
        entry_threshold,
        hold_until_close_threshold,
        second_entry_mode,
        second_entry_threshold,
        market_window_minutes=market_window_minutes,
        precomputed_groups=precomputed_groups,
        precomputed_target_order=precomputed_target_order,
    )

    autotune_records, strike_records = _split_trade_records(trade_records)
    if history_segment == "autotune":
        segment_records = autotune_records
    else:
        segment_records = strike_records

    total_count = len(segment_records)
    trade_records = [record for record in segment_records if record["outcome"] in {"Win", "Lose"}]
    trade_count = len(trade_records)
    wins = sum(1 for record in trade_records if record["outcome"] == "Win")
    strike_rate = (wins / trade_count * 100) if trade_count else np.nan
    pnl_values = [
        _calculate_trade_pnl_usd(record, trade_value_usd)
        for record in trade_records
        if record["entry_price"] is not None
        and record["exit_price"] is not None
        and not pd.isna(record["entry_price"])
        and not pd.isna(record["exit_price"])
    ]
    win_pnl_values = [pnl for pnl in pnl_values if pnl > 0]
    loss_pnl_values = [abs(pnl) for pnl in pnl_values if pnl <= 0]
    avg_win = (sum(win_pnl_values) / len(win_pnl_values)) if win_pnl_values else np.nan
    avg_loss = (sum(loss_pnl_values) / len(loss_pnl_values)) if loss_pnl_values else np.nan
    entry_prices = [record["entry_price"] for record in trade_records if record["entry_price"] is not None]
    if not pd.isna(avg_win) and not pd.isna(avg_loss) and avg_win > 0 and avg_loss > 0:
        win_rate_needed = avg_loss / (avg_win + avg_loss) * 100
    else:
        win_rate_needed = np.nan
    if entry_prices:
        avg_entry_price = sum(entry_prices) / len(entry_prices)
        min_entry_price = min(entry_prices)
        max_entry_price = max(entry_prices)
    else:
        avg_entry_price = np.nan
        min_entry_price = np.nan
        max_entry_price = np.nan
        win_rate_needed = np.nan
    expectancy = (sum(pnl_values) / len(pnl_values)) if pnl_values else np.nan
    expected_pnl = sum(pnl_values) if pnl_values else np.nan
    if return_dict:
        return {
            "strike_rate": strike_rate,
            "avg_entry_price": avg_entry_price,
            "min_entry_price": min_entry_price,
            "max_entry_price": max_entry_price,
            "win_rate_needed": win_rate_needed,
            "total_count": total_count,
            "expectancy": expectancy,
            "expected_pnl": expected_pnl,
        }
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
    market_window_minutes=MARKET_WINDOW_MINUTES,
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
        market_window_minutes=market_window_minutes,
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
            pnl_usd = _calculate_trade_pnl_usd(record, trade_value_usd)
        exit_price_display = record.get("exit_price_market", record["exit_price"])

        entry_mode = (record.get("entry_mode") or "off").title()
        second_entry_time = record.get("second_entry_time")
        second_entry_price = record.get("second_entry_price")
        if entry_mode == "Off":
            second_entry_result = "Off"
        elif second_entry_time is not None and second_entry_price is not None:
            second_entry_result = "Executed"
        else:
            second_entry_result = "No pullback"

        summary_rows.append(
            {
                "TargetTime": market_group["TargetTime"].iloc[0],
                "Market Open": record["market_open"],
                "First Crossing Side": record["expected_side"] or "None",
                "Crossing Time": record["entry_time"],
                "Entry Price": record["entry_price"],
                "Second Entry Mode": entry_mode,
                "Second Entry Result": second_entry_result,
                "Second Entry Time": second_entry_time,
                "Second Entry Price": second_entry_price,
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
    market_window_minutes=MARKET_WINDOW_MINUTES,
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
        market_window_minutes=market_window_minutes,
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
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None
    if "optimization_message" not in st.session_state:
        st.session_state.optimization_message = None
    if "optimization_notice" not in st.session_state:
        st.session_state.optimization_notice = None
    if "strike_sample_size" not in st.session_state:
        st.session_state.strike_sample_size = None
    if "autotune_sample_size" not in st.session_state:
        st.session_state.autotune_sample_size = None
    if "coarse_autotune_results_df" not in st.session_state:
        st.session_state.coarse_autotune_results_df = None
    if "coarse_autotune_save_path" not in st.session_state:
        st.session_state.coarse_autotune_save_path = _default_coarse_autotune_filename()
    if "coarse_autotune_save_enabled" not in st.session_state:
        st.session_state.coarse_autotune_save_enabled = False
    if "optimization_log_lines" not in st.session_state:
        st.session_state.optimization_log_lines = []


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
    trade_value_usd,
    current_open,
    market_window_minutes=MARKET_WINDOW_MINUTES,
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
            trade_value_usd,
            history_segment="strike",
            market_window_minutes=market_window_minutes,
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
            trade_value_usd,
            history_segment="autotune",
            market_window_minutes=market_window_minutes,
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
    selected_cadence_minutes,
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
            market_window_minutes=selected_cadence_minutes,
        )
        if latest_loss_target is not None:
            last_loss_target = st.session_state.window_summary_last_loss_target
            new_loss_seen = last_loss_target is None or latest_loss_target > last_loss_target
            if new_loss_seen:
                last_updated = st.session_state.window_summary_last_updated
                now = pd.Timestamp.utcnow()
                if pd.isna(last_updated) or now - last_updated >= pd.Timedelta(
                    minutes=selected_cadence_minutes
                ):
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
            market_window_minutes=selected_cadence_minutes,
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
    selected_cadence_minutes,
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
        if pd.isna(last_updated) or now - last_updated >= pd.Timedelta(
            minutes=selected_cadence_minutes
        ):
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

def build_market_summary_table(df_window, latest, time_column, selected_cadence_minutes):
    latest_timestamp = df_window[time_column].max()
    market_rows = df_window[df_window['TargetTime'] == latest['TargetTime']]
    market_start_time = market_rows[time_column].min()
    market_open_time = align_market_open(market_start_time, selected_cadence_minutes)
    if pd.isna(market_start_time):
        countdown_display = "N/A"
    else:
        market_end_time = market_open_time + pd.Timedelta(minutes=selected_cadence_minutes)
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
    selected_cadence_minutes,
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
        if st.button(f"Zoom Last {selected_cadence_minutes}m", key='zoom_last_market_button'):
            st.session_state.zoom_mode = 'last_market'

    # Calculate range based on mode
    current_range = None
    if st.session_state.zoom_mode == 'last_market':
        end_time = df_window[time_column].max()
        start_time = end_time - pd.Timedelta(minutes=selected_cadence_minutes)
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
        market_window_minutes=selected_cadence_minutes,
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

        if record["outcome"] in {"Win", "Lose"}:
            if record["exit_reason"] == "threshold":
                outcome_text = f"{record['outcome']} (exit)"
            else:
                outcome_text = f"{record['outcome']} (close)"

            outcome_color = "#00AA00" if record["outcome"] == "Win" else "#FF0000"
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
    selected_cadence_minutes,
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
        trade_value_usd,
        current_open,
        market_window_minutes=selected_cadence_minutes,
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
        selected_cadence_minutes,
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
        selected_cadence_minutes,
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
            market_window_minutes=selected_cadence_minutes,
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
        market_window_minutes=selected_cadence_minutes,
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
        allow_compute=True,
        market_window_minutes=selected_cadence_minutes,
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
    cadence_autotune_config,
    selected_cadence,
    selected_cadence_minutes,
    precomputed_groups=None,
    precomputed_target_order=None,
    optimization_log_placeholder=None,
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
                    {"range": [50, win_rate_needed_pct], "color": "red"},
                    {"range": [win_rate_needed_pct, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": gauge_value,
                },
            },
        )
    )
    gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
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
    edge_display = (
        f"{(strike_rate - win_rate_needed):+.2f}%"
        if not pd.isna(strike_rate) and not pd.isna(win_rate_needed)
        else "N/A"
    )
    if strike_sample_size is not None and autotune_sample_size is not None:
        st.caption(f"Samples: autotune={autotune_sample_size}, strike rate={strike_sample_size}")

    metrics_table = pd.DataFrame(
        {
            "Metric": ["Average Entry", "Win Rate Needed", "Edge"],
            "Value": [average_entry_display, win_rate_display, edge_display],
        }
    )
    st.table(metrics_table)

    optimization_clicked = st.button(
        "Run Optimization",
        key="optimization_button",
        use_container_width=True,
    )

    with st.expander("Advanced optimization settings", expanded=False):
        coarse_minutes_min = cadence_autotune_config["coarse_minutes_min"]
        coarse_minutes_max = cadence_autotune_config["coarse_minutes_max"]
        coarse_minutes_default = cadence_autotune_config["coarse_minutes_default"]
        coarse_minutes_step = cadence_autotune_config["coarse_minutes_step"]
        coarse_entry_bounds = cadence_autotune_config["coarse_entry_bounds"]
        coarse_hold_bounds = cadence_autotune_config["coarse_hold_bounds"]
        coarse_second_entry_bounds = cadence_autotune_config["coarse_second_entry_bounds"]
        coarse_entry_step = cadence_autotune_config["coarse_entry_step"]
        coarse_hold_step = cadence_autotune_config["coarse_hold_step"]
        coarse_second_entry_step = cadence_autotune_config["coarse_second_entry_step"]
        coarse_minutes_format = cadence_autotune_config["coarse_minutes_format"]

        coarse_slider_label = "Minutes after open range"
        if selected_cadence == "5min":
            coarse_slider_label = "Start/end after open (m:ss)"
        coarse_minutes_range = st.slider(
            coarse_slider_label,
            min_value=coarse_minutes_min,
            max_value=coarse_minutes_max,
            value=coarse_minutes_default,
            step=coarse_minutes_step,
            format=coarse_minutes_format,
            key="coarse_minutes_after_open_range",
        )
        st.caption(
            "Optimization objective: Max expected P/L | "
            f"Minimum samples: {int(st.session_state.get('min_autotune_samples', 200))}"
        )
        coarse_entry_range = st.slider(
            "Entry threshold range",
            min_value=coarse_entry_bounds[0],
            max_value=coarse_entry_bounds[1],
            value=coarse_entry_bounds,
            step=coarse_entry_step,
            format="%.2f",
            key="coarse_entry_threshold_range",
        )
        coarse_hold_range = st.slider(
            "Hold until close threshold range",
            min_value=coarse_hold_bounds[0],
            max_value=coarse_hold_bounds[1],
            value=coarse_hold_bounds,
            step=coarse_hold_step,
            format="%.2f",
            key="coarse_hold_threshold_range",
        )
        coarse_second_entry_threshold_range = st.slider(
            "Second entry threshold range (phase 2 only)",
            min_value=coarse_second_entry_bounds[0],
            max_value=coarse_second_entry_bounds[1],
            value=coarse_second_entry_bounds,
            step=coarse_second_entry_step,
            format="%.2f",
            key="coarse_second_entry_threshold_range",
        )
        coarse_second_entry_modes = st.multiselect(
            "Second entry modes (phase 2)",
            options=("additive", "sole"),
            default=("additive", "sole"),
            key="coarse_second_entry_modes",
        )
        save_results_enabled = st.checkbox(
            "Save optimization candidates to CSV",
            key="coarse_autotune_save_enabled",
        )
        st.text_input(
            "Save path",
            key="coarse_autotune_save_path",
            help="Relative paths are saved under the dashboard directory.",
            disabled=not save_results_enabled,
        )

    def _build_filter_summary(df, minimum_samples):
        total_rows = int(len(df))
        removed_df = df[df["total_count"] < minimum_samples].copy()
        retained_count = total_rows - int(len(removed_df))
        removed_count = int(len(removed_df))
        if not removed_df.empty:
            dropped_preview = removed_df.sort_values("total_count", ascending=True).head(10).copy()
            dropped_preview = dropped_preview[
                [
                    "minutes_after_open",
                    "entry_threshold",
                    "hold_until_close_threshold",
                    "second_entry_mode",
                    "total_count",
                ]
            ]
        else:
            dropped_preview = pd.DataFrame(
                columns=[
                    "minutes_after_open",
                    "entry_threshold",
                    "hold_until_close_threshold",
                    "second_entry_mode",
                    "total_count",
                ]
            )
        return {
            "total_rows": total_rows,
            "removed_rows": removed_count,
            "retained_rows": retained_count,
            "dropped_preview": dropped_preview,
        }

    if optimization_clicked:
        st.session_state.optimization_log_lines = []
        st.session_state.optimization_notice = None
        st.session_state.optimization_filter_summaries = None
        _append_optimization_log("Starting optimization run (phase 1 + phase 2)", optimization_log_placeholder)
        if not coarse_second_entry_modes:
            st.session_state.optimization_result = None
            st.session_state.optimization_message = "Select at least one second-entry mode"
            st.session_state.coarse_autotune_results_df = None
            _append_optimization_log("Select at least one second-entry mode.", optimization_log_placeholder)
        else:
            min_total_count = int(st.session_state.get("min_autotune_samples", 200))
            progress_container = st.empty()
            status_container = st.status("Optimizing strategy (phase 1 + phase 2)…", expanded=True)
            progress_bar = progress_container.progress(0)

            with status_container:
                def _coarse_progress_callback(current_step, total_steps, message):
                    if total_steps:
                        progress_bar.progress(current_step / total_steps)
                    humanized_message = _humanize_autotune_progress_message(message, selected_cadence)
                    status_container.write(humanized_message)
                    _append_optimization_log(humanized_message, optimization_log_placeholder)

                def _phase1_metrics(df, column, minutes, threshold, hold_threshold, mode, second_entry_value):
                    return _calculate_strike_rate_metrics(
                        df,
                        column,
                        minutes,
                        threshold,
                        hold_threshold,
                        "off",
                        second_entry_value,
                        trade_value_usd,
                        history_segment="autotune",
                        precomputed_groups=precomputed_groups,
                        precomputed_target_order=precomputed_target_order,
                        return_dict=True,
                        market_window_minutes=selected_cadence_minutes,
                    )

                def _phase2_metrics(df, column, minutes, threshold, hold_threshold, mode, second_entry_value):
                    return _calculate_strike_rate_metrics(
                        df,
                        column,
                        minutes,
                        threshold,
                        hold_threshold,
                        mode,
                        second_entry_value,
                        trade_value_usd,
                        history_segment="autotune",
                        precomputed_groups=precomputed_groups,
                        precomputed_target_order=precomputed_target_order,
                        return_dict=True,
                        market_window_minutes=selected_cadence_minutes,
                    )

                run_id = f"opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                save_path_value = st.session_state.coarse_autotune_save_path
                if save_results_enabled and not save_path_value:
                    save_path_value = _default_coarse_autotune_filename()
                    st.session_state.coarse_autotune_save_path = save_path_value
                resolved_save_path = _resolve_results_path(save_path_value) if save_results_enabled else None

                # Phase 1: second entry forced off for fast scout
                phase1_caption = "Phase 1/2: scouting base entry setup with second-entry disabled"
                status_container.caption(phase1_caption)
                _append_optimization_log(phase1_caption, optimization_log_placeholder)
                phase1_results = run_coarse_autotune(
                    history_df,
                    history_time_column,
                    _phase1_metrics,
                    minutes_range=np.arange(
                        coarse_minutes_range[0],
                        coarse_minutes_range[1] + (coarse_minutes_step / 2),
                        coarse_minutes_step,
                    ),
                    entry_threshold_range=np.arange(
                        coarse_entry_range[0],
                        coarse_entry_range[1] + 0.001,
                        coarse_entry_step,
                    ),
                    hold_until_close_threshold_range=np.arange(
                        coarse_hold_range[0],
                        coarse_hold_range[1] + 0.001,
                        coarse_hold_step,
                    ),
                    second_entry_threshold_range=[second_entry_threshold],
                    modes=["off"],
                    progress_callback=_coarse_progress_callback,
                    run_id=f"{run_id}_phase1",
                    min_total_count=0,
                )
                phase1_df = _prepare_coarse_results_df(phase1_results)
                phase1_df = phase1_df.dropna(subset=["expected_pnl", "total_count"])
                phase1_filter_summary = _build_filter_summary(phase1_df, min_total_count)
                if phase1_df.empty:
                    st.session_state.optimization_result = None
                    st.session_state.optimization_message = "No viable phase-1 candidates"
                    st.session_state.coarse_autotune_results_df = None
                else:
                    phase1_eligible_df = phase1_df[phase1_df["total_count"] >= min_total_count]
                    phase1_candidates_df = phase1_eligible_df if not phase1_eligible_df.empty else phase1_df
                    if phase1_eligible_df.empty:
                        top_phase1_fallback = _select_best_coarse_result(phase1_df, "expected_pnl")
                        fallback_summary = _format_optimization_candidate_summary(
                            top_phase1_fallback,
                            selected_cadence,
                        )
                        notice = (
                            f"No phase-1 candidates met minimum sample count ({min_total_count}). "
                            "Proceeding with lower-sample candidates. "
                            f"Top fallback phase-1 candidate: {fallback_summary}"
                        )
                        st.session_state.optimization_notice = notice
                        _append_optimization_log(notice, optimization_log_placeholder)
                    top_n = min(8, len(phase1_candidates_df))
                    top_phase1 = phase1_candidates_df.nlargest(top_n, "expected_pnl")
                    candidate_pairs = sorted(
                        {
                            (float(row["minutes_after_open"]), float(row["entry_threshold"]))
                            for _, row in top_phase1.iterrows()
                        }
                    )
                    phase2_caption = (
                        f"Phase 2/2: refining {len(candidate_pairs)} top phase-1 candidates with second-entry"
                    )
                    status_container.caption(phase2_caption)
                    _append_optimization_log(phase2_caption, optimization_log_placeholder)

                    pair_set = {(round(m, 6), round(e, 2)) for m, e in candidate_pairs}

                    def _phase2_filtered_metrics(df, column, minutes, threshold, hold_threshold, mode, second_entry_value):
                        if (round(float(minutes), 6), round(float(threshold), 2)) not in pair_set:
                            return {
                                "strike_rate": np.nan,
                                "win_rate_needed": np.nan,
                                "total_count": 0,
                                "expectancy": np.nan,
                                "expected_pnl": np.nan,
                            }
                        return _phase2_metrics(
                            df,
                            column,
                            minutes,
                            threshold,
                            hold_threshold,
                            mode,
                            second_entry_value,
                        )

                    phase2_results = run_coarse_autotune(
                        history_df,
                        history_time_column,
                        _phase2_filtered_metrics,
                        minutes_range=np.array([m for m, _ in candidate_pairs]),
                        entry_threshold_range=np.array([e for _, e in candidate_pairs]),
                        hold_until_close_threshold_range=np.arange(
                            coarse_hold_range[0],
                            coarse_hold_range[1] + 0.001,
                            coarse_hold_step,
                        ),
                        second_entry_threshold_range=np.arange(
                            coarse_second_entry_threshold_range[0],
                            coarse_second_entry_threshold_range[1] + 0.001,
                            coarse_second_entry_step,
                        ),
                        modes=[_normalize_second_entry_mode(mode) for mode in coarse_second_entry_modes],
                        progress_callback=_coarse_progress_callback,
                        save_path=resolved_save_path,
                        run_id=f"{run_id}_phase2",
                        incremental_save=bool(resolved_save_path),
                        min_total_count=0,
                    )
                    results_df = _prepare_coarse_results_df(phase2_results)
                    results_df = results_df.dropna(subset=["expected_pnl", "total_count"])
                    phase2_filter_summary = _build_filter_summary(results_df, min_total_count)
                    st.session_state.optimization_filter_summaries = {
                        "min_autotune_samples": min_total_count,
                        "phase_1": phase1_filter_summary,
                        "phase_2": phase2_filter_summary,
                    }
                    summary_lines = [
                        f"Min samples filter: {min_total_count}",
                        (
                            "Phase 1 — "
                            f"evaluated={phase1_filter_summary['total_rows']}, "
                            f"removed={phase1_filter_summary['removed_rows']}, "
                            f"retained={phase1_filter_summary['retained_rows']}"
                        ),
                        (
                            "Phase 2 — "
                            f"evaluated={phase2_filter_summary['total_rows']}, "
                            f"removed={phase2_filter_summary['removed_rows']}, "
                            f"retained={phase2_filter_summary['retained_rows']}"
                        ),
                    ]
                    status_container.warning("\n\n".join(summary_lines), icon="⚠️")
                    dropped_candidates_df = phase2_filter_summary["dropped_preview"]
                    if not dropped_candidates_df.empty:
                        status_container.caption(
                            "Top dropped phase-2 candidates (lowest sample counts):"
                        )
                        status_container.dataframe(dropped_candidates_df, width='stretch', hide_index=True)
                    eligible_results_df = results_df[results_df["total_count"] >= min_total_count]
                    final_results_df = eligible_results_df if not eligible_results_df.empty else results_df
                    if eligible_results_df.empty and not results_df.empty:
                        top_phase2_fallback = _select_best_coarse_result(results_df, "expected_pnl")
                        fallback_summary = _format_optimization_candidate_summary(
                            top_phase2_fallback,
                            selected_cadence,
                        )
                        notice = (
                            f"No phase-2 candidates met minimum sample count ({min_total_count}). "
                            "Showing best lower-sample result. "
                            f"Top fallback phase-2 candidate: {fallback_summary}"
                        )
                        st.session_state.optimization_notice = notice
                        _append_optimization_log(notice, optimization_log_placeholder)
                    st.session_state.coarse_autotune_results_df = final_results_df
                    best_result = _select_best_coarse_result(final_results_df, "expected_pnl")
                    if best_result:
                        st.session_state.optimization_result = best_result
                        st.session_state.optimization_message = None
                    else:
                        st.session_state.optimization_result = None
                        st.session_state.optimization_message = "No viable phase-2 candidates"

            progress_container.empty()
            status_container.update(state="complete", label="Optimization complete")
            _append_optimization_log("Optimization complete.", optimization_log_placeholder)

    optimization_notice = st.session_state.get("optimization_notice")
    if optimization_notice:
        st.info(optimization_notice)

    if st.session_state.optimization_result:
        result = st.session_state.optimization_result
        minutes_value_display = _format_minutes_for_ui(result["minutes_after_open"], selected_cadence)
        expected_pnl_display = (
            f"{result['expected_pnl']:.2f}"
            if result.get("expected_pnl") is not None and not pd.isna(result.get("expected_pnl"))
            else "N/A"
        )
        st.caption(
            "Best optimized setup: "
            f"minutes_after_open={minutes_value_display}, "
            f"entry_threshold={result['entry_threshold']:.2f}, "
            f"hold_until_close_threshold={result['hold_until_close_threshold']:.2f}, "
            f"second_entry_threshold={result['second_entry_threshold']:.2f}, "
            f"second_entry_mode={result['second_entry_mode']}, "
            f"expected_pnl={expected_pnl_display}, "
            f"samples={int(result['total_count']) if not pd.isna(result['total_count']) else 0}"
        )
        result_sample_count = int(result["total_count"]) if not pd.isna(result.get("total_count")) else 0
        min_total_count = int(st.session_state.get("min_autotune_samples", 200))
        if result_sample_count <= (min_total_count + 20):
            st.warning(
                (
                    "⚠️ Caution: best result is near the minimum sample threshold "
                    f"({result_sample_count} samples vs minimum {min_total_count})."
                )
            )
    elif st.session_state.optimization_message:
        st.caption(st.session_state.optimization_message)

    render_coarse_results_explorer(
        st.session_state.get("coarse_autotune_results_df"),
        "expected_pnl",
        selected_cadence,
    )

def render_coarse_results_explorer(results_df, objective, selected_cadence):
    if results_df is None or results_df.empty:
        return
    st.subheader("Optimization results explorer")
    filter_summaries = st.session_state.get("optimization_filter_summaries")
    if filter_summaries:
        min_samples = int(filter_summaries.get("min_autotune_samples", 0))
        phase_1 = filter_summaries.get("phase_1", {})
        phase_2 = filter_summaries.get("phase_2", {})
        st.caption(
            "Sample filtering summary "
            f"(min={min_samples}) — "
            f"Phase 1: evaluated={phase_1.get('total_rows', 0)}, "
            f"removed={phase_1.get('removed_rows', 0)}, retained={phase_1.get('retained_rows', 0)}; "
            f"Phase 2: evaluated={phase_2.get('total_rows', 0)}, "
            f"removed={phase_2.get('removed_rows', 0)}, retained={phase_2.get('retained_rows', 0)}"
        )

        dropped_candidates_df = phase_2.get("dropped_preview")
        if isinstance(dropped_candidates_df, pd.DataFrame) and not dropped_candidates_df.empty:
            st.caption("Dropped phase-2 candidates preview")
            st.dataframe(dropped_candidates_df, width='stretch', hide_index=True)
    objective_column = "expected_pnl" if objective == "expected_pnl" else "edge"
    minutes_values = sorted(
        {
            float(value)
            for value in results_df["minutes_after_open"].dropna().unique().tolist()
        }
    )
    mode_values = sorted(
        {str(value) for value in results_df["second_entry_mode"].dropna().unique().tolist()}
    )
    minutes_selection = st.multiselect(
        "Filter minutes after open"
        if selected_cadence != "5min"
        else "Filter start after open (m:ss)",
        options=minutes_values,
        default=minutes_values,
        format_func=lambda value: _format_minutes_for_ui(value, selected_cadence),
        key=f"coarse_results_minutes_filter_{selected_cadence}",
    )
    mode_selection = st.multiselect(
        "Filter second entry modes",
        options=mode_values,
        default=mode_values,
        key="coarse_results_mode_filter",
    )
    filtered_df = results_df.copy()
    if minutes_selection:
        filtered_df = filtered_df[filtered_df["minutes_after_open"].isin(minutes_selection)]
    if mode_selection:
        filtered_df = filtered_df[filtered_df["second_entry_mode"].isin(mode_selection)]
    filtered_df = filtered_df.dropna(
        subset=[
            "entry_threshold",
            "hold_until_close_threshold",
            objective_column,
        ]
    )
    if filtered_df.empty:
        st.info("No optimization results match the selected filters.")
        return

    filtered_df = filtered_df.copy()
    filtered_df["minutes_after_open_label"] = filtered_df["minutes_after_open"].apply(
        lambda value: _format_minutes_for_ui(value, selected_cadence)
    )
    facet_col = "minutes_after_open_label" if len(minutes_selection) > 1 else None
    facet_row = "second_entry_mode" if len(mode_selection) > 1 else None
    hover_columns = [
        "minutes_after_open_label",
        "second_entry_mode",
        "second_entry_threshold",
        "strike_rate",
        "win_rate_needed",
        "edge",
        "expectancy",
        "expected_pnl",
        "total_count",
    ]
    fig = px.scatter(
        filtered_df,
        x="entry_threshold",
        y="hold_until_close_threshold",
        color=objective_column,
        facet_col=facet_col,
        facet_row=facet_row,
        hover_data=hover_columns,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=450 + 200 * max(0, len(mode_selection) - 1),
        xaxis_title="Entry threshold",
        yaxis_title="Hold until close threshold",
        coloraxis_colorbar=dict(title=objective_column.replace("_", " ").title()),
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown("Raw data")
    display_df = filtered_df.sort_values(
        by=[objective_column],
        ascending=False,
    )
    st.dataframe(display_df, width="stretch")


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
    cadence_key = st.sidebar.selectbox(
        "Market cadence",
        options=tuple(CADENCE_OPTIONS.keys()),
        index=tuple(CADENCE_OPTIONS.keys()).index(DEFAULT_CADENCE_KEY),
        help="Choose which cadence folder to load dated CSV files from.",
    )
    expected_cadence_minutes = CADENCE_OPTIONS[cadence_key]
    cadence_autotune_config = _get_cadence_autotune_config(cadence_key)
    active_cadence_dir = os.path.join(SCRIPT_DIR, "data", cadence_key)
    st.sidebar.caption(f"Active source: `{active_cadence_dir}`")
    selected_cadence_minutes = CADENCE_OPTIONS[cadence_key]

    minutes_after_open = st.sidebar.number_input(
        "Minutes after open",
        min_value=cadence_autotune_config["minutes_after_open_min"],
        max_value=cadence_autotune_config["minutes_after_open_max"],
        value=cadence_autotune_config["minutes_after_open_default"],
        step=cadence_autotune_config["minutes_after_open_step"],
        format=cadence_autotune_config["minutes_display_format"],
        help=cadence_autotune_config["minutes_after_open_help"],
    )
    st.sidebar.caption(
        "Market window: "
        + cadence_autotune_config["minutes_display_label"].format(
            value=cadence_autotune_config["market_window_minutes"]
        )
    )
    min_autotune_samples = st.sidebar.number_input(
        "Minimum optimization samples",
        min_value=50,
        max_value=5000,
        value=int(st.session_state.get("min_autotune_samples", 200)),
        step=25,
        help="Candidates below this count are excluded from optimization ranking.",
    )
    st.session_state.min_autotune_samples = int(min_autotune_samples)

    selected_date_state_key = f"selected_date_{cadence_key}"
    files_by_date, legacy_path = _get_available_data_files_for_cadence(cadence_key)
    available_dates = sorted(files_by_date)
    latest_available_date = max(available_dates) if available_dates else None
    min_available_date = min(available_dates) if available_dates else None

    if available_dates:
        default_selected_date = st.session_state.get(selected_date_state_key, latest_available_date)
        if default_selected_date not in available_dates:
            default_selected_date = latest_available_date
        selected_date = st.sidebar.date_input(
            "Data date",
            value=default_selected_date,
            min_value=min_available_date,
            max_value=latest_available_date,
            key=selected_date_state_key,
            help="Select a historical data file by date. Defaults to the latest available file.",
        )
    else:
        selected_date = None
        st.sidebar.caption(
            "No dated CSV files found for selected cadence; falling back to "
            "`market_data.csv` only if it exists."
        )

    jump_container = st.sidebar.container()

    progress_container = st.empty()
    status_container = st.status("Loading dashboard data…", expanded=True)
    progress_bar = progress_container.progress(0, text="Loading data files…")

    df, resolved_date, load_warnings = load_data(
        selected_date, files_by_date, legacy_path, selected_cadence_minutes, cadence_key
    )
    progress_bar.progress(0.25, text="Loaded selected data file.")
    history_df, history_warnings = load_all_data(
        files_by_date, legacy_path, selected_cadence_minutes, cadence_key
    )
    progress_bar.progress(0.45, text="Loaded historical data.")
    if history_df is None or history_df.empty:
        history_df = df
    if selected_date and resolved_date and selected_date != resolved_date:
        st.info(
            f"No data file found for {selected_date.strftime('%Y-%m-%d')}. "
            f"Showing {resolved_date.strftime('%Y-%m-%d')} instead."
        )

    all_load_warnings = list(load_warnings) + list(history_warnings)
    if all_load_warnings:
        unique_warnings = list(dict.fromkeys(all_load_warnings))
        for warning in unique_warnings:
            st.warning(warning)

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
        current_open = align_market_open(history_latest_timestamp, selected_cadence_minutes)
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
            selected_cadence_minutes,
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
            st.caption("Optimization log")
            optimization_log_placeholder = st.empty()
            _render_optimization_log_window(optimization_log_placeholder)
        with header_cols[1]:
            render_strike_rate_section(
                summary_state,
                history_df,
                history_time_column,
                second_entry_mode,
                second_entry_threshold,
                cadence_autotune_config,
                cadence_key,
                selected_cadence_minutes,
                precomputed_groups=history_market_groups,
                precomputed_target_order=history_target_order,
                optimization_log_placeholder=optimization_log_placeholder,
            )
        with header_cols[2]:
            market_summary_table = build_market_summary_table(
                probability_window["df_window"],
                probability_window["latest"],
                time_column,
                selected_cadence_minutes,
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
            selected_cadence_minutes,
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
