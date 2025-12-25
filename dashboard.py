import numpy as np
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
import re


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
win_threshold = st.sidebar.number_input(
    "Win threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.99,
    step=0.01,
    format="%.2f",
)
resample_interval = st.sidebar.selectbox(
    "Resample interval",
    options=("5s", "15s", "30s", "60s"),
    index=0,
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

def load_data(selected_date, files_by_date, legacy_path):
    try:
        data_file, resolved_date = _resolve_data_file(selected_date, files_by_date, legacy_path)
        if not data_file:
            return None, None
        df = pd.read_csv(data_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=TIME_FORMAT)
        if 'Timestamp_UK' in df.columns:
            df['Timestamp_UK'] = pd.to_datetime(df['Timestamp_UK'], format=TIME_FORMAT)
        return df, resolved_date
    except FileNotFoundError:
        return None, None
    except Exception as e: # Catch other potential errors during loading/parsing
        st.error(f"Error loading data: {e}")
        return None, None

# Top-row controls
files_by_date, legacy_path = _get_available_data_files()
available_dates = sorted(files_by_date)
latest_available_date = max(available_dates) if available_dates else None
min_available_date = min(available_dates) if available_dates else None

col_top1, col_top2 = st.columns(2)
with col_top1:
    if st.button('Refresh Data', key='refresh_data_button', width='stretch'):
        st.rerun()
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
    if df.empty or not interval:
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

df, resolved_date = load_data(selected_date, files_by_date, legacy_path)
if selected_date and resolved_date and selected_date != resolved_date:
    st.info(
        f"No data file found for {selected_date.strftime('%Y-%m-%d')}. "
        f"Showing {resolved_date.strftime('%Y-%m-%d')} instead."
    )

if df is not None and not df.empty:
    time_options = ["Polymarket Time (ET)"]
    if "Timestamp_UK" in df.columns:
        time_options.append("UK Time")
    time_axis = st.sidebar.selectbox(
        "Chart time axis",
        options=tuple(time_options),
        index=0,
        help="Switch the chart between Polymarket (ET) and UK timestamps.",
    )
    time_column = "Timestamp" if time_axis == "Polymarket Time (ET)" else "Timestamp_UK"
    if time_column not in df.columns:
        st.warning("UK timestamps are not available in this data file.")
        time_column = "Timestamp"

    df = df.sort_values(time_column)
    df['TargetTime_dt'] = pd.to_datetime(df['TargetTime'], format=TIME_FORMAT, errors='coerce')

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
            st.rerun()

    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("Back", key="window_back_button", disabled=st.session_state.window_offset >= max_offset):
            st.session_state.window_offset = min(max_offset, st.session_state.window_offset + 1)
            st.rerun()
    with col_nav2:
        if st.button("Forward", key="window_forward_button", disabled=st.session_state.window_offset <= 0):
            st.session_state.window_offset = max(0, st.session_state.window_offset - 1)
            st.rerun()
    with col_nav3:
        if st.button("Latest", key="window_latest_button", disabled=st.session_state.window_offset == 0):
            st.session_state.window_offset = 0
            st.rerun()

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

    interval_seconds = pd.Timedelta(resample_interval).total_seconds()
    momentum_window_points = max(1, int(momentum_window_seconds / interval_seconds))

    df_chart["UpPrice_Momentum"] = (
        df_chart.groupby("group")["UpPrice"]
        .transform(lambda x: x.diff().rolling(momentum_window_points, min_periods=1).mean())
    )

    latest_up_vol = latest.get("UpVol")
    latest_down_vol = latest.get("DownVol")
    total_liquidity = latest_up_vol + latest_down_vol
    liquidity_imbalance = np.nan
    if pd.notna(total_liquidity) and total_liquidity > 0:
        liquidity_imbalance = (latest_up_vol - latest_down_vol) / total_liquidity

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        latest_timestamp = df_window[time_column].max()
        market_rows = df_window[df_window['TargetTime'] == latest['TargetTime']]
        market_start_time = market_rows[time_column].min()
        if pd.isna(market_start_time):
            countdown_display = "N/A"
        else:
            market_end_time = market_start_time + pd.Timedelta(minutes=15)
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
            st.rerun()
    with col_z2:
        if st.button("Zoom Last 15m", key='zoom_15m_button'):
            st.session_state.zoom_mode = 'last_15m'
            st.rerun()

    # Calculate range based on mode
    current_range = None
    if st.session_state.zoom_mode == 'last_15m':
        end_time = df_window[time_column].max()
        start_time = end_time - pd.Timedelta(minutes=15)
        current_range = [start_time, end_time]

    trace_mode = "lines+markers" if show_markers else "lines"
    colors = {
        "up": "rgba(34, 139, 34, 0.7)",
        "down": "rgba(220, 20, 60, 0.5)",
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
            marker_color=colors["up"],
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
            marker_color=colors["down"],
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

    minutes_threshold = pd.Timedelta(minutes=int(minutes_after_open))
    probability_threshold = float(entry_threshold)
    expected_winner_traces = []

    for _, market_group in df_window.groupby('TargetTime', sort=False):
        market_group = market_group.sort_values(time_column)
        if market_group.empty:
            continue
        market_open = market_group[time_column].iloc[0]
        open_threshold_time = market_open + minutes_threshold

        add_vline_all_rows(
            fig,
            open_threshold_time,
            line_width=1,
            line_dash="solid",
            line_color="rgba(200, 200, 200, 0.4)",
        )

        eligible = market_group[market_group[time_column] >= open_threshold_time].copy()
        if eligible.empty:
            continue

        def find_crossing(series):
            above = series >= probability_threshold
            crossings = above & ~above.shift(fill_value=False)
            if crossings.any():
                return crossings[crossings].index[0]
            return None

        up_cross_index = find_crossing(eligible['UpPrice'])
        down_cross_index = find_crossing(eligible['DownPrice'])
        candidates = []
        if up_cross_index is not None:
            candidates.append(("up", eligible.loc[up_cross_index, time_column], eligible.loc[up_cross_index, 'UpPrice']))
        if down_cross_index is not None:
            candidates.append(("down", eligible.loc[down_cross_index, time_column], eligible.loc[down_cross_index, 'DownPrice']))

        if candidates:
            expected_side, cross_time, cross_value = min(candidates, key=lambda item: item[1])
            expected_winner_traces.append(
                go.Scatter(
                    x=[cross_time],
                    y=[cross_value],
                    mode='markers+text',
                    marker=dict(color='#1E90FF', size=9),
                    text=["expected winner"],
                    textposition='top center',
                    textfont=dict(size=10, color='#1E90FF'),
                    showlegend=False,
                )
            )
            final_up_values = market_group['UpPrice'].replace(0, np.nan).dropna()
            final_down_values = market_group['DownPrice'].replace(0, np.nan).dropna()
            if final_up_values.empty or final_down_values.empty:
                continue
            final_up = final_up_values.iloc[-1]
            final_down = final_down_values.iloc[-1]
            win = final_up >= win_threshold if expected_side == "up" else final_down >= win_threshold
            fig.add_annotation(
                x=market_group[time_column].iloc[-1],
                y=1.03,
                text="Win" if win else "Lose",
                showarrow=False,
                font=dict(color="#00AA00" if win else "#FF0000", size=16),
                row=1,
                col=1,
            )

    for trace in expected_winner_traces:
        fig.add_trace(trace, row=1, col=1)

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
    
    st.plotly_chart(fig, width='stretch', config={'scrollZoom': True})

    summary_rows = []
    probability_threshold = float(entry_threshold)
    minutes_threshold = pd.Timedelta(minutes=int(minutes_after_open))
    if total_markets:
        target_order = active_targets
    else:
        target_order = df_window['TargetTime_dt'].dropna().drop_duplicates().tolist()

    for target_time in target_order:
        market_group = df_window[df_window['TargetTime_dt'] == target_time].sort_values(time_column)
        if market_group.empty:
            continue
        market_open = market_group[time_column].iloc[0]
        eligible = market_group[market_group[time_column] >= market_open + minutes_threshold].copy()

        def find_crossing(series):
            above = series >= probability_threshold
            crossings = above & ~above.shift(fill_value=False)
            if crossings.any():
                return crossings[crossings].index[0]
            return None

        expected_side = None
        cross_time = None
        cross_value = None
        if not eligible.empty:
            up_cross_index = find_crossing(eligible['UpPrice'])
            down_cross_index = find_crossing(eligible['DownPrice'])
            candidates = []
            if up_cross_index is not None:
                candidates.append(("Up", eligible.loc[up_cross_index, time_column], eligible.loc[up_cross_index, 'UpPrice']))
            if down_cross_index is not None:
                candidates.append(("Down", eligible.loc[down_cross_index, time_column], eligible.loc[down_cross_index, 'DownPrice']))
            if candidates:
                expected_side, cross_time, cross_value = min(candidates, key=lambda item: item[1])

        final_up_values = market_group['UpPrice'].replace(0, np.nan).dropna()
        final_down_values = market_group['DownPrice'].replace(0, np.nan).dropna()
        final_up = final_up_values.iloc[-1] if not final_up_values.empty else np.nan
        final_down = final_down_values.iloc[-1] if not final_down_values.empty else np.nan

        outcome = "N/A"
        if expected_side == "Up" and pd.notna(final_up):
            outcome = "Win" if final_up >= win_threshold else "Lose"
        elif expected_side == "Down" and pd.notna(final_down):
            outcome = "Win" if final_down >= win_threshold else "Lose"

        total_liq_series = market_group['UpVol'] + market_group['DownVol']
        up_price_series = market_group['UpPrice'].replace(0, np.nan)
        summary_rows.append(
            {
                "TargetTime": market_group['TargetTime'].iloc[0],
                "Market Open": market_open,
                "First Crossing Side": expected_side or "None",
                "Crossing Time": cross_time,
                "Crossing Price": cross_value,
                "Final UpPrice": final_up,
                "Final DownPrice": final_down,
                "Win/Lose": outcome,
                "Mean Total Liquidity": total_liq_series.mean(),
                "Max Total Liquidity": total_liq_series.max(),
                "Max UpPrice": up_price_series.max(),
                "Min UpPrice": up_price_series.min(),
            }
        )

    with st.expander("Window summary"):
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, width='stretch')
    
    st.caption(f"Last updated: {latest['Timestamp']}")

else:
    st.title("Polymarket 15m BTC Monitor")
    st.warning("No data found yet. Please ensure data_logger.py is running.")
    
# --- Auto-refresh logic (periodic check) ---
if auto_refresh:
    # A short sleep to create a periodic refresh effect.
    # This will cause the Streamlit app to refresh after this delay.
    # Be aware that this will block the Streamlit server for this duration,
    # and might make Ctrl+C less responsive for longer sleep times.
    time.sleep(refresh_interval_seconds)
    st.rerun() # Explicitly rerun to ensure a full page refresh
