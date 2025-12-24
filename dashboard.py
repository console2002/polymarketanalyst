import numpy as np
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
import pytz


# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(__file__)
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATE_FORMAT = "%d%m%Y"
TIMEZONE_ET = pytz.timezone("US/Eastern")
TIMEZONE_UK = pytz.timezone("Europe/London")


def _get_latest_data_file():
    candidates = []
    for filename in os.listdir(SCRIPT_DIR):
        if filename.endswith(".csv") and len(filename) == 12 and filename[:8].isdigit():
            try:
                file_date = datetime.datetime.strptime(filename[:8], DATE_FORMAT).date()
            except ValueError:
                continue
            candidates.append((file_date, os.path.join(SCRIPT_DIR, filename)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    legacy_path = os.path.join(SCRIPT_DIR, "market_data.csv")
    if os.path.exists(legacy_path):
        return legacy_path
    return None


st.set_page_config(page_title="Polymarket BTC Monitor", layout="wide")

def load_data():
    try:
        data_file = _get_latest_data_file()
        if not data_file:
            return None
        df = pd.read_csv(data_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=TIME_FORMAT)
        if 'Timestamp_UK' in df.columns:
            df['Timestamp_UK'] = pd.to_datetime(df['Timestamp_UK'], format=TIME_FORMAT)
        return df
    except FileNotFoundError:
        return None
    except Exception as e: # Catch other potential errors during loading/parsing
        st.error(f"Error loading data: {e}")
        return None

# Top-row controls
col_top1, col_top2, col_top3 = st.columns(3)
with col_top1:
    if st.button('Refresh Data', key='refresh_data_button', width="stretch"):
        st.rerun()
    auto_refresh = st.checkbox("Auto-refresh", value=True)
with col_top2:
    smoothing_window = st.selectbox(
        "Derivative Smoothing Window (Moving Average)",
        options=list(range(1, 31)),
        index=0,
        help="Set the window size for the moving average applied to the derivative to smooth out zig-zag. A value of 1 means no smoothing.",
        width="stretch",
    )
with col_top3:
    lookback_period = st.number_input(
        "Lookback Period (Markets)",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="Number of markets to display in the window, including the current one.",
        width="stretch",
    )

df = load_data()

if df is not None and not df.empty:
    time_options = ["Polymarket Time (ET)"]
    if "Timestamp_UK" in df.columns:
        time_options.append("UK Time")
    time_axis = st.selectbox(
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
    target_time_uk = df['TargetTime_dt'].dt.tz_localize(TIMEZONE_UK, nonexistent="shift_forward", ambiguous="NaT")
    target_time_et = target_time_uk.dt.tz_convert(TIMEZONE_ET).dt.tz_localize(None)
    target_time_uk = target_time_uk.dt.tz_localize(None)

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
    
    # Treat 0 prices as gaps
    df_chart.loc[df_chart['UpPrice'] == 0, 'UpPrice'] = np.nan
    df_chart.loc[df_chart['DownPrice'] == 0, 'DownPrice'] = np.nan

    # Calculate numerical derivative of UpPrice
    # Calculate diff only within each group (market segment)
    df_chart['UpPrice_diff'] = df_chart.groupby('group')['UpPrice'].diff()
    df_chart['Timestamp_diff_seconds'] = df_chart.groupby('group')[time_column].diff().dt.total_seconds()
    
    # Avoid division by zero and handle cases where Timestamp_diff_seconds might be 0 or NaN
    df_chart['UpPrice_Derivative'] = df_chart['UpPrice_diff'] / df_chart['Timestamp_diff_seconds']
    # Filter out infinite values that might result from division by very small numbers close to zero
    df_chart.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Apply moving average to smooth the derivative
    # Using .transform() with apply allows for grouping and then rejoining the result to the original DataFrame
    if smoothing_window > 1:
        df_chart['UpPrice_Derivative'] = df_chart.groupby('group')['UpPrice_Derivative'].transform(
            lambda x: x.rolling(window=smoothing_window, min_periods=1, center=True).mean()
        )
    

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_time = latest['TargetTime_dt']
        if pd.isna(target_time):
            countdown_display = "N/A"
        else:
            latest_timestamp = df_window[time_column].max()
            if time_column == "Timestamp":
                target_time_value = target_time_et.loc[latest.name]
            else:
                target_time_value = target_time_uk.loc[latest.name]
            remaining_seconds = int((target_time_value - latest_timestamp).total_seconds())
            remaining_seconds = max(0, remaining_seconds)
            minutes_left = remaining_seconds // 60
            seconds_left = remaining_seconds % 60
            countdown_display = f"{minutes_left:02d}:{seconds_left:02d}"
        st.metric("Minutes Left (MM:SS)", countdown_display)
    with col3:
        st.metric("Yes (Up) Cost", f"${latest['UpPrice']:.2f}")
    with col4:
        st.metric("No (Down) Cost", f"${latest['DownPrice']:.2f}")


    
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

    # Create Subplots with shared x-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=("Probability History", "Volume", "Up Price Derivative"))

    # Probability Chart (Row 1)
    fig.add_trace(go.Scatter(x=df_chart[time_column], y=df_chart['UpPrice'], name="Yes (Up)", line=dict(color='#00FF00', width=2), mode='lines+markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart[time_column], y=df_chart['DownPrice'], name="No (Down)", line=dict(color='rgba(255, 0, 0, 0.3)', dash='dash', width=2), mode='lines+markers'), row=1, col=1)
    
    # Volume Chart (Row 2)
    fig.add_trace(go.Scatter(x=df_chart[time_column], y=df_chart['UpVol'], name="Yes (Up) Volume", line=dict(color='#00FF00', width=2), mode='lines+markers'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_chart[time_column], y=df_chart['DownVol'], name="No (Down) Volume", line=dict(color='rgba(255, 0, 0, 0.3)', dash='dash', width=2), mode='lines+markers'), row=2, col=1)

    # Derivative Chart (Row 3)
    fig.add_trace(go.Scatter(x=df_chart[time_column], y=df_chart['UpPrice_Derivative'], name="Up Price Derivative", line=dict(color='#FFA500', width=2), mode='lines+markers'), row=3, col=1)

    five_min_threshold = pd.Timedelta(minutes=5)
    probability_threshold = 0.6
    expected_winner_traces = []

    for _, market_group in df_window.groupby('TargetTime', sort=False):
        market_group = market_group.sort_values(time_column)
        if market_group.empty:
            continue
        market_open = market_group[time_column].iloc[0]
        five_min_mark = market_open + five_min_threshold

        fig.add_vline(x=five_min_mark, line_width=1, line_dash="solid", line_color="rgba(200, 200, 200, 0.4)", row=1, col=1)
        fig.add_vline(x=five_min_mark, line_width=1, line_dash="solid", line_color="rgba(200, 200, 200, 0.4)", row=2, col=1)
        fig.add_vline(x=five_min_mark, line_width=1, line_dash="solid", line_color="rgba(200, 200, 200, 0.4)", row=3, col=1)

        eligible = market_group[market_group[time_column] >= five_min_mark].copy()
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
            win = final_up >= 0.99 if expected_side == "up" else final_down >= 0.99
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
        fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="gray", row=1, col=1)
        fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="gray", row=3, col=1)
    
    # Update Layout
    fig.update_layout(
        height=1000,
        hovermode="x unified",
        xaxis_title="Time",
        yaxis=dict(title="Probability", range=[0, 1.05]),
        yaxis2=dict(title="Volume"),
        yaxis3=dict(title="Rate of Change"),
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
    
    st.plotly_chart(fig, width="stretch", config={'scrollZoom': True})
    
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
    time.sleep(1) # Refresh every 1 second
    st.rerun() # Explicitly rerun to ensure a full page refresh
