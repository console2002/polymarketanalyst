import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os

DATA_FILE = "market_data.csv"

st.set_page_config(page_title="Polymarket BTC Monitor", layout="wide")

st.title("Polymarket 15m BTC Monitor")

def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except FileNotFoundError:
        return None

# Auto-refresh logic (Top of page)
col_top1, col_top2 = st.columns([1, 4])
with col_top1:
    if st.button('Refresh Data'):
        st.rerun()
with col_top2:
    auto_refresh = st.checkbox("Auto-refresh", value=True)

df = load_data()

if df is not None and not df.empty:
    # Get latest from raw data
    latest = df.iloc[-1]
    
    # Process data for charts (add gaps between different markets)
    df_chart = df.copy().sort_values('Timestamp')
    df_chart['group'] = (df_chart['TargetTime'] != df_chart['TargetTime'].shift()).cumsum()
    
    segments = []
    for _, group in df_chart.groupby('group'):
        segments.append(group)
        # Add gap row
        gap_row = group.iloc[[-1]].copy()
        gap_row['Timestamp'] += pd.Timedelta(seconds=1) 
        # Set values to None to break the line
        for col in ['Strike', 'CurrentPrice', 'UpPrice', 'DownPrice']:
            gap_row[col] = None
        segments.append(gap_row)
    
    df_chart = pd.concat(segments).reset_index(drop=True)
    
    # Treat 0 prices as gaps
    df_chart.loc[df_chart['UpPrice'] == 0, 'UpPrice'] = None
    df_chart.loc[df_chart['DownPrice'] == 0, 'DownPrice'] = None
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Strike Price", f"${latest['Strike']:,.2f}")
    with col2:
        st.metric("Current Price", f"${latest['CurrentPrice']:,.2f}", 
                  delta=f"{latest['CurrentPrice'] - latest['Strike']:.2f}")
    with col3:
        st.metric("Yes (Up) Cost", f"${latest['UpPrice']:.3f}")
    with col4:
        st.metric("No (Down) Cost", f"${latest['DownPrice']:.3f}")


    
    # Initialize zoom mode
    if 'zoom_mode' not in st.session_state:
        st.session_state.zoom_mode = None

    # Zoom Controls
    col_z1, col_z2 = st.columns([1, 10])
    with col_z1:
        if st.button("Reset Zoom"):
            st.session_state.zoom_mode = None
            st.rerun()
    with col_z2:
        if st.button("Zoom Last 15m"):
            st.session_state.zoom_mode = 'last_15m'
            st.rerun()

    # Calculate range based on mode
    current_range = None
    if st.session_state.zoom_mode == 'last_15m':
        end_time = df['Timestamp'].max()
        start_time = end_time - pd.Timedelta(minutes=15)
        current_range = [start_time, end_time]

    # Create Subplots with shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=("BTC Price vs Strike", "Probability History"))

    # Price Chart (Row 1)
    fig.add_trace(go.Scatter(x=df_chart['Timestamp'], y=df_chart['CurrentPrice'], name="Current Price", mode='lines+markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart['Timestamp'], y=df_chart['Strike'], name="Strike Price", line=dict(dash='dash')), row=1, col=1)

    # Probability Chart (Row 2)
    fig.add_trace(go.Scatter(x=df_chart['Timestamp'], y=df_chart['UpPrice'], name="Yes (Up)", line=dict(color='#00FF00', width=2), mode='lines+markers'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_chart['Timestamp'], y=df_chart['DownPrice'], name="No (Down)", line=dict(color='rgba(255, 0, 0, 0.3)', dash='dash', width=2), mode='lines+markers'), row=2, col=1)
    
    # Add vertical lines for market transitions to both plots
    # Add vertical lines for market transitions to both plots
    # Identify where TargetTime changes
    transitions = df.loc[df['TargetTime'].shift() != df['TargetTime'], 'Timestamp'].iloc[1:]
    
    for t in transitions:
        fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="gray", row=1, col=1)
        fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Update Layout
    fig.update_layout(
        height=800,
        hovermode="x unified",
        xaxis2_title="Time", # x-axis title on the bottom plot
        yaxis_title="Price (USD)", # Row 1 y-axis
        yaxis2_title="Probability", # Row 2 y-axis
        yaxis2=dict(range=[0, 1]),
        xaxis=dict(rangeslider=dict(visible=False)), # Hide slider on top plot if present
        xaxis2=dict(rangeslider=dict(visible=True), type="date", range=current_range)
    )

    # Enable crosshair (spike lines) across both subplots
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='dash')
    
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    
    st.caption(f"Last updated: {latest['Timestamp']}")

else:
    st.warning("No data found yet. Please ensure data_logger.py is running.")
    


if auto_refresh:
    # Get current modification time
    try:
        last_mtime = os.path.getmtime(DATA_FILE)
    except FileNotFoundError:
        last_mtime = 0
        
    while True:
        time.sleep(1)
        try:
            current_mtime = os.path.getmtime(DATA_FILE)
            if current_mtime > last_mtime:
                st.rerun()
        except FileNotFoundError:
            pass
