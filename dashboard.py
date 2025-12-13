import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

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

# Auto-refresh logic
if st.button('Refresh Data'):
    st.rerun()

df = load_data()

if df is not None and not df.empty:
    # Get latest
    latest = df.iloc[-1]
    
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

    st.subheader("Price History")
    
    # Price Chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df['Timestamp'], y=df['CurrentPrice'], name="Current Price"))
    fig_price.add_trace(go.Scatter(x=df['Timestamp'], y=df['Strike'], name="Strike Price", line=dict(dash='dash')))
    fig_price.update_layout(title="BTC Price vs Strike", xaxis_title="Time", yaxis_title="Price (USD)")
    st.plotly_chart(fig_price, use_container_width=True)

    # Probability Chart
    st.subheader("Probability History")
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(x=df['Timestamp'], y=df['UpPrice'], name="Yes (Up)", line=dict(color='green')))
    fig_prob.add_trace(go.Scatter(x=df['Timestamp'], y=df['DownPrice'], name="No (Down)", line=dict(color='red')))
    fig_prob.update_layout(title="Contract Prices (Probability)", xaxis_title="Time", yaxis_title="Price (USD)")
    st.plotly_chart(fig_prob, use_container_width=True)
    
    st.caption(f"Last updated: {latest['Timestamp']}")

else:
    st.warning("No data found yet. Please ensure data_logger.py is running.")
    
# Auto-refresh using empty container and sleep (basic way)
# time.sleep(15)
# st.rerun()
