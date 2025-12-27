import asyncio
import json
import queue
import threading
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import websockets

from get_current_markets import get_available_market_urls, get_current_market_urls

DEFAULT_WS_URL = "ws://127.0.0.1:8765"


def _listener_worker(url, out_queue, stop_event):
    async def _listen():
        while not stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    while not stop_event.is_set():
                        message = await ws.recv()
                        out_queue.put(message)
            except Exception:
                await asyncio.sleep(1)

    asyncio.run(_listen())


def _ensure_listener(url):
    if "listener_thread" in st.session_state:
        if st.session_state.get("listener_url") == url:
            return
        st.session_state.listener_stop.set()
    out_queue = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_listener_worker,
        args=(url, out_queue, stop_event),
        daemon=True,
    )
    thread.start()
    st.session_state.listener_thread = thread
    st.session_state.listener_queue = out_queue
    st.session_state.listener_stop = stop_event
    st.session_state.listener_url = url
    st.session_state.last_rows = []
    st.session_state.last_update = None


def _drain_messages():
    out_queue = st.session_state.listener_queue
    latest_rows = None
    latest_update = None
    while True:
        try:
            message = out_queue.get_nowait()
        except queue.Empty:
            break
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            continue
        rows = payload.get("rows") or []
        if rows:
            latest_rows = rows
            latest_update = payload.get("timestamp")
    if latest_rows is not None:
        st.session_state.last_rows = latest_rows
        st.session_state.last_update = latest_update


def _format_market_label(market):
    start_et = market["target_time_et"]
    expiration_utc = market["expiration_time_utc"]
    slug = market["polymarket"].split("/")[-1]
    return f"{start_et:%b %d %I:%M %p ET} → {expiration_utc:%H:%M UTC} ({slug})"


st.set_page_config(page_title="Polymarket Logger GUI", layout="wide")

st.sidebar.header("Logger GUI")
ws_url = st.sidebar.text_input("WebSocket URL", value=DEFAULT_WS_URL)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.number_input(
    "Refresh interval (seconds)",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.5,
)
st.sidebar.subheader("Market Selection")
market_options = get_available_market_urls()
current_market = get_current_market_urls()
market_by_url = {market["polymarket"]: market for market in market_options}
option_urls = list(market_by_url.keys())
current_url = current_market["polymarket"]
default_index = option_urls.index(current_url) if current_url in option_urls else 0
override_market = st.sidebar.checkbox(
    "Override auto market",
    value=False,
    help="Overrides the GUI display only. The logger remains unchanged unless restarted.",
)
selected_url = st.sidebar.selectbox(
    "Available markets",
    options=option_urls,
    index=default_index,
    format_func=lambda url: _format_market_label(market_by_url[url]),
    disabled=not override_market,
    key="market_selection_url",
)
if override_market:
    selected_market = market_by_url[selected_url]
    if selected_url != current_url:
        st.sidebar.info(
            "Selection affects the GUI display only. Restart the logger to switch feeds."
        )
else:
    selected_market = current_market
    st.sidebar.caption("Auto-advance enabled (next 15-minute market).")

_ensure_listener(ws_url)
_drain_messages()

st.title("Live Logger Feed")
st.caption(f"GUI Market: {selected_market['polymarket']}")
last_update = st.session_state.last_update
if last_update:
    st.caption(f"Last update (UTC): {last_update}")
else:
    st.caption("Waiting for updates from the logger...")

rows = st.session_state.last_rows
if rows:
    df = pd.DataFrame(rows)
    st.subheader("Latest Rows")
    st.dataframe(df, use_container_width=True)

    st.subheader("Latest Values")
    metric_cols = st.columns(len(rows))
    for idx, row in enumerate(rows):
        outcome = row.get("outcome", "Outcome")
        with metric_cols[idx]:
            st.metric(
                label=outcome,
                value=row.get("best_ask", ""),
                delta=f"Bid {row.get('best_bid', '')} | Mid {row.get('mid', '')}",
            )
else:
    st.info("No data yet. Ensure the logger is running with --ui-stream.")

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
