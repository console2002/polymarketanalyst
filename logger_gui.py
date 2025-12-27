"""Read-only GUI for the Polymarket logger.

Run this in a separate process from the logger. It only consumes the logger's
WebSocket stream and never calls into logger code paths.
"""

import asyncio
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

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
    st.session_state.rolling_rows = []


def _drain_messages():
    out_queue = st.session_state.listener_queue
    latest_rows = None
    latest_update = None
    rolling_rows = st.session_state.get("rolling_rows", [])
    max_rows = st.session_state.get("buffer_size", 100)
    while True:
        try:
            message = out_queue.get_nowait()
        except queue.Empty:
            break
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            continue
        payload_timestamp = payload.get("timestamp")
        if payload_timestamp:
            latest_update = payload_timestamp
        rows = payload.get("rows") or []
        if rows:
            latest_rows = rows
            if not latest_update:
                latest_update = payload.get("timestamp")
            for row in rows:
                rolling_rows.append(
                    {
                        "timestamp": latest_update,
                        "outcome": row.get("outcome"),
                        "best_bid": row.get("best_bid"),
                        "best_ask": row.get("best_ask"),
                        "mid": row.get("mid"),
                        "last_trade": row.get("last_trade_price"),
                        "stale": row.get("is_stale"),
                    }
                )
    if latest_rows is not None:
        st.session_state.last_rows = latest_rows
        st.session_state.last_update = latest_update
    if rolling_rows:
        st.session_state.rolling_rows = rolling_rows[-max_rows:]
    else:
        st.session_state.rolling_rows = []


def _parse_last_update(value):
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_market_label(market):
    start_et = market["target_time_et"]
    expiration_utc = market["expiration_time_utc"]
    slug = market["polymarket"].split("/")[-1]
    return f"{start_et:%b %d %I:%M %p ET} → {expiration_utc:%H:%M UTC} ({slug})"


def _get_logger_process():
    proc = st.session_state.get("logger_process")
    if proc and proc.poll() is not None:
        st.session_state.logger_process = None
        return None
    return proc


def _parse_ws_target(ws_url):
    parsed = urlparse(ws_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765
    scheme = parsed.scheme or "ws"
    return scheme, host, port


def _start_logger_process(host, port):
    logger_path = os.path.join(os.path.dirname(__file__), "data_logger.py")
    command = [
        sys.executable,
        logger_path,
        "--ui-stream",
        "--ui-stream-host",
        host,
        "--ui-stream-port",
        str(port),
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


st.set_page_config(page_title="Polymarket Logger GUI", layout="wide")

st.sidebar.header("Logger GUI")
st.sidebar.caption(
    "Read-only consumer. Updates are best-effort and may skip under load to keep the logger fast."
)
ws_url = st.sidebar.text_input("WebSocket URL", value=DEFAULT_WS_URL)
scheme, host, port = _parse_ws_target(ws_url)
logger_proc = _get_logger_process()
logger_running = logger_proc is not None
can_manage_logger = scheme in {"ws", ""} and host in {"127.0.0.1", "localhost"}
st.sidebar.subheader("Logger Control")
if not can_manage_logger:
    st.sidebar.info("Start/Stop is available only for local ws:// endpoints.")

start_clicked = st.sidebar.button(
    "Start Logger",
    disabled=logger_running or not can_manage_logger,
    help="Launch data_logger.py in a background process with UI streaming enabled.",
)
stop_clicked = st.sidebar.button(
    "Stop Logger",
    disabled=not logger_running,
    help="Send SIGINT for a graceful shutdown.",
)
if start_clicked and not logger_running and can_manage_logger:
    logger_proc = _start_logger_process(host, port)
    st.session_state.logger_process = logger_proc
    logger_running = True
    st.sidebar.success("Logger process started.")
if stop_clicked and logger_running:
    logger_proc.send_signal(signal.SIGINT)
    logger_running = False
    st.sidebar.info("Stop signal sent to logger.")

auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.number_input(
    "Refresh interval (seconds)",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.5,
)
buffer_size = st.sidebar.slider(
    "Rolling buffer size (rows)",
    min_value=50,
    max_value=200,
    value=100,
    step=10,
)
st.session_state.buffer_size = buffer_size
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

last_update = st.session_state.last_update
last_update_dt = _parse_last_update(last_update)
stream_active = (
    last_update_dt is not None
    and (datetime.now(timezone.utc) - last_update_dt).total_seconds() < 15
)

status_label = "Running (stream active)" if stream_active else "Stopped/No data"
if stream_active:
    st.sidebar.success(f"Status: {status_label}")
else:
    st.sidebar.warning(f"Status: {status_label}")
if logger_running:
    st.sidebar.caption(f"Local logger process: running (PID {logger_proc.pid})")
else:
    st.sidebar.caption("Local logger process: not running")

st.title("Live Logger Feed")
st.caption(f"Logger status: {status_label}")
st.caption(f"GUI Market: {selected_market['polymarket']}")
if last_update:
    st.caption(f"Last update (UTC): {last_update}")
else:
    st.caption("Waiting for updates from the logger...")

rows = st.session_state.last_rows
rolling_rows = st.session_state.get("rolling_rows", [])
if rolling_rows:
    table_columns = ["timestamp", "outcome", "best_bid", "best_ask", "mid", "last_trade", "stale"]
    table_df = pd.DataFrame(rolling_rows).reindex(columns=table_columns)
    st.subheader("Latest Logged Entries")
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )

if rows:
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
