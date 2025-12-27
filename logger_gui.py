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
LAUNCH_LOG_MAX_LINES = 20
STREAM_INACTIVE_THRESHOLD_SECONDS = 20
LOGGER_PID_FILE = os.path.join(os.path.dirname(__file__), "data_logger_ui.pid")


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
    st.session_state.last_message_time = None
    st.session_state.rolling_rows = []


def _ensure_launch_log():
    if "launch_log" not in st.session_state:
        st.session_state.launch_log = []
    if "launch_log_queue" not in st.session_state:
        st.session_state.launch_log_queue = queue.Queue()


def _append_launch_log(line):
    logs = st.session_state.launch_log
    logs.append(line)
    st.session_state.launch_log = logs[-LAUNCH_LOG_MAX_LINES:]


def _drain_launch_log_queue():
    log_queue = st.session_state.get("launch_log_queue")
    if not log_queue:
        return
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    while True:
        try:
            line = log_queue.get_nowait()
        except queue.Empty:
            break
        _append_launch_log(f"{timestamp} {line}")


def _drain_messages():
    out_queue = st.session_state.listener_queue
    latest_rows = None
    latest_update = None
    rolling_rows = st.session_state.get("rolling_rows", [])
    last_message_time = st.session_state.get("last_message_time")
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
            parsed_timestamp = _parse_last_update(payload_timestamp)
            if parsed_timestamp is not None:
                last_message_time = parsed_timestamp
            else:
                last_message_time = datetime.now(timezone.utc)
        else:
            last_message_time = datetime.now(timezone.utc)
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
    st.session_state.last_message_time = last_message_time
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


def _read_logger_pid():
    if not os.path.exists(LOGGER_PID_FILE):
        return None
    try:
        with open(LOGGER_PID_FILE, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        return int(raw)
    except (OSError, ValueError):
        return None


def _is_pid_running(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_logger_pid(pid):
    if not pid:
        return False, "No PID available for stop request."
    try:
        if os.name == "nt":
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return False, f"Failed to stop logger PID {pid}: {exc}"
    return True, f"Stop signal sent to PID {pid}."


def _parse_ws_target(ws_url):
    parsed = urlparse(ws_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765
    scheme = parsed.scheme or "ws"
    return scheme, host, port


def _run_coroutine(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _is_ws_endpoint_active(ws_url, timeout=0.5):
    async def _probe():
        try:
            async with websockets.connect(
                ws_url,
                open_timeout=timeout,
                close_timeout=timeout,
                ping_interval=None,
            ):
                return True
        except Exception:
            return False

    return _run_coroutine(_probe())


def _enqueue_process_output(stream, label, out_queue):
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        if not line:
            break
        out_queue.put(f"{label}: {line.rstrip()}")
    stream.close()


def _start_logger_process(host, port):
    logger_path = os.path.join(os.path.dirname(__file__), "data_logger.py")
    command = [
        sys.executable,
        "-u",
        logger_path,
        "--ui-stream",
        "--ui-stream-host",
        host,
        "--ui-stream-port",
        str(port),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    creationflags = 0
    start_new_session = True
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        start_new_session = False
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=start_new_session,
            creationflags=creationflags,
            env=env,
        )
    except OSError as exc:
        return None, f"Failed to launch logger: {exc}"

    log_queue = st.session_state.get("launch_log_queue")
    if log_queue is not None:
        stdout_thread = threading.Thread(
            target=_enqueue_process_output,
            args=(proc.stdout, "stdout", log_queue),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_enqueue_process_output,
            args=(proc.stderr, "stderr", log_queue),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
    return proc, None


def _stop_logger_process(proc):
    if os.name == "nt":
        if hasattr(signal, "CTRL_BREAK_EVENT"):
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                proc.terminate()
        else:
            proc.terminate()
    else:
        proc.send_signal(signal.SIGINT)


def _stop_logger(logger_proc=None):
    logger_pid = _read_logger_pid()
    if _is_pid_running(logger_pid):
        ok, message = _stop_logger_pid(logger_pid)
        return ok, message, "pid"
    if logger_proc and logger_proc.poll() is None:
        _stop_logger_process(logger_proc)
        return True, "Stop signal sent to logger subprocess.", "process"
    return False, "No running logger found to stop.", "none"


def _wait_for_logger_exit(proc, timeout=5):
    if proc is None:
        return True
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def _format_launch_log_for_error():
    launch_log = st.session_state.get("launch_log", [])
    if not launch_log:
        return "No stdout/stderr captured."
    return "\n".join(launch_log)


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
_ensure_launch_log()
_drain_launch_log_queue()
st.sidebar.subheader("Logger Control")
if not can_manage_logger:
    st.sidebar.info("Start/Stop is available only for local ws:// endpoints.")

endpoint_active = can_manage_logger and _is_ws_endpoint_active(ws_url)
logger_pid = _read_logger_pid()
logger_pid_running = _is_pid_running(logger_pid)
if logger_pid and not logger_pid_running:
    try:
        os.remove(LOGGER_PID_FILE)
    except OSError:
        pass
    logger_pid = None
logger_already_running = endpoint_active and not logger_running and logger_pid_running
if logger_already_running:
    st.sidebar.warning("Logger already running")

start_clicked = st.sidebar.button(
    "Start Logger",
    disabled=logger_running or logger_already_running or not can_manage_logger,
    help="Launch data_logger.py in a background process with UI streaming enabled.",
)
stop_clicked = st.sidebar.button(
    "Stop Logger",
    disabled=not (logger_running or logger_pid_running),
    help="Send a stop signal for a graceful shutdown.",
)
if start_clicked and not logger_running and can_manage_logger:
    st.session_state.logger_error = None
    _append_launch_log("Attempting logger start.")
    logger_proc, start_error = _start_logger_process(host, port)
    time.sleep(0.5)
    _drain_launch_log_queue()
    if start_error:
        st.session_state.logger_error = start_error
        logger_proc = None
        logger_running = False
    elif logger_proc and logger_proc.poll() is None:
        st.session_state.logger_process = logger_proc
        logger_running = True
        st.sidebar.success("Logger process started.")
        _append_launch_log("Logger start succeeded.")
    else:
        exit_code = logger_proc.poll() if logger_proc else "unknown"
        _drain_launch_log_queue()
        launch_output = _format_launch_log_for_error()
        st.session_state.logger_error = (
            "Logger failed to start (exit code "
            f"{exit_code}).\n{launch_output}"
        )
        logger_proc = None
        logger_running = False
if stop_clicked and logger_running:
    try:
        ok, message, stop_mode = _stop_logger(logger_proc)
        if ok and stop_mode == "process":
            stopped = _wait_for_logger_exit(logger_proc)
            if not stopped:
                logger_proc.terminate()
                stopped = _wait_for_logger_exit(logger_proc, timeout=2)
            if stopped:
                st.session_state.logger_process = None
                logger_running = False
            st.sidebar.info("Logger stop requested.")
            _append_launch_log("Logger stop requested.")
        elif ok:
            st.sidebar.info(message)
            _append_launch_log(message)
        else:
            st.session_state.logger_error = message
            st.sidebar.error(st.session_state.logger_error)
            _append_launch_log(message)
    except Exception as exc:
        st.session_state.logger_error = f"Unable to stop logger process: {exc}"
        st.sidebar.error(st.session_state.logger_error)
        _append_launch_log(f"Stop signal failed: {exc}")
elif stop_clicked and logger_pid_running:
    ok, message, _ = _stop_logger(logger_proc)
    if ok:
        st.sidebar.info(message)
        _append_launch_log(message)
    else:
        st.session_state.logger_error = message
        st.sidebar.error(st.session_state.logger_error)
        _append_launch_log(message)

if st.session_state.get("logger_error"):
    st.sidebar.error(st.session_state.logger_error)

with st.sidebar.expander(
    f"Launch log (last {LAUNCH_LOG_MAX_LINES} lines)",
    expanded=False,
):
    launch_log = st.session_state.get("launch_log", [])
    if launch_log:
        st.text_area(
            "Console output",
            value="\n".join(launch_log),
            height=200,
            disabled=True,
        )
    else:
        st.caption("No launch messages yet.")

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
last_message_time = st.session_state.get("last_message_time") or last_update_dt
stream_active = (
    last_message_time is not None
    and (datetime.now(timezone.utc) - last_message_time).total_seconds()
    < STREAM_INACTIVE_THRESHOLD_SECONDS
)

status_label = "Running (stream active)" if stream_active else "Stopped"
if stream_active:
    st.sidebar.success(f"Status: {status_label}")
else:
    st.sidebar.warning(f"Status: {status_label} (no recent stream activity)")
if logger_running:
    st.sidebar.caption(f"Local logger process: running (PID {logger_proc.pid})")
elif logger_pid_running:
    st.sidebar.caption(f"Local logger process: running (PID {logger_pid})")
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
        width="stretch",
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
