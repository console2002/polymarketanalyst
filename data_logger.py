"""Polymarket logger.

The logger runs headless by default and can optionally expose a lightweight
WebSocket feed intended for read-only GUI consumers. The GUI should run in a
separate process and must never block the logger's critical path. Messages are
best-effort and may be dropped under load to preserve logging performance.
"""

import argparse
import asyncio
import contextlib
import csv
import datetime
import json
import os
import signal

import pytz
import websockets
from websockets.exceptions import InvalidMessage

from fetch_current_polymarket import resolve_current_market, resolve_market_by_expiration
from websocket_logger import PolymarketWebsocketLogger, STALE_THRESHOLD_SECONDS

LOGGING_INTERVAL_SECONDS = 1
NO_UPDATE_WARNING_SECONDS = 45
STATUS_CHECK_INTERVAL_SECONDS = 5
TIMEZONE_ET = pytz.timezone("US/Eastern")
TIMEZONE_UK = pytz.timezone("Europe/London")
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATE_FORMAT = "%d%m%Y"
MARKET_DURATION = datetime.timedelta(minutes=15)
SCRIPT_DIR = os.path.dirname(__file__)
PID_FILE = os.path.join(SCRIPT_DIR, "data_logger_ui.pid")
CSV_HEADERS = [
    "timestamp_et",
    "timestamp_uk",
    "target_time_uk",
    "expiration_uk",
    "server_time_utc",
    "local_time_utc",
    "stream_seq_id",
    "token_id",
    "outcome",
    "best_bid",
    "best_ask",
    "mid",
    "spread",
    "spread_pct",
    "best_bid_size",
    "best_ask_size",
    "last_trade_price",
    "last_trade_size",
    "last_trade_side",
    "last_trade_ts",
    "heartbeat_last_seen",
    "reconnect_count",
    "is_stale",
    "stale_age_seconds",
]


def _format_timestamp(value, timezone):
    if not value:
        return ""
    if value.tzinfo is None:
        value = pytz.utc.localize(value)
    return value.astimezone(timezone).strftime(TIME_FORMAT)


def _format_timestamp_utc(value):
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = pytz.utc.localize(value)
    return value.astimezone(pytz.utc).isoformat()


def _get_data_file(timestamp_dt):
    if not timestamp_dt:
        timestamp_dt = datetime.datetime.now(pytz.utc)
    if timestamp_dt.tzinfo is None:
        timestamp_dt = pytz.utc.localize(timestamp_dt)
    date_str = timestamp_dt.astimezone(TIMEZONE_ET).strftime(DATE_FORMAT)
    return os.path.join(SCRIPT_DIR, f"{date_str}.csv")


def _ensure_csv(file_path):
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Required fields: timestamps, market times, IDs, best quotes/sizes, and health/staleness.
            # Optional fields: last trade details (price/size/side/timestamp).
            writer.writerow(CSV_HEADERS)
        print(f"Created {file_path}")


def _ensure_utc(value):
    if not value:
        return None
    if value.tzinfo is None:
        return pytz.utc.localize(value)
    return value.astimezone(pytz.utc)


class LoggerStreamBroadcaster:
    def __init__(
        self,
        host="127.0.0.1",
        port=8765,
        port_fallback_attempts=5,
        max_queue=1000,
        send_timeout=0.25,
        allowed_origins=None,
    ):
        self.host = host
        self.port = port
        self.port_fallback_attempts = max(1, int(port_fallback_attempts))
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.send_timeout = send_timeout
        self.allowed_origins = set(allowed_origins) if allowed_origins else None
        self._clients = set()
        self._server = None
        self._task = None
        self._shutdown = asyncio.Event()

    async def start(self):
        for attempt in range(2):
            if attempt == 0 and self._terminate_existing_pid():
                await asyncio.sleep(0.5)
            for offset in range(self.port_fallback_attempts):
                candidate_port = self.port + offset
                try:
                    self._server = await websockets.serve(
                        self._handle_client,
                        self.host,
                        candidate_port,
                        process_request=self._process_request,
                        reuse_address=True,
                    )
                    self.port = candidate_port
                    break
                except OSError as exc:
                    if exc.errno in {98, 10048}:
                        print(
                            "UI stream unavailable: could not bind to "
                            f"ws://{self.host}:{candidate_port} (address already in use). "
                            "Another logger may still be running, or the socket has not been "
                            "released yet."
                        )
                        continue
                    if exc.errno in {10013}:
                        print(
                            "UI stream unavailable: could not bind to "
                            f"ws://{self.host}:{candidate_port} (permission denied). "
                            "Windows may block binding to this port."
                        )
                        continue
                    raise
            if self._server:
                break
        if not self._server:
            print(
                "UI stream unavailable: could not bind to any configured port "
                f"starting at {self.host}:{self.port}. The UI stream will remain disabled."
            )
            return False
        self._task = asyncio.create_task(self._broadcast_loop())
        self._write_pid()
        print(f"UI stream available at ws://{self.host}:{self.port}")
        return True

    async def stop(self):
        self._shutdown.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._remove_pid()

    def publish(self, payload):
        """Queue payloads for UI consumers without blocking the logger."""
        if self.queue.full():
            return
        self.queue.put_nowait(payload)

    async def _handle_client(self, websocket):
        self._clients.add(websocket)
        try:
            await websocket.wait_closed()
        except (InvalidMessage, EOFError):
            return
        finally:
            self._clients.discard(websocket)

    async def _process_request(self, path, request_headers):
        headers = request_headers
        if hasattr(request_headers, "headers"):
            headers = request_headers.headers
        upgrade = headers.get("Upgrade", "")
        if upgrade.lower() != "websocket":
            return (
                426,
                [("Content-Type", "text/plain; charset=utf-8"), ("Connection", "close")],
                b"Expected WebSocket connection.\n",
            )
        if self.allowed_origins is None:
            return None
        origin = headers.get("Origin")
        if origin not in self.allowed_origins:
            return (
                403,
                [("Content-Type", "text/plain; charset=utf-8"), ("Connection", "close")],
                b"Origin not allowed.\n",
            )
        return None

    async def _broadcast_loop(self):
        while not self._shutdown.is_set():
            payload = await self.queue.get()
            message = json.dumps(payload, default=str)
            if not self._clients:
                continue
            clients = list(self._clients)
            results = await asyncio.gather(
                *[self._send_to_client(client, message) for client in clients],
                return_exceptions=True,
            )
            for client, ok in zip(clients, results):
                if ok is True:
                    continue
                self._clients.discard(client)

    async def _send_to_client(self, client, message):
        try:
            await asyncio.wait_for(client.send(message), timeout=self.send_timeout)
        except (asyncio.TimeoutError, Exception):
            return False
        return True

    def _read_pid(self):
        if not os.path.exists(PID_FILE):
            return None
        try:
            with open(PID_FILE, "r", encoding="utf-8") as handle:
                raw = handle.read().strip()
            return int(raw)
        except (OSError, ValueError):
            return None

    def _write_pid(self):
        try:
            with open(PID_FILE, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
        except OSError:
            pass

    def _remove_pid(self):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass

    def _terminate_existing_pid(self):
        pid = self._read_pid()
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            self._remove_pid()
            return False
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except OSError:
            return False


class PriceAggregator:
    def __init__(self, market_info, broadcaster=None):
        self.market_info = market_info
        self.latest = {}
        self.last_logged = {}
        self.last_log_time = None
        self.last_update_ts = None
        self._last_warning_time = None
        self._start_time = datetime.datetime.now(pytz.utc)
        self.outcome_order = market_info.get("outcomes") or []
        self._current_file_path = None
        self._current_file_handle = None
        self._current_writer = None
        self._broadcaster = broadcaster

    async def handle_update(self, update):
        outcome = update["outcome"]
        update_timestamp = (
            update.get("last_update_ts")
            or update.get("timestamp")
            or datetime.datetime.now(pytz.utc)
        )
        if update_timestamp.tzinfo is None:
            update_timestamp = pytz.utc.localize(update_timestamp)
        self.last_update_ts = update_timestamp
        self.latest[outcome] = update
        if not self._has_both_outcomes():
            return

        now = update_timestamp
        prices_changed = self._prices_changed()
        interval_elapsed = (
            self.last_log_time is None
            or (now - self.last_log_time).total_seconds() >= LOGGING_INTERVAL_SECONDS
        )

        if prices_changed or interval_elapsed:
            self._log_row(now)

    async def monitor_no_updates(
        self,
        warning_seconds=NO_UPDATE_WARNING_SECONDS,
        check_interval=STATUS_CHECK_INTERVAL_SECONDS,
    ):
        while True:
            await asyncio.sleep(check_interval)
            now = datetime.datetime.now(pytz.utc)
            last_update = self.last_update_ts or self._start_time
            age_seconds = max(0.0, (now - last_update).total_seconds())
            warn_due = age_seconds >= warning_seconds
            should_warn = warn_due and (
                self._last_warning_time is None
                or (now - self._last_warning_time).total_seconds() >= warning_seconds
            )
            if should_warn:
                timestamp_et = _format_timestamp(now, TIMEZONE_ET)
                last_update_str = _format_timestamp(last_update, TIMEZONE_ET)
                print(
                    f"[{timestamp_et}] Warning: No websocket updates for "
                    f"{round(age_seconds, 1)}s (last update {last_update_str}). "
                    "Waiting for new data before writing CSV."
                )
                self._last_warning_time = now
            if self._broadcaster:
                self._broadcaster.publish(
                    {
                        "type": "status",
                        "timestamp": _format_timestamp_utc(now),
                        "last_update": _format_timestamp_utc(last_update),
                        "seconds_since_update": round(age_seconds, 1),
                        "warning": warn_due,
                        "status": "warning" if warn_due else "ok",
                    }
                )

    def _has_both_outcomes(self):
        return len(self.latest) >= 2

    def _prices_changed(self):
        for outcome, update in self.latest.items():
            last = self.last_logged.get(outcome)
            if not last:
                return True
            if (
                update["best_bid"] != last["best_bid"]
                or update["best_ask"] != last["best_ask"]
                or update["best_bid_size"] != last["best_bid_size"]
                or update["best_ask_size"] != last["best_ask_size"]
            ):
                return True
        return False

    def _log_row(self, timestamp_dt):
        if len(self.latest) < 2:
            return
        up_key, down_key = self._ordered_outcomes()
        if not up_key or not down_key:
            return
        up_update = self.latest[up_key]
        down_update = self.latest[down_key]

        timestamp_et = _format_timestamp(timestamp_dt, TIMEZONE_ET)
        timestamp_uk = _format_timestamp(timestamp_dt, TIMEZONE_UK)
        target_time = self.market_info.get("target_time_utc")
        expiration = self.market_info.get("expiration_time_utc")
        target_time_str = _format_timestamp(target_time, TIMEZONE_UK)
        expiration_str = _format_timestamp(expiration, TIMEZONE_UK)
        local_time_utc = datetime.datetime.now(pytz.utc)
        rows = []
        for update in (up_update, down_update):
            is_stale, stale_age = self._stale_info(update, timestamp_dt)
            row = [
                timestamp_et,
                timestamp_uk,
                target_time_str,
                expiration_str,
                _format_timestamp_utc(update.get("server_time_utc")),
                _format_timestamp_utc(local_time_utc),
                update.get("stream_seq_id"),
                update.get("token_id"),
                update.get("outcome"),
                update["best_bid"],
                update["best_ask"],
                update["mid"],
                update["spread"],
                update["spread_pct"],
                update["best_bid_size"],
                update["best_ask_size"],
                update.get("last_trade_price"),
                update.get("last_trade_size"),
                update.get("last_trade_side"),
                _format_timestamp_utc(update.get("last_trade_ts")),
                _format_timestamp_utc(update.get("heartbeat_last_seen")),
                update.get("reconnect_count"),
                is_stale,
                stale_age,
            ]
            rows.append(row)

        data_file = _get_data_file(self._event_timestamp(up_update, down_update, timestamp_dt))
        writer = self._get_writer(data_file)
        writer.writerows(rows)
        if self._broadcaster:
            payload_rows = [dict(zip(CSV_HEADERS, row)) for row in rows]
            self._broadcaster.publish(
                {
                    "file": data_file,
                    "timestamp": _format_timestamp_utc(timestamp_dt),
                    "rows": payload_rows,
                }
            )

        self.last_logged = {
            up_key: up_update,
            down_key: down_update,
        }
        self.last_log_time = timestamp_dt
        up_is_stale, up_stale_age = self._stale_info(up_update, timestamp_dt)
        down_is_stale, down_stale_age = self._stale_info(down_update, timestamp_dt)
        print(
            f"[{timestamp_et}] Logged: Up={up_update['best_ask']}, "
            f"Down={down_update['best_ask']}, "
            f"Stale={up_is_stale or down_is_stale}"
        )

    def _ordered_outcomes(self):
        if len(self.outcome_order) >= 2:
            return self.outcome_order[0], self.outcome_order[1]
        outcomes = list(self.latest.keys())
        if len(outcomes) >= 2:
            return outcomes[0], outcomes[1]
        return None, None

    def _event_timestamp(self, up_update, down_update, fallback_timestamp):
        for update in (up_update, down_update):
            server_time = update.get("server_time_utc")
            if server_time:
                return server_time
        return fallback_timestamp or datetime.datetime.now(pytz.utc)

    def _get_writer(self, data_file):
        if self._current_file_path != data_file:
            if self._current_file_handle:
                self._current_file_handle.close()
            _ensure_csv(data_file)
            self._current_file_handle = open(data_file, mode="a", newline="")
            self._current_file_path = data_file
            self._current_writer = csv.writer(self._current_file_handle)
        return self._current_writer

    @staticmethod
    def _stale_info(update, timestamp_dt):
        last_update = update.get("last_update_ts") or update.get("timestamp")
        if not last_update:
            return True, None
        if last_update.tzinfo is None:
            last_update = pytz.utc.localize(last_update)
        age_seconds = max(0.0, (timestamp_dt - last_update).total_seconds())
        is_stale = age_seconds > STALE_THRESHOLD_SECONDS
        return is_stale, round(age_seconds, 3)

    @staticmethod
    def _max_stale_age(*ages):
        numeric_ages = [age for age in ages if isinstance(age, (int, float))]
        if not numeric_ages:
            return ""
        return max(numeric_ages)


async def run_logger(broadcaster=None, stop_event=None):
    if stop_event is None:
        stop_event = asyncio.Event()
    if broadcaster:
        started = await broadcaster.start()
        if not started:
            broadcaster = None

    next_expiration = None
    try:
        while not stop_event.is_set():
            if next_expiration:
                market_info, err = resolve_market_by_expiration(next_expiration)
            else:
                market_info, err = resolve_current_market()
            if err:
                print(f"Error: {err}")
                await asyncio.sleep(STATUS_CHECK_INTERVAL_SECONDS)
                continue

            now = datetime.datetime.now(pytz.utc)
            target_time = _ensure_utc(market_info.get("target_time_utc"))
            expiration = _ensure_utc(market_info.get("expiration_time_utc"))
            if expiration and now >= expiration:
                next_expiration = expiration + MARKET_DURATION
                continue
            if target_time and now < target_time:
                next_expiration = target_time
                continue

            aggregator = PriceAggregator(market_info, broadcaster=broadcaster)
            _ensure_csv(_get_data_file(now))
            ws_logger = PolymarketWebsocketLogger(market_info, aggregator.handle_update)
            stop_task = asyncio.create_task(stop_event.wait())
            logger_task = asyncio.create_task(ws_logger.run())
            monitor_task = asyncio.create_task(aggregator.monitor_no_updates())
            expiry_task = None
            if expiration:
                seconds_until_expiry = max(0.0, (expiration - now).total_seconds())
                expiry_task = asyncio.create_task(asyncio.sleep(seconds_until_expiry))

            try:
                tasks = [logger_task, stop_task]
                if expiry_task:
                    tasks.append(expiry_task)
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done and stop_event.is_set():
                    await ws_logger.shutdown()
                    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(logger_task, timeout=5)
                    break
                if expiry_task and expiry_task in done:
                    await ws_logger.shutdown()
                    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(logger_task, timeout=5)
                    next_expiration = expiration + MARKET_DURATION if expiration else None
                    continue
                if logger_task in done:
                    await ws_logger.shutdown()
                    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(logger_task, timeout=5)
                    next_expiration = expiration
                    continue
            finally:
                stop_task.cancel()
                monitor_task.cancel()
                if expiry_task:
                    expiry_task.cancel()
                for task in (logger_task, monitor_task, expiry_task):
                    if task:
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
    finally:
        if broadcaster:
            await broadcaster.stop()


def _install_signal_handlers(stop_event):
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop_event.set)


async def _run_with_signals(broadcaster):
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    await run_logger(broadcaster, stop_event=stop_event)


def main():
    parser = argparse.ArgumentParser(description="Polymarket price logger")
    parser.add_argument(
        "--ui-stream",
        action="store_true",
        help=(
            "Enable a local WebSocket feed for GUI consumers. The GUI should run in a "
            "separate process and remains read-only."
        ),
    )
    parser.add_argument(
        "--ui-stream-host",
        default="127.0.0.1",
        help="Host interface for the GUI WebSocket feed.",
    )
    parser.add_argument(
        "--ui-stream-port",
        type=int,
        default=8765,
        help="Port for the GUI WebSocket feed.",
    )
    parser.add_argument(
        "--ui-stream-port-fallbacks",
        type=int,
        default=5,
        help=(
            "Number of sequential ports to try if binding fails "
            "(starting at --ui-stream-port)."
        ),
    )
    args = parser.parse_args()
    print("Starting Data Logger...")
    broadcaster = None
    if args.ui_stream:
        broadcaster = LoggerStreamBroadcaster(
            host=args.ui_stream_host,
            port=args.ui_stream_port,
            port_fallback_attempts=args.ui_stream_port_fallbacks,
        )
    try:
        asyncio.run(_run_with_signals(broadcaster))
    except KeyboardInterrupt:
        print("\nStopping logger...")


if __name__ == "__main__":
    main()
