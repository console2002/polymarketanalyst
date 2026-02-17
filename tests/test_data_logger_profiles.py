import asyncio
import csv
import datetime
from pathlib import Path

import pytz

import data_logger
from market_profiles import get_market_profile


class _FrozenDateTime(datetime.datetime):
    """datetime replacement with a fixed UTC now()."""

    frozen_now = datetime.datetime(2026, 1, 1, 12, 7, 42, tzinfo=pytz.utc)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.frozen_now.replace(tzinfo=None)
        return cls.frozen_now.astimezone(tz)


def test_current_market_window_rounds_for_5m_profile(monkeypatch):
    monkeypatch.setattr(data_logger.datetime, "datetime", _FrozenDateTime)

    start, expiration = data_logger._current_market_window(profile_key="btc_5m")

    assert start == datetime.datetime(2026, 1, 1, 12, 5, tzinfo=pytz.utc)
    assert expiration == datetime.datetime(2026, 1, 1, 12, 10, tzinfo=pytz.utc)


def test_current_market_window_rounds_for_15m_profile(monkeypatch):
    monkeypatch.setattr(data_logger.datetime, "datetime", _FrozenDateTime)

    start, expiration = data_logger._current_market_window(profile_key="btc_15m")

    assert start == datetime.datetime(2026, 1, 1, 12, 0, tzinfo=pytz.utc)
    assert expiration == datetime.datetime(2026, 1, 1, 12, 15, tzinfo=pytz.utc)


def test_get_data_file_uses_profile_specific_subdirs():
    ts = datetime.datetime(2026, 1, 2, 1, 0, tzinfo=pytz.utc)

    file_5m = data_logger._get_data_file(ts, "btc_5m")
    file_15m = data_logger._get_data_file(ts, "btc_15m")

    assert file_5m.endswith("data/5min/01012026.csv")
    assert file_15m.endswith("data/15min/01012026.csv")


def test_ensure_csv_creates_directories_and_writes_headers(tmp_path):
    csv_path = tmp_path / "nested" / "deeper" / "prices.csv"

    data_logger._ensure_csv(str(csv_path))

    assert csv_path.exists()
    with csv_path.open("r", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == data_logger.CSV_HEADERS


def test_run_logger_rollover_uses_profile_duration(monkeypatch):
    profile = get_market_profile("btc_5m")
    now = datetime.datetime.now(pytz.utc)
    expiration = now + datetime.timedelta(minutes=profile.window_minutes)
    captured_start_times = []

    stop_event = asyncio.Event()

    class StubAggregator:
        def __init__(self, market_info, broadcaster=None):
            self.market_info = market_info

        async def monitor_no_updates(self):
            await asyncio.sleep(3600)

        async def handle_update(self, update):
            return None

    class StubWsLogger:
        def __init__(self, market_info, on_price_update):
            self.market_info = market_info

        async def run(self):
            return None

        async def shutdown(self):
            return None

    def fake_resolve_current_market(profile_key=None):
        return ({"expiration_time_utc": expiration, "profile_key": profile_key}, None)

    def fake_resolve_market_by_start_time(start_time_utc, profile_key=None):
        captured_start_times.append(start_time_utc)
        stop_event.set()
        return (None, "stop")

    monkeypatch.setattr(data_logger, "PriceAggregator", StubAggregator)
    monkeypatch.setattr(data_logger, "PolymarketWebsocketLogger", StubWsLogger)
    monkeypatch.setattr(data_logger, "_resolve_current_market", fake_resolve_current_market)
    monkeypatch.setattr(
        data_logger,
        "_resolve_market_by_start_time",
        fake_resolve_market_by_start_time,
    )
    monkeypatch.setattr(data_logger, "_ensure_csv", lambda path: None)
    monkeypatch.setattr(data_logger, "_get_data_file", lambda now_dt, profile_or_key=None: "x.csv")
    monkeypatch.setattr(data_logger, "STATUS_CHECK_INTERVAL_SECONDS", 0)

    asyncio.run(data_logger.run_logger(profile_key="btc_5m", stop_event=stop_event))

    assert captured_start_times == [expiration - datetime.timedelta(minutes=5)]
