import asyncio
import csv
import datetime
import os

import pytz

from fetch_current_polymarket import resolve_current_market
from websocket_logger import PolymarketWebsocketLogger, STALE_THRESHOLD_SECONDS

LOGGING_INTERVAL_SECONDS = 1
TIMEZONE_ET = pytz.timezone("US/Eastern")
TIMEZONE_UK = pytz.timezone("Europe/London")
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATE_FORMAT = "%d%m%Y"
SCRIPT_DIR = os.path.dirname(__file__)


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
            writer.writerow(
                [
                    "Timestamp",
                    "Timestamp_UK",
                    "TargetTime",
                    "Expiration",
                    "UpBestBid",
                    "DownBestBid",
                    "UpBestAsk",
                    "DownBestAsk",
                    "UpMidPrice",
                    "DownMidPrice",
                    "UpSpread",
                    "DownSpread",
                    "UpSpreadPct",
                    "DownSpreadPct",
                    "UpBidSize",
                    "DownBidSize",
                    "UpAskSize",
                    "DownAskSize",
                    "UpLastTradePrice",
                    "DownLastTradePrice",
                    "UpLastTradeSize",
                    "DownLastTradeSize",
                    "UpLastTradeSide",
                    "DownLastTradeSide",
                    "UpLastTradeTs",
                    "DownLastTradeTs",
                    "UpServerTimeUtc",
                    "DownServerTimeUtc",
                    "UpStreamSeqId",
                    "DownStreamSeqId",
                    "HeartbeatLastSeen",
                    "ReconnectCount",
                    "UpIsStale",
                    "DownIsStale",
                    "UpStaleAgeSeconds",
                    "DownStaleAgeSeconds",
                    "LocalTimeUtc",
                ]
            )
        print(f"Created {file_path}")


class PriceAggregator:
    def __init__(self, market_info):
        self.market_info = market_info
        self.latest = {}
        self.last_logged = {}
        self.last_log_time = None
        self.outcome_order = market_info.get("outcomes") or []

    async def handle_update(self, update):
        outcome = update["outcome"]
        self.latest[outcome] = update
        if not self._has_both_outcomes():
            return

        now = update["timestamp"]
        prices_changed = self._prices_changed()
        interval_elapsed = (
            self.last_log_time is None
            or (now - self.last_log_time).total_seconds() >= LOGGING_INTERVAL_SECONDS
        )

        if prices_changed or interval_elapsed:
            self._log_row(now)

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
        up_is_stale, up_stale_age = self._stale_info(up_update, timestamp_dt)
        down_is_stale, down_stale_age = self._stale_info(down_update, timestamp_dt)

        timestamp_et = _format_timestamp(timestamp_dt, TIMEZONE_ET)
        timestamp_uk = _format_timestamp(timestamp_dt, TIMEZONE_UK)
        target_time = self.market_info.get("target_time_utc")
        expiration = self.market_info.get("expiration_time_utc")
        target_time_str = _format_timestamp(target_time, TIMEZONE_UK)
        expiration_str = _format_timestamp(expiration, TIMEZONE_UK)
        local_time_utc = datetime.datetime.now(pytz.utc)

        row = [
            timestamp_et,
            timestamp_uk,
            target_time_str,
            expiration_str,
            up_update["best_bid"],
            down_update["best_bid"],
            up_update["best_ask"],
            down_update["best_ask"],
            up_update["mid"],
            down_update["mid"],
            up_update["spread"],
            down_update["spread"],
            up_update["spread_pct"],
            down_update["spread_pct"],
            up_update["best_bid_size"],
            down_update["best_bid_size"],
            up_update["best_ask_size"],
            down_update["best_ask_size"],
            up_update.get("last_trade_price"),
            down_update.get("last_trade_price"),
            up_update.get("last_trade_size"),
            down_update.get("last_trade_size"),
            up_update.get("last_trade_side"),
            down_update.get("last_trade_side"),
            _format_timestamp_utc(up_update.get("last_trade_ts")),
            _format_timestamp_utc(down_update.get("last_trade_ts")),
            _format_timestamp_utc(up_update.get("server_time_utc")),
            _format_timestamp_utc(down_update.get("server_time_utc")),
            up_update.get("stream_seq_id"),
            down_update.get("stream_seq_id"),
            _format_timestamp_utc(up_update.get("heartbeat_last_seen")),
            up_update.get("reconnect_count"),
            up_is_stale,
            down_is_stale,
            up_stale_age,
            down_stale_age,
            _format_timestamp_utc(local_time_utc),
        ]

        data_file = _get_data_file(timestamp_dt)
        _ensure_csv(data_file)
        with open(data_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        self.last_logged = {
            up_key: up_update,
            down_key: down_update,
        }
        self.last_log_time = timestamp_dt
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

    @staticmethod
    def _stale_info(update, timestamp_dt):
        last_update = update.get("last_update_ts") or update.get("timestamp")
        if not last_update:
            return True, ""
        if last_update.tzinfo is None:
            last_update = pytz.utc.localize(last_update)
        age_seconds = max(0.0, (timestamp_dt - last_update).total_seconds())
        is_stale = age_seconds > STALE_THRESHOLD_SECONDS
        return is_stale, round(age_seconds, 3)


async def run_logger():
    market_info, err = resolve_current_market()
    if err:
        print(f"Error: {err}")
        return

    aggregator = PriceAggregator(market_info)
    ws_logger = PolymarketWebsocketLogger(market_info, aggregator.handle_update)
    await ws_logger.run()


def main():
    print("Starting Data Logger...")
    try:
        asyncio.run(run_logger())
    except KeyboardInterrupt:
        print("\nStopping logger...")


if __name__ == "__main__":
    main()
