import asyncio
import contextlib
import datetime
import json
import logging
import random
from dataclasses import dataclass, field

import pytz
import websockets

CLOB_MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CLOB_PING_SECONDS = 10
MAX_BACKOFF_SECONDS = 5
STALE_THRESHOLD_SECONDS = 15

logger = logging.getLogger(__name__)


@dataclass
class OrderBook:
    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)

    def update_levels(self, side, levels):
        book = self.bids if side == "bids" else self.asks
        for price, size in levels:
            if size <= 0:
                book.pop(price, None)
            else:
                book[price] = size

    def replace_levels(self, side, levels):
        book = self.bids if side == "bids" else self.asks
        book.clear()
        for price, size in levels:
            if size > 0:
                book[price] = size

    def best_bid(self):
        if not self.bids:
            return 0.0, 0.0
        price = max(self.bids)
        return price, self.bids.get(price, 0.0)

    def best_ask(self):
        if not self.asks:
            return 0.0, 0.0
        price = min(self.asks)
        return price, self.asks.get(price, 0.0)

    def mid(self):
        bid, _ = self.best_bid()
        ask, _ = self.best_ask()
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return 0.0


def _parse_levels(raw_levels):
    levels = []
    if not raw_levels:
        return levels
    for level in raw_levels:
        if isinstance(level, dict):
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
        else:
            price = float(level[0]) if len(level) > 0 else 0
            size = float(level[1]) if len(level) > 1 else 0
        levels.append((price, size))
    return levels


def _extract_book_sides(payload):
    bids = payload.get("bids")
    asks = payload.get("asks")
    if bids is None and asks is None:
        buys = payload.get("buys")
        sells = payload.get("sells")
        if buys is not None or sells is not None:
            bids = buys
            asks = sells
    return bids, asks


def _find_token_id(payload):
    for key in ("token_id", "tokenId", "token", "tokenID", "asset_id", "assetId"):
        if key in payload:
            return str(payload[key])
    return None


def _parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(datetime.timezone.utc)
    if isinstance(value, (int, float)):
        seconds = value
        if value > 1_000_000_000_000:
            seconds = value / 1000
        elif value > 10_000_000_000:
            seconds = value / 1000
        return datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return _parse_timestamp(float(raw))
        except ValueError:
            try:
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                parsed = datetime.datetime.fromisoformat(raw)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=datetime.timezone.utc)
                return parsed.astimezone(datetime.timezone.utc)
            except ValueError:
                return None
    return None


def _find_server_time(payload):
    for key in (
        "server_time_utc",
        "server_time",
        "serverTime",
        "server_timestamp",
        "timestamp",
        "ts",
    ):
        if key in payload:
            return _parse_timestamp(payload[key])
    return None


def _normalize_trade_side(value):
    if not value:
        return ""
    return str(value).lower()


def _coerce_asset_id(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            try:
                return int(stripped)
            except ValueError:
                return stripped
        return stripped
    return value


class PolymarketWebsocketLogger:
    def __init__(self, market_info, on_price_update, clob_ping_interval=CLOB_PING_SECONDS):
        self.market_info = market_info
        self.on_price_update = on_price_update
        self._clob_ping_interval = clob_ping_interval
        self._asset_ids = [
            _coerce_asset_id(token_id)
            for token_id in market_info.get("clob_token_ids", [])
        ]
        self.token_id_to_outcome = {
            str(token_id): outcome
            for outcome, token_id in zip(
                market_info["outcomes"], market_info["clob_token_ids"]
            )
        }
        self.order_books = {
            str(token_id): OrderBook() for token_id in market_info["clob_token_ids"]
        }
        self.last_trade = {
            str(token_id): {
                "price": None,
                "size": None,
                "side": "",
                "timestamp": None,
            }
            for token_id in market_info["clob_token_ids"]
        }
        self.last_server_time = {str(token_id): None for token_id in market_info["clob_token_ids"]}
        self.last_update = {str(token_id): None for token_id in market_info["clob_token_ids"]}
        self.last_heartbeat = None
        self.clob_frames_received = 0
        self._shutdown = asyncio.Event()

    async def run(self):
        await self._run_clob_socket()

    async def shutdown(self):
        self._shutdown.set()

    async def _run_clob_socket(self):
        backoff = 1
        while not self._shutdown.is_set():
            try:
                async with websockets.connect(
                    CLOB_MARKET_WS_URL,
                    ping_interval=None,
                    max_queue=None,
                ) as ws:
                    print("CLOB connected")
                    await self._subscribe_clob(ws)
                    heartbeat = asyncio.create_task(self._clob_heartbeat(ws))
                    backoff = 1
                    try:
                        async for message in ws:
                            await self._handle_clob_message(message)
                    finally:
                        heartbeat.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await heartbeat
            except websockets.exceptions.InvalidStatusCode as exc:
                if exc.status_code == 403:
                    print(
                        "CLOB error: websocket handshake failed with HTTP 403. "
                        "Check geoblock status before retrying."
                    )
                    self._shutdown.set()
                    return
                await asyncio.sleep(self._next_backoff(backoff))
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            except Exception:
                await asyncio.sleep(self._next_backoff(backoff))
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

    async def _subscribe_clob(self, ws):
        payload = {"type": "market", "asset_ids": self._asset_ids}
        print(f"CLOB subscribe: {json.dumps(payload)}")
        await ws.send(json.dumps(payload))

    async def _clob_heartbeat(self, ws):
        while not self._shutdown.is_set():
            await asyncio.sleep(self._clob_ping_interval)
            try:
                await ws.send("PING")
            except Exception:
                break

    async def _handle_clob_message(self, message):
        self.clob_frames_received += 1
        if message == "PONG":
            self._mark_heartbeat()
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("CLOB message parse failure: %s", message)
            return

        if not isinstance(payload, dict):
            return

        self._mark_heartbeat()
        token_id = _find_token_id(payload)
        event_type = payload.get("event_type") or payload.get("type")
        bids, asks = _extract_book_sides(payload)
        books = payload.get("books") or payload.get("order_books")
        price_changes = payload.get("price_changes") or payload.get("priceChanges")
        server_time = _find_server_time(payload)
        if event_type in ("heartbeat", "heart_beat"):
            return

        if token_id is not None:
            if token_id not in self.order_books:
                return
            if server_time:
                self.last_server_time[token_id] = server_time
            book = self.order_books[token_id]
        else:
            book = None

        if event_type in ("book", "snapshot", "book_snapshot"):
            if bids is not None or asks is not None:
                if not book:
                    return
                if bids is not None:
                    book.replace_levels("bids", _parse_levels(bids))
                if asks is not None:
                    book.replace_levels("asks", _parse_levels(asks))
                await self._emit_price_update(token_id)
                return
            if isinstance(books, list):
                for entry in books:
                    if not isinstance(entry, dict):
                        continue
                    entry_token = _find_token_id(entry)
                    if entry_token not in self.order_books:
                        continue
                    if server_time:
                        self.last_server_time[entry_token] = server_time
                    entry_book = self.order_books[entry_token]
                    entry_bids, entry_asks = _extract_book_sides(entry)
                    if entry_bids is not None:
                        entry_book.replace_levels("bids", _parse_levels(entry_bids))
                    if entry_asks is not None:
                        entry_book.replace_levels("asks", _parse_levels(entry_asks))
                    await self._emit_price_update(entry_token)
                return

        if event_type in ("book_delta", "delta", "update") and (
            bids is not None or asks is not None
        ):
            if not book:
                return
            if bids is not None:
                book.update_levels("bids", _parse_levels(bids))
            if asks is not None:
                book.update_levels("asks", _parse_levels(asks))
            await self._emit_price_update(token_id)
            return

        if event_type in ("price_change", "price_change_event") and price_changes:
            for change in price_changes:
                if not isinstance(change, dict):
                    continue
                change_token = _find_token_id(change)
                if change_token not in self.order_books:
                    continue
                if server_time:
                    self.last_server_time[change_token] = server_time
                side = _normalize_trade_side(change.get("side"))
                price = change.get("price")
                size = change.get("size")
                if price is None or size is None:
                    continue
                try:
                    price = float(price)
                    size = float(size)
                except (TypeError, ValueError):
                    continue
                book = self.order_books[change_token]
                if side == "buy":
                    book.update_levels("bids", [(price, size)])
                elif side == "sell":
                    book.update_levels("asks", [(price, size)])
                await self._emit_price_update(change_token)
            return

        if event_type in ("last_trade_price", "last_trade", "trade"):
            if token_id is None or token_id not in self.last_trade:
                return
            price = payload.get("price") or payload.get("trade_price")
            size = payload.get("size") or payload.get("amount") or payload.get("quantity")
            side = payload.get("side") or payload.get("trade_side")
            timestamp = (
                _parse_timestamp(payload.get("timestamp"))
                or _parse_timestamp(payload.get("ts"))
                or _parse_timestamp(payload.get("created_at"))
            )
            if price is not None:
                try:
                    price = float(price)
                except (TypeError, ValueError):
                    price = None
            if size is not None:
                try:
                    size = float(size)
                except (TypeError, ValueError):
                    size = None
            if timestamp is None:
                timestamp = _find_server_time(payload)
            self.last_trade[token_id] = {
                "price": price,
                "size": size,
                "side": _normalize_trade_side(side),
                "timestamp": timestamp,
            }

    async def _emit_price_update(self, token_id):
        book = self.order_books[token_id]
        best_bid, best_bid_size = book.best_bid()
        best_ask, best_ask_size = book.best_ask()
        mid = book.mid()
        spread = best_ask - best_bid if best_ask and best_bid else 0.0
        spread_pct = (spread / mid) if mid else 0.0
        outcome = self.token_id_to_outcome.get(token_id, token_id)
        last_trade = self.last_trade.get(token_id, {})
        server_time = self.last_server_time.get(token_id)
        heartbeat_last_seen = self.last_heartbeat
        now = datetime.datetime.now(pytz.utc)
        self.last_update[token_id] = now
        payload = {
            "timestamp": now,
            "token_id": token_id,
            "outcome": outcome,
            "best_bid": best_bid,
            "best_bid_size": best_bid_size,
            "best_ask": best_ask,
            "best_ask_size": best_ask_size,
            "mid": mid,
            "spread": spread,
            "spread_pct": spread_pct,
            "last_trade_price": last_trade.get("price"),
            "last_trade_size": last_trade.get("size"),
            "last_trade_side": last_trade.get("side"),
            "last_trade_ts": last_trade.get("timestamp"),
            "server_time_utc": server_time,
            "heartbeat_last_seen": heartbeat_last_seen,
            "last_update_ts": self.last_update[token_id],
        }
        await self.on_price_update(payload)

    def _mark_heartbeat(self):
        self.last_heartbeat = datetime.datetime.now(pytz.utc)

    @staticmethod
    def _next_backoff(current):
        jitter = random.uniform(0.0, 0.5)
        return min(current + jitter, MAX_BACKOFF_SECONDS)
