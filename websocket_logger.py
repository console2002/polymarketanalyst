import asyncio
import datetime
import json
import random
from dataclasses import dataclass, field

import pytz
import websockets

CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
RTDS_WS_URL = "wss://ws-live-data.polymarket.com"
CLOB_HEARTBEAT_SECONDS = 10
RTDS_HEARTBEAT_SECONDS = 5
MAX_BACKOFF_SECONDS = 60


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


def _find_token_id(payload):
    for key in ("token_id", "tokenId", "token", "tokenID"):
        if key in payload:
            return str(payload[key])
    return None


class PolymarketWebsocketLogger:
    def __init__(self, market_info, on_price_update):
        self.market_info = market_info
        self.on_price_update = on_price_update
        self.token_id_to_outcome = {
            str(token_id): outcome
            for outcome, token_id in zip(
                market_info["outcomes"], market_info["clob_token_ids"]
            )
        }
        self.order_books = {
            str(token_id): OrderBook() for token_id in market_info["clob_token_ids"]
        }
        self._shutdown = asyncio.Event()

    async def run(self):
        await asyncio.gather(
            self._run_clob_socket(),
            self._run_rtds_socket(),
        )

    async def shutdown(self):
        self._shutdown.set()

    async def _run_clob_socket(self):
        backoff = 1
        while not self._shutdown.is_set():
            try:
                async with websockets.connect(
                    CLOB_WS_URL,
                    ping_interval=None,
                    max_queue=None,
                ) as ws:
                    await self._subscribe_clob(ws)
                    heartbeat = asyncio.create_task(self._clob_heartbeat(ws))
                    backoff = 1
                    async for message in ws:
                        await self._handle_clob_message(message)
                    heartbeat.cancel()
            except Exception:
                await asyncio.sleep(self._next_backoff(backoff))
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

    async def _run_rtds_socket(self):
        backoff = 1
        while not self._shutdown.is_set():
            try:
                async with websockets.connect(
                    RTDS_WS_URL,
                    ping_interval=None,
                    max_queue=None,
                ) as ws:
                    await self._subscribe_rtds(ws)
                    heartbeat = asyncio.create_task(self._rtds_heartbeat(ws))
                    backoff = 1
                    async for _message in ws:
                        if self._shutdown.is_set():
                            break
                    heartbeat.cancel()
            except Exception:
                await asyncio.sleep(self._next_backoff(backoff))
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

    async def _subscribe_clob(self, ws):
        token_ids = [str(token_id) for token_id in self.market_info["clob_token_ids"]]
        subscribe = {"type": "subscribe", "channel": "book", "token_ids": token_ids}
        await ws.send(json.dumps(subscribe))
        snapshot_request = {
            "type": "snapshot",
            "channel": "book",
            "token_ids": token_ids,
        }
        await ws.send(json.dumps(snapshot_request))

    async def _subscribe_rtds(self, ws):
        token_ids = [str(token_id) for token_id in self.market_info["clob_token_ids"]]
        payload = {
            "type": "subscribe",
            "channel": "trades",
            "token_ids": token_ids,
        }
        await ws.send(json.dumps(payload))

    async def _clob_heartbeat(self, ws):
        while not self._shutdown.is_set():
            await asyncio.sleep(CLOB_HEARTBEAT_SECONDS)
            try:
                await ws.send("PING")
            except Exception:
                break

    async def _rtds_heartbeat(self, ws):
        while not self._shutdown.is_set():
            await asyncio.sleep(RTDS_HEARTBEAT_SECONDS)
            try:
                await ws.ping()
            except Exception:
                break

    async def _handle_clob_message(self, message):
        if message == "PONG":
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        token_id = _find_token_id(payload)
        if token_id is None:
            return
        if token_id not in self.order_books:
            return

        book = self.order_books[token_id]
        event_type = payload.get("event_type") or payload.get("type")
        bids = payload.get("bids")
        asks = payload.get("asks")

        if event_type in ("book", "snapshot", "book_snapshot") and (bids or asks):
            if bids is not None:
                book.replace_levels("bids", _parse_levels(bids))
            if asks is not None:
                book.replace_levels("asks", _parse_levels(asks))
            await self._emit_price_update(token_id)
            return

        if event_type in ("book_delta", "delta", "update") and (bids or asks):
            if bids is not None:
                book.update_levels("bids", _parse_levels(bids))
            if asks is not None:
                book.update_levels("asks", _parse_levels(asks))
            await self._emit_price_update(token_id)

    async def _emit_price_update(self, token_id):
        book = self.order_books[token_id]
        best_bid, best_bid_size = book.best_bid()
        best_ask, best_ask_size = book.best_ask()
        mid = book.mid()
        outcome = self.token_id_to_outcome.get(token_id, token_id)
        payload = {
            "timestamp": datetime.datetime.now(pytz.utc),
            "token_id": token_id,
            "outcome": outcome,
            "best_bid": best_bid,
            "best_bid_size": best_bid_size,
            "best_ask": best_ask,
            "best_ask_size": best_ask_size,
            "mid": mid,
        }
        await self.on_price_update(payload)

    @staticmethod
    def _next_backoff(current):
        jitter = random.uniform(0.0, 0.5)
        return min(current + jitter, MAX_BACKOFF_SECONDS)
