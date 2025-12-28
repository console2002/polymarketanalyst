import asyncio
import json

import websocket_logger


def _sample_market_info():
    return {
        "clob_token_ids": ["1", "2"],
        "outcomes": ["Yes", "No"],
    }


def test_build_clob_ws_url():
    assert (
        websocket_logger.build_clob_ws_url()
        == "wss://ws-subscriptions-clob.polymarket.com"
    )


def test_handle_clob_book_emits_price_update():
    updates = []

    async def on_price_update(payload):
        updates.append(payload)

    logger = websocket_logger.PolymarketWebsocketLogger(
        _sample_market_info(), on_price_update
    )
    message = json.dumps(
        {
            "event_type": "book",
            "token_id": "1",
            "bids": [[0.4, 10]],
            "asks": [[0.6, 5]],
        }
    )

    async def run():
        await logger._handle_clob_message(message)

    asyncio.run(run())
    assert len(updates) == 1
    assert updates[0]["token_id"] == "1"
    assert updates[0]["best_bid"] == 0.4
    assert updates[0]["best_ask"] == 0.6


def test_clob_keepalive_sends_ping():
    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    updates = []

    async def on_price_update(payload):
        updates.append(payload)

    logger = websocket_logger.PolymarketWebsocketLogger(
        _sample_market_info(), on_price_update, clob_ping_interval=0.01
    )
    ws = FakeWebSocket()

    async def run():
        task = asyncio.create_task(logger._clob_heartbeat(ws))
        await asyncio.sleep(0.03)
        logger._shutdown.set()
        await asyncio.wait_for(task, timeout=0.1)

    asyncio.run(run())
    assert "PING" in ws.sent
