"""Preflight diagnostics for Polymarket data logger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import requests

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
GAMMA_MARKET_SLUG_URL = "https://gamma-api.polymarket.com/markets/slug"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
DEFAULT_TIMEOUT = (3, 10)


class PreflightError(RuntimeError):
    """Raised when a preflight check fails."""


@dataclass(frozen=True)
class GeoblockResult:
    blocked: bool
    ip: str | None
    country: str | None
    region: str | None
    raw: dict


def check_geoblock(session: requests.sessions.Session | None = None) -> GeoblockResult:
    session = session or requests
    response = session.get(GEOBLOCK_URL, timeout=DEFAULT_TIMEOUT)
    if response.status_code != 200:
        raise PreflightError(
            f"preflight.error geoblock status={response.status_code}"
        )
    payload = response.json() if response.content else {}
    return GeoblockResult(
        blocked=bool(payload.get("blocked")),
        ip=payload.get("ip"),
        country=payload.get("country"),
        region=payload.get("region"),
        raw=payload,
    )


def fetch_market_by_slug(
    slug: str, session: requests.sessions.Session | None = None
) -> dict:
    session = session or requests
    response = session.get(
        f"{GAMMA_MARKET_SLUG_URL}/{slug}",
        timeout=DEFAULT_TIMEOUT,
    )
    if response.status_code == 404:
        raise PreflightError(f"preflight.error gamma_slug_not_found slug={slug}")
    if response.status_code != 200:
        raise PreflightError(
            f"preflight.error gamma_slug status={response.status_code} slug={slug}"
        )
    return response.json()


def parse_clob_token_ids(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        cleaned = raw.strip("[]")
        tokens = [token.strip().strip("'\"") for token in cleaned.split(",")]
        return [token for token in tokens if token]
    return []


def validate_clob_token_ids(
    token_ids: Iterable[str],
    slug: str | None = None,
    session: requests.sessions.Session | None = None,
) -> tuple[str, int | None]:
    token_list = [str(token_id) for token_id in token_ids if str(token_id).strip()]
    if not token_list:
        raise PreflightError("preflight.error gamma_validate empty_token_ids")
    session = session or requests
    response = session.get(
        GAMMA_MARKETS_URL,
        params={"clob_token_ids": token_list},
        timeout=DEFAULT_TIMEOUT,
    )
    if response.status_code == 200:
        payload = response.json() if response.content else []
        markets = _parse_gamma_markets(payload)
        if markets:
            if slug and any(
                market.get("slug") == slug for market in markets if isinstance(market, dict)
            ):
                return "gamma", len(markets)
            if any(
                _market_has_tokens(market, token_list)
                for market in markets
                if isinstance(market, dict)
            ):
                return "gamma", len(markets)
            return "gamma", len(markets)
        return _validate_clob_books(token_list, session)
    if response.status_code == 422 or response.status_code >= 500:
        return _validate_clob_books(token_list, session)
    body_snippet = _format_response_snippet(response)
    message = (
        "preflight.error gamma_validate "
        f"status={response.status_code} token_ids={','.join(token_list)}"
    )
    if body_snippet:
        message = f"{message} body={body_snippet}"
    raise PreflightError(message)


def run_preflight(slug: str, emit=print) -> list[str]:
    geoblock = check_geoblock()
    emit(
        "preflight.geoblock "
        f"blocked={str(geoblock.blocked).lower()} "
        f"ip={geoblock.ip or ''} "
        f"country={geoblock.country or ''} "
        f"region={geoblock.region or ''}"
    )
    if geoblock.blocked:
        raise PreflightError(
            "preflight.error geoblock blocked=true "
            f"ip={geoblock.ip or ''} "
            f"country={geoblock.country or ''} "
            f"region={geoblock.region or ''}"
        )

    market = fetch_market_by_slug(slug)
    token_ids = parse_clob_token_ids(market.get("clobTokenIds"))
    emit(
        "preflight.gamma.market "
        f"slug={slug} token_ids={','.join(token_ids)} count={len(token_ids)}"
    )
    if len(token_ids) < 2:
        raise PreflightError(
            "preflight.error gamma_slug invalid_token_count "
            f"slug={slug} count={len(token_ids)}"
        )

    validator, count = validate_clob_token_ids(token_ids, slug=slug)
    if validator == "gamma":
        emit(f"preflight.gamma_validate ok count={count}")
    else:
        for token_id in token_ids:
            emit(f"preflight.clob_book_validate ok token_id={token_id}")
    return token_ids


def _parse_gamma_markets(payload: object) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    return []


def _market_has_tokens(market: dict, token_ids: list[str]) -> bool:
    raw_tokens = market.get("clobTokenIds")
    parsed = parse_clob_token_ids(raw_tokens)
    return all(token_id in parsed for token_id in token_ids)


def _format_response_snippet(
    response: requests.Response, limit: int = 200
) -> str:
    text = getattr(response, "text", "") or ""
    snippet = text.strip()
    if not snippet:
        return ""
    if len(snippet) > limit:
        return f"{snippet[:limit]}..."
    return snippet


def _validate_clob_books(
    token_ids: list[str], session: requests.sessions.Session
) -> tuple[str, int | None]:
    for token_id in token_ids:
        response = session.get(
            CLOB_BOOK_URL,
            params={"token_id": token_id},
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code != 200:
            body_snippet = _format_response_snippet(response)
            message = (
                "preflight.error clob_book_validate "
                f"status={response.status_code} token_id={token_id}"
            )
            if body_snippet:
                message = f"{message} body={body_snippet}"
            raise PreflightError(message)
    return "clob_book", None
