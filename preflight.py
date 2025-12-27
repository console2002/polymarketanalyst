"""Preflight diagnostics for Polymarket data logger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import requests

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
GAMMA_MARKET_SLUG_URL = "https://gamma-api.polymarket.com/markets/slug"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
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
    token_ids: Iterable[str], session: requests.sessions.Session | None = None
) -> bool:
    token_list = [str(token_id) for token_id in token_ids if str(token_id).strip()]
    if not token_list:
        raise PreflightError("preflight.error gamma_validate empty_token_ids")
    session = session or requests
    response = session.get(
        GAMMA_MARKETS_URL,
        params={"clob_token_ids": ",".join(token_list)},
        timeout=DEFAULT_TIMEOUT,
    )
    if response.status_code != 200:
        raise PreflightError(
            "preflight.error gamma_validate "
            f"status={response.status_code} token_ids={','.join(token_list)}"
        )
    return True


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

    validate_clob_token_ids(token_ids)
    emit(
        "preflight.gamma.validate "
        f"status=ok token_ids={','.join(token_ids)}"
    )
    return token_ids
