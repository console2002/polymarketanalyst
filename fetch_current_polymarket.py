import datetime
import json
import time
from email.utils import parsedate_to_datetime

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import SSLError
from urllib3.util.retry import Retry

from get_current_markets import get_current_market_urls

# Configuration
POLYMARKET_API_URL = "https://gamma-api.polymarket.com/events"
POLYMARKET_TIMEOUT = (3, 10)
POLYMARKET_RATE_LIMIT_SECONDS = 1
_POLYMARKET_SESSION = None
_LAST_POLYMARKET_CALL_AT = None


def _get_polymarket_session():
    global _POLYMARKET_SESSION
    if _POLYMARKET_SESSION is None:
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _POLYMARKET_SESSION = session
    return _POLYMARKET_SESSION


def _rate_limit_polymarket_calls():
    global _LAST_POLYMARKET_CALL_AT
    now = time.monotonic()
    if _LAST_POLYMARKET_CALL_AT is not None:
        elapsed = now - _LAST_POLYMARKET_CALL_AT
        if elapsed < POLYMARKET_RATE_LIMIT_SECONDS:
            time.sleep(POLYMARKET_RATE_LIMIT_SECONDS - elapsed)
    _LAST_POLYMARKET_CALL_AT = time.monotonic()


def _get_polymarket_event(slug):
    session = _get_polymarket_session()
    _rate_limit_polymarket_calls()
    for attempt in range(2):
        try:
            response = session.get(
                POLYMARKET_API_URL,
                params={"slug": slug},
                timeout=POLYMARKET_TIMEOUT,
            )
            response.raise_for_status()
            date_header = response.headers.get("Date")
            server_time = parsedate_to_datetime(date_header) if date_header else None
            if server_time:
                server_time = server_time.astimezone(datetime.timezone.utc)
            return response.json(), None, server_time
        except SSLError:
            if attempt == 0:
                time.sleep(1)
                continue
            return None, "Transient network/TLS issue while fetching Polymarket data", None
        except Exception as e:
            return None, str(e), None


def _parse_list_field(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def _parse_iso_datetime(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            return datetime.datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    return None


def _extract_market_times(market):
    start_time = _parse_iso_datetime(market.get("startDate")) or _parse_iso_datetime(market.get("startTime"))
    end_time = _parse_iso_datetime(market.get("endDate")) or _parse_iso_datetime(market.get("endTime"))
    if not end_time:
        end_time = _parse_iso_datetime(market.get("expiration"))
    return start_time, end_time


def get_polymarket_metadata(slug):
    try:
        data, fetch_err, server_time = _get_polymarket_event(slug)
        if fetch_err:
            return None, fetch_err

        if not data:
            return None, "Event not found"

        event = data[0]
        markets = event.get("markets", [])
        if not markets:
            return None, "Markets not found in event"

        market = markets[0]
        clob_token_ids = _parse_list_field(market.get("clobTokenIds", []))
        outcomes = _parse_list_field(market.get("outcomes", []))
        if len(clob_token_ids) < 2:
            return None, "Unexpected number of tokens"

        start_time, end_time = _extract_market_times(market)

        return {
            "slug": slug,
            "clob_token_ids": clob_token_ids,
            "outcomes": outcomes,
            "start_time": start_time,
            "end_time": end_time,
            "polymarket_time_utc": server_time,
        }, None
    except Exception as e:
        return None, str(e)


def fetch_polymarket_data_struct():
    """
    Fetches current Polymarket market metadata (token IDs, outcomes, times).
    """
    try:
        market_info = get_current_market_urls()
        polymarket_url = market_info["polymarket"]
        target_time_utc = market_info["target_time_utc"]
        expiration_time_utc = market_info["expiration_time_utc"]

        slug = polymarket_url.split("/")[-1]

        poly_data, poly_err = get_polymarket_metadata(slug)

        if poly_err:
            return None, f"Polymarket Error: {poly_err}"

        target_time_utc = poly_data.get("start_time") or target_time_utc
        expiration_time_utc = poly_data.get("end_time") or expiration_time_utc

        result = {
            "slug": slug,
            "clob_token_ids": poly_data.get("clob_token_ids", []),
            "outcomes": poly_data.get("outcomes", []),
            "target_time_utc": target_time_utc,
            "expiration_time_utc": expiration_time_utc,
            "polymarket_time_utc": poly_data.get("polymarket_time_utc"),
        }
        return result, None
    except Exception as e:
        return None, str(e)
