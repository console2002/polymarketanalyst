import datetime
import json
import logging
import re
import time
from email.utils import parsedate_to_datetime

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import SSLError
from urllib3.util.retry import Retry

from get_current_markets import get_current_market_urls
from find_new_market import generate_market_url


# Configuration
POLYMARKET_API_URL = "https://gamma-api.polymarket.com/events"
GAMMA_MARKET_SLUG_URL = "https://gamma-api.polymarket.com/markets/slug"
POLYMARKET_TIMEOUT = (3, 10)
POLYMARKET_RATE_LIMIT_SECONDS = 1
_POLYMARKET_SESSION = None
_LAST_POLYMARKET_CALL_AT = None
_FIFTEEN_MINUTES = datetime.timedelta(minutes=15)
_LOGGER = logging.getLogger(__name__)


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


def _get_gamma_market_by_slug(slug):
    session = _get_polymarket_session()
    _rate_limit_polymarket_calls()
    try:
        response = session.get(
            f"{GAMMA_MARKET_SLUG_URL}/{slug}",
            timeout=POLYMARKET_TIMEOUT,
        )
    except Exception as e:
        return None, f"Gamma request failed: {e}", None

    if response.status_code == 404:
        return None, "gamma_slug_not_found", response.status_code
    if response.status_code != 200:
        return None, f"Gamma status {response.status_code}", response.status_code

    try:
        return response.json(), None, response.status_code
    except Exception as e:
        return None, f"Gamma invalid JSON: {e}", response.status_code


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


def _parse_clob_token_ids(value):
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


def _market_window_from_slug(slug):
    match = re.search(r"btc-updown-15m-(\d+)", slug or "")
    if not match:
        return None, None
    try:
        target_time = datetime.datetime.fromtimestamp(
            int(match.group(1)),
            tz=datetime.timezone.utc,
        )
    except (ValueError, OSError, OverflowError):
        return None, None
    expiration = target_time + _FIFTEEN_MINUTES
    return target_time, expiration


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


def resolve_market_by_slug(slug):
    gamma_market, gamma_err, status_code = _get_gamma_market_by_slug(slug)
    if gamma_err:
        if status_code == 404:
            return None, f"gamma_slug_not_found slug={slug}"
        fallback_data, fallback_err = get_polymarket_metadata(slug)
        if fallback_err:
            return None, (
                f"Gamma Error: {gamma_err}; Fallback Error: {fallback_err}"
            )
        _LOGGER.warning(
            "Gamma slug lookup failed; falling back to event metadata. "
            "slug=%s error=%s",
            slug,
            gamma_err,
        )
        return {
            "slug": slug,
            "clob_token_ids": fallback_data.get("clob_token_ids", []),
            "outcomes": fallback_data.get("outcomes", []),
            "start_time": fallback_data.get("start_time"),
            "end_time": fallback_data.get("end_time"),
            "polymarket_time_utc": fallback_data.get("polymarket_time_utc"),
        }, None

    clob_token_ids = _parse_clob_token_ids(gamma_market.get("clobTokenIds"))
    if len(clob_token_ids) != 2:
        return (
            None,
            "gamma_slug_invalid_clob_token_ids "
            f"slug={slug} count={len(clob_token_ids)}",
        )

    outcomes = _parse_list_field(gamma_market.get("outcomes", []))
    start_time, end_time = _extract_market_times(gamma_market)

    return {
        "slug": slug,
        "clob_token_ids": clob_token_ids,
        "outcomes": outcomes,
        "start_time": start_time,
        "end_time": end_time,
        "polymarket_time_utc": None,
    }, None


def fetch_polymarket_data_struct():
    """
    Fetches current Polymarket market metadata (token IDs, outcomes, times).
    """
    return resolve_current_market()


def resolve_market_by_start_time(start_time_utc):
    if start_time_utc is None:
        return None, "Start time is required"
    if start_time_utc.tzinfo is None:
        start_time_utc = start_time_utc.replace(tzinfo=datetime.timezone.utc)
    polymarket_url = generate_market_url(start_time_utc)
    slug = polymarket_url.split("/")[-1]
    poly_data, poly_err = resolve_market_by_slug(slug)
    if poly_err:
        return None, f"Polymarket Error: {poly_err}"
    target_time_utc, expiration_time_utc = _market_window_from_slug(slug)
    result = {
        "slug": slug,
        "polymarket_url": polymarket_url,
        "clob_token_ids": poly_data.get("clob_token_ids", []),
        "outcomes": poly_data.get("outcomes", []),
        "target_time_utc": target_time_utc,
        "expiration_time_utc": expiration_time_utc,
        "polymarket_time_utc": poly_data.get("polymarket_time_utc"),
    }
    return result, None


def resolve_market_by_expiration(expiration_time_utc):
    if expiration_time_utc is None:
        return None, "Expiration time is required"
    if expiration_time_utc.tzinfo is None:
        expiration_time_utc = expiration_time_utc.replace(tzinfo=datetime.timezone.utc)
    start_time_utc = expiration_time_utc - _FIFTEEN_MINUTES
    return resolve_market_by_start_time(start_time_utc)


def resolve_current_market():
    """
    Resolve the active market slug and token IDs with a single Polymarket API call.
    """
    try:
        market_info = get_current_market_urls()
        target_time_utc = market_info["target_time_utc"]
        expiration_time_utc = market_info["expiration_time_utc"]

        candidate_times = [target_time_utc + _FIFTEEN_MINUTES * offset for offset in range(3)]
        last_error = None

        for candidate_time in candidate_times:
            polymarket_url = generate_market_url(candidate_time)
            slug = polymarket_url.split("/")[-1]

            poly_data, poly_err = resolve_market_by_slug(slug)
            if poly_err:
                last_error = poly_err
                if "gamma_slug_not_found" in poly_err:
                    continue
                return None, f"Polymarket Error: {poly_err}"

            slug_target_time, slug_expiration_time = _market_window_from_slug(slug)
            target_time_utc = slug_target_time or poly_data.get("start_time") or candidate_time
            expiration_time_utc = (
                slug_expiration_time
                or poly_data.get("end_time")
                or (target_time_utc + _FIFTEEN_MINUTES)
            )

            result = {
                "slug": slug,
                "polymarket_url": polymarket_url,
                "clob_token_ids": poly_data.get("clob_token_ids", []),
                "outcomes": poly_data.get("outcomes", []),
                "target_time_utc": target_time_utc,
                "expiration_time_utc": expiration_time_utc,
                "polymarket_time_utc": poly_data.get("polymarket_time_utc"),
            }
            return result, None

        return None, (
            "Polymarket Error: no valid slug found after checking 3 windows "
            f"last_error={last_error}"
        )
    except Exception as e:
        return None, str(e)
