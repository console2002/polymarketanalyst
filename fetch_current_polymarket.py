import requests
import time
import datetime
import pytz
from email.utils import parsedate_to_datetime
from requests.adapters import HTTPAdapter
from requests.exceptions import SSLError
from urllib3.util.retry import Retry
from get_current_markets import get_current_market_urls

# Configuration
POLYMARKET_API_URL = "https://gamma-api.polymarket.com/events"
CLOB_API_URL = "https://clob.polymarket.com/book"
POLYMARKET_TIMEOUT = (3, 10)
POLYMARKET_RATE_LIMIT_SECONDS = 1
_POLYMARKET_SESSION = None
_LAST_POLYMARKET_CALL_AT = None
_LAST_GOOD_POLYMARKET_DATA = None


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

def get_clob_best_quotes(token_id):
    try:
        response = requests.get(CLOB_API_URL, params={"token_id": token_id})
        response.raise_for_status()
        data = response.json()
        
        # data structure: {'bids': [{'price': '0.38', 'size': '...'}, ...], 'asks': ...}
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        best_bid = 0.0
        best_bid_size = 0.0
        best_ask = 0.0
        best_ask_size = 0.0
        
        if bids:
            # Bids: We want the HIGHEST price someone is willing to pay
            best_bid_quote = max(bids, key=lambda b: float(b["price"]))
            best_bid = float(best_bid_quote["price"])
            best_bid_size = float(best_bid_quote.get("size", 0.0))
            
        if asks:
            # Asks: We want the LOWEST price someone is willing to sell for
            best_ask_quote = min(asks, key=lambda a: float(a["price"]))
            best_ask = float(best_ask_quote["price"])
            best_ask_size = float(best_ask_quote.get("size", 0.0))
            
        return {
            "best_bid": best_bid,
            "best_bid_size": best_bid_size,
            "best_ask": best_ask,
            "best_ask_size": best_ask_size,
        }
    except Exception as e:
        return None

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

def get_polymarket_data(slug):
    try:
        # 1. Get Event Details to find Token IDs
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
        
        # Get Token IDs
        # clobTokenIds is a list of strings
        clob_token_ids = eval(market.get("clobTokenIds", "[]"))
        outcomes = eval(market.get("outcomes", "[]"))
        
        if len(clob_token_ids) != 2:
            return None, "Unexpected number of tokens"
            
        # 2. Fetch Price for each Token from CLOB
        prices = {}
        volumes = {}
        # Assuming order is [Up, Down] or matches outcomes
        # Usually outcomes are ["Up", "Down"] and clobTokenIds correspond.
        
        for outcome, token_id in zip(outcomes, clob_token_ids):
            quote = get_clob_best_quotes(token_id)
            if quote is not None:
                prices[outcome] = quote["best_ask"] if quote["best_ask"] > 0 else 0.0
                volumes[outcome] = quote["best_ask_size"]
            else:
                prices[outcome] = 0.0
                volumes[outcome] = 0.0
            
        start_time, end_time = _extract_market_times(market)

        return {
            "prices": prices,
            "volumes": volumes,
            "start_time": start_time,
            "end_time": end_time,
            "polymarket_time_utc": server_time,
        }, None
    except Exception as e:
        return None, str(e)



def fetch_polymarket_data_struct():
    """
    Fetches current Polymarket data and returns a structured dictionary.
    """
    global _LAST_GOOD_POLYMARKET_DATA
    try:
        # Get current market info
        market_info = get_current_market_urls()
        polymarket_url = market_info["polymarket"]
        target_time_utc = market_info["target_time_utc"]
        expiration_time_utc = market_info["expiration_time_utc"]
        
        # Extract slug from URL
        slug = polymarket_url.split("/")[-1]
        
        # Fetch Data
        poly_data, poly_err = get_polymarket_data(slug)
        
        if poly_err:
            if _LAST_GOOD_POLYMARKET_DATA:
                cached = dict(_LAST_GOOD_POLYMARKET_DATA)
                cached["is_cached"] = True
                cached["cache_age_seconds"] = time.time() - cached.get("fetched_at", time.time())
                return cached, None
            return None, f"Polymarket Error: {poly_err}"
            
        target_time_utc = poly_data.get("start_time") or target_time_utc
        expiration_time_utc = poly_data.get("end_time") or expiration_time_utc

        result = {
            "prices": poly_data.get("prices", {}), # {'Up': 0.xx, 'Down': 0.xx}
            "slug": slug,
            "target_time_utc": target_time_utc,
            "expiration_time_utc": expiration_time_utc,
            "polymarket_time_utc": poly_data.get("polymarket_time_utc"),
            "fetched_at": time.time(),
        }
        _LAST_GOOD_POLYMARKET_DATA = result
        return result, None        
    except Exception as e:
        return None, str(e)

def main():
    data, err = fetch_polymarket_data_struct()
    
    if err:
        print(f"Error: {err}")
        return

    print(f"Fetching data for: {data['slug']}")
    print(f"Target Time (UTC): {data['target_time_utc']}")
    print("-" * 50)
    
    up_price = data['prices'].get("Up", 0)
    down_price = data['prices'].get("Down", 0)
    print(f"BUY: UP ${up_price:.3f} & DOWN ${down_price:.3f}")

if __name__ == "__main__":
    main()
