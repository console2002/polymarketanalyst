import datetime
import pytz
from find_new_market import generate_market_url as generate_polymarket_url
from market_profiles import default_market_profile_key, get_market_profile


_MARKET_TYPE_TO_PROFILE_KEY = {
    "15m": "btc_15m",
    "5m": "btc_5m",
}


def _resolve_profile_key(profile_key=None, market_type=None):
    if profile_key:
        return profile_key
    if market_type:
        return _MARKET_TYPE_TO_PROFILE_KEY.get(market_type, market_type)
    return default_market_profile_key()


def get_current_market_urls(profile_key=None, market_type=None):
    """
    Returns a dictionary with the current active market URL for Polymarket.
    'Current' is defined as the market starting at the current profile interval
    (for example, 5m or 15m depending on ``profile_key``).
    """
    profile = get_market_profile(_resolve_profile_key(profile_key=profile_key, market_type=market_type))
    now = datetime.datetime.now(pytz.utc)
    
    # Target time is the current profile interval mark.
    base_time = now.replace(second=0, microsecond=0)
    minutes = base_time.minute
    remainder = minutes % profile.window_minutes
    start_time_utc = base_time - datetime.timedelta(minutes=remainder)
    
    # Expiration is profile-window minutes after start.
    expiration_time_utc = start_time_utc + datetime.timedelta(minutes=profile.window_minutes)

    # Interval slugs use the start timestamp.
    polymarket_url = generate_polymarket_url(start_time_utc, profile_key=profile.key)
    
    return {
        "polymarket": polymarket_url,
        "target_time_utc": start_time_utc,
        "expiration_time_utc": expiration_time_utc,
        "target_time_et": start_time_utc.astimezone(pytz.timezone('US/Eastern')),
        "cadence_label": f"{profile.window_minutes}m",
    }


def get_available_market_urls(num_markets=12, profile_key=None, market_type=None):
    """
    Returns a list of upcoming Polymarket URLs starting from the current market
    using the selected profile cadence (e.g. 5m/15m).
    """
    profile = get_market_profile(_resolve_profile_key(profile_key=profile_key, market_type=market_type))
    current_market = get_current_market_urls(profile_key=profile.key)
    start_time_utc = current_market["target_time_utc"]
    expiration_time_utc = current_market["expiration_time_utc"]
    et_timezone = pytz.timezone("US/Eastern")
    markets = []

    for i in range(num_markets):
        market_start = start_time_utc + datetime.timedelta(minutes=profile.window_minutes * i)
        market_expiration = expiration_time_utc + datetime.timedelta(minutes=profile.window_minutes * i)
        markets.append(
            {
                "polymarket": generate_polymarket_url(market_start, profile_key=profile.key),
                "target_time_utc": market_start,
                "expiration_time_utc": market_expiration,
                "target_time_et": market_start.astimezone(et_timezone),
                "cadence_label": f"{profile.window_minutes}m",
            }
        )

    return markets

if __name__ == "__main__":
    urls = get_current_market_urls()
    
    print(f"Current Time (UTC): {datetime.datetime.now(pytz.utc)}")
    print(f"Market Start Time (UTC): {urls['target_time_utc']}")
    print(f"Expiration Time (UTC):   {urls['expiration_time_utc']}")
    print("-" * 50)
    print(f"Polymarket: {urls['polymarket']}")
