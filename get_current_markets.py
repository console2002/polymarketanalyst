import datetime
import pytz
from find_new_market import generate_market_url as generate_polymarket_url
from market_profiles import default_market_profile_key, get_market_profile

def get_current_market_urls(profile_key=None):
    """
    Returns a dictionary with the current active market URL for Polymarket (15-min markets).
    'Current' is defined as the market starting at the current 15-minute interval.
    """
    profile = get_market_profile(profile_key or default_market_profile_key())
    now = datetime.datetime.now(pytz.utc)
    
    # Target time is the current 15-minute mark
    # Example: If now is 12:05, target is 12:00.
    base_time = now.replace(second=0, microsecond=0)
    minutes = base_time.minute
    remainder = minutes % profile.window_minutes
    start_time_utc = base_time - datetime.timedelta(minutes=remainder)
    
    # The "Expiration Time" of this 15-min candle is 15 minutes after start.
    expiration_time_utc = start_time_utc + datetime.timedelta(minutes=profile.window_minutes)

    # 15m Polymarket slugs use the start timestamp.
    polymarket_url = generate_polymarket_url(start_time_utc, profile_key=profile.key)
    
    return {
        "polymarket": polymarket_url,
        "target_time_utc": start_time_utc,
        "expiration_time_utc": expiration_time_utc,
        "target_time_et": start_time_utc.astimezone(pytz.timezone('US/Eastern'))
    }


def get_available_market_urls(num_markets=12, profile_key=None):
    """
    Returns a list of upcoming 15-minute Polymarket URLs starting from the current market.
    """
    profile = get_market_profile(profile_key or default_market_profile_key())
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
