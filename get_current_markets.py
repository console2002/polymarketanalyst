import datetime
import pytz
from find_new_market import generate_market_url as generate_polymarket_url

def get_current_market_urls():
    """
    Returns a dictionary with the current active market URL for Polymarket (15-min markets).
    'Current' is defined as the market starting at the current 15-minute interval.
    """
    now = datetime.datetime.now(pytz.utc)
    
    # Target time is the current 15-minute mark
    # Example: If now is 12:05, target is 12:00.
    base_time = now.replace(second=0, microsecond=0)
    minutes = base_time.minute
    remainder = minutes % 15
    start_time_utc = base_time - datetime.timedelta(minutes=remainder)
    
    # The "Expiration Time" of this 15-min candle is 15 minutes after start.
    expiration_time_utc = start_time_utc + datetime.timedelta(minutes=15)

    # 15m Polymarket slugs use the start timestamp.
    polymarket_url = generate_polymarket_url(start_time_utc)
    
    return {
        "polymarket": polymarket_url,
        "target_time_utc": start_time_utc,
        "expiration_time_utc": expiration_time_utc,
        "target_time_et": start_time_utc.astimezone(pytz.timezone('US/Eastern'))
    }


def get_available_market_urls(num_markets=12):
    """
    Returns a list of upcoming 15-minute Polymarket URLs starting from the current market.
    """
    current_market = get_current_market_urls()
    start_time_utc = current_market["target_time_utc"]
    expiration_time_utc = current_market["expiration_time_utc"]
    et_timezone = pytz.timezone("US/Eastern")
    markets = []

    for i in range(num_markets):
        market_start = start_time_utc + datetime.timedelta(minutes=15 * i)
        market_expiration = expiration_time_utc + datetime.timedelta(minutes=15 * i)
        markets.append(
            {
                "polymarket": generate_polymarket_url(market_start),
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
