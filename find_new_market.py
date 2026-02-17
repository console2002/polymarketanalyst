import datetime
import pytz

from market_profiles import default_market_profile_key, get_market_profile

# Base URL for Polymarket events
BASE_URL = "https://polymarket.com/event/"


def _market_profile(profile_key=None):
    selected_key = profile_key or default_market_profile_key()
    return get_market_profile(selected_key)


def _normalize_utc(target_time):
    """Return a timezone-aware UTC datetime."""

    if target_time.tzinfo is None:
        return pytz.utc.localize(target_time)
    return target_time.astimezone(pytz.utc)

def generate_slug(target_time):
    """
    Generates the Polymarket event slug for a given datetime.
    Format: bitcoin-up-or-down-[month]-[day]-[hour][am/pm]-et
    Example: bitcoin-up-or-down-november-26-1pm-et
    """
    # Ensure time is in Eastern Time
    et_tz = pytz.timezone('US/Eastern')
    if target_time.tzinfo is None:
        # Assume UTC if no timezone is provided, then convert to ET
        target_time = pytz.utc.localize(target_time).astimezone(et_tz)
    else:
        target_time = target_time.astimezone(et_tz)

    # Format components
    month = target_time.strftime("%B").lower()
    day = target_time.day
    
    # Hour formatting: 12-hour format with am/pm, lowercase, no leading zero for single digits
    hour_int = int(target_time.strftime("%I"))
    am_pm = target_time.strftime("%p").lower()
    
    slug = f"bitcoin-up-or-down-{month}-{day}-{hour_int}{am_pm}-et"
    return slug

def generate_interval_slug(target_time, slug_prefix):
    """Return a timestamp-based slug for a market interval."""

    normalized_time = _normalize_utc(target_time)
    timestamp = int(normalized_time.timestamp())
    return f"{slug_prefix}-{timestamp}"


def generate_profile_slug(target_time, profile_key=None):
    """
    Generates the Polymarket event slug for a configured market profile.
    Format: <slug_prefix>-[TIMESTAMP]
    The timestamp is the market start time (Unix timestamp).
    """
    profile = _market_profile(profile_key)

    return generate_interval_slug(target_time, profile.slug_prefix)


def generate_15m_slug(target_time):
    """Compatibility wrapper for legacy callers expecting 15-minute BTC slugs."""

    return generate_profile_slug(target_time, profile_key="btc_15m")

def generate_market_url_for_profile(target_time, profile):
    """Generate the full Polymarket URL for a resolved profile."""

    slug = generate_interval_slug(target_time, profile.slug_prefix)
    return f"{BASE_URL}{slug}"


def generate_market_url(target_time, profile_key=None):
    """
    Generates the full Polymarket URL for a given datetime.
    Detects if it should be an hourly or 15-minute market based on minutes?
    Actually, let's switch entirely to 15m markets as requested.
    """
    selected_profile_key = profile_key or default_market_profile_key()
    profile = _market_profile(selected_profile_key)
    return generate_market_url_for_profile(target_time, profile)

def get_next_market_urls(num_hours=5, profile_key=None, cadence_minutes=None):
    """
    Generates URLs for the next 'num_hours' 15-minute markets.
    """
    profile = _market_profile(profile_key)
    urls = []
    now = datetime.datetime.now(pytz.utc)
    
    # Start from the current 15-minute interval start.
    # Example: 12:07 -> 12:00 start
    # Example: 12:14 -> 12:00 start
    
    base_time = now.replace(second=0, microsecond=0)
    minutes = base_time.minute
    cadence = cadence_minutes or profile.window_minutes
    remainder = minutes % cadence
    current_quarter = base_time - datetime.timedelta(minutes=remainder)

    windows_per_hour = int(60 / cadence)
    for i in range(num_hours * windows_per_hour): # Fetch enough for X hours
        target_time = current_quarter + datetime.timedelta(minutes=cadence * i)
        urls.append(generate_market_url(target_time, profile_key=profile.key))
        
    return urls

def get_current_market_url(profile_key=None, cadence_minutes=None):
    """
    Determines the URL for the 'current' necessary market.
    Logic: The current 15-min market start.
    """
    profile = _market_profile(profile_key)
    now = datetime.datetime.now(pytz.utc)
    
    # Calculate current 15-minute interval start
    base_time = now.replace(second=0, microsecond=0)
    minutes = base_time.minute
    cadence = cadence_minutes or profile.window_minutes
    remainder = minutes % cadence
    current_quarter = base_time - datetime.timedelta(minutes=remainder)
    return generate_market_url(current_quarter, profile_key=profile.key)

def generate_urls_until_year_end():
    """
    Generates URLs for every hour from now until Jan 1, 2026.
    Saves them to 'market_urls_2025.txt'.
    """
    urls = []
    now = datetime.datetime.now(pytz.utc)
    
    # Start from the next full hour
    current_target = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    
    # End date: Jan 1, 2026 00:00 UTC (approx, depends on ET)
    # Let's just go until the year changes in ET
    et_tz = pytz.timezone('US/Eastern')
    
    print(f"Generating URLs starting from: {current_target.astimezone(et_tz)}")
    
    while True:
        # Check if we reached 2026 in ET
        et_time = current_target.astimezone(et_tz)
        if et_time.year >= 2026:
            break
            
        urls.append(generate_market_url(current_target))
        current_target += datetime.timedelta(hours=1)
        
    with open("market_urls_2025.txt", "w") as f:
        for url in urls:
            f.write(url + "\n")
            
    print(f"Generated {len(urls)} URLs and saved to 'market_urls_2025.txt'")

if __name__ == "__main__":
    print("--- Polymarket URL Generator ---")
    
    # Test with the user's specific example time to verify logic
    # User example: bitcoin-up-or-down-november-26-1pm-et
    # This corresponds to Nov 26, 1 PM ET.
    
    et_tz = pytz.timezone('US/Eastern')
    test_time = et_tz.localize(datetime.datetime(2025, 11, 26, 13, 0, 0))
    print(f"Test Time (ET): {test_time}")
    print(f"Generated URL: {generate_market_url(test_time)}")
    
    print("\n--- Current Market URL ---")
    print(f"Current Time (UTC): {datetime.datetime.now(pytz.utc)}")
    print(f"Current Market URL: {get_current_market_url()}")
    
    print("\n--- Generating URLs until 2026 ---")
    generate_urls_until_year_end()
