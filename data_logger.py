import time
import datetime
import csv
import os
import math
import pytz
from fetch_current_polymarket import fetch_polymarket_data_struct

LOGGING_INTERVAL_SECONDS = 5
TIMEZONE_ET = pytz.timezone("US/Eastern")
TIMEZONE_UK = pytz.timezone("Europe/London")
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATE_FORMAT = "%d%m%Y"
SCRIPT_DIR = os.path.dirname(__file__)


def _format_timestamp(value, timezone):
    if not value:
        return ""
    if value.tzinfo is None:
        value = pytz.utc.localize(value)
    return value.astimezone(timezone).strftime(TIME_FORMAT)


def _get_data_file(timestamp_dt):
    if not timestamp_dt:
        timestamp_dt = datetime.datetime.now(pytz.utc)
    if timestamp_dt.tzinfo is None:
        timestamp_dt = pytz.utc.localize(timestamp_dt)
    date_str = timestamp_dt.astimezone(TIMEZONE_ET).strftime(DATE_FORMAT)
    return os.path.join(SCRIPT_DIR, f"{date_str}.csv")


def _ensure_csv(file_path):
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp",
                "Timestamp_UK",
                "TargetTime",
                "Expiration",
                "UpPrice",
                "DownPrice",
                "UpVol",
                "DownVol",
            ])
        print(f"Created {file_path}")


def _next_log_time(current_time, target_time):
    if not current_time:
        current_time = datetime.datetime.now(pytz.utc)
    if current_time.tzinfo is None:
        current_time = pytz.utc.localize(current_time)
    if not target_time:
        return current_time + datetime.timedelta(seconds=LOGGING_INTERVAL_SECONDS)
    if target_time.tzinfo is None:
        target_time = pytz.utc.localize(target_time)
    if current_time < target_time:
        return target_time
    elapsed_seconds = (current_time - target_time).total_seconds()
    steps = math.floor(elapsed_seconds / LOGGING_INTERVAL_SECONDS) + 1
    return target_time + datetime.timedelta(seconds=steps * LOGGING_INTERVAL_SECONDS)

def log_data(fetched_data=None, timestamp_dt=None):
    if fetched_data is None:
        fetched_data, err = fetch_polymarket_data_struct()
    else:
        err = None

    if timestamp_dt is None:
        timestamp_dt = fetched_data.get("polymarket_time_utc") if fetched_data else None
    if not timestamp_dt:
        timestamp_dt = datetime.datetime.now(pytz.utc)
    timestamp_et = _format_timestamp(timestamp_dt, TIMEZONE_ET)
    timestamp_uk = _format_timestamp(timestamp_dt, TIMEZONE_UK)
    
    # Check if data is complete
    data = None
    if fetched_data and \
       fetched_data.get('prices') is not None:
         data = fetched_data
    else:
        error_msg = err if err else "Missing fields"
        print(f"[{timestamp_et}] Error: Could not retrieve full data ({error_msg})")
        return

    # Extract values
    target_time = data.get('target_time_utc', '')
    expiration = data.get('expiration_time_utc', '')

    target_time_str = _format_timestamp(target_time, TIMEZONE_UK)
    expiration_str = _format_timestamp(expiration, TIMEZONE_UK)
    up_price = data['prices'].get('Up', 0.0)
    down_price = data['prices'].get('Down', 0.0)
    up_volume = data.get('volumes', {}).get('Up', 0.0)
    down_volume = data.get('volumes', {}).get('Down', 0.0)
    
    row = [
        timestamp_et,
        timestamp_uk,
        target_time_str,
        expiration_str,
        up_price,
        down_price,
        up_volume,
        down_volume,
    ]
    
    data_file = _get_data_file(timestamp_dt)
    _ensure_csv(data_file)

    with open(data_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
        
    print(f"[{timestamp_et}] Logged: Up={up_price}, Down={down_price}, UpVol={up_volume}, DownVol={down_volume}")

def main():
    print("Starting Data Logger...")
    
    while True:
        try:
            fetched_data, err = fetch_polymarket_data_struct()
            timestamp_dt = fetched_data.get("polymarket_time_utc") if fetched_data else None
            if not timestamp_dt:
                timestamp_dt = datetime.datetime.now(pytz.utc)
            target_time = fetched_data.get("target_time_utc") if fetched_data else None

            if err:
                print(f"[{_format_timestamp(timestamp_dt, TIMEZONE_ET)}] Error: {err}")
                time.sleep(LOGGING_INTERVAL_SECONDS)
                continue

            if target_time and timestamp_dt < target_time:
                sleep_seconds = (target_time - timestamp_dt).total_seconds()
                time.sleep(max(0, sleep_seconds))
                continue

            log_data(fetched_data=fetched_data, timestamp_dt=timestamp_dt)
            next_log_time = _next_log_time(timestamp_dt, target_time)
            sleep_seconds = (next_log_time - timestamp_dt).total_seconds()
            time.sleep(max(0, sleep_seconds))
        except KeyboardInterrupt:
            print("\nStopping logger...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(LOGGING_INTERVAL_SECONDS)   

if __name__ == "__main__":
    main()
