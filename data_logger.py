import time
import datetime
import csv
import os
import pytz
from fetch_current_polymarket import fetch_polymarket_data_struct

DATA_FILE = "market_data.csv"
LOOGING_INTERVAL_SECONDS = 3
TIMEZONE_ET = pytz.timezone("US/Eastern")
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"


def _format_et_timestamp(value):
    if not value:
        return ""
    if value.tzinfo is None:
        value = pytz.utc.localize(value)
    return value.astimezone(TIMEZONE_ET).strftime(TIME_FORMAT)


def init_csv():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "TargetTime", "Expiration", "UpPrice", "DownPrice", "UpVol", "DownVol"])
        print(f"Created {DATA_FILE}")

def log_data():
    fetched_data, err = fetch_polymarket_data_struct()

    timestamp_dt = fetched_data.get("polymarket_time_utc") if fetched_data else None
    if not timestamp_dt:
        timestamp_dt = datetime.datetime.now(pytz.utc)
    timestamp = _format_et_timestamp(timestamp_dt)
    
    # Check if data is complete
    data = None
    if fetched_data and \
       fetched_data.get('prices') is not None:
         data = fetched_data
    else:
        error_msg = err if err else "Missing fields"
        print(f"[{timestamp}] Error: Could not retrieve full data ({error_msg})")
        return

    # Extract values
    target_time = data.get('target_time_utc', '')
    expiration = data.get('expiration_time_utc', '')

    target_time_str = _format_et_timestamp(target_time)
    expiration_str = _format_et_timestamp(expiration)
    up_price = data['prices'].get('Up', 0.0)
    down_price = data['prices'].get('Down', 0.0)
    up_volume = data.get('volumes', {}).get('Up', 0.0)
    down_volume = data.get('volumes', {}).get('Down', 0.0)
    
    row = [timestamp, target_time_str, expiration_str, up_price, down_price, up_volume, down_volume]
    
    with open(DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
        
    print(f"[{timestamp}] Logged: Up={up_price}, Down={down_price}, UpVol={up_volume}, DownVol={down_volume}")

def main():
    print("Starting Data Logger...")
    init_csv()
    
    while True:
        try:
            log_data()
            time.sleep(LOOGING_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nStopping logger...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(LOOGING_INTERVAL_SECONDS)   

if __name__ == "__main__":
    main()
