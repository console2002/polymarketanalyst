import time
import datetime
import csv
import os
from fetch_current_polymarket import fetch_polymarket_data_struct

DATA_FILE = "market_data.csv"

def init_csv():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "TargetTime", "Expiration", "Strike", "CurrentPrice", "UpPrice", "DownPrice"])
        print(f"Created {DATA_FILE}")

def log_data():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Fetching data...")
    
    data, err = fetch_polymarket_data_struct()
    
    if err:
        print(f"Error fetching data: {err}")
        return

    if not data:
        print("No data returned.")
        return

    # Extract values
    target_time = data.get('target_time_utc', '')
    expiration = data.get('expiration_time_utc', '')
    strike = data.get('price_to_beat')
    current_price = data.get('current_price')
    up_price = data['prices'].get('Up', 0.0)
    down_price = data['prices'].get('Down', 0.0)
    
    row = [timestamp, target_time, expiration, strike, current_price, up_price, down_price]
    
    with open(DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
        
    print(f"Logged: Strike={strike}, Current={current_price}, Up={up_price}, Down={down_price}")

def main():
    print("Starting Data Logger...")
    init_csv()
    
    while True:
        try:
            log_data()
            time.sleep(15) # Log every 15 seconds
        except KeyboardInterrupt:
            print("\nStopping logger...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
