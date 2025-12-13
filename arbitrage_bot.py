import time
import datetime
from fetch_current_polymarket import fetch_polymarket_data_struct

def check_market():
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] Checking Polymarket...")
    
    # Fetch Data
    poly_data, poly_err = fetch_polymarket_data_struct()
    
    if poly_err:
        print(f"Polymarket Error: {poly_err}")
        return
        
    if not poly_data:
        print("Missing data.")
        return

    # Polymarket Data extraction
    poly_strike = poly_data['price_to_beat']
    current_price = poly_data['current_price']
    poly_up_cost = poly_data['prices'].get('Up', 0.0)
    poly_down_cost = poly_data['prices'].get('Down', 0.0)
    target_time_utc = poly_data.get('target_time_utc')
    expiration_time_utc = poly_data.get('expiration_time_utc', 'N/A')
    
    if poly_strike is None:
        print("Polymarket Strike (Open Price) is None")
        # Proceeding to show other data if available
        str_strike = "N/A"
    else:
        str_strike = f"${poly_strike:,.2f}"

    if current_price is None:
        str_current = "N/A"
    else:
        str_current = f"${current_price:,.2f}"
        
    print(f"MARKET INFO | Start: {target_time_utc} | Exp: {expiration_time_utc}")
    print(f"PRICES      | Strike: {str_strike} | Current: {str_current}")
    print(f"CONTRACTS   | Up: ${poly_up_cost:.3f} | Down: ${poly_down_cost:.3f}")
    
    # Simple Analysis
    if poly_strike and current_price:
        diff = current_price - poly_strike
        direction = "ABOVE" if diff > 0 else "BELOW"
        print(f"STATUS      | Current is {direction} Strike by ${abs(diff):.2f}")

    print("-" * 50)

def main():
    print("Starting Polymarket Monitor (15m BTC)...")
    print("Press Ctrl+C to stop.")
    while True:
        try:
            check_market()
            # Wait 15 seconds to be polite and since candles don't move THAT fast
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
