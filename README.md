# Polymarket BTC Monitor

A Python-based tool to monitor **Polymarket's 15-minute Bitcoin (BTC) Up/Down prediction markets**. Features real-time data logging and an interactive dashboard for analyzing market behavior and identifying potential opportunities.

## Quickstart (Windows)

### 1) Install dependencies

Run the installer from the project root:

```bat
install.bat
```

### 2) Activate the virtual environment

Either activate it:

```bat
.\.venv\Scripts\activate
```

Or run Python directly from the venv in every command:

```bat
.\.venv\Scripts\python.exe --version
```

### 3) Start the logger

```bat
.\.venv\Scripts\python.exe data_logger.py --ui-stream
```

### 4) Start the GUI (in a new terminal)

IMPORTANT: Streamlit must be launched with `streamlit run`, not `python logger_gui.py`.
If Streamlit reports file in use, close any previous Streamlit processes (Task Manager or `taskkill /f /im streamlit.exe`).

```bat
.\.venv\Scripts\python.exe -m streamlit run logger_gui.py
```

Running `python logger_gui.py` directly will show a warning and won’t launch the Streamlit UI. Always use `streamlit run`.

### Logger only (no GUI)

```bat
.\.venv\Scripts\python.exe data_logger.py --ui-stream
```

### GUI + Logger (recommended)

Terminal 1:

```bat
.\.venv\Scripts\python.exe data_logger.py --ui-stream
```

Terminal 2:

```bat
.\.venv\Scripts\python.exe -m streamlit run logger_gui.py
```

## Features
Recent updates have brought significant improvements to both the backtesting capabilities and the data logging service, enhancing overall analysis and strategy development.

### Data Collection
- **Automated Market Detection**: Automatically finds the currently active 15-minute BTC market
- **Real-time Data Logging**: Continuously fetches and logs market data.
- **CSV Storage**: Historical data stored in daily CSV files (one per day) for analysis.

### Interactive Dashboard
- **Live Auto-Refresh**: Automatically updates periodically (every second) to ensure the latest data is displayed.
- **Advanced Visualization**:
  - Probability trends for Up/Down contracts
  - Market transition indicators (vertical lines)
  - Data point markers for clarity
  - Gaps for zero/missing values
- **Interactive Controls**:
  - Manual refresh button
  - Auto-refresh toggle
  - "Reset Zoom" - view all historical data
  - "Zoom Last 15m" - focus on recent activity (follows new data)
  - Scroll zoom and range slider

### Backtester
A script (`backtester.py`) that simulates the execution of a trading strategy on historical Polymarket data.

#### Strategy
The backtester now employs the `DerivativeMovingAverageStrategy`. This strategy analyzes the price movement by calculating a moving average of price derivatives.

- It tracks the `DERIVATIVE_MA_PERIOD` (e.g., 10 periods) moving average of price changes.
- If the derivative moving average for 'UpPrice' exceeds a `DERIVATIVE_THRESHOLD` (e.g., 0.005), it suggests an upward trend, and the strategy buys 'Up' shares.
- If the derivative moving average for 'DownPrice' falls below `-DERIVATIVE_THRESHOLD`, it suggests a downward trend, and the strategy buys 'Down' shares.
- Trades are considered on every `N_TICK_POINT`-th data point for a given market, allowing for less frequent trading.
- The backtester supports multiple open positions for the same market ID.

#### Winning Condition
A direction (Up or Down) is considered to have "won" if its own price is exactly $0 at the last available data point before the market's expiration date. Each share of the winning side pays out $1. If neither side's price reaches exactly $0, the trade is considered a loss for the position held.

#### Configuration
The backtester's behavior can be easily adjusted by modifying the following global variables at the beginning of `backtester.py`:
- `INITIAL_CAPITAL`: The starting capital for the simulation (default: $1000.0).
- `SHARE_SIZE`: The number of shares to buy in each trade (default: 2).
- `DERIVATIVE_MA_PERIOD`: The number of periods for calculating the moving average of the derivative (default: 10).
- `DERIVATIVE_THRESHOLD`: A small threshold (e.g., 0.005) used to filter out minor fluctuations in the derivative moving average.
- `N_TICK_POINT`: Determines trading frequency, e.g., trade on every N-th data point (default: 1).

#### Consolidated Market Summaries
Upon market resolution, the backtester now provides a single, consolidated summary for each market. This summary includes the total PnL, total shares traded for 'Up' and 'Down' sides, and their average entry prices, offering a clear overview of market performance rather than individual position resolutions.


#### How to Run Data Logger and GUI

**Step 0: Run setup**
Open a terminal and run:
```bat
install.bat
```

Then run the logger and GUI in separate terminals:
```bat
.\.venv\Scripts\python.exe data_logger.py
```
```bat
.\.venv\Scripts\python.exe -m streamlit run logger_gui.py
```

This will continuously fetch market data and save it to daily CSV files (one per day, named `DDMMYYYY.csv`). The GUI will open in your browser at `http://localhost:8501`.

#### How to Run backtester

Ensure `market_data.csv` contains historical data (generated by running `data_logger.py` for a sufficient period). Then, execute the backtester:
```bash
python backtester.py
```
The script will output a report including final capital, total PnL, and the number of winning and losing trades.

### Arbitrage Backtester
A new script (`backtester_arbitrage.py`) has been introduced to backtest arbitrage strategies. This backtester focuses on identifying and exploiting price discrepancies between different markets or contracts within Polymarket.

#### How to Run Arbitrage Backtester
Ensure `market_data.csv` contains historical data. Then, execute the arbitrage backtester:
```bash
python backtester_arbitrage.py
```
This script will analyze historical data for arbitrage opportunities and report on potential profits.

### Dashboard Tips
- Enable **Auto-refresh** to see updates in real-time as data is logged
- Use **Zoom Last 15m** to track the most recent market activity
- Hover over the charts to see exact values with the unified crosshair
- The charts are linked - zooming one automatically zooms the other

## What gets logged
The logger writes a row per outcome (Up and Down) each time it logs. Files are created per day using US/Eastern dates (e.g., `24032025.csv`).

| Column | Description |
| --- | --- |
| `timestamp_et` | Logger timestamp in US/Eastern (ET). |
| `timestamp_uk` | Logger timestamp in Europe/London. |
| `target_time_uk` | Market start time in UK time (15-minute window start). |
| `expiration_uk` | Market expiration time in UK time (15-minute window end). |
| `server_time_utc` | Polymarket server time (UTC) reported with the quote. |
| `local_time_utc` | Local logger time (UTC) when the row was written. |
| `stream_seq_id` | Sequence ID from the quote stream. |
| `token_id` | CLOB token ID for the outcome. |
| `outcome` | Outcome label (e.g., Up/Down). |
| `best_bid` | Current best bid price. |
| `best_ask` | Current best ask price. |
| `mid` | Midpoint of best bid and best ask. |
| `spread` | Best ask minus best bid. |
| `spread_pct` | Spread as a percentage of the mid. |
| `best_bid_size` | Size at the best bid. |
| `best_ask_size` | Size at the best ask. |
| `last_trade_price` | Most recent trade price (if available). |
| `last_trade_size` | Most recent trade size (if available). |
| `last_trade_side` | Most recent trade side (buy/sell). |
| `last_trade_ts` | Most recent trade timestamp (UTC, if available). |
| `heartbeat_last_seen` | Last heartbeat time from the stream (UTC). |
| `reconnect_count` | Number of reconnects since start. |
| `is_stale` | Whether the data is stale per the logger threshold. |
| `stale_age_seconds` | Age in seconds since the last update. |

## Daily file rotation
The logger writes to one CSV file per day using the US/Eastern date (format `DDMMYYYY.csv`). At midnight ET, it switches to a new file automatically.

## Default market selection
By default, the logger tracks the **current 15-minute market**: the contract that expires at the next 15-minute boundary. The GUI mirrors this behavior and auto-advances, but you can override the display in the sidebar without changing the logger (restart the logger to switch feeds).

## FAQ
**Why does the GUI say “Waiting for updates”?**  
Make sure the logger is running with `--ui-stream` and that the GUI is pointing to the same WebSocket URL (default `ws://127.0.0.1:8765`). Note that this is a WebSocket endpoint, not an HTTP URL.

**Why is the data marked stale?**  
If the stream pauses or Polymarket stops sending updates, the logger flags the row as stale once it exceeds the staleness threshold. This is informational and does not stop logging.

**Why do I see reconnects?**  
The WebSocket client will reconnect automatically on transient network issues. Reconnect counts are tracked in the CSV to help diagnose interruptions.

## How It Works

The system identifies the **Active Market** by finding the 15-minute interval that has started but not yet expired:

- **Start Time**: Beginning of the 15-minute candle
- **Expiration**: End of the 15-minute candle  
- **Contract Prices**: Live "Yes" (Up) and "No" (Down) prices from Polymarket's CLOB

Contracts pay out based on whether the price at **Expiration** is higher ("Up") or lower ("Down") than the **Strike Price**.

## Project Structure

```
PolymarketAnalyst/
├── backtester.py                 # Script to backtest trading strategies
├── backtester_arbitrage.py       # Script to backtest arbitrage strategies
├── backtest8020.py               # 80/20 backtest variant
├── backtest_metrics.py           # Backtest metrics helpers
├── data_logger.py                # Background data collection service
├── fetch_current_polymarket.py  # Core market data fetching logic
├── find_new_market.py           # Script to find new markets
├── get_current_markets.py       # Script to get current markets
├── logger_gui.py                # Streamlit GUI (connects to logger stream)
├── websocket_logger.py          # WebSocket logging client
├── .gitignore                   # Specifies intentionally untracked files to ignore
├── README.md                    # This file
└── *.csv                         # Daily historical data (auto-generated and used by the GUI)
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT License - feel free to use this project for your own analysis and trading strategies.

## Disclaimer

This tool is for informational and educational purposes only. Trading cryptocurrencies and prediction markets involves risk. Always do your own research and never invest more than you can afford to lose.
