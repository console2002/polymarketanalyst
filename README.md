# Polymarket Analyst (v0.2)

A Python-based tool to monitor **Polymarket's 15-minute Bitcoin (BTC) Up/Down prediction markets**. It fetches real-time contract prices and compares them with the current Binance spot price to identify potential arbitrage opportunities or mispricings.

## Features

- **Automated Market Detection**: Automatically finds the currently active 15-minute BTC market.
- **Real-time Monitoring**: Fetches live contract prices from Polymarket's CLOB (Central Limit Order Book).
- **Strike Comparison**: Compares the market's strike price (Open Price) with the current Binance spot price.
- **Probability Analysis**: Displays real-time "Yes" (Up) and "No" (Down) contract prices.
- **Data Logging & Visualization**: Record market history and view it in an interactive dashboard.

## Installation

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install requests pytz streamlit pandas plotly
   ```

## Usage

### 1. Basic Monitor
Run the simple console bot:
```bash
python arbitrage_bot.py
```

### 2. Data Dashboard (New!)
To visualize market data over time:

**Step A: Start the Data Logger**
Open a terminal and run this to start collecting data to `market_data.csv`:
```bash
python data_logger.py
```

**Step B: Launch the Dashboard**
Open a **new** terminal and run:
```bash
python -m streamlit run dashboard.py
```

## Logic Explained

The bot identifies the **Active Market** by looking for the 15-minute interval that has already **started** but not yet **expired**.
- **Start Time**: The beginning of the 15-minute candle.
- **Expiration**: The end of the 15-minute candle.
- **Strike Price**: The price of BTC at the **Start Time**.

Contracts pay out based on whether the price at **Expiration** is higher ("Up") or lower ("Down") than the **Strike Price**.
