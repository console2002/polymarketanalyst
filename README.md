# Polymarket Analyst (v0.1)

A Python-based tool to monitor **Polymarket's 15-minute Bitcoin (BTC) Up/Down prediction markets**. It fetches real-time contract prices and compares them with the current Binance spot price to identify potential arbitrage opportunities or mispricings.

## Features

- **Automated Market Detection**: Automatically finds the currently active 15-minute BTC market.
- **Real-time Monitoring**: Fetches live contract prices from Polymarket's CLOB (Central Limit Order Book).
- **Strike Comparison**: Compares the market's strike price (Open Price) with the current Binance spot price.
- **Probability Analysis**: Displays real-time "Yes" (Up) and "No" (Down) contract prices.

## Installation

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install requests pytz
   ```

## Usage

Run the main bot script:

```bash
python arbitrage_bot.py
```

### Sample Output

```text
[16:58:24] Checking Polymarket...
MARKET INFO | Start: 2025-12-13 15:45:00+00:00 | Exp: 2025-12-13 16:00:00+00:00
PRICES      | Strike: $90,102.58 | Current: $90,089.43
CONTRACTS   | Up: $0.290 | Down: $0.730
STATUS      | Current is BELOW Strike by $13.15
```

## Logic Explained

The bot identifies the **Active Market** by looking for the 15-minute interval that has already **started** but not yet **expired**.
- **Start Time**: The beginning of the 15-minute candle.
- **Expiration**: The end of the 15-minute candle.
- **Strike Price**: The price of BTC at the **Start Time**.

Contracts pay out based on whether the price at **Expiration** is higher ("Up") or lower ("Down") than the **Strike Price**.
