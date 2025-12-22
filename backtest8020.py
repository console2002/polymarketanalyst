import pandas as pd
import datetime
import os

DATA_FILE = "market_data.csv"

# Global Configuration Variables
INITIAL_CAPITAL = 1000.0

class EightyTwentyStrategy:
    def __init__(self):
        self.MINUTES_AFTER_OPEN = 5
        self.ENTRY_THRESHOLD = 0.60
        self.EXIT_THRESHOLD = 0.80
        self.HOLD_TO_CLOSE_THRESHOLD = 0.79
        self.MAX_TRADE_SIZE = 50

    def decide_entry(self, market_data_point, market_open_time, current_capital):
        entry_time = market_open_time + datetime.timedelta(minutes=self.MINUTES_AFTER_OPEN)
        if market_data_point['Timestamp'] < entry_time:
            return None

        up_price = market_data_point['UpPrice']
        down_price = market_data_point['DownPrice']

        eligible_up = up_price >= self.ENTRY_THRESHOLD
        eligible_down = down_price >= self.ENTRY_THRESHOLD

        if not (eligible_up or eligible_down):
            return None

        if eligible_up and eligible_down:
            side = 'Up' if up_price >= down_price else 'Down'
            price = up_price if side == 'Up' else down_price
        else:
            side = 'Up' if eligible_up else 'Down'
            price = up_price if eligible_up else down_price

        max_qty_by_capital = int(current_capital / price) if price > 0 else 0
        quantity = min(self.MAX_TRADE_SIZE, max_qty_by_capital)

        if quantity <= 0:
            return None

        return (side, quantity, price)

    def should_sell(self, market_data_point, position):
        if position.get('hold_to_close'):
            return False
        price = market_data_point['UpPrice'] if position['side'] == 'Up' else market_data_point['DownPrice']
        return price >= self.EXIT_THRESHOLD

class Backtester:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transactions = []
        self.open_positions = {}
        self.market_data = pd.DataFrame()
        self.market_history = {}
        self.market_open_times = {}
        self.pending_market_summaries = {}

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        self.market_data = pd.read_csv(file_path)

        self.market_data['Timestamp'] = pd.to_datetime(self.market_data['Timestamp']).dt.tz_localize('UTC').dt.tz_convert('UTC')
        self.market_data['TargetTime'] = pd.to_datetime(self.market_data['TargetTime']).dt.tz_localize('UTC').dt.tz_convert('UTC')
        self.market_data['Expiration'] = pd.to_datetime(self.market_data['Expiration']).dt.tz_localize('UTC').dt.tz_convert('UTC')

        self.market_data.sort_values(by='Timestamp', inplace=True)

        for _, row in self.market_data.iterrows():
            market_id = (row['TargetTime'], row['Expiration'])
            if market_id not in self.market_history:
                self.market_history[market_id] = []
            self.market_history[market_id].append(row)

        for market_id, rows in self.market_history.items():
            self.market_open_times[market_id] = rows[0]['Timestamp']

        print(f"Loaded {len(self.market_data)} data points from {file_path}")

    def _resolve_single_position(self, market_id_tuple, position, current_timestamp):
        if market_id_tuple not in self.market_history:
            return {'pnl': 0, 'winning_side': 'Error'}

        market_specific_data = self.market_history[market_id_tuple]
        last_dp = market_specific_data[-1]

        winning_side = None
        if last_dp['UpPrice'] == 0:
            winning_side = 'Up'
        elif last_dp['DownPrice'] == 0:
            winning_side = 'Down'
        elif last_dp['DownPrice'] > last_dp['UpPrice']:
            winning_side = 'Down'
        else:
            winning_side = 'Up'

        pnl = 0
        if position['side'] == winning_side:
            pnl = position['quantity'] * (1 - position['entry_price'])
            self.capital += position['quantity']
        else:
            pnl = - (position['quantity'] * position['entry_price'])

        self.transactions.append({
            'Timestamp': current_timestamp,
            'Type': 'Resolution',
            'MarketID': market_id_tuple,
            'Side': position['side'],
            'Quantity': position['quantity'],
            'EntryPrice': position['entry_price'],
            'EntryTime': position.get('entry_time'),
            'Value': position['quantity'] * position['entry_price'],
            'PnL': pnl,
            'WinningSide': winning_side,
            'ExitTime': current_timestamp
        })

        return {
            'market_id': market_id_tuple,
            'side': position['side'],
            'quantity': position['quantity'],
            'entry_price': position['entry_price'],
            'pnl': pnl,
            'winning_side': winning_side
        }

    def _record_market_summary(self, market_id_tuple, resolved_info):
        if market_id_tuple not in self.pending_market_summaries:
            self.pending_market_summaries[market_id_tuple] = []
        self.pending_market_summaries[market_id_tuple].append(resolved_info)

    def _print_market_summary(self, market_id_tuple, resolved_positions_data):
        market_id_formatted = f"({market_id_tuple[0].strftime('%Y-%m-%d %H:%M:%S')}, {market_id_tuple[1].strftime('%Y-%m-%d %H:%M:%S')})"

        total_up_shares = 0
        total_up_cost = 0.0
        total_down_shares = 0
        total_down_cost = 0.0
        total_market_pnl = 0.0

        for res in resolved_positions_data:
            total_market_pnl += res['pnl']
            if res['side'] == 'Up':
                total_up_shares += res['quantity']
                total_up_cost += res['quantity'] * res['entry_price']
            elif res['side'] == 'Down':
                total_down_shares += res['quantity']
                total_down_cost += res['quantity'] * res['entry_price']

        avg_up_price = total_up_cost / total_up_shares if total_up_shares > 0 else 0.0
        avg_down_price = total_down_cost / total_down_shares if total_down_shares > 0 else 0.0

        print(f"\n--- Market Resolution Summary for {market_id_formatted} ---")
        print(f"Total PnL for market: ${total_market_pnl:.2f}")
        print(f"Up Shares: {total_up_shares}, Avg Entry Price: ${avg_up_price:.2f}")
        print(f"Down Shares: {total_down_shares}, Avg Entry Price: ${avg_down_price:.2f}")
        print("--------------------------------------------------")

    def run_strategy(self, strategy_instance):
        current_timestamp = None
        unique_timestamps = self.market_data['Timestamp'].unique()
        traded_windows = set()

        for ts_np in unique_timestamps:
            current_timestamp = pd.to_datetime(ts_np)
            current_data_points = self.market_data[self.market_data['Timestamp'] == current_timestamp]

            for _, row in current_data_points.iterrows():
                market_id_tuple = (row['TargetTime'], row['Expiration'])

                if market_id_tuple in self.open_positions:
                    position = self.open_positions[market_id_tuple]
                    if current_timestamp < position['expiration'] and strategy_instance.should_sell(row, position):
                        sell_price = row['UpPrice'] if position['side'] == 'Up' else row['DownPrice']
                        proceeds = position['quantity'] * sell_price
                        pnl = position['quantity'] * (sell_price - position['entry_price'])
                        self.capital += proceeds
                        self.transactions.append({
                            'Timestamp': current_timestamp,
                            'Type': 'Sell',
                            'MarketID': market_id_tuple,
                            'Side': position['side'],
                            'Quantity': position['quantity'],
                            'EntryPrice': position['entry_price'],
                            'EntryTime': position.get('entry_time'),
                            'ExitPrice': sell_price,
                            'ExitTime': current_timestamp,
                            'Value': proceeds,
                            'PnL': pnl,
                            'WinningSide': 'Sold'
                        })
                        self._record_market_summary(market_id_tuple, {
                            'market_id': market_id_tuple,
                            'side': position['side'],
                            'quantity': position['quantity'],
                            'entry_price': position['entry_price'],
                            'pnl': pnl,
                            'winning_side': 'Sold'
                        })
                        del self.open_positions[market_id_tuple]

                market_open_time = self.market_open_times[market_id_tuple]
                market_window_start = market_open_time.floor('15min')

                if market_window_start in traded_windows or market_id_tuple in self.open_positions:
                    continue

                trade_decision = strategy_instance.decide_entry(row, market_open_time, self.capital)
                if trade_decision:
                    side, quantity, entry_price = trade_decision
                    cost = quantity * entry_price
                    if self.capital >= cost:
                        hold_to_close = entry_price >= strategy_instance.HOLD_TO_CLOSE_THRESHOLD
                        self.capital -= cost
                        self.open_positions[market_id_tuple] = {
                            'market_id': market_id_tuple,
                            'side': side,
                            'quantity': quantity,
                            'entry_price': entry_price,
                            'entry_time': current_timestamp,
                            'expiration': row['Expiration'],
                            'hold_to_close': hold_to_close
                        }
                        self.transactions.append({
                            'Timestamp': current_timestamp,
                            'Type': 'Buy',
                            'MarketID': market_id_tuple,
                            'Side': side,
                            'Quantity': quantity,
                            'EntryPrice': entry_price,
                            'EntryTime': current_timestamp,
                            'Value': cost,
                            'PnL': -cost
                        })
                        traded_windows.add(market_window_start)

            positions_to_resolve = []
            for market_id_tuple, position in list(self.open_positions.items()):
                if current_timestamp >= position['expiration']:
                    resolved_info = self._resolve_single_position(market_id_tuple, position, current_timestamp)
                    self._record_market_summary(market_id_tuple, resolved_info)
                    positions_to_resolve.append(market_id_tuple)

            for market_id_tuple in positions_to_resolve:
                del self.open_positions[market_id_tuple]

        for position in list(self.open_positions.values()):
            market_id_tuple = position['market_id']
            resolved_info = self._resolve_single_position(market_id_tuple, position, current_timestamp)
            self._record_market_summary(market_id_tuple, resolved_info)
            del self.open_positions[market_id_tuple]

        for market_id_tuple, resolutions_data in self.pending_market_summaries.items():
            self._print_market_summary(market_id_tuple, resolutions_data)
        self.pending_market_summaries.clear()

    def generate_report(self):
        print("\n--- Backtest Report ---")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital:   ${self.capital:.2f}")

        total_pnl = self.capital - self.initial_capital
        print(f"Total PnL:       ${total_pnl:.2f}")

        buy_trades = [t for t in self.transactions if t['Type'] == 'Buy']
        closed_trades = [t for t in self.transactions if t['Type'] in ('Sell', 'Resolution')]

        total_up_shares = sum(t['Quantity'] for t in buy_trades if t['Side'] == 'Up')
        total_down_shares = sum(t['Quantity'] for t in buy_trades if t['Side'] == 'Down')

        markets_played = set(t['MarketID'] for t in buy_trades)
        num_markets_played = len(markets_played)

        market_pnl = {}
        for t in closed_trades:
            market_id = t['MarketID']
            pnl = t.get('PnL', 0.0)
            market_pnl.setdefault(market_id, 0.0)
            market_pnl[market_id] += pnl

        num_markets_won = sum(1 for pnl in market_pnl.values() if pnl > 0)

        winning_trades_count = sum(1 for t in closed_trades if t['PnL'] > 0)
        losing_trades_count = sum(1 for t in closed_trades if t['PnL'] <= 0)

        print(f"Number of Buy Trades: {len(buy_trades)}")
        print(f"Number of Markets Traded: {num_markets_played}")
        print(f"Number of Markets Won: {num_markets_won}")
        print(f"Total Up Shares: {total_up_shares}")
        print(f"Total Down Shares: {total_down_shares}")
        print(f"Number of Winning Trades: {winning_trades_count}")
        print(f"Number of Losing Trades: {losing_trades_count}")

if __name__ == "__main__":
    backtester = Backtester(initial_capital=INITIAL_CAPITAL)

    try:
        backtester.load_data(DATA_FILE)
    except FileNotFoundError as e:
        print(e)
        exit()

    strategy = EightyTwentyStrategy()
    backtester.run_strategy(strategy)
    backtester.generate_report()
