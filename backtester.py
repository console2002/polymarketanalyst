import pandas as pd
import datetime
import os
from collections import deque

DATA_FILE = "market_data.csv"
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
TIMEZONE_ET = "US/Eastern"

# Global Configuration Variables
INITIAL_CAPITAL = 1000.0

class RebalancingStrategy:
    def __init__(self):
        # Parameters from plan
        self.SAFETY_MARGIN_M = 0.98
        self.MAX_TRADE_SIZE = 500
        self.MIN_BALANCE_QTY = 1

        # Portfolio state per market
        self.portfolio_state = {}  # key: market_id, value: {'qty_yes': int, 'qty_no': int, 'cost_yes': float, 'cost_no': float}

    def _get_or_init_portfolio(self, market_id):
        if market_id not in self.portfolio_state:
            self.portfolio_state[market_id] = {'qty_yes': 0, 'qty_no': 0, 'cost_yes': 0.0, 'cost_no': 0.0}
        return self.portfolio_state[market_id]

    def check_safety_margin(self, portfolio, target_side, qty_to_buy, price):
        qty_yes = portfolio['qty_yes']
        qty_no = portfolio['qty_no']
        cost_yes = portfolio['cost_yes']
        cost_no = portfolio['cost_no']

        avg_p_yes = cost_yes / qty_yes if qty_yes > 0 else 0
        avg_p_no = cost_no / qty_no if qty_no > 0 else 0
        
        new_combined_avg_p = -1

        if target_side == 'Up':  # 'Up' is YES
            new_cost_yes = cost_yes + qty_to_buy * price
            new_qty_yes = qty_yes + qty_to_buy
            if new_qty_yes == 0: return False
            new_avg_p_yes = new_cost_yes / new_qty_yes
            new_combined_avg_p = new_avg_p_yes + avg_p_no
        elif target_side == 'Down':  # 'Down' is NO
            new_cost_no = cost_no + qty_to_buy * price
            new_qty_no = qty_no + qty_to_buy
            if new_qty_no == 0: return False
            new_avg_p_no = new_cost_no / new_qty_no
            new_combined_avg_p = avg_p_yes + new_avg_p_no
        else:
            return False

        return new_combined_avg_p < self.SAFETY_MARGIN_M

    def decide(self, market_data_point, current_capital):
        market_id = (market_data_point['TargetTime'], market_data_point['Expiration'])
        portfolio = self._get_or_init_portfolio(market_id)

        qty_yes = portfolio['qty_yes']
        qty_no = portfolio['qty_no']

        # --- LOGIC FOR BALANCED PORTFOLIO (Increase Position) ---
        if qty_yes == qty_no:
            if qty_yes >= self.MAX_TRADE_SIZE:
                return None

            price_yes = market_data_point['UpPrice']
            price_no = market_data_point['DownPrice']

            # Only increase position if buying a pair is profitable
            if price_yes > 0 and price_no > 0 and (price_yes + price_no < self.SAFETY_MARGIN_M):
                side_to_buy = 'Up'
                price_to_buy = price_yes
                
                # Determine the max quantity we can possibly add
                qty_to_try = self.MAX_TRADE_SIZE - qty_yes
                
                # Search for the largest, safe, and affordable quantity to buy
                while qty_to_try > 0:
                    cost = qty_to_try * price_to_buy
                    if cost > current_capital:
                        qty_to_try -= 1
                        continue

                    if self.check_safety_margin(portfolio, side_to_buy, qty_to_try, price_to_buy):
                        # Found the optimal amount, return it
                        return (side_to_buy, qty_to_try, price_to_buy)

                    qty_to_try -= 1
            
            return None # Conditions not met to increase position

        # --- LOGIC FOR UNBALANCED PORTFOLIO (Rebalancing) ---
        quantity_delta = abs(qty_yes - qty_no)

        if quantity_delta < self.MIN_BALANCE_QTY:
            return None 

        price_yes = market_data_point['UpPrice']
        price_no = market_data_point['DownPrice']

        target_side = None
        target_price = 0.0

        if qty_yes > qty_no:
            target_side = 'Down'
            target_price = price_no
        else: # qty_no > qty_yes
            target_side = 'Up'
            target_price = price_yes
        
        if target_price <= 0:
            return None

        qty_to_buy = int(min(quantity_delta, self.MAX_TRADE_SIZE))

        while qty_to_buy > 0:
            cost = qty_to_buy * target_price
            if cost > current_capital:
                qty_to_buy -= 1
                continue

            if self.check_safety_margin(portfolio, target_side, qty_to_buy, target_price):
                return (target_side, qty_to_buy, target_price)

            qty_to_buy -= 1

        return None

    def update_portfolio(self, market_id, side, quantity, price):
        portfolio = self._get_or_init_portfolio(market_id)
        cost = quantity * price
        if side == 'Up':
            portfolio['qty_yes'] += quantity
            portfolio['cost_yes'] += cost
        elif side == 'Down':
            portfolio['qty_no'] += quantity
            portfolio['cost_no'] += cost

class Backtester:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transactions = [] # List of (timestamp, type, market_id, side, quantity, price, value, PnL)
        self.open_positions = [] # Changed to a list to allow multiple open positions per market
        self.market_data = pd.DataFrame()
        self.market_history = {} # Stores historical data grouped by market for resolution
        self.pending_market_summaries = {} # Key: market_id_tuple, Value: list of resolved_position_info dictionaries

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        self.market_data = pd.read_csv(file_path)
        
        # Convert relevant columns to datetime objects
        self.market_data['Timestamp'] = (
            pd.to_datetime(self.market_data['Timestamp'], format=TIME_FORMAT)
            .dt.tz_localize(TIMEZONE_ET)
            .dt.tz_convert('UTC')
        )
        self.market_data['TargetTime'] = (
            pd.to_datetime(self.market_data['TargetTime'], format=TIME_FORMAT)
            .dt.tz_localize(TIMEZONE_ET)
            .dt.tz_convert('UTC')
        )
        self.market_data['Expiration'] = (
            pd.to_datetime(self.market_data['Expiration'], format=TIME_FORMAT)
            .dt.tz_localize(TIMEZONE_ET)
            .dt.tz_convert('UTC')
        )
        
        # Sort data by timestamp to ensure chronological processing
        self.market_data.sort_values(by='Timestamp', inplace=True)

        # Group data by market identifier for easier lookup during resolution
        # A market is uniquely identified by its TargetTime and Expiration
        for _, row in self.market_data.iterrows():
            market_id = (row['TargetTime'], row['Expiration'])
            if market_id not in self.market_history:
                self.market_history[market_id] = []
            self.market_history[market_id].append(row)

        print(f"Loaded {len(self.market_data)} data points from {file_path}")

    def _resolve_single_position(self, market_id_tuple, position, current_timestamp):
        """Resolves a single expired market position and returns its PnL details."""
        if market_id_tuple not in self.market_history:
            # This should ideally not happen if data is loaded correctly
            # and market_id_tuple comes from an existing position
            return {'pnl': 0, 'winning_side': 'Error'} 

        market_specific_data = self.market_history[market_id_tuple]
        last_dp = market_specific_data[-1] # The final state of the market

        winning_side = None
        if last_dp['UpPrice'] == 0:  # If Up price is 0, then Up wins
            winning_side = 'Up'
        elif last_dp['DownPrice'] == 0: # If Down price is 0, then Down wins
            winning_side = 'Down'
        elif last_dp['DownPrice'] > last_dp['UpPrice']: # If Down price is 0, then Down wins
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
            'Value': position['quantity'] * position['entry_price'],
            'PnL': pnl,
            'WinningSide': winning_side
        })
        
        # Return details for aggregation
        return {
            'market_id': market_id_tuple,
            'side': position['side'],
            'quantity': position['quantity'],
            'entry_price': position['entry_price'],
            'pnl': pnl,
            'winning_side': winning_side
        }

    def _print_market_summary(self, market_id_tuple, resolved_positions_data):
        """Prints a consolidated summary for a fully resolved market."""
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
        
        for ts_np in unique_timestamps:
            current_timestamp = pd.to_datetime(ts_np)
            
            positions_to_remove_indices = []
            
            # Process open positions for expiration at current_timestamp or earlier
            for i, position in enumerate(self.open_positions):
                if current_timestamp >= position['expiration']:
                    market_id_tuple = position['market_id']
                    
                    resolved_info = self._resolve_single_position(market_id_tuple, position, current_timestamp)
                    
                    if market_id_tuple not in self.pending_market_summaries:
                        self.pending_market_summaries[market_id_tuple] = []
                    self.pending_market_summaries[market_id_tuple].append(resolved_info)
                    
                    positions_to_remove_indices.append(i)
            
            # Remove resolved positions from self.open_positions (iterate in reverse to avoid index issues)
            for index in sorted(positions_to_remove_indices, reverse=True):
                del self.open_positions[index]

            # Check for markets that are now fully resolved and print their summary
            # This logic is moved to the end of run_strategy to ensure all positions for all markets are processed.

            # Get all data points for the current timestamp
            current_data_points = self.market_data[self.market_data['Timestamp'] == current_timestamp]

            for _, row in current_data_points.iterrows():
                market_id_tuple = (row['TargetTime'], row['Expiration'])
                
                trade_decision = strategy_instance.decide(row, self.capital)
                
                if trade_decision:
                    side, quantity, entry_price = trade_decision
                    cost = quantity * entry_price

                    if self.capital >= cost:

                        self.capital -= cost

                        if hasattr(strategy_instance, 'update_portfolio'):
                            strategy_instance.update_portfolio(market_id_tuple, side, quantity, entry_price)

                        self.open_positions.append({ # Append to list
                            'market_id': market_id_tuple, # Store market_id explicitly
                            'side': side,
                            'quantity': quantity,
                            'entry_price': entry_price,
                            'expiration': row['Expiration']
                        })
                        self.transactions.append({
                            'Timestamp': current_timestamp,
                            'Type': 'Buy',
                            'MarketID': market_id_tuple,
                            'Side': side,
                            'Quantity': quantity,
                            'EntryPrice': entry_price,
                            'Value': cost,
                            'PnL': -cost # Initial PnL is the cost of investment
                        })
                        #print(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] BOUGHT {quantity} shares of {side} in market ({market_id_tuple[0].strftime('%Y-%m-%d %H:%M:%S')}, {market_id_tuple[1].strftime('%Y-%m-%d %H:%M:%S')}) at ${entry_price:.2f}. Capital: ${self.capital:.2f}")
                    else:
                        print(f"[{current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Insufficient capital to buy {quantity} shares of {side} in market ({market_id_tuple[0].strftime('%Y-%m-%d %H:%M:%S')}, {market_id_tuple[1].strftime('%Y-%m-%d %H:%M:%S')}) (Cost: ${cost:.2f}, Capital: ${self.capital:.2f})")
        
        # After iterating through all timestamps, resolve any remaining open positions
        # This will collect any positions that expired after the last recorded current_timestamp
        # or were still open at the end of the backtest data.
        for position in self.open_positions[:]: # Use a copy to allow modification
            market_id_tuple = position['market_id']
            resolved_info = self._resolve_single_position(market_id_tuple, position, current_timestamp) # Use the last current_timestamp or the position's expiration
            
            if market_id_tuple not in self.pending_market_summaries:
                self.pending_market_summaries[market_id_tuple] = []
            self.pending_market_summaries[market_id_tuple].append(resolved_info)
            self.open_positions.remove(position) # Remove from original list

        # Print summaries for any remaining markets in pending_market_summaries
        for market_id_tuple, resolutions_data in self.pending_market_summaries.items():
            self._print_market_summary(market_id_tuple, resolutions_data)
        self.pending_market_summaries.clear() # Clear after all summaries are printed

    def generate_report(self):
        print("\n--- Backtest Report ---")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital:   ${self.capital:.2f}")
        
        total_pnl = self.capital - self.initial_capital
        print(f"Total PnL:       ${total_pnl:.2f}")
        
        buy_trades = [t for t in self.transactions if t['Type'] == 'Buy']
        resolution_trades = [t for t in self.transactions if t['Type'] == 'Resolution']
        
        total_up_shares = sum(t['Quantity'] for t in buy_trades if t['Side'] == 'Up')
        total_down_shares = sum(t['Quantity'] for t in buy_trades if t['Side'] == 'Down')

        markets_played = set(t['MarketID'] for t in buy_trades)
        num_markets_played = len(markets_played)

        market_pnl = {}
        for t in resolution_trades:
            market_id = t['MarketID']
            pnl = t.get('PnL', 0.0)
            market_pnl.setdefault(market_id, 0.0)
            market_pnl[market_id] += pnl
        
        num_markets_won = sum(1 for pnl in market_pnl.values() if pnl > 0)

        winning_trades = [t for t in resolution_trades if t['PnL'] > 0]
        losing_trades = [t for t in resolution_trades if t['PnL'] <= 0]
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)

        avg_win = sum(t['PnL'] for t in winning_trades) / winning_trades_count if winning_trades_count > 0 else 0.0
        avg_loss = (sum(abs(t['PnL']) for t in losing_trades) / losing_trades_count) if losing_trades_count > 0 else 0.0

        total_trades_count = len(resolution_trades)
        win_rate = (winning_trades_count / total_trades_count) if total_trades_count > 0 else 0.0
        loss_rate = (losing_trades_count / total_trades_count) if total_trades_count > 0 else 0.0

        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        profit_factor = (
            sum(t['PnL'] for t in winning_trades) / sum(abs(t['PnL']) for t in losing_trades)
            if losing_trades_count > 0
            else 0.0
        )
        break_even_win_rate = (avg_loss / (avg_loss + avg_win)) if (avg_loss + avg_win) > 0 else 0.0

        sorted_trades = sorted(resolution_trades, key=lambda t: t['Timestamp'])
        current_loss_streak = 0
        max_consecutive_losses = 0
        for trade in sorted_trades:
            if trade['PnL'] <= 0:
                current_loss_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
            else:
                current_loss_streak = 0

        equity = self.initial_capital
        equity_curve = []
        for trade in sorted(self.transactions, key=lambda t: t['Timestamp']):
            if trade['Type'] == 'Buy':
                equity -= trade['Value']
            elif trade['Type'] == 'Resolution':
                payout = trade['PnL'] + trade['Value']
                if payout > 0:
                    equity += payout
            equity_curve.append(equity)

        max_drawdown_pct = 0.0
        max_drawdown_duration = 0
        if equity_curve:
            peak = equity_curve[0]
            current_duration = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                    current_duration = 0
                elif value < peak:
                    current_duration += 1
                    max_drawdown_duration = max(max_drawdown_duration, current_duration)
                if peak > 0:
                    drawdown = (peak - value) / peak
                    max_drawdown_pct = max(max_drawdown_pct, drawdown)
        max_drawdown_pct *= 100

        worst_loss = min((t['PnL'] for t in resolution_trades), default=0.0)
        worst_loss_pct = (abs(worst_loss) / self.initial_capital * 100) if self.initial_capital > 0 else 0.0

        pnl_series = [t['PnL'] for t in sorted_trades]

        def worst_rolling_sum(values, window):
            if len(values) < window:
                return None
            current_sum = sum(values[:window])
            min_sum = current_sum
            for i in range(window, len(values)):
                current_sum += values[i] - values[i - window]
                if current_sum < min_sum:
                    min_sum = current_sum
            return min_sum

        worst_5_trade = worst_rolling_sum(pnl_series, 5)
        worst_10_trade = worst_rolling_sum(pnl_series, 10)

        print(f"Number of Buy Trades: {len(buy_trades)}")
        print(f"Number of Markets Traded: {num_markets_played}")
        print(f"Number of Markets Won: {num_markets_won}")
        print(f"Total Up Shares: {total_up_shares}")
        print(f"Total Down Shares: {total_down_shares}")
        print(f"Number of Winning Trades: {winning_trades_count}")
        print(f"Number of Losing Trades: {losing_trades_count}")
        print(f"Max Drawdown: {max_drawdown_pct:.2f}%")
        print(f"Max Drawdown Duration (Trades): {max_drawdown_duration}")
        print(f"Avg Win (USD): ${avg_win:.2f}")
        print(f"Avg Loss (USD): ${avg_loss:.2f}")
        print(f"Worst Loss (USD): ${worst_loss:.2f}")
        print(f"Worst Loss (% of Bank): {worst_loss_pct:.2f}%")
        print(f"Max Consecutive Losses: {max_consecutive_losses}")
        print(f"Worst 5-Trade PnL (USD): {worst_5_trade:.2f}" if worst_5_trade is not None else "Worst 5-Trade PnL (USD): N/A")
        print(f"Worst 10-Trade PnL (USD): {worst_10_trade:.2f}" if worst_10_trade is not None else "Worst 10-Trade PnL (USD): N/A")
        print(f"Payoff Ratio: {payoff_ratio:.2f}")
        print(f"Expectancy (USD): ${expectancy:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Break-even Win Rate: {break_even_win_rate:.2%}")

        # Optional: Print transaction history
        # print("\n--- Transaction History ---")
        # for t in self.transactions:
        #     print(t)

        # --- DEBUGGING IMBALANCED MARKETS ---
        print("\n--- Imbalanced Market Analysis ---")
        market_shares = {}

        for trade in buy_trades:
            market_id = trade['MarketID']
            side = trade['Side']
            quantity = trade['Quantity']

            if market_id not in market_shares:
                market_shares[market_id] = {'Up': 0, 'Down': 0}
            
            market_shares[market_id][side] += quantity

        imbalanced_count = 0
        for market_id, shares in market_shares.items():
            if shares['Up'] != shares['Down']:
                imbalanced_count += 1
                market_id_formatted = f"({market_id[0].strftime('%Y-%m-%d %H:%M:%S')}, {market_id[1].strftime('%Y-%m-%d %H:%M:%S')})"
                print(f"Market {market_id_formatted} is imbalanced: Up={shares['Up']}, Down={shares['Down']}")
        
        if imbalanced_count == 0:
            print("All traded markets are balanced.")
        print("------------------------------------")

if __name__ == "__main__":
    # The user has indicated that data_logger.py is running in a separate process
    # and market_data.csv already contains data.
    # Therefore, we skip automatic data generation.


    backtester = Backtester(initial_capital=INITIAL_CAPITAL)
    
    try:
        backtester.load_data(DATA_FILE)
    except FileNotFoundError as e:
        print(e)
        exit()

    strategy = RebalancingStrategy() # Use the new strategy
    backtester.run_strategy(strategy)
    backtester.generate_report()

    # Clean up generated data file (optional)
    # os.remove(DATA_FILE)
    # print(f"Removed {DATA_FILE}")
