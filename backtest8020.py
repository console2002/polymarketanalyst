import pandas as pd
import datetime
import os
import re

from backtest_metrics import (
    compute_r_metrics,
    compute_ulcer_index_pct,
    market_outcome_stats,
    market_pnls_by_close,
    pnl_distribution_stats,
    wins_between_losses_stats,
)

DATA_FILE = "market_data.csv"
DATE_FILE_PATTERN = re.compile(r"^\d{8}\.csv$")
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
TIMEZONE_ET = "US/Eastern"
TIMEZONE_UK = "Europe/London"

# Global Configuration Variables
INITIAL_CAPITAL = 1000.0

class EightyTwentyStrategy:
    def __init__(self):
        self.MINUTES_AFTER_OPEN = 5
        self.ENTRY_THRESHOLD = 0.60
        self.EXIT_THRESHOLD = 0.80
        self.HOLD_TO_CLOSE_THRESHOLD = 0.79
        self.MAX_TRADE_SIZE = 50
        self.USE_RISK_PERCENT = True
        self.RISK_PERCENT = 0.01
        self.USE_LOTS = True
        self.FIXED_VALUE = 10

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
        if self.USE_RISK_PERCENT:
            risk_amount = current_capital * self.RISK_PERCENT
            target_qty = int(risk_amount / price) if price > 0 else 0
            quantity = min(target_qty, max_qty_by_capital)
        elif self.USE_LOTS:
            quantity = min(self.MAX_TRADE_SIZE, max_qty_by_capital)
        else:
            target_qty = int(self.FIXED_VALUE / price) if price > 0 else 0
            quantity = min(target_qty, max_qty_by_capital)

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
        self.risk_percent = None

    def load_data(self, file_path):
        self.load_data_files([file_path])

    def _normalize_latest_csv(self, raw_data):
        required_columns = {"timestamp_et", "target_time_uk", "expiration_uk", "outcome"}
        missing_columns = required_columns - set(raw_data.columns)
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns in latest CSV format: {missing_list}")

        data = raw_data.copy()
        data["Timestamp"] = (
            pd.to_datetime(data["timestamp_et"], format=TIME_FORMAT)
            .dt.tz_localize(TIMEZONE_ET)
            .dt.tz_convert("UTC")
        )
        data["TargetTime"] = (
            pd.to_datetime(data["target_time_uk"], format=TIME_FORMAT)
            .dt.tz_localize(TIMEZONE_UK)
            .dt.tz_convert("UTC")
        )
        data["Expiration"] = (
            pd.to_datetime(data["expiration_uk"], format=TIME_FORMAT)
            .dt.tz_localize(TIMEZONE_UK)
            .dt.tz_convert("UTC")
        )

        if "local_time_utc" in data.columns:
            data["LocalTimeUtc"] = pd.to_datetime(data["local_time_utc"], errors="coerce", utc=True)
        elif "server_time_utc" in data.columns:
            data["LocalTimeUtc"] = pd.to_datetime(data["server_time_utc"], errors="coerce", utc=True)
        else:
            data["LocalTimeUtc"] = data["Timestamp"]

        data["best_bid"] = pd.to_numeric(data.get("best_bid"), errors="coerce")
        data["best_ask"] = pd.to_numeric(data.get("best_ask"), errors="coerce")
        data["EffectivePrice"] = data["best_ask"].where(data["best_ask"] > 0, data["best_bid"])
        data["EffectivePrice"] = data["EffectivePrice"].fillna(0.0)

        outcome_raw = data["outcome"].astype(str).str.strip().str.lower()
        outcome_mapping = {"up": "Up", "yes": "Up", "down": "Down", "no": "Down"}
        data["OutcomeNormalized"] = outcome_raw.map(outcome_mapping).fillna(data["outcome"].astype(str).str.strip())

        data = data.sort_values(
            by=["Timestamp", "TargetTime", "Expiration", "OutcomeNormalized", "LocalTimeUtc"]
        )
        grouped = (
            data.groupby(["Timestamp", "TargetTime", "Expiration", "OutcomeNormalized"], as_index=False)
            .last()
        )
        pivoted = (
            grouped.pivot_table(
                index=["Timestamp", "TargetTime", "Expiration"],
                columns="OutcomeNormalized",
                values="EffectivePrice",
                aggfunc="last",
            )
            .reset_index()
        )
        pivoted.rename(columns={"Up": "UpPrice", "Down": "DownPrice"}, inplace=True)
        if "UpPrice" not in pivoted.columns:
            pivoted["UpPrice"] = 0.0
        if "DownPrice" not in pivoted.columns:
            pivoted["DownPrice"] = 0.0
        pivoted["UpPrice"] = pivoted["UpPrice"].fillna(0.0)
        pivoted["DownPrice"] = pivoted["DownPrice"].fillna(0.0)
        return pivoted

    def load_data_files(self, file_paths):
        missing_files = [file_path for file_path in file_paths if not os.path.exists(file_path)]
        if missing_files:
            missing_list = ", ".join(missing_files)
            raise FileNotFoundError(f"Data file(s) not found: {missing_list}")

        data_frames = [pd.read_csv(file_path) for file_path in file_paths]
        if not data_frames:
            raise ValueError("No data files provided for backtest.")

        self.market_data = pd.concat(data_frames, ignore_index=True)
        if "timestamp_et" in self.market_data.columns:
            self.market_data = self._normalize_latest_csv(self.market_data)
        else:
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

        self.market_data.sort_values(by='Timestamp', inplace=True)

        self.market_history = {}
        self.market_open_times = {}
        self.pending_market_summaries = {}

        for _, row in self.market_data.iterrows():
            market_id = (row['TargetTime'], row['Expiration'])
            if market_id not in self.market_history:
                self.market_history[market_id] = []
            self.market_history[market_id].append(row)

        for market_id, rows in self.market_history.items():
            self.market_open_times[market_id] = rows[0]['Timestamp']

        loaded_files = ", ".join(file_paths)
        print(f"Loaded {len(self.market_data)} data points from {loaded_files}")

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
            'entry_time': position.get('entry_time'),
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
        entry_times = []

        for res in resolved_positions_data:
            total_market_pnl += res['pnl']
            if res.get('entry_time') is not None:
                entry_times.append(res['entry_time'])
            if res['side'] == 'Up':
                total_up_shares += res['quantity']
                total_up_cost += res['quantity'] * res['entry_price']
            elif res['side'] == 'Down':
                total_down_shares += res['quantity']
                total_down_cost += res['quantity'] * res['entry_price']

        avg_up_price = total_up_cost / total_up_shares if total_up_shares > 0 else 0.0
        avg_down_price = total_down_cost / total_down_shares if total_down_shares > 0 else 0.0
        entry_time_display = "N/A"
        if entry_times:
            entry_time_display = min(entry_times).strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n--- Market Resolution Summary for {market_id_formatted} ---")
        print(f"Entry Time: {entry_time_display}")
        print(f"Up Shares: {total_up_shares}, Avg Entry Price: ${avg_up_price:.2f}")
        print(f"Down Shares: {total_down_shares}, Avg Entry Price: ${avg_down_price:.2f}")
        print(f"----Total PnL for market: ${total_market_pnl:.2f}----")

    def run_strategy(self, strategy_instance):
        self.risk_percent = getattr(strategy_instance, "RISK_PERCENT", None)
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
                            'entry_time': position.get('entry_time'),
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

        winning_trades = [t for t in closed_trades if t['PnL'] > 0]
        losing_trades = [t for t in closed_trades if t['PnL'] <= 0]
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        total_trades_count = len(buy_trades)
        strike_rate = (winning_trades_count / total_trades_count * 100) if total_trades_count > 0 else 0.0

        avg_win = sum(t['PnL'] for t in winning_trades) / winning_trades_count if winning_trades_count > 0 else 0.0
        avg_loss = (sum(abs(t['PnL']) for t in losing_trades) / losing_trades_count) if losing_trades_count > 0 else 0.0

        total_closed = len(closed_trades)
        win_rate = (winning_trades_count / total_closed) if total_closed > 0 else 0.0
        loss_rate = (losing_trades_count / total_closed) if total_closed > 0 else 0.0

        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        profit_factor = (
            sum(t['PnL'] for t in winning_trades) / sum(abs(t['PnL']) for t in losing_trades)
            if losing_trades_count > 0
            else 0.0
        )
        break_even_win_rate = (avg_loss / (avg_loss + avg_win)) if (avg_loss + avg_win) > 0 else 0.0

        sorted_trades = sorted(closed_trades, key=lambda t: t['Timestamp'])
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
            elif trade['Type'] == 'Sell':
                equity += trade['Value']
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

        worst_loss = min((t['PnL'] for t in closed_trades), default=0.0)
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

        market_pnls = market_pnls_by_close(closed_trades, market_pnl)
        total_return_pct = (
            (self.capital / self.initial_capital - 1.0) * 100.0 if self.initial_capital > 0 else 0.0
        )
        avg_return_per_market_pct = (
            total_return_pct / num_markets_played if num_markets_played > 0 else 0.0
        )
        market_stats = market_outcome_stats(market_pnls)
        r_metrics = compute_r_metrics(
            self.initial_capital,
            self.risk_percent,
            market_stats["avg_win_usd"],
            market_stats["avg_loss_usd"],
            market_stats["worst_loss_usd"],
            market_stats["expectancy_usd"],
        )
        recovery_factor = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0
        ulcer_index_pct = compute_ulcer_index_pct(self.initial_capital, market_pnls)
        wins_between = wins_between_losses_stats(market_pnls)
        pnl_stats = pnl_distribution_stats(market_pnls)

        print(f"Number of Buy Trades: {len(buy_trades)}")
        print(f"Number of Markets Traded: {num_markets_played}")
        print(f"Number of Markets Won: {num_markets_won}")
        print(f"Total Up Shares: {total_up_shares}")
        print(f"Total Down Shares: {total_down_shares}")
        print(f"Number of Winning Trades: {winning_trades_count}")
        print(f"Number of Losing Trades: {losing_trades_count}")
        print(f"Strike Rate: {strike_rate:.2f}%")
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
        print(f"Total Return (%): {total_return_pct:.2f}%")
        print(f"Avg Return per Market (%): {avg_return_per_market_pct:.2f}%")
        print(f"Risk per Market (%): {r_metrics['risk_per_market_pct']:.2f}%")
        print(f"R (USD): ${r_metrics['r_usd']:.2f}")
        print(f"Avg Win (R): {r_metrics['avg_win_r']:.3f}")
        print(f"Avg Loss (R): {r_metrics['avg_loss_r']:.3f}")
        print(f"Worst Loss (R): {r_metrics['worst_loss_r']:.3f}")
        print(f"Expectancy (R): {r_metrics['expectancy_r']:.3f}")
        print(f"Recovery Factor: {recovery_factor:.2f}")
        print(f"Ulcer Index (%): {ulcer_index_pct:.2f}%")
        print(f"Wins Between Losses Count: {wins_between['count']}")
        print(f"Wins Between Losses Min: {wins_between['min']}")
        print(f"Wins Between Losses Median: {wins_between['median']:.2f}")
        print(f"Wins Between Losses P10: {wins_between['p10']:.2f}")
        print(f"PnL Std Dev (USD): ${pnl_stats['pnl_std_usd']:.2f}")
        print(f"Sharpe per Market: {pnl_stats['sharpe_per_market']:.2f}")

if __name__ == "__main__":
    backtester = Backtester(initial_capital=INITIAL_CAPITAL)

    try:
        date_files = sorted(
            [file_name for file_name in os.listdir(".") if DATE_FILE_PATTERN.match(file_name)],
            key=lambda file_name: datetime.datetime.strptime(file_name.split(".")[0], "%d%m%Y"),
        )
        if date_files:
            backtester.load_data_files(date_files)
        else:
            backtester.load_data(DATA_FILE)
    except FileNotFoundError as e:
        print(e)
        exit()

    strategy = EightyTwentyStrategy()
    backtester.run_strategy(strategy)
    backtester.generate_report()
