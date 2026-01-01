import pandas as pd
import numpy as np


def build_trade_pnl_records(trade_records, trade_value_usd):
    closed_trades = []
    for record in trade_records:
        entry_price = record.get("entry_price")
        exit_price = record.get("exit_price")
        if entry_price is None or exit_price is None:
            continue
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        exit_timestamp = record.get("exit_time") or record.get("market_close_time")
        if exit_timestamp is None or pd.isna(exit_timestamp):
            continue
        pnl_usd = (exit_price - entry_price) * trade_value_usd
        closed_trades.append(
            {
                "exit_time": pd.Timestamp(exit_timestamp),
                "pnl_usd": pnl_usd,
            }
        )
    return closed_trades


def summarize_profit_loss(closed_trades, reference_time):
    if not closed_trades:
        return {
            "today": np.nan,
            "week_to_date": np.nan,
            "month_to_date": np.nan,
            "all_time": np.nan,
        }

    reference_time = pd.Timestamp(reference_time).normalize()
    start_of_week = reference_time - pd.Timedelta(days=reference_time.weekday())
    start_of_month = reference_time.replace(day=1)

    def _sum_since(start_time):
        return sum(
            trade["pnl_usd"]
            for trade in closed_trades
            if trade["exit_time"] >= start_time
        )

    return {
        "today": _sum_since(reference_time),
        "week_to_date": _sum_since(start_of_week),
        "month_to_date": _sum_since(start_of_month),
        "all_time": sum(trade["pnl_usd"] for trade in closed_trades),
    }


def summarize_drawdowns(closed_trades, reference_time, test_balance_start):
    if not closed_trades:
        return {
            "today": np.nan,
            "week_to_date": np.nan,
            "month_to_date": np.nan,
        }

    reference_time = pd.Timestamp(reference_time).normalize()
    start_of_week = reference_time - pd.Timedelta(days=reference_time.weekday())
    start_of_month = reference_time.replace(day=1)

    def _max_drawdown_since(start_time):
        period_trades = [
            trade for trade in closed_trades if trade["exit_time"] >= start_time
        ]
        if not period_trades:
            return np.nan
        period_trades = sorted(period_trades, key=lambda trade: trade["exit_time"])
        pnl_series = pd.Series([trade["pnl_usd"] for trade in period_trades])
        equity = test_balance_start + pnl_series.cumsum()
        equity = pd.concat([pd.Series([test_balance_start]), equity], ignore_index=True)
        rolling_peak = equity.cummax()
        rolling_peak = rolling_peak.replace(0, np.nan)
        drawdowns = (rolling_peak - equity) / rolling_peak
        return drawdowns.max()

    return {
        "today": _max_drawdown_since(reference_time),
        "week_to_date": _max_drawdown_since(start_of_week),
        "month_to_date": _max_drawdown_since(start_of_month),
    }
