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


def _rolling_window_dates(closed_trades, reference_time, window_size):
    reference_date = pd.Timestamp(reference_time).date()
    unique_dates = sorted(
        {
            trade["exit_time"].date()
            for trade in closed_trades
            if trade["exit_time"].date() < reference_date
        }
    )
    if not unique_dates:
        return []
    return unique_dates[-window_size:]


def _filter_trades_by_dates(closed_trades, allowed_dates, reference_time=None):
    if not allowed_dates:
        return []
    date_set = set(allowed_dates)
    reference_ts = pd.Timestamp(reference_time) if reference_time is not None else None
    filtered = []
    for trade in closed_trades:
        exit_time = trade["exit_time"]
        if exit_time.date() not in date_set:
            continue
        if reference_ts is not None and exit_time > reference_ts:
            continue
        filtered.append(trade)
    return filtered


def _filter_trades_since(closed_trades, start_time, reference_time=None):
    if start_time is None:
        return []
    start_ts = pd.Timestamp(start_time)
    reference_ts = pd.Timestamp(reference_time) if reference_time is not None else None
    return [
        trade
        for trade in closed_trades
        if trade["exit_time"] >= start_ts
        and (reference_ts is None or trade["exit_time"] <= reference_ts)
    ]


def summarize_profit_loss(closed_trades, reference_time, today_start_time=None):
    if not closed_trades:
        return {
            "today": np.nan,
            "week_to_date": np.nan,
            "month_to_date": np.nan,
            "all_time": np.nan,
        }

    reference_ts = pd.Timestamp(reference_time)
    today_start = pd.Timestamp(today_start_time) if today_start_time is not None else reference_ts.normalize()
    rolling_week_dates = _rolling_window_dates(closed_trades, reference_ts, 7)
    rolling_month_dates = _rolling_window_dates(closed_trades, reference_ts, 30)

    def _sum_trades(trades):
        return sum(trade["pnl_usd"] for trade in trades)

    return {
        "today": _sum_trades(_filter_trades_since(closed_trades, today_start, reference_ts)),
        "week_to_date": _sum_trades(_filter_trades_by_dates(closed_trades, rolling_week_dates, reference_ts)),
        "month_to_date": _sum_trades(_filter_trades_by_dates(closed_trades, rolling_month_dates, reference_ts)),
        "all_time": _sum_trades(_filter_trades_since(closed_trades, pd.Timestamp.min, reference_ts)),
    }


def summarize_drawdowns(closed_trades, reference_time, test_balance_start, today_start_time=None):
    if not closed_trades:
        return {
            "today": np.nan,
            "week_to_date": np.nan,
            "month_to_date": np.nan,
        }

    reference_ts = pd.Timestamp(reference_time)
    today_start = pd.Timestamp(today_start_time) if today_start_time is not None else reference_ts.normalize()
    rolling_week_dates = _rolling_window_dates(closed_trades, reference_ts, 7)
    rolling_month_dates = _rolling_window_dates(closed_trades, reference_ts, 30)

    def _max_drawdown(trades):
        if not trades:
            return np.nan
        period_trades = sorted(trades, key=lambda trade: trade["exit_time"])
        pnl_series = pd.Series([trade["pnl_usd"] for trade in period_trades])
        equity = test_balance_start + pnl_series.cumsum()
        equity = pd.concat([pd.Series([test_balance_start]), equity], ignore_index=True)
        rolling_peak = equity.cummax()
        rolling_peak = rolling_peak.replace(0, np.nan)
        drawdowns = (rolling_peak - equity) / rolling_peak
        return drawdowns.max()

    return {
        "today": _max_drawdown(_filter_trades_since(closed_trades, today_start, reference_ts)),
        "week_to_date": _max_drawdown(_filter_trades_by_dates(closed_trades, rolling_week_dates, reference_ts)),
        "month_to_date": _max_drawdown(_filter_trades_by_dates(closed_trades, rolling_month_dates, reference_ts)),
    }
