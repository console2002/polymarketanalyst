import numpy as np


def normalize_risk_percent(risk_percent):
    if risk_percent is None:
        return 0.0
    if risk_percent <= 1:
        return risk_percent * 100.0
    return float(risk_percent)


def market_pnls_by_close(trades, market_pnl):
    market_last_timestamp = {}
    for trade in trades:
        market_id = trade["MarketID"]
        timestamp = trade["Timestamp"]
        if market_id not in market_last_timestamp or timestamp > market_last_timestamp[market_id]:
            market_last_timestamp[market_id] = timestamp

    ordered_markets = [
        market_id
        for market_id, _timestamp in sorted(market_last_timestamp.items(), key=lambda item: item[1])
    ]
    return [market_pnl.get(market_id, 0.0) for market_id in ordered_markets]


def market_outcome_stats(market_pnls):
    if not market_pnls:
        return {
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "expectancy_usd": 0.0,
            "worst_loss_usd": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
        }

    wins = [pnl for pnl in market_pnls if pnl > 0]
    losses = [pnl for pnl in market_pnls if pnl <= 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if losses else 0.0
    win_rate = len(wins) / len(market_pnls)
    loss_rate = len(losses) / len(market_pnls)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    worst_loss = min(market_pnls) if market_pnls else 0.0
    return {
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "expectancy_usd": expectancy,
        "worst_loss_usd": worst_loss,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
    }


def compute_r_metrics(initial_capital, risk_percent, avg_win_usd, avg_loss_usd, worst_loss_usd, expectancy_usd):
    risk_per_market_pct = normalize_risk_percent(risk_percent)
    r_usd = initial_capital * (risk_per_market_pct / 100.0) if initial_capital > 0 else 0.0
    if r_usd <= 0:
        return {
            "risk_per_market_pct": risk_per_market_pct,
            "r_usd": r_usd,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "worst_loss_r": 0.0,
            "expectancy_r": 0.0,
        }
    return {
        "risk_per_market_pct": risk_per_market_pct,
        "r_usd": r_usd,
        "avg_win_r": avg_win_usd / r_usd,
        "avg_loss_r": abs(avg_loss_usd) / r_usd,
        "worst_loss_r": abs(worst_loss_usd) / r_usd,
        "expectancy_r": expectancy_usd / r_usd,
    }


def compute_ulcer_index_pct(initial_capital, market_pnls):
    equity = initial_capital
    peak = equity
    drawdowns = []
    for pnl in market_pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            drawdown_pct = (peak - equity) / peak * 100.0
        else:
            drawdown_pct = 0.0
        drawdowns.append(drawdown_pct)
    if not drawdowns:
        return 0.0
    return float(np.sqrt(np.mean(np.square(drawdowns))))


def wins_between_losses_stats(market_pnls):
    wins_between = []
    wins_since_loss = 0
    for pnl in market_pnls:
        if pnl > 0:
            wins_since_loss += 1
        else:
            wins_between.append(wins_since_loss)
            wins_since_loss = 0

    count = len(wins_between)
    if count == 0:
        return {
            "count": 0,
            "min": 0,
            "median": 0.0,
            "p10": 0.0,
        }
    try:
        p10 = float(np.percentile(wins_between, 10, method="linear"))
    except TypeError:
        p10 = float(np.percentile(wins_between, 10, interpolation="linear"))
    return {
        "count": count,
        "min": min(wins_between),
        "median": float(np.median(wins_between)),
        "p10": p10,
    }


def pnl_distribution_stats(market_pnls):
    if not market_pnls:
        return {
            "pnl_std_usd": 0.0,
            "sharpe_per_market": 0.0,
        }
    pnl_array = np.array(market_pnls, dtype=float)
    pnl_std = float(np.std(pnl_array))
    pnl_mean = float(np.mean(pnl_array))
    sharpe = pnl_mean / pnl_std if pnl_std > 0 else 0.0
    return {
        "pnl_std_usd": pnl_std,
        "sharpe_per_market": sharpe,
    }
