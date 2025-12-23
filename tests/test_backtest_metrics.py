import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backtest_metrics import (
    compute_r_metrics,
    compute_ulcer_index_pct,
    wins_between_losses_stats,
)


def test_wins_between_losses_stats():
    market_pnls = [10, -5, 7, 8, -3, -1, 4]
    stats = wins_between_losses_stats(market_pnls)
    assert stats["count"] == 3
    assert stats["min"] == 0
    assert stats["median"] == 1.0
    try:
        expected_p10 = np.percentile([1, 2, 0], 10, method="linear")
    except TypeError:
        expected_p10 = np.percentile([1, 2, 0], 10, interpolation="linear")
    assert stats["p10"] == expected_p10


def test_ulcer_index_pct():
    initial_capital = 100.0
    market_pnls = [10, -5, -5, 10]
    equity = initial_capital
    peak = equity
    drawdowns = []
    for pnl in market_pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdowns.append((peak - equity) / peak * 100.0)
    expected = float(np.sqrt(np.mean(np.square(drawdowns))))
    assert compute_ulcer_index_pct(initial_capital, market_pnls) == expected


def test_r_metrics_normalization():
    metrics = compute_r_metrics(
        initial_capital=1000.0,
        risk_percent=0.01,
        avg_win_usd=20.0,
        avg_loss_usd=10.0,
        worst_loss_usd=-30.0,
        expectancy_usd=5.0,
    )
    assert metrics["risk_per_market_pct"] == 1.0
    assert metrics["r_usd"] == 10.0
    assert metrics["avg_win_r"] == 2.0
    assert metrics["avg_loss_r"] == 1.0
    assert metrics["worst_loss_r"] == 3.0
    assert metrics["expectancy_r"] == 0.5
